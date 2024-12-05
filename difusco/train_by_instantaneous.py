import os
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.stats import binom
from scipy.optimize import bisect
import gc
from models.ema import ExponentialMovingAverage
from models.gnn_encoder import GNNEncoder
from utils.utils import save_checkpoint_withEval as save_checkpoint
from utils.utils import restore_checkpoint_withEval as restore_checkpoint
from co_datasets.tsp_graph_dataset import TSPGraphDataset
from tqdm import tqdm

# Load the train and validation datasets
train_dataset = TSPGraphDataset(
    data_file=os.path.join("./", "tsp50_train_concorde.txt"),
    sparse_factor=-1,
)

val_dataset = TSPGraphDataset(
    data_file=os.path.join("./", "tsp50_test_concorde.txt"),
    sparse_factor=-1,
)

# Define constants
tEnd = 15.
T = 1000

def f(x):
    return np.log(x/(1-x))

xEnd = np.exp(-tEnd)
fGrid = np.linspace(-f(xEnd), f(xEnd), T)
xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(T)])
observationTimes = -np.log(xGrid)    

# Analytically derived reverse-time transition rate
num_states = 2 # state space spans only zero and one
brTable = np.zeros((num_states, num_states, T))
for tIndex in range(T):
    for n in range(num_states):
        for m in range(n):
            brTable[n, m, tIndex] = n-m 

# Analytical forward solution, PDF
support = np.arange(0, num_states)
solArray = np.zeros((T+1, num_states, num_states))
solArray[0,:,:] = np.eye(num_states)

for tIndex in range(T):
    p = np.exp(-observationTimes[tIndex])
    for initial_condition in range(num_states):
        solArray[tIndex + 1, :, initial_condition] =  binom(initial_condition, p).pmf(support)    
        
# Analytical forward solution, CDF
cumSolArray = np.zeros_like(solArray)

for i in range(T+1):
    for j in range(num_states):
        cumSolArray[i, :, j] = np.cumsum(solArray[i, :, j])    

# Move arrays to GPU
device = "cuda"
cumSolArrayGPU = torch.from_numpy(cumSolArray).to(device)
brTableGPU = torch.from_numpy(np.ravel(brTable)).to(device) # flatten
observationTimeGPU = torch.from_numpy(observationTimes).to(device)

eobservationTimes = np.hstack([0, observationTimes])

ps = np.exp(-eobservationTimes[:-1])
pt = np.exp(-eobservationTimes[1:])

samplingProb = np.ones_like(pt)
samplingProb /= np.sum(samplingProb)

pi = samplingProb

weights = pt/pi * (eobservationTimes[1:]-eobservationTimes[:-1])
weightsGPU = torch.from_numpy(weights).to(device)

def diffusion_forward_process(x_0):
    with torch.no_grad():
        batch_size, node_size, _ = x_0.shape
        tIndex = torch.multinomial(torch.ones(T, device=device), batch_size, replacement=True).reshape(batch_size, 1, 1)

        cp = cumSolArrayGPU[(tIndex + 1), :, x_0].to(device)
        u = torch.rand((batch_size, node_size, node_size, 1), device=device)
        x_t = torch.argmax((u < cp).to(torch.int), dim=-1).view(batch_size, node_size, node_size)

        birthRate_t = brTableGPU[(x_0 * num_states * T + x_t * T + tIndex)]
        p = torch.exp(-observationTimeGPU[tIndex])

        return x_t, birthRate_t, tIndex[:, 0]

# Initialize model, optimizer, and EMA
model = GNNEncoder(
    n_layers = 12,
    hidden_dim = 256,
    out_channels = 1,
    aggregation = "sum",
    sparse = False,
    use_activation_checkpoint = False,
    node_feature_only = False,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Set batch size and other configurations
batch_size = 96
initial_step = 0
lossHistory = []
evalLossHistory = []

checkpoint_dir = "./checkpoints"

# Restore checkpoint if available
state = {
    'model': model,
    'optimizer': optimizer,
    'ema': ema,
    'step': initial_step,
    'lossHistory': lossHistory,
    'evalLossHistory': evalLossHistory,
}

# state = restore_checkpoint(os.path.join(checkpoint_dir, f'checkpoint.pth'), state, device)

initial_step = state['step']
lossHistory = state['lossHistory']
evalLossHistory = state['evalLossHistory']

def val():
    model.eval()
    """Evaluates the model on the entire validation set."""
    total_eval_loss = 0.0
    ema.store(state['model'].parameters())
    ema.copy_to(state['model'].parameters())

    for start_idx in range(0, len(val_dataset), batch_size):
        with torch.no_grad():
            end_idx = min(start_idx + batch_size, len(val_dataset))
            batch_data = [val_dataset[idx] for idx in range(start_idx, end_idx)]
            
            points_batch = torch.stack([data[1] for data in batch_data]).to(device)
            adj_matrix_0_batch = torch.stack([data[2] for data in batch_data]).to(device)
            adj_matrix_t_batch, birthRate_t_batch, tIndex_batch = diffusion_forward_process(adj_matrix_0_batch)
            
            y_batch = torch.sigmoid(model(
                points_batch, 
                tIndex_batch.float().view(adj_matrix_0_batch.shape[0]), 
                adj_matrix_t_batch.float(), 
                None
            ))
            y_batch = y_batch.squeeze(1) 
            
            # eval_loss = torch.mean(weightsGPU[tIndex_batch].reshape([-1, 1, 1]) * (y_batch - birthRate_t_batch * torch.log(y_batch)))
            eval_loss = (y_batch - birthRate_t_batch * torch.log(y_batch)).mean()
            total_eval_loss += eval_loss.item() * len(batch_data)

    ema.restore(state['model'].parameters())
    
    avg_eval_loss = total_eval_loss / len(val_dataset)
    evalLossHistory.append(avg_eval_loss)
    model.train()
    return avg_eval_loss

num_epochs = 50  # Define the number of epochs
steps_per_epoch = len(train_dataset) // batch_size

for epoch in range(num_epochs):
    total_train_loss = 0.0  # Initialize total training loss for the epoch

    # Training loop over the entire dataset for the current epoch
    for batch_start in tqdm(range(0, len(train_dataset), batch_size)):
        batch_end = min(batch_start + batch_size, len(train_dataset))
        batch_indices = range(batch_start, batch_end)
        batch_data = [train_dataset[idx] for idx in batch_indices]
        
        points_batch = torch.stack([data[1] for data in batch_data]).to(device)
        adj_matrix_0_batch = torch.stack([data[2] for data in batch_data]).to(device)
        
        adj_matrix_t_batch, birthRate_t_batch, tIndex_batch = diffusion_forward_process(adj_matrix_0_batch)
        optimizer.zero_grad()
    
        y_batch = torch.sigmoid(model(
            points_batch, 
            tIndex_batch.float().view(adj_matrix_0_batch.shape[0]), 
            adj_matrix_t_batch.float(), 
            None)) # [batch_size, 1, node_size, node_size])
        
        y_batch = y_batch.squeeze(1) 
    
        # loss = torch.mean(weightsGPU[tIndex_batch].reshape(-1, 1, 1) * (y_batch - birthRate_t_batch * torch.log(y_batch)))
        loss = (y_batch - birthRate_t_batch * torch.log(y_batch)).mean()
        # print("train", loss[0].item(), tIndex_batch[0].item())
        loss.backward()
    
        state['ema'].update(state['model'].parameters())
        
        optimizer.step()
        
        total_train_loss += loss.item() * len(batch_data)  # Accumulate the loss
    
    # Calculate average training loss for the epoch
    avg_train_loss = total_train_loss / len(train_dataset)

    # At the end of each epoch, evaluate the model and save a checkpoint
    avg_eval_loss = val()
    print(f'Epoch: {epoch+1}, Avg Train Loss: {avg_train_loss}, Eval Loss: {avg_eval_loss}')
    
    state['step'] = epoch + 1
    state['lossHistory'] = lossHistory
    state['evalLossHistory'] = evalLossHistory

    # Save checkpoint after every epoch
    save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'), state)
    
    scheduler.step()

    gc.collect()
    torch.cuda.empty_cache()