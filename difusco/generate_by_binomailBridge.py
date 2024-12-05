import torch
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from scipy.optimize import bisect
from utils.utils import restore_checkpoint_withEval as restore_checkpoint
from models.gnn_encoder import GNNEncoder
from models.ema import ExponentialMovingAverage
from co_datasets.tsp_graph_dataset import TSPGraphDataset
import imageio
import io

# Define constants
tEnd = 15
T = 1000

def f(x):
    return np.log(x/(1-x))

xEnd = np.exp(-tEnd)
fGrid = np.linspace(-f(xEnd), f(xEnd), T)
xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(T)])
observationTimes = -np.log(xGrid)    

# Analytically derived reverse-time transition rate
num_states = 2  # state space spans only zero and one
brTable = np.zeros((num_states, num_states, T))
for tIndex in range(T):
    p = np.exp(-observationTimes[tIndex])
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
        solArray[tIndex + 1, :, initial_condition] = binom(initial_condition, p).pmf(support)

# Analytical forward solution, CDF
cumSolArray = np.zeros_like(solArray)
for i in range(T+1):
    for j in range(num_states):
        cumSolArray[i, :, j] = np.cumsum(solArray[i, :, j])

# Move arrays to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cumSolArrayGPU = torch.from_numpy(cumSolArray).to(device)
brTableGPU = torch.from_numpy(np.ravel(brTable)).to(device) # flatten
observationTimesGPU = torch.from_numpy(observationTimes).to(device)

eobservationTimes = np.hstack([0, observationTimes])
ps = np.exp(-eobservationTimes[:-1])
pt = np.exp(-eobservationTimes[1:])
samplingProb = np.ones_like(pt)
samplingProb /= np.sum(samplingProb)
pi = samplingProb
weights = pt/pi * (eobservationTimes[1:] - eobservationTimes[:-1])
weightsGPU = torch.from_numpy(weights).to(device)

# Initialize the model
model = GNNEncoder(
    n_layers=12,
    hidden_dim=256,
    out_channels=1,
    aggregation="sum",
    sparse=False,
    use_activation_checkpoint=False,
    node_feature_only=False,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
state = {'model': model, 'optimizer': optimizer, 'ema': ema, 'step': 0, 'lossHistory': [], 'evalLossHistory': []}

# Restore the checkpoint
checkpoint_dir = "./checkpoints"
checkpoint_meta_dir = os.path.join(checkpoint_dir, "checkpoint_epoch_16.pth")
state = restore_checkpoint(checkpoint_meta_dir, state, device)
ema.copy_to(model.parameters())

# Load the validation dataset
val_dataset = TSPGraphDataset(
    data_file=os.path.join("./", "tsp50_test_concorde.txt"),
    sparse_factor=-1,
)

# Define the sampling function
def sample_adjacency_matrix(points, adj_matrix_0, T, generation='binomial', save_gif=True, sample_idx=0):
    with torch.no_grad():
        adj_matrix_t = torch.zeros_like(adj_matrix_0).to(device)  # Start with a zero matrix
        batch_size, num_nodes, _ = adj_matrix_0.shape
        frames = []

        skip = 1
        
        for i in range(T-skip, -1, -skip):
            t = eobservationTimes[i+skip]
            s = eobservationTimes[i]

            pt = np.exp(-t)
            ps = np.exp(-s)    
            pp = (ps-pt)/(1-pt)
            
            dn = torch.sigmoid(model(
                points.to(device),  # Shape: (batch_size, num_nodes, num_features)
                torch.full((batch_size,), i, dtype=torch.float32, device=device),  # Shape: (batch_size,)
                (adj_matrix_t).float().to(device),  # Shape: (batch_size, num_nodes, num_nodes)
                None  # If there is an additional input, handle it here
            ))

            dn = dn.squeeze(1) 
            
            if generation=='binomial':
                dn_np = np.clip(np.round((dn).detach().cpu().numpy()).astype('int'), 0, 1 - adj_matrix_t.cpu().numpy()).astype('int')                      
                print("dn_np min max", dn_np.min(), dn_np.max())
                draw_np = binom(dn_np, pp).rvs()
                
            elif generation=='poisson':
                draw_np = poisson(dn.detach().cpu().numpy() * pp).rvs() # why pp??
                
            else:
                raise NotImplementedError(f'Sampling method is not implemented.')

            print(i, dn.min().item(), dn.max().item(), draw_np.min(), draw_np.max(), pp)
            adj_matrix_t = torch.clip(adj_matrix_t + torch.from_numpy(draw_np).to(device), 0, 1)
            
            if save_gif:
                # Generate the graph image and append it to the frames list
                fig, ax = plt.subplots(figsize=(5, 5))
                adj_matrix_t_np = adj_matrix_t.cpu().numpy()[0]
                G = nx.Graph()
                for node in range(len(points[0])):
                    G.add_node(node, pos=(points[0][node, 0], points[0][node, 1]))
                for node in range(len(points[0])):
                    for j in range(len(points[0])):
                        if adj_matrix_t_np[node, j] > 0:
                            G.add_edge(node, j)
                pos = nx.get_node_attributes(G, 'pos')
                nx.draw(G, pos, with_labels=True, node_size=50, node_color='lightblue', edge_color='gray', ax=ax)
                
                # Save plot to a bytes buffer instead of a file
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                frames.append(imageio.imread(buf))

        # Create a GIF from the frames
        if save_gif:
            gif_filename = f'recovered_adj_matrix_sample_{sample_idx}.gif'
            imageio.mimsave(gif_filename, frames, duration=0.1)
            print(f"Saved GIF of recovered adjacency matrix evolution as {gif_filename}")

    return adj_matrix_t

def plot_graph(points, adj_matrix, filename):
    """Plot and save the graph."""
    G = nx.Graph()
    num_nodes = len(points)
    for i in range(num_nodes):
        G.add_node(i, pos=(points[i, 0], points[i, 1]))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray')
    plt.savefig(filename)
    plt.close()

# Perform the reverse diffusion for each sample in the validation dataset
for idx, data in enumerate(val_dataset):
    points, adj_matrix_0 = data[1], data[2]
    points = points.unsqueeze(0)
    adj_matrix_0 = adj_matrix_0.unsqueeze(0)  # Add batch dimension

    print(f"\nProcessing sample {idx + 1}/{len(val_dataset)}")
    
    # Perform reverse diffusion to recover adjacency matrix
    recovered_adj_matrix = sample_adjacency_matrix(points, adj_matrix_0, T, sample_idx=idx, save_gif=False)  

    # Convert tensors to numpy arrays for plotting
    adj_matrix_0_np = adj_matrix_0.cpu().numpy()[0]
    recovered_adj_matrix_np = recovered_adj_matrix.cpu().numpy()[0]

    # Plot and save original and recovered adjacency matrices
    plot_graph(points[0].numpy(), adj_matrix_0.cpu().numpy()[0], f'original_adj_matrix_{idx + 1}.png')
    plot_graph(points[0].numpy(), recovered_adj_matrix.cpu().numpy()[0], f'recovered_adj_matrix_{idx + 1}.png')

    print(f"Saved original and recovered adjacency matrices for sample {idx + 1} as PNG images.")
