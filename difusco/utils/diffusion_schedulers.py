"""Schedulers for Denoising Diffusion Probabilistic Models"""

import math

import numpy as np
import torch
from scipy.optimize import bisect
from scipy.stats import binom


class GaussianDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    self.betabar = np.cumprod(self.beta)
    self.alpha = np.concatenate((np.array([1.0]), 1 - self.beta))
    self.alphabar = np.cumprod(self.alpha)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0, t):
    # Select noise scales
    noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))
    atbar = torch.from_numpy(self.alphabar[t]).view(noise_dims).to(x0.device)
    assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'

    # Sample noise and add to x0
    epsilon = torch.randn_like(x0)
    xt = torch.sqrt(atbar) * x0 + torch.sqrt(1.0 - atbar) * epsilon
    return xt, epsilon


class CategoricalDiffusion(object):
  """Gaussian Diffusion process with linear beta scheduling"""

  def __init__(self, T, schedule):
    # Diffusion steps
    self.T = T

    # Noise schedule
    if schedule == 'linear':
      b0 = 1e-4
      bT = 2e-2
      self.beta = np.linspace(b0, bT, T)
    elif schedule == 'cosine':
      self.alphabar = self.__cos_noise(np.arange(0, T + 1, 1)) / self.__cos_noise(
          0)  # Generate an extra alpha for bT
      self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)

    beta = self.beta.reshape((-1, 1, 1))
    eye = np.eye(2).reshape((1, 2, 2))
    ones = np.ones((2, 2)).reshape((1, 2, 2))

    self.Qs = (1 - beta) * eye + (beta / 2) * ones

    Q_bar = [np.eye(2)]
    for Q in self.Qs:
      Q_bar.append(Q_bar[-1] @ Q)
    self.Q_bar = np.stack(Q_bar, axis=0)

  def __cos_noise(self, t):
    offset = 0.008
    return np.cos(math.pi * 0.5 * (t / self.T + offset) / (1 + offset)) ** 2

  def sample(self, x0_onehot, t):
    # Select noise scales
    Q_bar = torch.from_numpy(self.Q_bar[t]).float().to(x0_onehot.device)
    xt = torch.matmul(x0_onehot, Q_bar.reshape((Q_bar.shape[0], 1, 2, 2)))
    return torch.bernoulli(xt[..., 1].clamp(0, 1))

class BlackoutDiffusion(object):
  """Original Blackout Diffusion process from https://arxiv.org/abs/2305.11089"""

  def __init__(self, T, schedule, num_states = 3, alpha = 0.1):
    self.T = T # Diffusion steps
    self.num_states = num_states # state space spans only zero and one
    
    # Brith rates for computational convenience
    brTable = np.zeros((self.num_states, self.num_states, T))
    for tIndex in range(T):
        for n in range(self.num_states):
            for m in range(n):
                brTable[n, m, tIndex] = n-m 
    self.brTable = np.ravel(brTable)
    
    # observationTimes
    if schedule == 'original':
      self.observationTimes = self.__original_observationTimes()
    elif schedule == 'improved':
      self.observationTimes = self.__improved_observationTimes()
    elif schedule == 'more_improved':
      self.observationTimes = self.__more_improved_observationTimes(alpha = alpha)
    else:
      raise ValueError(f"Unknown Blackout Diffusion schedule type {schedule}")
    
    # CDF
    self.cdf = self.__cdf(self.observationTimes)
    
    self.cdfGPU = None
    self.brTableGPU = None
    self.num_statesGPU = None
    
    
  def __original_observationTimes(self, tEnd = 15.0):
    def f(x):
      return np.log(x/(1-x))
    
    xEnd = np.exp(-tEnd)
    fGrid = np.linspace(-f(xEnd), f(xEnd), self.T)
    xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(self.T)])
    observationTimes = -np.log(xGrid)    
        
    return observationTimes
  
  def __improved_observationTimes(self, tEnd = 15.0):
    half_T = self.T // 2
    linear_root_p_values = np.concatenate([
        np.linspace(0, 0.5, half_T),  # Increase from 0 to 0.5
        np.linspace(0.5, 0, self.T - half_T)  # Decrease back from 0.5 to 0
    ])
    
    p_values_estimated = np.zeros(self.T)
    p_values_estimated[:half_T] = (1 + np.sqrt(1 - 4 * linear_root_p_values[:half_T]**2)) / 2
    p_values_estimated[half_T:] = (1 - np.sqrt(1 - 4 * linear_root_p_values[half_T:]**2)) / 2

    # observationTimes is monotonically increasing function with regard to tIdx
    observationTimes = -np.log(p_values_estimated + 1e-8)
    observationTimes = np.clip(observationTimes, 1e-6, tEnd)
    
    return observationTimes
  
  def __more_improved_observationTimes(self, alpha, tEnd = 15.0):
    half_T = self.T // 2
    linear_root_p_values = np.concatenate([
        np.linspace(0, 0.25, int(half_T * alpha)),  # Increase to 0.5
        np.linspace(0.25, 0.5, half_T - int(half_T * alpha)),  # Increase to 0.5
        np.linspace(0.5, 0.25, half_T - int(half_T * alpha)),  # Decrease back to 0
        np.linspace(0.25, 0, int(half_T * alpha)) # Decrease back to 0
    ])
    
    p_values_estimated = np.zeros(self.T)
    p_values_estimated[:half_T] = (1 + np.sqrt(1 - 4 * linear_root_p_values[:half_T]**2)) / 2
    p_values_estimated[half_T:] = (1 - np.sqrt(1 - 4 * linear_root_p_values[half_T:]**2)) / 2

    # observationTimes is monotonically increasing function with regard to tIdx
    observationTimes = -np.log(p_values_estimated + 1e-8)
    observationTimes = np.clip(observationTimes, 1e-6, tEnd)
    
    return observationTimes
  
  def __cdf(self, observationTimes):
    support = np.arange(0, self.num_states)
    pdf = np.zeros((self.T+1, self.num_states, self.num_states))
    pdf[0,:,:] = np.eye(self.num_states)

    for tIndex in range(self.T):
        p = np.exp(-observationTimes[tIndex])
        for initial_condition in range(self.num_states):
            pdf[tIndex + 1, :, initial_condition] =  binom(initial_condition, p).pmf(support)    
            
    cdf = np.zeros_like(pdf)

    for i in range(self.T + 1):
        for j in range(self.num_states):
            cdf[i, :, j] = np.cumsum(pdf[i, :, j])    
            
    return cdf
  
  def sample(self, x0_onehot, tIndex):    
    with torch.no_grad():
      device = x0_onehot.device
      
      if self.cdfGPU == None:
        self.cdfGPU = torch.from_numpy(self.cdf).to(device)
        self.brTableGPU = torch.from_numpy(self.brTable).to(device) # flatten
        self.num_statesGPU = torch.tensor(self.num_states, device = device)
      
      T = torch.tensor(self.T, device = device)
      tIndex = tIndex.unsqueeze(-1).unsqueeze(-1).to(device)
      
      batch_size, node_size, _, = x0_onehot.shape
      cp = self.cdfGPU[(tIndex).int(), :, x0_onehot].to(device)
      u = torch.rand((batch_size, node_size, node_size, 1), device=device)
      xt = torch.argmax((u < cp).int(), dim=-1).reshape(batch_size, node_size, node_size)

      birthRatet = self.brTableGPU[(x0_onehot * self.num_statesGPU * T + xt * T + tIndex - 1)]

      return xt, birthRatet

class InferenceSchedule(object):
  def __init__(self, inference_schedule="linear", T=1000, inference_T=1000, tEnd = 15.0, alpha = 0.1):
    self.inference_schedule = inference_schedule
    self.T = T
    self.inference_T = inference_T
    
    if inference_schedule == "original":
      def f(x):
        return np.log(x/(1-x))
      
      xEnd = np.exp(-tEnd)
      fGrid = np.linspace(-f(xEnd), f(xEnd), inference_T)
      xGrid = np.array([bisect(lambda x: f(x)-fGrid[i], xEnd/2, 1-xEnd/2) for i in range(inference_T)])
      observationTimes = -np.log(xGrid)    
      eobservationTimes = np.hstack([0, observationTimes])
      
      self.t1_list = eobservationTimes[:-1][::-1]
      self.t2_list = eobservationTimes[1:][::-1]

    elif inference_schedule == "improved":
      half_T = inference_T // 2
      linear_root_p_values = np.concatenate([
          np.linspace(0, 0.5, half_T),  # Increase from 0 to 0.5
          np.linspace(0.5, 0, inference_T - half_T)  # Decrease back from 0.5 to 0
      ])
      
      p_values_estimated = np.zeros(inference_T)
      p_values_estimated[:half_T] = (1 + np.sqrt(1 - 4 * linear_root_p_values[:half_T]**2)) / 2
      p_values_estimated[half_T:] = (1 - np.sqrt(1 - 4 * linear_root_p_values[half_T:]**2)) / 2

      # observationTimes is monotonically increasing function with regard to tIdx
      observationTimes = -np.log(p_values_estimated + 1e-8)
      observationTimes = np.clip(observationTimes, 1e-6, tEnd)      
      eobservationTimes = np.hstack([0, observationTimes])
      
      self.t1_list = eobservationTimes[:-1][::-1]
      self.t2_list = eobservationTimes[1:][::-1]
    
    elif inference_schedule == "more_improved":
      half_T = inference_T // 2
      linear_root_p_values = np.concatenate([
        np.linspace(0, 0.25, int(half_T * alpha)),  # Increase to 0.5
        np.linspace(0.25, 0.5, half_T - int(half_T * alpha)),  # Increase to 0.5
        np.linspace(0.5, 0.25, half_T - int(half_T * alpha)),  # Decrease back to 0
        np.linspace(0.25, 0, int(half_T * alpha)) # Decrease back to 0
      ])

      p_values_estimated = np.zeros(inference_T)
      p_values_estimated[:half_T] = (1 + np.sqrt(1 - 4 * linear_root_p_values[:half_T]**2)) / 2
      p_values_estimated[half_T:] = (1 - np.sqrt(1 - 4 * linear_root_p_values[half_T:]**2)) / 2

      # observationTimes is monotonically increasing function with regard to tIdx
      observationTimes = -np.log(p_values_estimated + 1e-8)
      observationTimes = np.clip(observationTimes, 1e-6, tEnd)
      eobservationTimes = np.hstack([0, observationTimes])
      
      self.t1_list = eobservationTimes[:-1][::-1]
      self.t2_list = eobservationTimes[1:][::-1]

  def __call__(self, i):
    assert 0 <= i < self.inference_T

    if self.inference_schedule == "linear":
      t1 = self.T - int((float(i) / self.inference_T) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    
    elif self.inference_schedule == "cosine":
      t1 = self.T - int(
          np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
      t1 = np.clip(t1, 1, self.T)

      t2 = self.T - int(
          np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
      t2 = np.clip(t2, 0, self.T - 1)
      return t1, t2
    
    elif self.inference_schedule == "original" or  self.inference_schedule == "improved" or  self.inference_schedule == "more_improved":
      return self.t1_list[i], self.t2_list[i]

    else:
      raise ValueError("Unknown inference schedule: {}".format(self.inference_schedule))
