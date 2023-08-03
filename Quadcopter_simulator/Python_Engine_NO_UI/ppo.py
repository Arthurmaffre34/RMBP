import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        pass

class Actor(nn.Module):
    def __init__(self, action_dim, input_dims, hidden_dims):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, action_dim),
            nn.Sigmoid()
        )
