import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    '''
    critic network responsible for detemining the q values for state-action pair
    chosen by the actor network
    '''
    def __init__(self, lr_critic, input_dims, fc1_dim, fc2_dim, ckp_dir, name):
        super(CriticNetwork, self).__init__()
        self.ckp_file = os.path.join(ckp_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        layer1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = self.q(layer2)
        return layer3

    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))
    
class ActorNetwork(nn.Module):
    '''
    actor network responsible for choosing policy based on a given state
    '''
    def __init__(self, lr_actor, input_dims, fc1_dim, fc2_dim, n_actions, ckp_dir, name):
        super(ActorNetwork, self).__init__()
        self.ckp_file = os.path.join(ckp_dir, name)
        print(f"Checkpoint file for Actor: {self.ckp_file}")
        self.fc1 = nn.Linear(input_dims, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.pi = nn.Linear(fc2_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = T.tensor(state, dtype=T.float, device=self.device)
        layer1  = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = T.tanh(self.pi(layer2))
        return layer3
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.ckp_file))

