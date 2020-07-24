import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic(nn.Module):
    """ Critic Model Architecture for Agent
    """ 
    def __init__(self, critic_config):
        '''
        Assume critic_config:dictionary contains:
            state_dim: int
            hidden_dim: int
            action_dim: int
        '''
        super(Critic, self).__init__()
        state_dim = critic_config['state_dim']
        hidden_dim = critic_config['hidden_dim']
        action_dim = critic_config['action_dim']
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.action_fc1 = nn.Linear(action_dim, hidden_dim // 2)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, states, actions):
        '''
        Args:
            states: pytorch tensor of shape [n, state_dim]
            actions: pytorch tensor of shape [n, action_dim]
        '''
        s1 = F.relu(self.state_fc1(states))
        s2 = F.relu(self.state_fc2(s1))

        a1 = F.relu(self.action_fc1(actions))
        x = torch.cat([s2, a1], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor(nn.Module):
    """ Actor Model Architecture for Agent
    """ 

    def __init__(self, actor_config):
        '''
        Assume actor_config:dictionary contains:
            state_dim: int
            hidden_dim: int
            action_dim: int
        '''
        super(Actor, self).__init__()
        state_dim = actor_config['state_dim']
        hidden_dim = actor_config['hidden_dim']
        action_dim = actor_config['action_dim']
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        '''
        Args:
            states: pytorch tensor of shape [n, state_dim]
        '''
        x = F.relu(self.fc1(states))
        x = torch.tanh(self.fc2(x))
        return x