import numpy as np
import random
import copy
from collections import namedtuple, deque

from maddpg_model import Actor_Critic_Models

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)        # reply buffer size
BATCH_SIZE = 128              # minibatch size
GAMMA = 0.99                  # discount factor
TAU = 1e-3                    # for soft update of target parameters
LR_ACTOR = 2e-4               # learning rate of the actor
LR_CRITIC = 2e-4              # learning rate of the critic
WEIGHT_DECAY = 0              # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetaAgent():
    """Meta agent that contains two DDPG agents and shared replay buffer"""
    def __init__(self):
        pass

class DDPGAgent():
    """Interacts with and learns from the environment."""
    def __init__(self):
        pass


class OUNoise():
    """"Ornstein-Uhlenbeck process"""
    def __init__(self):
        pass

class ReplayBuffer():
    """"Buffer to store experience tuples"""
    def __init__(self):
        pass
