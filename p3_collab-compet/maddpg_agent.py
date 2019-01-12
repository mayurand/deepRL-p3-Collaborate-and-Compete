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
    def __init__(self, agent_id, model, seed=0, action_size=2, tau=1e-3, lr_actor=2e-4, lr_critic=2e-4, weight_decay=0):
        """
        Params
        ======
        model: Model containing critic and actor models
        action_size(int): Dimension of each action
        seed (int): Random seed
        tau (float): Soft update for target parameters
        lr_actor (float): Learning rate of the actor
        lr_critic (float): Learning rate of the critic
        weight_decay (float): L2 weight decay
        """
        self.agent_id = agent_id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        # Actor models
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.optimizer = optim.Adam(self.actor_local.parameters(), self.lr_actor)

        # Critic models
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.optimizer = optim.Adam(self.critic_local.parameters(), self.lr_critic)

        # Initialize same weights for local and target actor and critic
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        self.noise = OUNoise(self.action_size, self.seed)

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def act(self,state, noise_weight=1.0, add_noise=True):
        """
        Give actions to take in a given state according to the policy

        Params
        ======
        state (numpy array): Input state of the environment
        :return: actions (int)

        """
        state = torch.from_numpy(state).float().to(device) # Convert state to torch tensor
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target
                Params
                ======
                    local_model (PyTorch model): weights will be copied from
                    target_model (PyTorch model): weights will be copied to
                    tau (float): interpolation parameter
                """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by its actor)
            all_actions (list): each agent's action (as calculated by its actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # q_targets = reward of this timestep + discount * Q(st+1,at+1) from target network
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        # compute critic loss
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        # detach actions from other agents
        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)



class ReplayBuffer():
    """Buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
