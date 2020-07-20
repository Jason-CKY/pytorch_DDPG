from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer
from .q_network import DQN
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

class DDPG_Agent(BaseAgent):
    def __init__(self):
        pass
    
    def agent_init(self, agent_config):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the agent.

        Assume agent_config dict contains:
        {
            network_config: dictionary,
            optimizer_config: dictionary,
            replay_buffer_size: integer,
            minibatch_sz: integer, 
            num_replay_updates_per_step: float
            discount_factor: float,
        }
        """
        self.name = agent_config['name']
        self.device = agent_config['device']
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                        agent_config['minibatch_size'],
                                        agent_config.get('seed'))
        # define network
        optim_config = agent_config['optimizer_config']
        self.num_actions = agent_config['network_config']['num_actions']
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.rand_generator = np.random.RandomState(agent_config.get('seed'))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.epsiode_steps = 0

    def optimize_network(self, experiences):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
        """
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
        states = torch.tensor(np.concatenate(states)).to(self.device)
        next_states = torch.tensor(np.concatenate(next_states)).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        terminals = torch.tensor(terminals).to(self.device)
        raise NotImplementedError
        

    def policy(self, state):
        """
        Args:
            state (Numpy array): the state
        Returns:
            the action
        """
        raise NotImplementedError

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1

        # Make state an array of shape (1, state_dim) to add a batch dimension and
        # to later match the get_action_values() and get_TD_update() functions
        state = np.array([state])

        # Select action
        action = self.policy(state)
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()     
                self.optimize_network(experiences)
                
        # Update the last state and last action.
        self.last_state = state
        self.last_action = action
        
        return action
        
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        # Set terminal state to an array of zeros
        state = np.zeros_like(self.last_state)

        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.current_q.load_state_dict(self.network.state_dict())
            for _ in range(self.num_replay):
                # Get sample experiences from the replay buffer
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences)
                

    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
        pass

    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognized Message!")
