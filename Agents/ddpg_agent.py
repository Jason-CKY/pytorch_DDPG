from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer
from .nets import Actor, Critic
from .noise import OUNoise
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
            checkpoint_dir: str
        }
        """
        self.name = agent_config['name']
        self.device = agent_config['device']
        self.replay_buffer = ReplayBuffer(agent_config['replay_buffer_size'],
                                        agent_config['minibatch_size'],
                                        agent_config.get('seed'))
        # define network
        self.actor = Actor(agent_config['network_config']).to(self.device)
        self.actor_target = Actor(agent_config['network_config']).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(agent_config['network_config']).to(self.device)
        self.critic_target = Critic(agent_config['network_config']).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        optim_config = agent_config['optimizer_config']
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=optim_config['step_size'], betas=optim_config['betas'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=optim_config['step_size'], betas=optim_config['betas'])
        self.num_replay = agent_config['num_replay_updates_per_step']
        self.discount = agent_config['gamma']
        self.tau = agent_config['tau']

        self.noise = OUNoise(agent_config['network_config']['action_dim'])
        self.rand_generator = np.random.RandomState(agent_config.get('seed'))

        self.last_state = None
        self.last_action = None

        self.sum_rewards = 0
        self.epsiode_steps = 0

        checkpoint_dir = agent_config.get('checkpoint_dir')
        if checkpoint_dir is None:
            self.checkpoint_dir = 'saved_models'
        else:
            self.checkpoint_dir = checkpoint_dir

    def optimize_network(self, experiences):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions, 
                                    rewards, terminals, and next_states.
        """
        # Get states, action, rewards, terminals, and next_states from experiences
        states, actions, rewards, terminals, next_states = experiences
        states = torch.tensor(states).to(self.device).float()
        next_states = torch.tensor(next_states).to(self.device).float()
        actions = torch.tensor(actions).to(self.device).float()
        rewards = torch.tensor(rewards).to(self.device).float()
        terminals = torch.tensor(terminals).to(self.device).float()
        
        # ------------------- optimize critic ----------------------------
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach())
        Qprime = rewards + self.discount * next_Q
        # Qprime = Qprime.float()
        critic_loss = F.smooth_l1_loss(Qprime, Qvals)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ------------------- optimize actor ----------------------------
        policy_loss = -1*self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def policy(self, state):
        """
        Args:
            state (Numpy array)/(torch tensor)/(list): the state
        Returns:
            the action
        if state is in the shape [n, state_dim], output will be [n, action_dim]
        if state is in the shape [state_dim], output will be [action_dim]
        """
        state = torch.tensor(state).to(self.device).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = self.actor(state).cpu().detach().numpy()[0]
        else:
            action = self.actor(state).cpu().detach().numpy()

        action = self.noise.get_action(action)  # add noise
        action = np.clip(action, -1, 1)         # clip to tanh range [-1, 1]
                
        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        self.noise.reset()
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array(state)
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
        state = np.array(state)
        action = self.policy(state)
        
        # Append new experience to replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # Perform replay steps:
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
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

    def set_train(self):
        '''
        Set actor and critic networks into train mode
        '''
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def set_eval(self):
        '''
        Set actor and critic networks into eval mode
        '''
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        