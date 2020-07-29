# from Environments.lunar_lander import LunarLanderEnvironment
from Environments.environment import Environment
from Agents.ddpg_agent import DDPG_Agent
from rl_glue import RLGlue
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
import os

def update_agent_parameters(environment_parameters, agent_parameters):
    env = gym.make(environment_parameters['gym_environment'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim

    checkpoint_dir = agent_parameters['checkpoint_dir']
    agent_parameters['checkpoint_dir'] = os.path.join(checkpoint_dir, environment_parameters['gym_environment'])

    return agent_parameters

def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    rl_glue = RLGlue(environment, agent)
        
    # save sum of reward at the end of each episode
    agent_sum_reward = []

    env_info = environment_parameters

    agent_info = agent_parameters

    rl_glue.rl_init(agent_info, env_info)
    starting_episode = 0
    max_reward = 0

    gym_name = env_info['gym_environment']
    save_name = "{}.npy".format(rl_glue.agent.name)    
    path = os.path.join(rl_glue.agent.checkpoint_dir, "sum_reward_{}".format(save_name))

    # load checkpoint if any
    if experiment_parameters['load_checkpoint'] is not None:
        rl_glue.agent.load_checkpoint(experiment_parameters['load_checkpoint'])
        fname = experiment_parameters['load_checkpoint'].split(os.path.sep)[-1]
        # starting_episode = int(fname.split('_')[1])
        print(f"starting from episode {starting_episode}")
        
    for episode in tqdm(range(1 + starting_episode, experiment_parameters["num_episodes"]+1+starting_episode)):
        # run episode
        rl_glue.rl_episode(experiment_parameters["timeout"])
        
        episode_reward = rl_glue.rl_agent_message("get_sum_reward")
        agent_sum_reward.append(episode_reward)
        print('Episode {}/{} | Reward {}'.format(episode, experiment_parameters['num_episodes'], episode_reward))

        if episode == 1:
            max_reward = episode_reward

        if episode % experiment_parameters['checkpoint_freq'] == 0:
            rl_glue.agent.save_checkpoint(episode + starting_episode)
            np.save(path, np.array(agent_sum_reward))
            
        if episode_reward >= max_reward:
            max_reward = episode_reward
            print(f"new max reward: {max_reward}")
            rl_glue.agent.save_checkpoint(episode + starting_episode, save_best=True)

        if episode_reward >= env_info['solved_threshold']:
            print("Task Solved with reward = {}".format(episode_reward))
            rl_glue.agent.save_checkpoint(episode + starting_episode, solved=True)
            break

    np.save(path, agent_sum_reward)
    x = np.load(path)
    plt.plot(np.arange(experiment_parameters['num_episodes']), x[0])
    plt.savefig(os.path.join(rl_glue.agent.checkpoint_dir, 'sum_rewards.png'))

def main():
    # Run Experiment

    # Experiment parameters
    experiment_parameters = {
        "num_runs" : 1,
        "num_episodes" : 5000,
        "checkpoint_freq": 100,
        "load_checkpoint": f'model_weights{os.path.sep}BipedalWalker-v3{os.path.sep}best.pth',    # None to start new experiment, path to checkpoint to resume training
        # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
        # some number of timesteps.
        "timeout" : 2000
    }

    # Environment parameters
    environment_parameters = {
        "gym_environment": 'BipedalWalker-v3',
        "record_frequency": 500,
        'solved_threshold': 300,
        'seed': 0
    }

    current_env = Environment

    # Agent parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_parameters = {
        'network_config': {
            'state_dim': 3,
            'action_dim': 1,
            'seed': 0
        },
        'optimizer_config': {
            'actor_lr': 0.0001,
            'critic_lr': 0.001,
            'weight_decay': 0.01
        },
        'name': 'DDPG_actor-critic_agent',
        'device': device,
        'replay_buffer_size': 1000000,
        'minibatch_size': 64,
        'num_replay_updates_per_step': 1,
        'gamma': 0.99,
        'tau': 0.001,
        'checkpoint_dir': 'model_weights',
        'seed': 0
    }
    current_agent = DDPG_Agent
    agent_parameters = update_agent_parameters(environment_parameters, agent_parameters)
    # run experiment
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

if __name__ == '__main__':
    main()