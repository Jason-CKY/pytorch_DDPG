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
    network_config = agent_parameters['network_config']
    action_high = env.action_space.high
    action_low = env.action_space.low
    action_dim = len(action_low)
    state_dim = len(env.observation_space.sample())
    action_lim = []
    for low, high in zip(action_low, action_high):
        action_lim.append([low, high])
    action_lim = np.array(action_lim, dtype='float32')

    agent_parameters['network_config']['state_dim'] = state_dim
    agent_parameters['network_config']['action_dim'] = action_dim
    agent_parameters['network_config']['action_lim'] = action_lim

    return agent_parameters

def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    
    rl_glue = RLGlue(environment, agent)
        
    # save sum of reward at the end of each episode
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"], 
                                 experiment_parameters["num_episodes"]))

    env_info = environment_parameters

    agent_info = agent_parameters

    # one agent setting
    for run in range(1, experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run

        rl_glue.rl_init(agent_info, env_info)
        
        for episode in tqdm(range(1, experiment_parameters["num_episodes"]+1)):
            # run episode
            rl_glue.rl_episode(experiment_parameters["timeout"])
            
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            agent_sum_reward[run - 1, episode - 1] = episode_reward

            if episode % experiment_parameters['checkpoint_freq'] == 0:
                rl_glue.agent.save_checkpoint(episode)

    save_name = "{}".format(rl_glue.agent.name)    
    path = os.path.join("sum_reward_{}".format(save_name))
    np.save(path, agent_sum_reward)
    x = np.load(path)
    plt.plot(np.arange(experiment_parameters['num_episodes']), x[0])
    gym_name = env_info['gym_environment']
    plt.savefig(f'{gym_name}_sum_rewards.png')

def main():
    # Run Experiment

    # Experiment parameters
    experiment_parameters = {
        "num_runs" : 1,
        "num_episodes" : 500,
        "checkpoint_freq": 100,
        "load_checkpoint": None,    # None to start new experiment, path to checkpoint to resume training
        # OpenAI Gym environments allow for a timestep limit timeout, causing episodes to end after 
        # some number of timesteps.
        "timeout" : 2000
    }

    # Environment parameters
    environment_parameters = {
        "gym_environment": 'Pendulum-v0',
        "record_frequency": 500
    }

    current_env = Environment

    # Agent parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent_parameters = {
        'network_config': {
            'state_dim': 3,
            'hidden_dim': 256,
            'action_dim': 1,
            'action_lim': 2
        },
        'optimizer_config': {
            'step_size': 1e-3,
            'betas': (0.9, 0.999)
        },
        'name': 'DDPG actor-critic agent',
        'device': device,
        'replay_buffer_size': 50000,
        'minibatch_size': 8,
        'num_replay_updates_per_step': 4,
        'gamma': 0.99,
        'tau': 0.01,
        'checkpoint_dir': 'model_weights'
    }
    current_agent = DDPG_Agent
    agent_parameters = update_agent_parameters(environment_parameters, agent_parameters)
    # run experiment on pendulum-v0
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
    # run experiment on BipedalWalker-v3
    environment_parameters['gym_environment'] = "BipedalWalker-v3"
    agent_parameters = update_agent_parameters(environment_parameters, agent_parameters)

    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

if __name__ == '__main__':
    main()