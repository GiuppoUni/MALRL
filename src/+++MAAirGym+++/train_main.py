
import argparse
from gym_airsim.envs.collectMTEnv import CollectMTEnv

import gym

import gym_airsim.envs
import gym_airsim

from navMapWindow import NavMapper
import utils
import time
import numpy as np

import random
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='RL for ma-gym')
    parser.add_argument('--episodes', type=int, default=4,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Log into file (default: %(default)s)')
    args = parser.parse_args()

    # env = gym.make("AirSimEnv-v1")
    
    if args.debug:
        print("DEBUG")
        logger = utils.initiate_logger()
        print = logger.info


    # TODO replace for variables
    # env = CollectGameEnv()
    env = CollectMTEnv()
    
    #navMapper = NavMapper(env.myClient)
    
    q_table = dict()
    # np.zeros([(env.width* env.height)**env.n_agents,
    #     env.n_actions**env.n_agents] )
    for i_a in range(0,env.n_agents):
        for x in range(22,223):
            for y in range(-69,121):    
                q_table[[x,y]]
    all_penalties = []
    
    print("Starting episodes...")
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()
        env.printTargets()
        # env.render()
        penalties = [0]*env.n_agents
        while not all(done_n):
        # for _ in range(0,150): # DEBUG ONLY
            if random.uniform(0, 1) < epsilon:
                # TODO single random
                action_n = env.action_space.sample() # Random actions DEBUG ONLY
            # action_n = [0 for _ in range(env.n_agents)] # DEBUG ONLY
            else:
                action_n = np.argmax(q_table[obs_n]) # Exploit learned values

            next_obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            old_value = q_table[obs_n][action_n]

            next_max = np.max(q_table[next_obs_n])
        
            new_value = (1 - alpha) * old_value + alpha * (reward_n + gamma * next_max)
            q_table[obs_n][action_n] = new_value

            for i,r in enumerate(reward_n) :
                if r <= -10:
                    penalties[i] += 1

            obs_n = next_obs_n

            # env.render()
            
            # navMapper.update_nav_fig()
            # TODO remove this and other slowers
            time.sleep(2)

        print("="*40)    
        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()

