
import argparse

import gym

import gym_airsim.envs
import gym_airsim
from gym_airsim.envs import AirSimEnv

from navMapWindow import NavMapper
import utils
import time

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='RL for ma-gym')
    parser.add_argument('--episodes', type=int, default=4,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    # env = gym.make("AirSimEnv-v1")
    
    
    # TODO replace for variables
    env = AirSimEnv(n_agents=int(utils.g_config["rl"]["n_agents"]),
        n_actions = 3,step_cost = -1)
    
    #navMapper = NavMapper(env.myClient)
    
    print("Starting episodes...")
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()

        # env.render()
        
        while not all(done_n):
        # for _ in range(0,5): # DEBUG ONLY
            # action_n = env.action_space.sample() # Random actions
            action_n = [0 for _ in range(env.n_agents)] # DEBUG ONLY

            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            # env.render()
            
            # navMapper.update_nav_fig()
            time.sleep(2)

        print("="*40)    
        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()

