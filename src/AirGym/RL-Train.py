import argparse

import gym

import gym_airsim.envs
import gym_airsim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--episodes', type=int, default=1,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    env = gym.make("AirSimEnv-v42")
    print("Starting episodes...")
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)
        obs_n = env.reset()

        # env.render()
        
        while not all(done_n):
            action_n = env.action_space.sample()
            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            # env.render()

        print('Episode #{} Reward: {}'.format(ep_i, ep_reward))
    env.close()

