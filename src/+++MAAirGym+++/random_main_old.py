import argparse
import datetime
from trajectoryTrackerClient import TrajectoryTrackerClient
from gym_airsim.envs.collectMTEnv import CollectMTEnv

import gym

import gym_airsim.envs
import gym_airsim
from gym_airsim.envs import AirSimEnv
import utils
import time
from gym_maze.envs.maze_env import MazeEnv


episode_cooldown = 0.1

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='RL for ma-gym')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')
    parser.add_argument('--actions-timeout', type=int, default=100,
                        help='episodes (default: %(default)s)')


    parser.add_argument('--debug', type=bool, default=False,
    help='Log into file (default: %(default)s)')
    
    parser.add_argument('--random-pos',action='store_true',  default=False,
        help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

    parser.add_argument('--env2D',action='store_true',  default=False,
        help='(default: %(default)s)')


    parser.add_argument('--track-trajectories',action='store_true',  default=False,
        help='Track trajectories into file (default: %(default)s)')

    parser.add_argument('--collision-trajectories', action='store_true', default=False,
    help='Track trajectories into file (default: %(default)s)')



    args = parser.parse_args()

    # env = gym.make("AirSimEnv-v1")
    
    if(args.debug):
        logger = utils.initiate_logger()
        print = logger.info

    if(args.env2D):
        env = MazeEnv(maze_name="My Maze (%s)",
                                        maze_file_path="maze"+str(datetime.datetime()).replace(":","--"),
                                        screen_size=(640, 640), 
                                        enable_render=True)

    # TODO replace f  or variables
    env = CollectMTEnv(trajColFlag = args.collision_trajectories)
    
    trackerClient = TrajectoryTrackerClient()
    #navMapper = NavMapper(env.myClient)
    

    print("Starting episodes...")
    for ep_i in range(args.episodes):
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0

        env.seed(ep_i)  
        obs_n = env.reset(random_pos = args.random_pos)
        if(args.env2D):
            env.render()
        
        if(args.track_trajectories):
            trackerClient.start_tracking(ep_i,vName="Drone0")
            time.sleep(0.01)
        
        n_actions_taken = 0

        while not all(done_n) :
        # for _ in range(0,150): # DEBUG ONLY
            action_n = env.action_space.sample() # Random actions DEBUG ONLY
            # action_n = [0 for _ in range(env.n_agents)] # DEBUG ONLY

            obs_n, reward_n, done_n, info = env.step(action_n)
            ep_reward += sum(reward_n)
            # env.render()
            
            n_actions_taken +=1
            
            if n_actions_taken == args.actions_timeout:
                print("Episode ended: actions timeout reached")
                break

            # navMapper.update_nav_fig()
            time.sleep(episode_cooldown)
        
        if(args.track_trajectories):
            trackerClient.stop_tracking()     

        print("="*40)    
        print('Episode #{} Reward: {}'.format(ep_i+1, ep_reward))
    env.close()


