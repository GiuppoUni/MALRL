import argparse
import datetime
from gym_airsim.envs.collectEnv import CollectEnv
from trajectoryTrackerClient import TrajectoryTrackerClient

import gym

import gym_airsim.envs
import gym_airsim
import utils
import time
from gym_maze.envs.maze_env import MazeEnv
import numpy as np

episode_cooldown = 3

ACTION_TO_IDX = {"LEFT":0, "FRONT":1, "RIGHT":2,"BACK" : 3}
IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}

def custom_random(past_action):

    action = env.action_space.sample() # Random actions DEBUG ONLY            
    if(past_action and past_action != ACTION_TO_IDX["FRONT"] ):
        while(action == past_action):
            action = env.action_space.sample() # Random actions DEBUG ONLY                    
    return action



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='RL for ma-gym')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')

    parser.add_argument('--actions-timeout', type=int, default=-1,
                        help='episodes (default: %(default)s)')
    
    parser.add_argument('--n-agents', type=int, default=1,
                        help='agents (default: %(default)s)')
    
    parser.add_argument('--thickness', type=int, default=140,
                        help='Draw line thickness (default: %(default)s)')
    
    # parser.add_argument('--ep-cooldown', type=int, default=1,
    #                     help='episode cooldown time sleeping (default: %(default)s)')

    parser.add_argument( '--fixed-action',action='store_true',  default=False,
        help='Same actions per episode (default: %(default)s)' )

    parser.add_argument( '--debug',action='store_true',  default=False,
        help='Log into file (default: %(default)s)' )
    
    parser.add_argument( '--crab-mode',action='store_true',  default=False,
        help='Move at fixed yaw (default: %(default)s)' )
    
    parser.add_argument('--draw-traj',action='store_true',  default=False,
        help='Draw past trajectories (red lines) (default: %(default)s)')

    parser.add_argument('--follow-traj',action='store_true',  default=False,
        help='Follow trajectory 3d (red lines) (default: %(default)s)')

    parser.add_argument('--random-pos',action='store_true',  default=False,
        help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

    parser.add_argument('--env2D',action='store_true',  default=False,
        help='(default: %(default)s)')

    parser.add_argument('--can-go-back',action='store_true',  default=False,
        help='(default: %(default)s)')

    parser.add_argument('--custom-random',action='store_true',  default=False,
        help='(default: %(default)s)')

    parser.add_argument('--track-traj',action='store_true',  default=False,
        help='Track trajectories into file (default: %(default)s)')

    parser.add_argument('--col-traj', action='store_true', default=False,
    help='Track trajectories into file (default: %(default)s)')




    args = parser.parse_args()

    # env = gym.make("AirSimEnv-v1")
    
    if(args.debug):
        logger = utils.initiate_logger()
        print = logger.info

    # TODO replace for variables
    n_actions = 4 if args.can_go_back else 3 
    env = CollectEnv(trajColFlag = args.col_traj,n_actions=n_actions,
        random_pos = args.random_pos,drawTrajectories = args.draw_traj,
        crabMode = args.crab_mode,thickness = args.thickness)
    trackerClient = TrajectoryTrackerClient()


    if(args.n_agents ==1):
        print("Starting episodes...")
        for ep_i in range(args.episodes):
            done = False
            ep_reward = 0

            env.seed(ep_i)  
            obs = env.reset()


            if(not args.env2D and args.track_traj):
                trackerClient.start_tracking(ep_i,vName="Drone0")
                time.sleep(0.01)
            
            n_actions_taken = 0
            past_action = None

          
            if(args.fixed_action):
                action = np.random.choice([0,1,2,3])
            while not done :
            # for _ in range(0,150): # DEBUG ONLY
                action = env.action_space.sample()
                if(not args.fixed_action):
                    if(args.custom_random):
                        action = custom_random(past_action)
                        past_action = action
                    else:
                        action = env.action_space.sample() # Random actions DEBUG ONLY            
                         # action = 0 if n_actions_taken % 2 == 0 else 1 # DEBUG ONLY
                if(not args.can_go_back and args.crab_mode and past_action is not None):
                    # Check if Left -> Right and vic or Up -> Down and vic
                    while(action +2 == past_action or action -2 == past_action):
                        action = env.action_space.sample()
                past_action = action

                obs, reward, done, info = env.step(action)
                ep_reward =  reward
                
                n_actions_taken +=1
                
                if n_actions_taken == args.actions_timeout:
                    print("Episode ended: actions timeout reached")
                    break
            
            if(not args.env2D and args.track_traj):
                trackerClient.stop_tracking()     

            print("="*40)    
            print('Episode #{} Reward: {}'.format(ep_i+1, ep_reward))
        env.close()

    