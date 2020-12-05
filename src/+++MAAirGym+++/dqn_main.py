import sys
import numpy as np
import math
import random

import gym
import gym_maze

import argparse
import datetime
from gym_airsim.envs.collectEnv import CollectEnv
from trajectoryTrackerClient import TrajectoryTrackerClient


import gym_airsim.envs
import gym_airsim
from gym_airsim.envs import AirSimEnv
import utils
import time
from gym_maze.envs.maze_env import MazeEnv


episode_cooldown = 3

ACTION_TO_IDX = {"LEFT":0, "FRONT":1, "RIGHT":2,"BACK" : 3}
IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}




def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(q_table[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)



    



# ==================================================================================================

def custom_random(past_action):

    action = env.action_space.sample() # Random actions DEBUG ONLY            
    if(past_action and past_action != ACTION_TO_IDX["FRONT"] ):
        while(action == past_action):
            action = env.action_space.sample() # Random actions DEBUG ONLY                    
    return action




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL for ma-gym')
    parser.add_argument('--episodes', type=int, default=10,
                        help='episodes (default: %(default)s)')

    parser.add_argument('--actions-timeout', type=int, default=100,
                        help='episodes (default: %(default)s)')

    # parser.add_argument('--ep-cooldown', type=int, default=1,
    #                     help='episode cooldown time sleeping (default: %(default)s)')



    parser.add_argument( '--debug',action='store_true',  default=False,
        help='Log into file (default: %(default)s)' )

    parser.add_argument('--random-pos',action='store_true',  default=False,
        help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

    parser.add_argument('--env2D',action='store_true',  default=True,
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

    if(args.env2D):
        env = MazeEnv( maze_file = "maze2d_002.npy",                  
            # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                        maze_size=(640, 640), 
                                        enable_render=True)

    else:
        # TODO replace for variables
        n_actions = 4 if args.can_go_back else 3 
        env = CollectEnv(trajColFlag = args.col_traj,n_actions=n_actions,random_pos = args.random_pos)
        trackerClient = TrajectoryTrackerClient()





    '''
    Defining the environment related constants
    '''
    # Number of discrete states (bucket) per state dimension
    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    NUM_BUCKETS = MAZE_SIZE  # one bucket per grid

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # ["N", "S", "E", "W"]
    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

    '''
    Learning related constants
    '''
    MIN_EXPLORE_RATE = 0.001
    MIN_LEARNING_RATE = 0.2
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    '''
    Defining the simulation related constants
    '''
    NUM_EPISODES = 50000
    MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0
    RENDER_MAZE = True

    '''
    Creating a Q-Table for each state-action pair
    '''
    q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

    #=================#
    #       '''        #
    # Begin simulation #
    #       '''        #
    #=================#

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99

    num_streaks = 0

    assert type(env) == MazeEnv

    # Render tha maze
    if(args.env2D and RENDER_MAZE):
        env.render()

    print("Learning starting...")
    for episode in range(args.episodes):

        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        total_reward = 0

        for t in range(MAX_T):

            # Select an action
            action = select_action(state_0, explore_rate)

            # execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)
            total_reward += reward

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if DEBUG_MODE == 2:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)
                print("")

            elif DEBUG_MODE == 1:
                if done or t >= MAX_T - 1:
                    print("\nEpisode = %d" % episode)
                    print("t = %d" % t)
                    print("Explore rate: %f" % explore_rate)
                    print("Learning rate: %f" % learning_rate)
                    print("Streaks: %d" % num_streaks)
                    print("Total reward: %f" % total_reward)
                    print("")

            # Render tha maze
            if RENDER_MAZE:
                env.render()

            if env.is_game_over():
                sys.exit()

            if done:
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                    % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                    % (episode, t, total_reward))

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)



    # print("Starting episodes...")
    # for ep_i in range(args.episodes):
    #     done = False
    #     ep_reward = 0

    #     env.seed(ep_i)  
    #     obs = env.reset()
        
        
    #     if(not args.env2D and args.track_traj):
    #         trackerClient.start_tracking(ep_i,vName="Drone0")
    #         time.sleep(0.01)
        
    #     n_actions_taken = 0
    #     past_action = None


    #     while not done :
    #     # for _ in range(0,150): # DEBUG ONLY
    #         if(args.custom_random):
    #             action = custom_random(past_action)
    #             past_action = action
    #         else:
    #             action = env.action_space.sample() # Random actions DEBUG ONLY            
    #         # action = 0 if n_actions_taken % 2 == 0 else 1 # DEBUG ONLY

    #         obs, reward, done, info = env.step(action)
    #         ep_reward =  reward
    #         # env.render()
            
    #         n_actions_taken +=1
            
    #         if n_actions_taken == args.actions_timeout:
    #             print("Episode ended: actions timeout reached")
    #             break

    #         if(args.env2D):
    #             env.render()

    #         # navMapper.update_nav_fig()
    #         if(not args.env2D):
    #             time.sleep(episode_cooldown)
    #         else:
    #             time.sleep(1)

        
    #     if(not args.env2D and args.track_traj):
    #         trackerClient.stop_tracking()     

    #     print("="*40)    
    #     print('Episode #{} Reward: {}'.format(ep_i+1, ep_reward))
    # env.close()

