import os
import sys
import numpy as np
import math
import random

import gym
import gym_maze

import argparse
import datetime


import gym_airsim.envs
import gym_airsim
import utils
import time
from gym_maze.envs.maze_env import MazeEnv
from gym_maze.envs.maze_env_cont import MazeEnvCont

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy


import matplotlib.pyplot as plt

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback






class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

episode_cooldown = 3

ACTION_TO_IDX = {"LEFT":0, "FRONT":1, "RIGHT":2,"BACK" : 3}
IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}



    



# ==================================================================================================


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

    parser.add_argument('--mode',type=str, default="train",
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

    env = MazeEnv( maze_file = "maze2d_002.npy",                  
        # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                    maze_size=(640, 640), 
                                    enable_render=False,
                                    do_track_trajectories=False,num_goals=1,verbose = False ,human_mode = False)


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
    if(args.mode=="train"):
        # Create log dir
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

        # Create and wrap the environment
        # env = Monitor(env, log_dir)

        # Create the callback: check every 1000 steps
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        # callback = None

        # Train the agent
        time_steps = 2e5
        
        # Instantiate the agent
        model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
        # Train the agent
        print("[TRAIN] Learning started...")
        model.learn(total_timesteps=int(time_steps),callback=callback)
        # Save the agent
        print("[TRAIN] Saving model")
        model.save("dqn_maze")
        del model  # delete trained model to demonstrate loading


            

        # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
        # plt.show()

    print("[TEST] Loading model...")
    # Load the trained agent
    model = DQN.load("dqn_maze")

    env = MazeEnv( maze_file = "maze2d_002.npy",                  
        # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                    maze_size=(640, 640), 
                                    enable_render=True,
                                    do_track_trajectories=False,num_goals=1)

    # # Evaluate the agent
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

    # Enjoy trained agent
    utils.play_audio_notification()

    input("Press any key to start testing")
    print("[TEST] Starting episodes...")
    for ep_i in range(args.episodes):
        done = False
        ep_reward = 0

        env.seed(ep_i)  
        obs = env.reset()
        
        n_actions_taken = 0
        past_action = None


        while not done :
        # for _ in range(0,150): # DEBUG ONLY
            print("predicting")
            action, _states = model.predict(obs)
            print("action",action)
            
            obs, reward, done, info = env.step(action)
            ep_reward =  reward

            n_actions_taken +=1
            
            if n_actions_taken == args.actions_timeout:
                print("Episode ended: actions timeout reached")
                break

            env.render()

            # navMapper.update_nav_fig()
            if(not args.env2D):
                time.sleep(episode_cooldown)
            else:
                time.sleep(1)

        

        print("="*40)    
        print('Episode #{} Reward: {}'.format(ep_i+1, ep_reward))
    env.close()
