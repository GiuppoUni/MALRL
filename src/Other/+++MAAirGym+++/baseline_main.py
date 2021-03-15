"""
    Baseline wrapper with monitor to train, monitor, test, benchmark
    different baseline RL algos (best for discrete non img: A2C)
    
    ISSUE: even with lot of episodes reward increased in training
    but testing predicted path is not sub-optimal
"""

import os
import sys
import numpy as np
import math
import random

import gym
from stable_baselines.common.policies import ActorCriticPolicy, MlpLnLstmPolicy
import gym_maze

import argparse
import datetime


import gym_airsim.envs
import gym_airsim
from stable_baselines.common.env_checker import check_env

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


from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env
import pandas




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

# episode_cooldown = 3

# ACTION_TO_IDX = {"LEFT":0, "FRONT":1, "RIGHT":2,"BACK" : 3}
# IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}

# STD_MAZE = "maze2d_002.npy"


    



# # ==================================================================================================


# if __name__ == "__main__":


    


#     parser = argparse.ArgumentParser(description='RL for ma-gym')
#     parser.add_argument('--episodes', type=int, default=100,
#                         help='episodes (default: %(default)s)')

#     parser.add_argument('--n-goals', type=int, default=1,
#                         help='episodes (default: %(default)s)')

#     parser.add_argument('--actions-timeout', type=int, default=100,
#                         help='episodes (default: %(default)s)')

#     parser.add_argument("--n-trajs",type=int, default=5,
#                          help='num trajs to track (default: %(default)s)')


#     parser.add_argument('--n-agents', type=int, default=1,
#                         help='num agents (default: %(default)s)')

#     # parser.add_argument('--ep-cooldown', type=int, default=1,
#     #                     help='episode cooldown time sleeping (default: %(default)s)')

#     parser.add_argument( '--debug',action='store_true',  default=False,
#         help='Log into file (default: %(default)s)' )
    
#     parser.add_argument( '--enable-render',action='store_true',  default=False,
#         help='Log into file (default: %(default)s)' )
    
#     parser.add_argument( '-v',action='store_true',  default=False,
#         help='verbose (default: %(default)s)' )

#     parser.add_argument('--random-pos',action='store_true',  default=False,
#         help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

#     parser.add_argument('--env2D',action='store_true',  default=True,
#         help='(default: %(default)s)')

#     parser.add_argument('--can-go-back',action='store_true',  default=False,
#         help='(default: %(default)s)')

#     parser.add_argument('--custom-random',action='store_true',  default=False,
#         help='(default: %(default)s)')

#     parser.add_argument('--slow',action='store_true',  default=False,
#         help='(default: %(default)s)')

#     parser.add_argument('--track-traj',action='store_true',  default=False,
#         help='Track trajectories into file (default: %(default)s)')

#     parser.add_argument('--col-traj', action='store_true', default=False,
#     help='Track trajectories into file (default: %(default)s)')

#     parser.add_argument('--load-qtable', type=str, 
#         help='qtable file (default: %(default)s)')

#     parser.add_argument('--load-maze', type=str, 
#         help='maze file (default: %(default)s)')

    

#     args = parser.parse_args()

#     # env = gym.make("AirSimEnv-v1")

#     if(args.debug):
#         logger = utils.initiate_logger()
#         print = logger.info

#     if(args.load_maze):
#         maze_file = args.load_maze
#     else:
#         maze_file = STD_MAZE
#     env = MazeEnv( maze_file = maze_file,                  
#         # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
#                                     maze_size=(640, 640), 
#                                     enable_render=args.enable_render,
#                                     do_track_trajectories=True,num_goals=args.n_goals, measure_distance = True,
#                                     verbose = args.v,n_trajs=args.n_trajs,random_pos = args.random_pos)




#     # Create log dir
#     log_dir = "tmp/"
#     os.makedirs(log_dir, exist_ok=True)

#     # Create and wrap the environment
#     # env = Monitor(env, log_dir)

#     # Create the callback: check every 1000 steps
#     callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
#     # callback = None

#     # Train the agent
#     time_steps = 2e5
    
#     # Instantiate the agent
#     model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
#     # Train the agent
#     print("[TRAIN] Learning started...")
#     model.learn(total_timesteps=int(time_steps),callback=callback)
#     # Save the agent
#     print("[TRAIN] Saving model")
#     model.save("dqn_maze")
#     del model  # delete trained model to demonstrate loading


        

#     # results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG LunarLander")
#     # plt.show()

#     print("[TEST] Loading model...")
#     # Load the trained agent
#     model = DQN.load("dqn_maze")

  

#     # # Evaluate the agent
#     # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

#     # Enjoy trained agent
#     utils.play_audio_notification()

#     input("Press any key to start testing")
#     print("[TEST] Starting episodes...")
#     for ep_i in range(args.episodes):
#         done = False
#         ep_reward = 0

#         env.seed(ep_i)  
#         obs = env.reset()
        
#         n_actions_taken = 0
#         past_action = None


#         while not done :
#         # for _ in range(0,150): # DEBUG ONLY
#             print("predicting")
#             action, _states = model.predict(obs)
#             print("action",action)
            
#             obs, reward, done, info = env.step(action)
#             ep_reward =  reward

#             n_actions_taken +=1
            
#             if n_actions_taken == args.actions_timeout:
#                 print("Episode ended: actions timeout reached")
#                 break

#             env.render()

#             # navMapper.update_nav_fig()
#             if(not args.env2D):
#                 time.sleep(episode_cooldown)
#             else:
#                 time.sleep(1)

        

#         print("="*40)    
#         print('Episode #{} Reward: {}'.format(ep_i+1, ep_reward))
#     env.close()



parser = argparse.ArgumentParser(description='RL for ma-gym')
parser.add_argument('--episodes', type=int, default=100,
                    help='episodes (default: %(default)s)')

parser.add_argument('--n-goals', type=int, default=1,
                    help='episodes (default: %(default)s)')

parser.add_argument('--actions-timeout', type=int, default=100,
                    help='episodes (default: %(default)s)')

parser.add_argument("--n-trajs",type=int, default=5,
                        help='num trajs to track (default: %(default)s)')


parser.add_argument('--n-agents', type=int, default=1,
                    help='num agents (default: %(default)s)')
# parser.add_argument('--ep-cooldown', type=int, default=1,
#                     help='episode cooldown time sleeping (default: %(default)s)')

parser.add_argument( '--debug',action='store_true',  default=False,
    help='Log into file (default: %(default)s)' )

parser.add_argument( '--enable-render',action='store_true',  default=False,
    help='Log into file (default: %(default)s)' )

parser.add_argument( '-v',action='store_true',  default=False,
    help='verbose (default: %(default)s)' )

parser.add_argument('--random-pos',action='store_true',  default=False,
    help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

parser.add_argument('--env2D',action='store_true',  default=True,
    help='(default: %(default)s)')

parser.add_argument('--can-go-back',action='store_true',  default=False,
    help='(default: %(default)s)')

parser.add_argument('--custom-random',action='store_true',  default=False,
    help='(default: %(default)s)')

parser.add_argument('--slow',action='store_true',  default=False,
    help='(default: %(default)s)')

parser.add_argument('--track-traj',action='store_true',  default=False,
    help='Track trajectories into file (default: %(default)s)')

parser.add_argument('--col-traj', action='store_true', default=False,
help='Track trajectories into file (default: %(default)s)')

parser.add_argument('--load-qtable', type=str, 
    help='qtable file (default: %(default)s)')

parser.add_argument('--load-maze', type=str, 
    help='maze file (default: %(default)s)')


parser.add_argument('--load-goals', type=str, 
    help='maze file (default: %(default)s)')

parser.add_argument('--load-init-pos', type=str, 
    help='maze file (default: %(default)s)')


args = parser.parse_args()

# env = gym.make("AirSimEnv-v1")



if(args.debug):
    logger = utils.initiate_logger()
    print = logger.info



df = pandas.read_csv("fixed_goals.csv", index_col='name')
# print(df)
fixed_goals = df.to_numpy()
if(len(fixed_goals)<1):
    raise Exception("Inavalid num of goals")
# print('fixed_goals: ', fixed_goals)


df = pandas.read_csv("init_pos.csv", index_col='name')
# print(df)
fixed_init_pos_list = df.to_numpy()
# print('fixed_goals: ', fixed_goals)



episode_cooldown = 3

ACTION_TO_IDX = {"LEFT":0, "FRONT":1, "RIGHT":2,"BACK" : 3}
IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}


STD_MAZE = "maze2d_002.npy"

INTERACTIVE = False
OUT_FORMAT = "csv"

SEED = 12
TIMESTEPS = int(1e5)
maze_file = STD_MAZE
mode = "train"

for fixed_init_pos in fixed_init_pos_list:
        
    env = MazeEnv( maze_file = maze_file,                  
                # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                            maze_size=(640, 640), 
                                            enable_render= args.enable_render if(mode=="train") else True,
                                            do_track_trajectories=True,num_goals=args.n_goals, measure_distance = True,
                                            verbose = args.v,n_trajs=args.n_trajs,random_pos = args.random_pos,seed_num = SEED,
                                            fixed_goals = fixed_goals,fixed_init_pos = fixed_init_pos)



    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    check_env(env)
    env = gym.make('uav-maze-v0')
    # Create and wrap the environment
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)

    # Instantiate the env
    # wrap it
    # env = make_vec_env(lambda: env, n_envs=1)
    print("Training started.")
    # model = DQN("MlpPolicy",env, verbose=1,)
    # model = ACKTR('MlpPolicy', env, verbose=1).learn(100000)
    # model = PPO2(MlpLnLstmPolicy,env,verbose=1,
    #     learning_rate=0.2,seed = SEED,nminibatches=1)
    model = A2C("MlpPolicy",env, verbose=1,seed=1,
    gamma = 0.99,learning_rate=0.8,epsilon=0.8,
    n_steps=3000)
    model.learn(total_timesteps=TIMESTEPS,callback=callback)

   
    utils.play_audio_notification(n_beeps=2)
    
    results_plotter.plot_results([log_dir], TIMESTEPS, results_plotter.X_TIMESTEPS, str(type(model)))

    plt.show()

    # env.maze_view.enable_render = True 
    env = MazeEnv( maze_file = maze_file,                  
                # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                            maze_size=(640, 640), 
                                            enable_render= True,
                                            do_track_trajectories=True,num_goals=args.n_goals, measure_distance = True,
                                            verbose = args.v,n_trajs=args.n_trajs,random_pos = args.random_pos,seed_num = SEED,
                                            fixed_goals = fixed_goals,fixed_init_pos = fixed_init_pos)


    obs = env.reset()
    n_steps = 10000

    print("Testing")
    env.render()
    qtrajectory = []
    done = False
    for step in range(n_steps):

        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        qtrajectory.append(list(obs))
        print('obs=', obs, 'reward=', reward, 'done=', done)
        env.render()
        if(args.slow):
            time.sleep(0.5)
        if done:
            # Note that the VecEnv resets automatically
            # when a done signal is encountered
            print("Goal reached!", "reward=", reward)
            break
    
    # Save traj file
    toBeSaved = np.array(qtrajectory,dtype=int)
    df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
    df["z_pos"] = -10
    outfile ="q_traj_"+str(fixed_init_pos[0]) +str(fixed_init_pos[1])+\
                    str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))
    df.to_csv("qtrajectories/csv/"+outfile+".csv")

