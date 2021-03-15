"""
    SOLO 2D MAZE, RL algo per avere  qlearning vs sarsa vs eSarsa
    ISSUE: Too exploration few exploitation
"""

import os
import sys
import numpy as np
import math
import random

import gym
from stable_baselines.common.policies import MlpLnLstmPolicy

import argparse
import datetime


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

import pandas



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


MAZE_SIZE = None
DECAY_FACTOR = None

MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2

def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))



# class SarsaAgent():
#     def __init__(self,env): 
#         self.env = env
#         self.maze=env.maze_view
#         nrow=self.maze.maze_size[0]
#         ncol=self.maze.maze_size[1]
#         state_number = nrow*ncol
#         action_number = env.actions.n
#         #implementing q as a dict - Not Yet
#         self.q={}
        
    
#     def get_q(self,state,action=None):
#         row,col = int(state[0]),int(state[1])
#         if action is None:
#             if (row,col) in self.q.keys():
#                 return self.q[(row,col)]
#             else:
#                 self.q[(row,col)]=np.full((4,),0.5)
#                 return self.q[(row,col)]
#         else:
#             if (row,col) in self.q.keys():
#                 return self.q[(row,col)][action]
#             else:
#                 self.q[(row,col)]=np.full((4,),0.5)
#                 return self.q[(row,col)][action]

#     def set_q(self,state,action,value):
#         row,col = int(state[0]),int(state[1]) 
#         if (row,col) not in self.q.keys():
#             self.q[(row,col)]=np.full((4,),0.5)
#         self.q[(row,col)][action]=value

#     def learn(self,episodes=1000, alpha=.5, gamma=.99, epsilon=0.1):
#         for i in range(episodes):
#             state = self.env.reset()
#             done=False
#             while not done:
#                 p=random.random()
#                 if p<epsilon:
#                     action=random.choice(self.env.actions)
#                 else:
#                     action=np.argmax(self.get_q(state))
                
#                 next_state,reward,done=self.maze.step(action)
                
#                 p=random.random()
#                 if p<epsilon:
#                     next_action=random.choice(self.env.actions)
#                 else:
#                     next_action=np.argmax(self.get_q(next_state))

#                 #Update Rule
#                 new_val=self.get_q(state,action)+alpha*(reward+gamma*self.get_q(next_state,next_action)-self.get_q(state,action))
#                 self.set_q(state,action,new_val)
#                 state = next_state
    
#     def get_policy(self,cell):
#         row,col=cell
#         state=(row,col,None)
#         action=np.argmax(self.get_q(state))
#         return action
    
#     # def get_policy_matrix():
#     #     policy=np.copy(self.maze.maze)
#     #     policy[maze.maze==1]==5


class Agent: 
    def __init__(self, method, env,start_alpha = 0.3, start_gamma = 0.9, start_epsilon = 0.5):
        """method: one of 'q_learning', 'sarsa' or 'expected_sarsa' """
        self.method = method
        self.env = env
        self.n_squares = env.maze_view.maze_size[0] * env.maze_view.maze_size[1] 

        self.n_actions = self.env.action_space.n
        self.epsilon = start_epsilon
        self.gamma = start_gamma
        self.alpha = start_alpha
        # one bucket per grid
        self.num_buckets = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
        
        self.max_t = np.prod(env.maze_view.maze_size, dtype=int) * 100

        # Set up initial q-table 
        # self.q = np.zeros(shape = (self.n_squares*self.n_passenger_locs*self.n_dropoffs, self.env.action_space.n))
        self.q = np.zeros(self.num_buckets + (self.n_actions,), dtype=float)
        # Set up policy pi, init as equiprobable random policy
        self.pi = np.zeros_like(self.q)
        for i in range(self.pi.shape[0]): 
            for a in range(self.n_actions): 
                self.pi[i,a] = 1/self.n_actions

    def fix_s(self,s):
        return  (int(s[0]),int(s[1]))

    def simulate_episode(self):
        s = self.env.reset()
        done = False
        r_sum = 0 
        n_steps = 0 
        gam = self.gamma
        for t in range(self.max_t):
            n_steps += 1
            # take action from policy
            x = np.random.random()
            s = self.fix_s(s)
            # print('self.pi[s,:]: ', self.pi,s)
            somma = np.cumsum(self.pi[s])
            # print('self.pi[s]: ', self.pi[s])
            # print('np.cumsum(self.pi[s]): ', np.cumsum(self.pi[s]))
            
            a = np.argmax( somma > x) 
            # take step 
            s_prime,r,done,info = self.env.step(a)  
            s_prime = self.fix_s(s_prime)

            if(args.enable_render):
                env.render()
            
            # print("Q:",self.q[(0,0)])
            index = s + (a,)
            if self.method == 'q_learning': 
                a_prime = np.random.choice(np.where(self.q[s_prime] == max(self.q[s_prime]))[0])
                index_prime = s_prime + (a_prime,) 
                self.q[index] +=   self.alpha * \
                    (r + gam*self.q[index_prime] - self.q[index])
            elif self.method == 'sarsa': 
                a_prime = np.argmax(np.cumsum(self.pi[s_prime,:]) > np.random.random())
                index_prime = s_prime + (a_prime,) 
                self.q[index] +=  self.alpha * \
                    (r + gam*self.q[index_prime] - self.q[index])
            elif self.method == 'expected_sarsa':
                # print('np.dot ', self.q[index])
                self.q[index] +=   self.alpha * \
                    (r + gam* np.dot(self.pi[s_prime],self.q[s_prime]) - self.q[index])
            else: 
                raise Exception("Invalid method provided")
            # update policy
            # print('self.q[s]: ', self.q[s],s)
            best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
            for i in range(self.n_actions): 
                if i == best_a:      self.pi[s,i] = 1 - (self.n_actions-1)*(self.epsilon / self.n_actions)
                else:                self.pi[s,i] = self.epsilon / self.n_actions

            # decay gamma close to the end of the episode
            # if n_steps > 185: 
            #     gam *= 0.875

            if(done): break

            s = s_prime
            r_sum += r
        return r_sum


def train_agent(agent: Agent, n_episodes= 50000, epsilon_decay = 0.99995, alpha_decay = 0.99995, print_trace = False):
    r_sums = []
    if(args.enable_render):
        env.render()
    print("Training is starting ...")
    for ep in range(n_episodes): 
        print("Episode:", ep)
        r_sum = agent.simulate_episode()
        # decrease epsilon and learning rate 
        # agent.epsilon *= epsilon_decay
        # agent.alpha *= alpha_decay
        agent.epsilon = get_explore_rate(ep)
        agent.alpha = get_learning_rate(ep)

        if print_trace: 
            print( "\tr:",r_sum,"alpha:", np.round(agent.alpha, 3), "epsilon:",  np.round(agent.epsilon, 3))
            if ep % 2000 == 0 and ep > 0 : 
                print ("Last 100 episodes avg reward: ", np.mean(r_sums[ep-100:ep]))
        r_sums.append(r_sum)
    try:
        utils.play_audio_notification(n_beeps=2)
    except:
        pass
    return r_sums



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


STD_MAZE = "maze2d_004.npy"

INTERACTIVE = False
OUT_FORMAT = "csv"

SEED = 12
TIMESTEPS = 100000
maze_file = STD_MAZE
mode = "train"


# Approccio 1 RL per init SOLO 2D 
for fixed_init_pos in fixed_init_pos_list:
        
    env = MazeEnv( maze_file = maze_file,                  
                # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                            maze_size=(640, 640), 
                                            enable_render= args.enable_render if(mode=="train") else True,
                                            do_track_trajectories=True,num_goals=args.n_goals, measure_distance = True,
                                            verbose = args.v,n_trajs=args.n_trajs,random_pos = args.random_pos,seed_num = SEED,
                                            fixed_goals = fixed_goals,fixed_init_pos = fixed_init_pos)



    MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)


    # Instantiate the env
    # wrap it
    # env = make_vec_env(lambda: env, n_envs=1)

    # model = ACKTR('MlpPolicy', env, verbose=1).learn(100000)
    # model = DQN("MlpPolicy",env, verbose=1)

    # Create agents 
    sarsa_agent = Agent(method='sarsa',env=env,
        start_epsilon=get_explore_rate(0),start_alpha=get_learning_rate(0),start_gamma=0.99)
    e_sarsa_agent = Agent(method='expected_sarsa',env=env,
        start_epsilon=get_explore_rate(0),start_alpha=get_learning_rate(0),start_gamma=0.99)
    q_learning_agent = Agent(method='q_learning',env=env,
        start_epsilon=get_explore_rate(0),start_alpha=get_learning_rate(0),start_gamma=0.99)

    # Train agents
    r_sums_q_learning = train_agent(q_learning_agent, 
    print_trace=True)
    r_sums_sarsa = train_agent(sarsa_agent, print_trace=True,epsilon_decay=DECAY_FACTOR)
    r_sums_e_sarsa = train_agent(e_sarsa_agent, print_trace=True)

        
    df = pandas.DataFrame({"Sarsa": r_sums_sarsa, 
                "Expected_Sarsa": r_sums_e_sarsa, 
                "Q-Learning": r_sums_q_learning})
    df_ma = df.rolling(100, min_periods = 100).mean()
    df_ma.iloc[1:1000].plot(title = "Init from: "+str(fixed_init_pos[0])
        +","+str(fixed_init_pos[1]),
        kind = "scatter" )