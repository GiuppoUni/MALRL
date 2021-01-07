import itertools
import os
import numpy as np
import sys
import math
import random

import gym
import gym_maze

import argparse
import datetime
from gym_airsim.envs.collectEnv import CollectEnv

import gym_airsim.envs
import gym_airsim
from gym_airsim.envs import AirSimEnv
import utils
import time
from gym_maze.envs.maze_env import MazeEnv
from trajectoryTrackerClient import TrajectoryTrackerClient

import signal
import sys
import pandas
import shutil
import queue
import trajs_utils 



IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}


STD_MAZE = "maze2d_004.npy"

INTERACTIVE = False
OUT_FORMAT = "csv"
RANDOM_TRAJECTORIES_FOLDER = "rtrajectories/csv/"
TRAJECTORIES_FOLDER = "qtrajectories/csv/" 
TRAJECTORIES_3D_FOLDER = "trajectories_3d/csv/"

SEED = random.randint(0,int(10e6))




def main(mode, fixed_init_pos=None, trainedQtable=None,visited_cells = [],i_trajectory = None):
    assert(mode in ["train","test","random"])
    

    if(args.load_maze):
        maze_file = args.load_maze
    else:
        maze_file = STD_MAZE

    env = MazeEnv( maze_file = maze_file,                  
        # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                    maze_size=(640, 640), 
                                    enable_render=args.render_train if(mode in ["train","random"]) else args.render_test,
                                    do_track_trajectories=True,num_goals=args.n_goals, measure_distance = True,
                                    verbose = args.v,n_trajs=args.n_random_init,random_pos = args.random_pos,seed_num = SEED,
                                    fixed_goals = fixed_goals,fixed_init_pos = fixed_init_pos,
                                    visited_cells = visited_cells)

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
    print('DECAY_FACTOR: ', DECAY_FACTOR)

    '''
    Defining the simulation related constants
    '''
    # MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
    MAX_T = 10000
    
    print('MAX_T: ', MAX_T)

    STREAK_TO_END = 100
    SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
    DEBUG_MODE = 0


            
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

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!\nSaving...')
        if(not args.load_qtable):
            print("ENDING OF TRAIN")
            np.save("results/q_table_"+str(i_trajectory)+"_"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M')),q_table )
            print("Table saved")
        sys.exit(0)
    '''
        Handle SIGINT saving qTable
    '''
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C to stop')

    qtable = []
    if(args.n_agents == 1):
        '''
        Creating a Q-Table for each state-action pair
        '''
        if(trainedQtable is None):
            if(args.load_qtable ):
                q_table = np.load(args.load_qtable)
            else:
                # Single agents
                q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)
        else:
            # print('trainedQtable: ', trainedQtable)
            q_table = trainedQtable

  

        # Instantiating the learning related parameters
        learning_rate = get_learning_rate(0)
        print('learning_rate: ', learning_rate)
        explore_rate = get_explore_rate(0)
        print('explore_rate: ', explore_rate)
        discount_factor = 0.99

        num_streaks = 0

        assert type(env) == MazeEnv

        # Render tha maze
        if(env.enable_render ):
            env.render()

        print("Learning starting...") 
        print("Init pose:", fixed_init_pos  )
        print("env.maze_size",env.maze_size)
        
        if(mode in ["train","random"]):
            n_episodes = args.episodes
        else:
            n_episodes = 1
            if(INTERACTIVE):
                print("Enter any key to start testing")
                input()
     
        
        if(args.log_reward and mode in ["train","random"] ):
            logOutfile ="logs/log-"+mode+"-"+str(i_trajectory)+\
                                "-"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt"
            rewLogFile=open(logOutfile, 'w')
        
        if(mode in ["train","test"]):
            #================#
            #       '''      #
            # Begin training/testing #
            #       '''      #
            #================#
            
            qtrajectory = []
            start = time.time()
            for episode in range(n_episodes):

                # Reset the environment
                obv = env.reset()
                env.seed(episode)
                # the initial state
                old_state = state_to_bucket(obv)
                total_reward = 0

                qtrajectory = []
                last_s_a_queue = []
                for t in range(MAX_T):
                    # Select an action
                    action = select_action(old_state, explore_rate)

                    # execute the action
                    obv, reward, done, info = env.step(action)
                    
                    if(mode=="test" and info["moved"]==True):
                        # Append to trajectory
                        # print("obv,qtrajectory",obv,qtrajectory)
                        qtrajectory.append(list(obv))

                    # Observe the result
                    state = state_to_bucket(obv)
                    total_reward += reward
                    
                    # Update queue
                

                    # Update the Q based on the result
                    best_q = np.amax(q_table[state])
                    
                    
                    # Q-Routing UPDATE
                    q_table[old_state + (action,)] += learning_rate * (reward +
                        discount_factor * (best_q) - q_table[old_state + (action,)])
                    
                    # Setting up for the next iteration
                    old_state = state
                    if(args.n_steps > 0 ):
                        if(len(last_s_a_queue)>= args.n_steps):
                            last_s_a_queue.pop(0)
                        last_s_a_queue += [ [state,action] ]
                    

                    # Print data
                    if DEBUG_MODE == 2 :
                        print("\n-----------------------------------------------")
                        print("Episode = %d" % episode)
                        print("-----------------------------------------------")
                        print("t = %d" % t)
                        print("Action: %d" % action)
                        print("State: %s" % str(state))
                        print("Reward: %f" % reward)
                        print("Total reward: %f" % total_reward)
                        print("Best Q: %f" % best_q)
                        print("Explore rate: %f" % explore_rate)
                        print("Learning rate: %f" % learning_rate)
                        print("Streaks: %d" % num_streaks)
                        print("")

                    elif DEBUG_MODE == 1 :
                        if done or t >= MAX_T - 1:
                            print("\nEpisode = %d" % episode)
                            print("t = %d" % t)
                            print("Explore rate: %f" % explore_rate)
                            print("Learning rate: %f" % learning_rate)
                            print("Streaks: %d" % num_streaks)
                            print("Total reward: %f" % total_reward)
                            print("")

                    # Render tha maze
                    if env.enable_render:
                        env.render()

                    if env.is_game_over():
                        sys.exit()

                    if done:
                        # print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                        #     % (episode, t, total_reward, num_streaks))
                        print("%d,%f,%f" % (episode, t, total_reward))
                        if(args.log_reward and mode in ["train","random"] ):  rewLogFile.write("%d,%f,%f\n" % (episode, t, total_reward))

                        # NOTE: Q-Routing N-STEPS 
                        for idx in range(len(last_s_a_queue)):
                            if(idx+1==len(last_s_a_queue)): break
                            cur_s, cur_a= last_s_a_queue[idx][0], last_s_a_queue[idx][1]
                            next_s,next_a = last_s_a_queue[idx+1][0], last_s_a_queue[idx+1][1]
                            td_best_q = q_table[next_s + (next_a,) ]

                            q_table[cur_s + (next_a,)] += learning_rate * (reward + 
                                discount_factor * (td_best_q) - q_table[ next_s + (next_a,)])


                        if t <= SOLVED_T:
                            num_streaks += 1
                        else:
                            num_streaks = 0
                        break

                    elif t >= MAX_T - 1:
                        print("%d,%f,%f,stopped\n" % (episode, t, total_reward) )
                        if(args.log_reward and mode in ["train","random"]):  rewLogFile.write("%d,%f,%f\n" % (episode, t, total_reward))

                    if(args.slow):
                        time.sleep(1)

                # It's considered done when it's solved over 120 times consecutively
                if not args.episodes and num_streaks > STREAK_TO_END:
                    break

                # Update parameters
                explore_rate = get_explore_rate(episode)
                learning_rate = get_learning_rate(episode)
                
            # EPISODES ENDED
            end = time.time()
            if(args.log_reward and mode in ["train","random"]):  
                rewLogFile.write("ELAPSEDTIME:"+str(end-start))
                rewLogFile.close()
            
            if(not args.load_qtable):
                print("ENDING OF TRAIN")
                np.save("results/q_table_"+str(i_trajectory)+"_"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M')),q_table )
                print("Table saved")

            if(mode =="test"):
                utils.play_audio_notification()
                outfile ="q_traj-"+str(i_trajectory)+"-"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))
                toBeSaved = np.array(qtrajectory,dtype=int)
                print('Saving in : ', outfile)
                
                if(OUT_FORMAT == "csv"):
                    # print('toBeSaved: ', toBeSaved)
                    df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
                    # df["z_pos"] = -10
                    df.index.name = "index"
                    df.to_csv(TRAJECTORIES_FOLDER+outfile+".csv")

                elif(OUT_FORMAT == "npy"):
                    np.save(TRAJECTORIES_FOLDER+outfile,toBeSaved )
                else:
                    raise Exception("Invalid out format:",OUT_FORMAT)
            # utils.play_audio_notification()

            return q_table,qtrajectory
        
        
        else:
            # RANDOM MODE
            maxTrajectory=[]
            max_reward = -1
            
            for i_episode in range(n_episodes):
                total_reward = 0
                observation = env.reset() # reset for each new trial
                trajectory = []
                for t in range(MAX_T): # run for 100 timesteps or until done, whichever is first
                    if(env.enable_render):
                        env.render()
                    action = env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
                    observation, reward, done, info = env.step(action)
                    total_reward += reward
                    if( info["moved"]==True):
                        # Append to trajectory
                        # print("obv,qtrajectory",obv,qtrajectory)
                        trajectory.append(list(observation))
                    if done:
                        print("%d,%f,%f" % (i_episode, t, total_reward))
                        if(total_reward > max_reward):
                            max_reward = total_reward
                            maxTrajectory = trajectory
                        break
                    elif t >= MAX_T - 1:
                        print("%d,%f,%f,stopped" % (i_episode, t, total_reward) )
                        if(args.log_reward and mode in ["train","random"]):  rewLogFile.write("%d,%f,%f\n" % (i_episode, t, total_reward))
                    
            utils.play_audio_notification()
            outfile ="r_traj-"+str(i_trajectory)+"-"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))
            toBeSaved = np.array(maxTrajectory,dtype=int)
            print('Saving in : ', outfile)
            
            if(OUT_FORMAT == "csv"):
                # print('toBeSaved: ', toBeSaved)
                df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
                # df["z_pos"] = -10
                df.index.name = "index"
                df.to_csv(RANDOM_TRAJECTORIES_FOLDER+outfile+".csv")

            elif(OUT_FORMAT == "npy"):
                np.save(RANDOM_TRAJECTORIES_FOLDER+outfile,toBeSaved )
            else:
                raise Exception("Invalid out format:",OUT_FORMAT)
            return None, maxTrajectory

    else:
        raise Exception("args.n_agents "+str(args.n_agents)+" not yet supported")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Layer 1')
    
    
    parser.add_argument('--episodes', type=int, default=100,
                        help='episodes (default: %(default)s)')

    parser.add_argument('--n-goals', type=int, default=1,
                        help='n goals to collect (default: %(default)s)')

    parser.add_argument("--seed",type=int,
                         help='seed value (default: %(default)s)')

    parser.add_argument('--n-agents', type=int, default=1,
                        help='num agents (supported 1 )(default: %(default)s)')

    parser.add_argument('--n-steps', type=int, default=0,
                        help='enforce n-steps qlearning if 0 is standard qlearning  (default: %(default)s)')

    parser.add_argument( '--debug',action='store_true',  default=False,
        help='Log debug in file (default: %(default)s)' )
    
    parser.add_argument('--render-train',action='store_true',  default=False,
        help='render maze while training/random  (default: %(default)s)' )
    
    parser.add_argument( '--render-test',action='store_true',  default=False,
        help='render maze while testing  (default: %(default)s)' )

    parser.add_argument('--random-mode',action='store_true',  default=False,
        help='Agent takes random actions (default: %(default)s)' )

    parser.add_argument( '--skip-train',action='store_true',  default=False,
        help='Just assign altitude to 2D trajecories in folder (default: %(default)s)' )
    
    parser.add_argument( '--show-plot',action='store_true',  default=False,
        help='Show generated trajectories each time (default: %(default)s)' )
    
    parser.add_argument( '-v',action='store_true',  default=False,
        help='verbose (default: %(default)s)' )

    parser.add_argument('--random-pos',action='store_true',  default=False,
        help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

    parser.add_argument('--slow',action='store_true',  default=False,
        help='Slow down training to observe behaviour (default: %(default)s)')
    
    # parser.add_argument('--avoid-traj',action='store_true',  default=False,
    #     help='reward on avoiding previous trajs path (default: %(default)s)')

    # parser.add_argument('--track-traj',action='store_true',  default=False,
    #     help='Track trajectories into file (default: %(default)s)')

    # parser.add_argument('--col-traj', action='store_true', default=False,
    # help='Track trajectories into file (default: %(default)s)')

    parser.add_argument('--n-random-init', type=int, default=5,
                        help='n sample pool for random init (default: %(default)s)')

    parser.add_argument('--log-reward', action='store_true', default=False,
    help='log reward file in out (default: %(default)s)')

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

    if(args.seed):
        SEED = args.seed
        random.seed(SEED)
        np.random.seed(seed=SEED)

         


    if(not args.skip_train):
        # Resetting folders
        for f in os.listdir(TRAJECTORIES_FOLDER):
            os.remove(TRAJECTORIES_FOLDER+f)


        if(args.debug):
            logger = utils.initiate_logger()
            print = logger.info


   

        if(args.load_init_pos ):
            df = pandas.read_csv(args.load_init_pos, index_col='name')
            # print(df)
            fixed_init_pos_list = df.to_numpy()
            # print('fixed_goals: ', fixed_goals)
        else:
            fixed_init_pos_list = None


        df = pandas.read_csv("fixed_goals.csv", index_col='name')
        # print(df)
        goals_pool = df.to_numpy()
        assert(len(goals_pool) > 0)
        df = pandas.read_csv("init_pos.csv", index_col='name')
        # print(df)
        fixed_init_pos_list = df.to_numpy()
        assert(len(fixed_init_pos_list) > 0)
        # print('fixed_goals: ', fixed_goals)

        if(fixed_init_pos_list is not None):
            if(not args.random_mode):
                visited_cells = []
                trajs = []
                for i_trajectory,fixed_init_pos in enumerate(fixed_init_pos_list):
                    fixed_goals = [ goals_pool[i_trajectory % len(goals_pool)] ]
                    # print('fixed_goals: ', fixed_goals)
                    qtable,_ = main(mode = "train",fixed_init_pos=fixed_init_pos,visited_cells = visited_cells,i_trajectory = i_trajectory)
                    _,traj =  main(mode = "test",trainedQtable=  qtable,fixed_init_pos=fixed_init_pos,i_trajectory = i_trajectory)
                    
                    trajs.append(traj)
                    
                    # Remove duplicates from single traj
                    visited_cells += list(num for num,_ in itertools.groupby(traj)) 
                    # Remove duplicates from all trajs
                    visited_cells = list(num for num,_ in itertools.groupby(visited_cells))
            else:
                visited_cells = []
                trajs = []
                for i_trajectory,fixed_init_pos in enumerate(fixed_init_pos_list):
                    fixed_goals = [ goals_pool[i_trajectory % len(goals_pool)] ]
                    _,traj = main(mode = "random",fixed_init_pos=fixed_init_pos,visited_cells = visited_cells,i_trajectory = i_trajectory)                    
                    trajs.append(traj)
                
        print("Trained and tested")


    print("Loading generated trajectories")
    traj_files_list = os.listdir(TRAJECTORIES_FOLDER)
    trajs = []
    for tf in traj_files_list:
        print("Reading:",tf)
        df = pandas.read_csv(TRAJECTORIES_FOLDER+tf,delimiter=",",index_col="index")
        # print(df)
        traj = df.to_numpy().tolist()
        trajs.append(traj)

    # print(trajs)

    print("Transforming 2D trajs into 3D trajs")
    # # trees = utils.build_trees(trajs)
    # # trajs3d, zs = trajs_utils.avoid_collision_in_empty_space(trajs,-50,-5,5,3)
    # # trajs_utils.plot_2d(trajs)
    # trajs3d = trajs_utils.avoid_collision_complex(trajs[2:4],min_height=-50,max_height=-5,sep_h = 1,radius=10)
    # # trajs_utils.plot_2d(trajs[0:2])
    # # trajs_utils.plot_2d([ trajs_utils.np_remove_z(t) for t in trajs3d[2:] ])
    # ntrajs3d = trajs_utils.avoid_collision_complex(trajs[0:2],assigned_trajs=trajs3d,min_height=-50,
    #     max_height=-5,sep_h = 1,radius=1,threshold=150)
    
    # trajs_utils.plot_3d(ntrajs3d+trajs3d)

    trajs3d = trajs_utils.avoid_collision_complex(trajs,min_height=-50,max_height=-5,sep_h = 1,radius=10)



    # Remove old trajectories files 
    for f in os.listdir(TRAJECTORIES_3D_FOLDER):
        os.remove(TRAJECTORIES_3D_FOLDER+f)
    for idx,traj in enumerate(trajs3d):
        traj = np.array(traj)
        df = pandas.DataFrame({'x_pos': traj[:, 0], 'y_pos': traj[:, 1],
        'z_pos': traj[:, 2]})
        df.index.name = "index"
        df.to_csv("trajectories_3d/csv/"+traj_files_list[idx])
        print("saved to","trajectories_3d/csv/"+traj_files_list[idx])

    trajs_utils.plot_3d(trajs3d)

    trajs_utils.plot_2d(trajs)
    # trajs_utils.height_algo(trajs)

    