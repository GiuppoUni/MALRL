import itertools
import logging
import os
import gym
import numpy as np
import sys
import math
import random

import argparse
import datetime

import utils
import time
from gym_maze.envs.maze_env import MazeEnv

import signal
import sys
import pandas

import trajs_utils
import yaml

configYml = utils.read_yaml("inputData/config.yaml")
c_paths = configYml["layer1"]["paths"]
c_settings = configYml["layer1"]["settings"]

IDX_TO_ACTION =  {0:"LEFT", 1:"FRONT", 2:"RIGHT", 3:"BACK"}

SEED = random.randint(0,int(10e6))

EXPERIMENT_DATE =  str(datetime.datetime.now().strftime('-D-%d-%m-%Y-H-%H-%M-%S-') ) # To be used in log and prints


def getNotWallsCells():
    goodCells = []
    for r in range(c_settings["NROWS"]):
        for c in range(c_settings["NCOLS"]):
            if( r % 7==0 or c % 7==0):
                goodCells.append([r,c])
    return goodCells


def select_action(state, explore_rate):
   # Select a random action
   if random.random() < explore_rate:
         action = env.action_space.sample()
   # Select the action with the highest q
   else:
         action = int(np.argmax(q_table[state]))
   return action


def get_explore_rate(t):
   return max(c_settings["MIN_EXPLORE_RATE"], min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
   return max(c_settings["MIN_LEARNING_RATE"], min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

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

'''
   Handle SIGINT saving qTable
'''
def signal_handler(sig, frame):
   print('You pressed Ctrl+C!\nSaving...')
   if(not args.load_qtable):
         print("ENDING OF TRAIN")
         np.save("generatedData/qTable/q_table_"+EXPERIMENT_DATE, q_table )
         print("Table saved")
   sys.exit(0)

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Layer 1')
   
   
   parser.add_argument('--nepisodes', type=int, default=100,
                     help='episodes (default: %(default)s)')

   parser.add_argument('--ngoals', type=int, default=1,
                     help='n goals to collect (default: %(default)s)')

   parser.add_argument("--seed",type=int,
                        help='seed value (default: %(default)s)')

   parser.add_argument("--ntrajs",type=int,
                        help='num trajectories value (default: %(default)s)')

   parser.add_argument("--nbuffer",type=int, default=3,
                        help='size of buffer for past trajectories (default: %(default)s)')


   parser.add_argument('--nagents', type=int, default=1,
                     help='num agents (supported 1 )(default: %(default)s)')

   parser.add_argument('--nsteps', type=int, default=0,
                     help='enforce n-steps qlearning if 0 is standard qlearning  (default: %(default)s)')

   parser.add_argument( '--debug',action='store_true',  default=False,
      help='Log debug in file (default: %(default)s)' )
   
   parser.add_argument('--render-train',action='store_true',  default=False,
      help='render maze while training/random  (default: %(default)s)' )
   
   parser.add_argument( '--render-test',action='store_true',  default=False,
      help='render maze while testing  (default: %(default)s)' )

   parser.add_argument( '--quiet',action='store_true',  default=False,
      help='render maze while testing  (default: %(default)s)' )

   parser.add_argument('--random-mode',action='store_true',  default=False,
      help='Agent takes random actions (default: %(default)s)' )

   parser.add_argument( '--skip-train',action='store_true',  default=False,
      help='Just assign altitude to 2D trajecories in folder (default: %(default)s)' )
   
   parser.add_argument( '--show-plot',action='store_true',  default=False,
      help='Show generated trajectories each time (default: %(default)s)' )
   
   parser.add_argument( '-v',action='store_true',  default=False,
      help='verbose (default: %(default)s)' )
   
   parser.add_argument( '--randomInitGoal',action='store_true',  default=False,
      help='Random init and goal per episode (default: %(default)s)' )
   
   parser.add_argument('--random-pos',action='store_true',  default=False,
      help='Drones start from random positions exctrateced from pool of 10 (default: %(default)s)')

   parser.add_argument('--slow',action='store_true',  default=False,
      help='Slow down training to observe behaviour (default: %(default)s)')

   parser.add_argument('--plot3d',action='store_true',  default=False,
      help='Render 3d plots(default: %(default)s)')

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




   if(args.seed):
      SEED = args.seed
      random.seed(SEED)
      np.random.seed(seed=SEED)

   """
      Creo celle pool da estrarre
   """

   goodCells = getNotWallsCells() # related to our maze design

   outs = 0
   trajsWithAltitude = []
   trajsBySteps = []
   fids = []

   logging.basicConfig(filename=utils.LOG_FOLDER+"log"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                           filemode='w',
                           format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logging.INFO)
   logger = logging.getLogger('RL Layer1')
   logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )


   if(args.load_init_pos ):
      df = pandas.read_csv(args.load_init_pos, index_col='name')
      # print(df)
      fixed_init_pos_list = df.to_numpy()
      # print('fixed_goals: ', fixed_goals)
   else:
      fixed_init_pos_list = None

   TRAJECTORIES_BUFFER_SIZE = args.nbuffer


   """
      Carico pool di input 
   """

   df = pandas.read_csv("inputData/fixed_goals.csv", index_col='name')
   # print(df)
   goals_pool = df.to_numpy()
   assert(len(goals_pool) > 0)
   df = pandas.read_csv("inputData/init_pos.csv", index_col='name')
   # print(df)
   fixed_init_pos_list = df.to_numpy()
   assert(len(fixed_init_pos_list) > 0)
   
   # print('fixed_goals: ', fixed_goals)


   ''''
   Preparo stuff pre training
   '''

   maze_file = c_paths["STD_MAZE"]
            
   # Creo Maze
   print("SEED",SEED)

   # # TODO bin flag here
   # render_val=0
   # if(args.render_train and args.render_test):
   #    render_value = 3
   # elif(args.render_test):
   #    render_value = 2
   # elif(args.render_train):
   #    render_value = 1

   """
   Init and goal cells are extracted randomly
   """

   np.random.shuffle(goodCells)
   # print(goodCells)
   fixed_init_pos = goodCells[0]
   print('fixed_init_pos: ', fixed_init_pos)
   fixed_goals = [goodCells[1]]


   env = gym.make("MALRLEnv-v0",maze_file = maze_file,                  
      # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                    maze_size=(640, 640), 
                                    enable_render= args.render_train,num_goals=args.ngoals, 
                                    verbose = args.v, n_trajs=args.n_random_init,random_pos = args.random_pos,seed_num = SEED,
                                    fixed_goals = fixed_goals ,fixed_init_pos = fixed_init_pos,
                                    visited_cells = [])

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

   DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
   print('DECAY_FACTOR: ', DECAY_FACTOR)

   signal.signal(signal.SIGINT, signal_handler)
   print('Press Ctrl+C to stop')

   # Render tha maze
   if(env.enable_render ):
      env.render()

   print("EXPERIMENT DATE:",EXPERIMENT_DATE)
   print("Seed:",SEED)
   print("Learning starting...") 

   print('MAX_T: ', c_settings["MAX_T"])

         # print("env.maze_size",env.maze_size)
   
   n_episodes = args.nepisodes

   
   # if(args.log_reward  ):
   logOutfile ="generatedData/logs/log-"+EXPERIMENT_DATE+".txt"
   rewLogFile=open(logOutfile, 'w')

   '''      
      Begin training/testing #
   '''      
            
   visited_cells = []
   trajs = []

   if(args.ntrajs):
      assert(args.ntrajs%TRAJECTORIES_BUFFER_SIZE==0)
      n_uavs = args.ntrajs
   else:
      n_uavs = c_settings["N_TRAJECTORIES_TO_GENERATE"]

   outDir ="qTrajs2D"+EXPERIMENT_DATE 
   outDir= (os.path.join(c_paths["TRAJECTORIES_FOLDER"], outDir) )
   os.makedirs( outDir)
   
   outDir3D ="trajs3D"+EXPERIMENT_DATE 
   outDir3D= (os.path.join(c_paths["TRAJECTORIES_3D_FOLDER"], outDir3D) )
   os.makedirs( outDir3D)
   
   outDirInt = "qTrajs2DINT"+EXPERIMENT_DATE 
   outDirInt= (os.path.join(c_paths["INT_TRAJECTORIES_FOLDER"], outDirInt) )
   os.makedirs( outDirInt)
 


   for uav_idx in range(0,n_uavs):
      print("||||||||||||||||||||||||||||| GENERATING TRAJECTORY ", uav_idx," |||||||||||||||||||||||||||||")
      if(uav_idx != 0): #alt. gia fatto
         """
         Prendo a caso init e goal
         """
         np.random.shuffle(goodCells)
         # print(goodCells)
         fixed_init_pos = goodCells[0]
         print('fixed_init_pos: ', fixed_init_pos)
         fixed_goals = [goodCells[1]]
         print('fixed_goals: ', fixed_goals)
         env.setNewEntrance(fixed_init_pos)
         env.setNewGoals(fixed_goals)
         env.setVisitedCells(visited_cells)

      # qtable,_ = main(mode = "train",fixed_init_pos=fixed_init_pos,visited_cells = visited_cells,i_trajectory = uav_idx)
      # _,traj =  main(mode = "test",trainedQtable=  qtable,fixed_init_pos=fixed_init_pos,i_trajectory = uav_idx,outDir=outDir)
      # print('traj: ', traj[0:3])

      #------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------  
      '''
      Preparo dati per nuova traj generata
      '''
      
      # Creating a Q-Table for each state-action pair
      qtable = []
      q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

      # Instantiating the learning related parameters
      learning_rate = get_learning_rate(0)
      explore_rate = get_explore_rate(0)
      discount_factor = 0.99

      num_streaks = 0
      qtrajectory = []
      last_s_a_queue = []

      rewLogFile.write("nrun %d,seed %d,max_t %d, buf_size %d\n" % (n_uavs, SEED,c_settings["MAX_T"],TRAJECTORIES_BUFFER_SIZE))
      start = time.time()
      for episode in range(n_episodes+1):
         
         if(episode == n_episodes and args.render_test):
            env.set_render(True)
            
         # Reset the environment
         obv = env.reset()
         env.seed(SEED+episode)
         # the initial state
         old_state = state_to_bucket(obv)
         total_reward = 0

         qtrajectory = []
         last_s_a_queue = []

         if(episode == n_episodes):
            print(">>>>>>>>>>>>>>> TESTING TRAJ. ",uav_idx," <<<<<<<<<<<<<<<<")
            # logger.info(', '.join( [str(x) for x in [episode,len(q_table),len(qtrajectory),len(last_s_a_queue)] ] ))

         for t in range(c_settings["MAX_T"]):
            # Select an action
            action = select_action(old_state, explore_rate)

            # execute the action
            obv, reward, done, info = env.step(action)
            # print('observation: ', obv)

            if(episode == n_episodes and info["moved"]):
                  # Append to trajectory
                  # print("obv,qtrajectory",obv,qtrajectory)
                  qtrajectory.append(list(obv))

            # Observe the result
            state = state_to_bucket(obv)
            # print("OBV:",obv,"STB",state_to_bucket(obv))
            
            total_reward += reward
            
            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            
            
            # Q-Routing UPDATE
            q_table[old_state + (action,)] += learning_rate * (reward +
                  discount_factor * (best_q) - q_table[old_state + (action,)])
            
            # Setting up for the next iteration
            old_state = state
            if(args.nsteps > 0 ):
                  if(len(last_s_a_queue)>= args.nsteps):
                     last_s_a_queue.pop(0)
                  last_s_a_queue += [ [state,action] ]

            # TODO qui le debug mode

            # Render tha maze
            if env.enable_render:
               env.render()

            if env.is_game_over():
               sys.exit()

            if done:
               # print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
               #     % (episode, t, total_reward, num_streaks))
               if(not args.quiet):
                  print("%d,%f,%f" % (episode, t, total_reward))
               if(episode != n_episodes ):  rewLogFile.write("%d,%f,%f\n" % (episode, t, total_reward))

               if(args.nsteps > 0):
                  # NOTE: Q-Routing N-STEPS More in the doc...
                  for idx in range(len(last_s_a_queue)):
                     if(idx+1==len(last_s_a_queue)): break
                     cur_s, cur_a= last_s_a_queue[idx][0], last_s_a_queue[idx][1]
                     next_s,next_a = last_s_a_queue[idx+1][0], last_s_a_queue[idx+1][1]
                     td_best_q = q_table[next_s + (next_a,) ]

                     q_table[cur_s + (next_a,)] += learning_rate * (reward + 
                        discount_factor * (td_best_q) - q_table[ next_s + (next_a,)])


               if t <= c_settings["SOLVED_T"]:
                  num_streaks += 1
               else:
                  num_streaks = 0
               break

            elif t >= c_settings["MAX_T"] - 1:
               if(not args.quiet):
                     print("%d,%f,%f,stopped\n" % (episode, t, total_reward) )
               if(episode != n_episodes):  rewLogFile.write("%d,%f,%f\n" % (episode, t, total_reward))

            if(args.slow):
                  time.sleep(1)

         # It's considered done when it's solved over 120 times consecutively
         if not args.nepisodes and num_streaks > c_settings["STREAK_TO_END"]:
            break

         # Update parameters
         explore_rate = get_explore_rate(episode)
         learning_rate = get_learning_rate(episode)
         
      # ----> EPISODES ENDED
      
      end = time.time()
      rewLogFile.write("ELAPSED TIME for:"+str(uav_idx)+","+str(end-start))
      
      toBeSaved = np.array(qtrajectory,dtype=int)
      print('Saving in : ', outDir)
      outfile = "traj2d_" + str(uav_idx)
      if(c_settings["OUT_FORMAT"] == "csv"):
         # print('toBeSaved: ', toBeSaved)
         df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
         # df["z_pos"] = -10
         df.index.name = "index"
         df.to_csv(os.path.join(outDir,outfile+".csv"))

      elif(c_settings["OUT_FORMAT"] == "npy"):
         np.save(os.path.join(outDir,outfile+".npy"),toBeSaved )


      #------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------
      # EPISODE HA GIA FINITO SU UNA TRAIETTORIA
      trajs.append(qtrajectory)
      
      # Remove duplicates from single traj
      for p in qtrajectory:
         if p not in visited_cells:
            visited_cells.append(p)  
      
      print("Num. trajs till now ",len(trajs))

      if(uav_idx!=0 and uav_idx  % TRAJECTORIES_BUFFER_SIZE == TRAJECTORIES_BUFFER_SIZE-1 ):
            # gtrajs = trajs_utils.fix_traj(trajs)
            
            mtrajs = trajs_utils.fromCellsToMeters(trajs,scale_factor = c_settings["SCALE_SIZE"]/2)

            gtrajs = trajs_utils.myInterpolate2D(mtrajs,step_size=1)
            print("2D interpolation completed.")    
            for t in gtrajs:
               toBeSaved=np.array(t)
               df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
               # df["z_pos"] = -10
               df.index.name = "index"
               df.to_csv(os.path.join(outDirInt,"traj_2d_int_"+str(uav_idx)+".csv"))
            # gtrajs = trajs
            # gtrajs = [ [ list(p) for p in t]  for t in gtrajs]
            

            print("Altitude assignment started...")    
            trajs3d, i_outs,local_fids = trajs_utils.vertical_separate(new_trajs_2d=gtrajs,
               fids=[ i + ((uav_idx-TRAJECTORIES_BUFFER_SIZE)+1) for i in range(TRAJECTORIES_BUFFER_SIZE)],
               assigned_trajs = trajsWithAltitude,min_height=50,max_height=300,sep_h = 10,radius=100, tolerance=0.0)
            
            print("gtrajs ",len(gtrajs))
            print("mtrajs ",len(mtrajs))
            print("trajwith ",len(trajsWithAltitude))
            print("trajwith ",len(trajs3d))

            outs += i_outs
            for t in trajs3d:
               trajsWithAltitude.append(t)
            trajsBySteps.append( (local_fids,trajs3d) )
            trajs_utils.plot_xy(trajs,cell_size=1,fids=local_fids,doScatter=True,doSave=True,isCell=True,name="rlTrajs_ep_"+str(uav_idx),date=EXPERIMENT_DATE)

            for id in local_fids:
               fids.append(id)
            # Resetto il buffer
            trajs = []
            visited_cells = []

            for i_t,t in enumerate( trajs3d):
               nptraj = np.array(t)
               df = pandas.DataFrame({'x_pos': nptraj[:, 0], 'y_pos': nptraj[:, 1],
               'z_pos': nptraj[:, 2]})
               df.index.name = "index"
               df.to_csv(os.path.join( outDir3D,"3dtraj"+str(i_t+ (uav_idx - TRAJECTORIES_BUFFER_SIZE)+1)+".csv" ) )
               print("saved traj",i_t ," in 3d to",outDir3D)

   # SONO FINITE TUTTE LE TRAIETTORIE

   rewLogFile.close()


   # 
   # PLOT
   # 
   print("Trained and tested runs",uav_idx+1)
   utils.play_audio_notification()

   print("OUTS:",outs)
   print("Start PLOTTING...")
   if(args.plot3d):
      trajs_utils.plot_3d(trajsWithAltitude,fids,also2d=False,doSave=False,name="test"+"3d",exploded=False,date=EXPERIMENT_DATE)
      trajs_utils.plot_xy(trajsWithAltitude,cell_size=20,fids=fids,doSave=False,date=EXPERIMENT_DATE)
      # trajs_utils.plot_z(trajsWithAltitude,fids,second_axis=0,name="test"+"xz")
      # trajs_utils.plot_z(trajsWithAltitude,fids,second_axis=1,name="test"+"yz")
   else:
      trajs_utils.plot_3d(trajsWithAltitude,fids,also2d=False,doSave=True,name="test"+"3d",exploded=False,date=EXPERIMENT_DATE)
      trajs_utils.plot_xy(trajsWithAltitude,cell_size=20,fids=fids,doSave=True,date=EXPERIMENT_DATE)


   print("DONE.")