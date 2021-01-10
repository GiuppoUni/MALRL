import itertools
import logging
import os
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



IDX_TO_ACTION =  {0:"LEFT",1:"FRONT",2:"RIGHT",3:"BACK"}


STD_MAZE = "maze2d_004.npy"

INTERACTIVE = False
OUT_FORMAT = "csv"
RANDOM_TRAJECTORIES_FOLDER = "generatedData/2dL1/csv/"
TRAJECTORIES_FOLDER = "generatedData/2dL1/" 
INT_TRAJECTORIES_FOLDER = "generatedData/2dL1INT/" 
TRAJECTORIES_3D_FOLDER = "generatedData/3dL2/csv/"

N_TRAJECTORIES_TO_GENERATE = 99
TRAJECTORIES_BUFFER_SIZE = 3
SCALE_SIZE = 20 # in meters unit


def strDate():
    return str(datetime.datetime.now().strftime('-D-%d-%m-%Y-H-%H-%M-%S-') )

EXPERIMENT_DATE =  strDate()





def cell_value(r,c,obs_blocks=1,street_blocks=1):
    if(obs_blocks ==1):
        if r %2 ==0 or c %2 ==0:
            return 15
        elif r != 0 and c !=0 and r != NROWS -1 and c != NCOLS -1 :
            return 0
        else:
            return 15
    else:
        if(r % ( obs_blocks+1)==0 or c % (obs_blocks +1 ) == 0 ):
            return 15
        else:
            return 0




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

   parser.add_argument("--nbuffer",type=int,
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

   SEED = random.randint(0,int(10e6))

   if(args.seed):
      SEED = args.seed
      random.seed(SEED)
      np.random.seed(seed=SEED)

   NROWS = 43
   NCOLS = 43

   # dummyMaze = [ [cell_value(r,c,obs_blocks=6) for c in range(43) ] for r in range(43)]
   # dummyMaze = np.array(dummyMaze)
   # rnd= np.where((dummyMaze!=0) & 
   # ((dummyMaze == 7)|(dummyMaze == 15) |(dummyMaze  ==13 )) )
   # rs= rnd[0]
   # cs= rnd[1]
   # goodCells = np.array( [[rs[i],cs[i]] for i in range(len(rs)) ] )

   """
      Creo celle pool da estrarre
   """
   goodCells = []
   for r in range(NROWS):
      for c in range(NCOLS):
         if( r % 7==0 or c % 7==0):
               goodCells.append([r,c])


   outs = 0
   trajsWithAltitude = []
   fids = []

   # if(args.debug):
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

   if(args.nbuffer):
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

   maze_file = STD_MAZE
            
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
   Prendo a caso init e goal
   """
   np.random.shuffle(goodCells)
   # print(goodCells)
   fixed_init_pos = goodCells[0]
   print('fixed_init_pos: ', fixed_init_pos)
   fixed_goals = [goodCells[1]]

   env = MazeEnv( maze_file = maze_file,                  
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
   STREAK_TO_END = 100
   SOLVED_T = 10
   DEBUG_MODE = 0
   

   signal.signal(signal.SIGINT, signal_handler)
   print('Press Ctrl+C to stop')

  

   assert type(env) == MazeEnv

   # Render tha maze
   if(env.enable_render ):
         env.render()

   print("EXPERIMENT DATE:",EXPERIMENT_DATE)
   print("Seed:",SEED)
   print("Learning starting...") 

   print('MAX_T: ', MAX_T)

         # print("env.maze_size",env.maze_size)
   
   n_episodes = args.nepisodes

   
   # if(args.log_reward  ):
   logOutfile ="generatedData/logs/log-"+EXPERIMENT_DATE+".txt"
   rewLogFile=open(logOutfile, 'w')
   
   #================#
   #       '''      #
   # Begin training/testing #
   #       '''      #
   #================#
         
   visited_cells = []
   trajs = []

   if(args.ntrajs):
      assert(args.ntrajs%TRAJECTORIES_BUFFER_SIZE==0)
      n_run = args.ntrajs
   else:
      n_run = N_TRAJECTORIES_TO_GENERATE

   outDir ="qTrajs2D"+EXPERIMENT_DATE 
   outDir= (os.path.join(TRAJECTORIES_FOLDER,outDir) )
   os.makedirs( outDir)
   
   outDir3D ="trajs3D"+EXPERIMENT_DATE 
   outDir3D= (os.path.join(TRAJECTORIES_3D_FOLDER,outDir3D) )
   os.makedirs( outDir3D)
   
   outDirInt = "qTrajs2DINT"+EXPERIMENT_DATE 
   outDirInt= (os.path.join(INT_TRAJECTORIES_FOLDER,outDirInt) )
   os.makedirs( outDirInt)
 


   for i_run in range(0,n_run):
      print("||||||||||||||||||||||||||||| GENERATING", i_run,"|||||||||||||||||||||||||||||")
      if(i_run != 0): #alt. gia fatto
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

      # qtable,_ = main(mode = "train",fixed_init_pos=fixed_init_pos,visited_cells = visited_cells,i_trajectory = i_run)
      # _,traj =  main(mode = "test",trainedQtable=  qtable,fixed_init_pos=fixed_init_pos,i_trajectory = i_run,outDir=outDir)
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

      print('learning_rate: ', learning_rate)
      print('explore_rate: ', explore_rate)
     
      rewLogFile.write("nrun %d,seed %d,max_t %d, buf_size %d\n" % (n_run, SEED,MAX_T,TRAJECTORIES_BUFFER_SIZE))
      start = time.time()
      for episode in range(n_episodes+1):

         # Reset the environment
         obv = env.reset()
         env.seed(SEED+episode)
         # the initial state
         old_state = state_to_bucket(obv)
         total_reward = 0

         qtrajectory = []
         last_s_a_queue = []

         if(episode == n_episodes):
            print(">>>>>>>>>>>>>>>TESTING<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            # logger.info(', '.join( [str(x) for x in [episode,len(q_table),len(qtrajectory),len(last_s_a_queue)] ] ))

         for t in range(MAX_T):
            # Select an action
            action = select_action(old_state, explore_rate)

            # execute the action
            obv, reward, done, info = env.step(action)
            # print('observation: ', obv)

            if(episode==n_episodes and info["moved"]):
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
               if(not args.quiet):
                     print("%d,%f,%f,stopped\n" % (episode, t, total_reward) )
               if(episode != n_episodes):  rewLogFile.write("%d,%f,%f\n" % (episode, t, total_reward))

            if(args.slow):
                  time.sleep(1)

         # It's considered done when it's solved over 120 times consecutively
         if not args.nepisodes and num_streaks > STREAK_TO_END:
            break

         # Update parameters
         explore_rate = get_explore_rate(episode)
         learning_rate = get_learning_rate(episode)
         
      # ----> EPISODES ENDED
      
      end = time.time()
      rewLogFile.write("ELAPSEDTIME for:"+str(i_run)+","+str(end-start))
      
      toBeSaved = np.array(qtrajectory,dtype=int)
      print('Saving in : ', outDir)
      outfile = "traj2d_"+str(i_run)
      if(OUT_FORMAT == "csv"):
         # print('toBeSaved: ', toBeSaved)
         df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
         # df["z_pos"] = -10
         df.index.name = "index"
         df.to_csv(os.path.join(outDir,outfile+".csv"))

      elif(OUT_FORMAT == "npy"):
         np.save(os.path.join(outDir,outfile+".npy"),toBeSaved )


#------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------
      # EPISODE HA GIA FINITO SU UNA TRAIETTORIA
      trajs.append(qtrajectory)
      
      # Remove duplicates from single traj
      for p in qtrajectory:
         if p not in visited_cells:
            visited_cells.append(p)  
      
      print("trajs",len(trajs))

      if(i_run!=0 and i_run  % TRAJECTORIES_BUFFER_SIZE == TRAJECTORIES_BUFFER_SIZE-1 ):
            # gtrajs = trajs_utils.fix_traj(trajs)
            
            mtrajs = trajs_utils.fromCellsToMeters(trajs,scale_factor = SCALE_SIZE/2)

            gtrajs = trajs_utils.myInterpolate2D(mtrajs,step_size=1)
            print("2D interpolation completed.")    
            for t in gtrajs:
               toBeSaved=np.array(t)
               df = pandas.DataFrame({'x_pos': toBeSaved[:, 0], 'y_pos': toBeSaved[:, 1]})
               # df["z_pos"] = -10
               df.index.name = "index"
               df.to_csv(os.path.join(outDirInt,"traj_2d_int_"+str(i_run)+".csv"))
            # gtrajs = trajs
            # gtrajs = [ [ list(p) for p in t]  for t in gtrajs]
            

            print("Altitude assignment started...")    
            trajs3d, i_outs,local_fids = trajs_utils.avoid_collision_complex(new_trajs_2d=gtrajs,
               fids=[i+ ((i_run-TRAJECTORIES_BUFFER_SIZE)+1) for i in range(TRAJECTORIES_BUFFER_SIZE)],
               assigned_trajs = trajsWithAltitude,min_height=50,max_height=300,sep_h = 1,radius=10, tolerance=0.0)
            
            print("gtrajs",len(gtrajs))
            print("mtrajs",len(mtrajs))
            print("trajwith",len(trajsWithAltitude))
            print("trajwith",len(trajs3d))

            outs += i_outs
            for t in trajs3d:
               trajsWithAltitude.append(t)
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
               df.to_csv(os.path.join( outDir3D,"3dtraj"+str(i_t+ (i_run - TRAJECTORIES_BUFFER_SIZE)+1)+".csv" ) )
               print("saved traj",i_t ," in 3d to",outDir3D)












   # SONO FINITE TUTTE LE TRAIETTORIE

   rewLogFile.close()


   # 
   # PLOT
   # 
   print("Trained and tested runs",i_run+1)
   utils.play_audio_notification()

   if(args.plot3d):
      print("OUTS:",outs)
      print("Start PLOTTING...")
      trajs_utils.plot_3d(trajsWithAltitude,fids,also2d=False,doSave=True,name="test"+"3d",exploded=False,date=EXPERIMENT_DATE)
      trajs_utils.plot_xy(trajsWithAltitude,fids,doSave=True,date=EXPERIMENT_DATE)
      # trajs_utils.plot_z(trajsWithAltitude,fids,second_axis=0,name="test"+"xz")
      # trajs_utils.plot_z(trajsWithAltitude,fids,second_axis=1,name="test"+"yz")
   
   print("DONE.")