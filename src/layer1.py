"""
Layer 1 
(Config files are inside inputData folder)

Returns:
      Saved 2D trajectories with altitude inside ./generatedData

"""
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
from gym_maze.envs.maze_env import MazeEnv, MazeView2D

import signal
import sys
import pandas

import trajs_utils
import yaml
from gym_maze.envs.maze_view_2d import Maze
from PIL import Image as img

configYml = utils.read_yaml("inputData/config.yaml")
c_paths = configYml["layer1"]["paths"]
c_settings = configYml["layer1"]["settings"]
c_verSep= configYml["layer1"]["vertical_separation"]

IDX_TO_ACTION =  {0:"LEFT", 1:"FRONT", 2:"RIGHT", 3:"BACK"}

EXPERIMENT_DATE =  str(datetime.datetime.now().strftime('-D-%d-%m-%Y-H-%H-%M-%S-') ) # To be used in log and prints


def getNotWallsCells():
   goodCells = []
   for r in range(c_settings["NROWS"]):
      for c in range(c_settings["NCOLS"]):
         if( r % 7==0 or c % 7==0):
               goodCells.append([r,c])
   return goodCells


def select_action(state, explore_rate,lastAction):
   if(lastAction is None):
      # Select a random action
      if random.random() < explore_rate:
         action = env.action_space.sample()
      # Select the action with the highest q
      else:
         action = int(np.argmax(q_table[state]))
      return action
   else:
        # Select a random action
      if random.random() < explore_rate:
         action = env.action_space.sample()
         while(action == lastAction):
            action = env.action_space.sample()
      # Select the action with the highest q
      else:
         action = int(np.argmax(q_table[state]))
      return action

def get_explore_rate(t):
   # return max(c_settings["MIN_EXPLORE_RATE"], min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))
   return 0.2

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

   parser.add_argument("--ntrajs",type=int,
                        help='num trajectories value (default: %(default)s)')

   parser.add_argument("--nbuffer",type=int, default=3,
                        help='size of buffer for past trajectories (default: %(default)s)')

   # parser.add_argument('--nagents', type=int, default=1,
   #                   help='num of simultaneous agents (supported 1 )(default: %(default)s)')

   parser.add_argument('--nsteps', type=int, default=0,
                     help='enforce n-steps qlearning if 0 is standard qlearning  (default: %(default)s)')

   parser.add_argument( '--debug',action='store_true',  default=False,
      help='Log debug in file (default: %(default)s)' )
   
   parser.add_argument('--render-train',action='store_true',  default=False,
      help='render maze while training/random  (default: %(default)s)' )
   
   parser.add_argument( '--render-test',action='store_true',  default=False,
      help='render maze while testing  (default: %(default)s)' )

   parser.add_argument( '--quiet',action='store_true',  default=False,
      help='Less info in output  (default: %(default)s)' )
   
   parser.add_argument( '-v',action='store_true',  default=False,
      help='verbose (default: %(default)s)' )
   
   parser.add_argument('--slow',action='store_true',  default=False,
      help='Slow down training to observe behaviour (default: %(default)s)')

   parser.add_argument('--n-random-init', type=int, default=5,
                     help='n sample pool for random init (default: %(default)s)')

   parser.add_argument('--log-reward', action='store_true', default=False,
   help='log reward file in out (default: %(default)s)')

   parser.add_argument('--load-qtable', type=str, 
      help='qtable file (default: %(default)s)')

   parser.add_argument('--load-maze', type=str, 
      help='maze file (default: %(default)s)')
   
   parser.add_argument("--show-maze-bm", action="store_true",default=False, 
      help='Show Bitmap used as maze')

   parser.add_argument("--train", action="store_true",default=False, 
      help='Start generating trajectories')


   parser.add_argument('--random-goal-pos', action="store_true",default=False, 
      help='Choose random start pos instead of the one inside csv file (default: %(default)s)')

   parser.add_argument('--generate-random-start', action="store_true",default=False, 
      help='Choose random start pos instead of the one inside csv file (default: %(default)s)')

   parser.add_argument('--random-start-pos',  action="store_true",default=False,  
      help='Choose random goal pos instead of the one inside csv file  (default: %(default)s)')

   parser.add_argument('--generate-random-goal',  action="store_true",default=False,  
      help='Choose random goal pos instead of the one inside csv file  (default: %(default)s)')

   args = parser.parse_args()


   if(args.show_maze_bm):
      data = np.load(os.path.join("gym_maze\envs\maze_samples",c_paths["STD_MAZE"]) )
      print("MAZE BitMap:",data.shape,data)

      data  =    np.where(data==0, 255, data) 
      data  =    np.where(data!=255, 0, data) 
      image = img.fromarray(data)
      # image.save('my.png')
      np.savetxt('np.csv', image, delimiter=',',fmt="%u")
      image.resize(size=(720, 1280))
      image.show()
   
   SEED = c_settings["SEED"]
   if(SEED==-1):
    SEED = random.seed()
   else:
    random.seed(SEED)
    np.random.seed(seed=SEED)
      

   # """
   #    Define cells allowed to be used as start or goal
   # """
   # goodCells = getNotWallsCells() # NOTE: relative to our maze design

   outs = 0
   trajsWithAltitude = []
   trajsBySteps = []
   fids = []

   # Init logger
   logging.basicConfig(filename=utils.LOG_FOLDER+"log"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                           filemode='w',
                           format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logging.INFO)
   logger = logging.getLogger('RL Layer1')
   logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )

   TRAJECTORIES_BUFFER_SIZE = args.nbuffer

   if(args.ntrajs):
      assert(args.ntrajs%TRAJECTORIES_BUFFER_SIZE==0)
      n_uavs = args.ntrajs
   else:
      n_uavs = c_settings["N_TRAJECTORIES_TO_GENERATE"]

   """
      Choose inputs: standard start and goals
   """
   
   if(args.generate_random_start or args.random_goal_pos):
      maze = Maze(maze_cells=Maze.load_maze(os.path.join("gym_maze\envs\maze_samples",c_paths["STD_MAZE"]) ) ,verbose = args.v)
      allGoodCells = utils.getGoodCells(maze) # good as entrance or goal
      with open("inputData/start_pos_table.csv","w") as fstart, open("inputData/goal_pos_table.csv","w") as fgoal:
         fstart.write("name,x_pos,y_pos\n")
         for i in range(0,n_uavs):
            if(args.generate_random_start):
               startRandomCell = allGoodCells[np.random.choice(allGoodCells.shape[0])]
               fstart.write("s"+str(i)+",")
               fstart.write(",".join([ str(x) for x in startRandomCell.tolist() ] ) +"\n")
            if(args.generate_random_goal):
               if(i==0): fgoal.write("name,x_pos,y_pos\n")
               goalRandomCell = startRandomCell
               while( np.array_equal(goalRandomCell , startRandomCell)):
                  goalRandomCell =  allGoodCells[np.random.choice(allGoodCells.shape[0])]
               fgoal.write("g"+str(i)+",")
               fgoal.write(",".join([ str(x) for x in goalRandomCell.tolist() ] ) +"\n")

   df = pandas.read_csv("inputData/start_pos_table.csv", index_col='name')
   fixed_start_pos_list= df.values.tolist()
   print('fixed_start_pos_list: ', fixed_start_pos_list)

   df = pandas.read_csv("inputData/goal_pos_table.csv", index_col='name')
   fixed_goals_list = df.values.tolist()
   # assert(len(fixed_goals_list) == len(fixed_start_pos_list))
 

   ''''
      Prepare  pre training
   '''

   
            
   print("SEED",SEED)

   '''
      Check
   '''

   if(not (args.random_start_pos and args.random_goal_pos ) and c_settings["N_TRAJECTORIES_TO_GENERATE"]>len(fixed_start_pos_list)):
      print('len(fixed_start_pos_list): ', len(fixed_start_pos_list))
      print('fixed_start_pos_list: ', fixed_start_pos_list)
      raise Exception("Too many trajectory, adjust fixed positions ")

   # Define start and goal for first env settings
   fixed_start_pos =  fixed_start_pos_list.pop(0)
   fixed_goals = [ fixed_goals_list.pop(0)]

   env = gym.make("MALRLEnv-v0",maze_file =  c_paths["STD_MAZE"],                  
      # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                    maze_size=(640, 640), 
                                    enable_render= args.render_train,num_goals=args.ngoals, 
                                    verbose = args.v, 
                                    random_start_pos = args.random_start_pos,
                                    random_goal_pos = args.random_goal_pos,
                                    seed_num = SEED,
                                    fixed_goals = fixed_goals ,fixed_start_pos = fixed_start_pos,
                                    visited_cells = [])

   '''
      Defining the constants related to env
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
      Create folders to allocate generated trajectories
   '''

   outDir ="qTrajs2D"+EXPERIMENT_DATE 
   outDir= (os.path.join(c_paths["TRAJECTORIES_FOLDER"], outDir) )
   os.makedirs( outDir)
   
   outDir3D ="trajs3D"+EXPERIMENT_DATE 
   outDir3D= (os.path.join(c_paths["TRAJECTORIES_3D_FOLDER"], outDir3D) )
   os.makedirs( outDir3D)
   
   outDirInt = "qTrajs2DINT"+EXPERIMENT_DATE 
   outDirInt= (os.path.join(c_paths["INT_TRAJECTORIES_FOLDER"], outDirInt) )
   os.makedirs( outDirInt)
 



   '''      
      Begin training and testing 
   '''      
   visited_cells = []
   trajs = []
   for uav_idx in range(0,n_uavs):
      print("||||||||||||||||||||||||||||| GENERATING TRAJECTORY ", uav_idx," |||||||||||||||||||||||||||||")
      
      if(uav_idx != 0): #oth. yet done
         # Need for new random start and goal
         fixed_start_pos = fixed_start_pos_list.pop(0)
         env.setNewEntrance(fixed_start_pos)
         fixed_goals = [ fixed_goals_list.pop(0)]
         env.setNewGoals(fixed_goals)
         env.setVisitedCells(visited_cells)
      
      print('fixed_start_pos: ', fixed_start_pos)
      print('fixed_goals: ', fixed_goals)

      #------------------------------------------------------------------------------------------------#------------------------------------------------------------------------------------------------  
      '''
         Update variables before generating new trajectory
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

      env.seed(SEED)

      for episode in range(n_episodes+1):
         
         if(episode == n_episodes and args.render_test):
            env.set_render(True)
         
         lastAction = None #To save last action avoiding repeat

         # Reset the environment
         obv = env.reset()
         # the initial state
         old_state = state_to_bucket(obv)
         total_reward = 0

         qtrajectory = []
         last_s_a_queue = []

         if(episode == n_episodes):
            print(">>>>>>>>>>>>>>> TESTING TRAJ. ",uav_idx," <<<<<<<<<<<<<<<<")
            # logger.info(', '.join( [str(x) for x in [episode,len(q_table),len(qtrajectory),len(last_s_a_queue)] ] ))

         elapsed_time1 = 0 
         elapsed_time2 = 0
         for t in range(c_settings["MAX_T"]):
            # Select an action
            action = select_action(old_state, explore_rate,lastAction=lastAction)
            lastAction=action

            t = time.process_time()
            
            # execute the action
            obv, reward, done, info = env.step(action)
            elapsed_time1 += time.process_time() - t

            t = time.process_time()

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
                  import time

         elapsed_time2 += time.process_time() - t
         # print(str( elapsed_time1)+","+str(elapsed_time2))

         # It's considered done when it's solved over 120 times consecutively
         if not args.nepisodes and num_streaks > c_settings["STREAK_TO_END"]:
            break

         # Update parameters
         explore_rate = get_explore_rate(episode)
         learning_rate = get_learning_rate(episode)
         
      # ----> EPISODES ENDED
      
      end = time.time()
      rewLogFile.write("ELAPSED TIME for all episodes of "+str(uav_idx)+" is "+str(end-start))

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
      # ONE TRAJECTORY IS COMPLETED
      trajs.append(qtrajectory)
      
      # Remove duplicates from single traj
      for p in qtrajectory:
         if p not in visited_cells:
            visited_cells.append(p)  
      
      print("Num. trajs generated: ",len(trajs))

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
               assigned_trajs = trajsWithAltitude,
               min_height=c_verSep["MIN_HEIGHT"],max_height=c_verSep["MAX_HEIGHT"],sep_h = c_verSep["SEP_H"],
               radius=c_verSep["RADIUS"], tolerance=c_verSep["TOLERANCE"], seed=SEED)
            
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
   trajs_utils.plot_3d(trajsWithAltitude,fids,also2d=False,doSave=c_settings["doSave_3dPlot"],name="test"+"3d",
      exploded=c_settings["exploded_3dPlot"],date=EXPERIMENT_DATE)
   trajs_utils.plot_xy(trajsWithAltitude,cell_size=20,fids=fids,doSave=c_settings["doSave_xyPlot"],date=EXPERIMENT_DATE)

   print("DONE.")