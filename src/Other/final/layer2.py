import argparse
import datetime
import logging
import math
import sys
from time import sleep
from airsim.types import Pose, Vector3r
from airsim.utils import to_quaternion
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import pandas

import scipy.interpolate
import utils
from sklearn.neighbors import KDTree
import os
import re
from airsimgeo.newMyAirSimClient import NewMyAirSimClient
import trajs_utils

TRAJECTORIES_3D_FOLDER = "generatedData/3dL2/csv/"


# if(args.debug):
logging.basicConfig(filename=utils.LOG_FOLDER+"L2log(AIRSIM)"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


logger = logging.getLogger('RL Layer2')
logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )





if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Layer 2')
   
   parser.add_argument("-i", type=str,required=False,
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = 20, type=float,required=False,
                     help='Speed value')


   # parser.add_argument( '--debug',action='store_true',  default=False,
   #    help='Log into file (default: %(default)s)' )

   # parser.add_argument('--load-qtable', type=str, 
   #    help='qtable file (default: %(default)s)')

   args = parser.parse_args()

   # Starting position of agent
   # s_x,s_y = args.sx, args.sy

   l_files = os.listdir(args.i)
   trajectory_file = None
   x,y = None,None

   # for f in l_files:
   #    if( f[-4:]==".csv" ):
   #       print("Reading:",f)
   #       with open(os.path.join(args.i,f),"r") as fin:
   #          lr= fin.readlines()
   #          x = float( lr[1].split(",")[1] )
   #          y = float( lr[1].split(",")[2] )
   #          # print('x: ', x)
   #          # print('y: ', y)

         
   #    # if(x==s_x and y==s_y):
   #    #    trajectory_file = f
   # trajectory_file = l_files[1]
   # if(x is None or y is None):
   #    raise Exception("Invalid Initial positions",s_x,s_y)
   # trajectory = None
   # # if(trajectory_file):
   # other_trajectories = []
   # if(True):
      
   #    df = pandas.read_csv(os.path.join(args.i,trajectory_file),delimiter=",",index_col="index")
   #      # print(df)
   #    trajectory = df.to_numpy()
   #    trajectory=trajectory.tolist()
   #    trajectory = trajs_utils.assign_dummy_z(trajectory,dummyZ=-50)
   #    trajs_utils.plot_xy([trajectory],cell_size=1,isCell=1,doScatter=True)
   #    trajectory = [ [p[0]*20,p[1]*20,p[2]] for p in trajectory]
   #    # trajectory[:,2] *= -1
   #    # for ff in l_files[1:]:
   #    #    df = pandas.read_csv(os.path.join(args.i,ff),delimiter=",",index_col="index")
   #    #    # print(df)
   #    #    o_t = df.to_numpy()
   #    #    o_t[:,2] *= -1
   #    #    other_trajectories.append( o_t )

   # else:
   #    print("Invalid trajectory")
   #    sys.exit(0)
   
   # # trajectory = np.array( trajs_utils.fix_traj([list(trajectory)])[0] )
   # print("FOLLOWING trajectory:",trajectory_file)
   # print("\t traj_sum",trajectory[:4],"...",trajectory[-4:])
   # print("\t num. of points:", np.shape(trajectory)[0] )
   
   # Create AirSim client
   asClient = NewMyAirSimClient(trajColFlag=False,
            canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])
   
   BUFFER_SIZE=3
   trajs2d = trajs_utils.random_gen_2d(0,860,0,860,
      step=120,n_trajs=BUFFER_SIZE)
   regex="Init"
   balls_name = asClient.simListSceneObjects(regex)

   def _vec2r_to_numpy_array(vec):
    return np.array([vec.x_val, vec.y_val]) 

                
   for j in range(0,3):
      trajectory=[] 
      if(j==1):
         for i in [5,18,6,7,17,46] : 
            bn = asClient.simListSceneObjects(regex+str(i))
            pose = asClient.simGetObjectPose(bn[0])
            trajectory.append (_vec2r_to_numpy_array(pose.position))
      elif(j==0):
         for i in [8,13,42,50,54,49,24] : 
            bn = asClient.simListSceneObjects(regex+str(i))
            print(bn)
            pose = asClient.simGetObjectPose(bn[0])
            trajectory.append (_vec2r_to_numpy_array(pose.position))
      elif(j==2):
         for i in [14,9,20,19,23,0,31,30,999,37] : 
            bn = asClient.simListSceneObjects(regex+str(i))
            pose = asClient.simGetObjectPose(bn[0])
            trajectory.append (_vec2r_to_numpy_array(pose.position))

      # print("Reading FILE:",trajs2d[i])
      # df = pandas.read_csv(os.path.join(args.i,l_files[i]),delimiter=",",index_col="index")
        # print(df)
      # trajectory = df.to_numpy()
      # trajectory=trajectory.tolist()
      # trajectory = trajs2d[i]

      trajectory = trajs_utils.assign_dummy_z(trajectory,
      # dummyZ= random.randint(-300,-50))
      dummyZ= -50)
      trajs_utils.plot_xy([trajectory],cell_size=20,isCell=False,doScatter=False)
      # trajectory = trajs_utils.fix_traj([trajectory])
      
      # trajectory_vecs = [utils.list_to_position(x,1,1) for x in trajectory]
      trajectory_vecs = [ Vector3r(p[0],p[1],p[2] )  for p in trajectory[1:] ]

      asClient.disable_trace_lines()
      print("moved to",trajectory[0] )
      p = trajectory[0]
      pose = Pose(Vector3r(p[0],p[1],p[2] ), to_quaternion(0, 0, 0) ) 
      asClient.simSetVehiclePose(pose,True,"Drone0")       
      asClient.enable_trace_lines()

      # pointer = asClient.moveToZAsync(trajectory[0][2],20)
      # pointer.join()

      pointer = asClient.moveOnPathAsync(
         trajectory_vecs,
         args.velocity,
         adaptive_lookahead=1,vehicle_name="Drone0")


      # for i in range(0,500):
      #    sleep(1)
      #    logger.info( ","+ str(i)+","+", ".join([ str(x) for x in utils.position_to_list( asClient.getPosition("Drone0"))]) )

      print("UAV following trajectory...")
      pointer.join()
      print("UAV completed its mission.")
      


                                       