import argparse
import math
import sys
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

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Layer 2')
   
   parser.add_argument("-i", type=str,required=True,
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = 10, type=float,required=False,
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

   for f in l_files:
      if( f[-4:]==".csv" ):
         print("Reading:",f)
         with open(os.path.join(args.i,f),"r") as fin:
            lr= fin.readlines()
            x = float( lr[1].split(",")[1] )
            y = float( lr[1].split(",")[2] )
            # print('x: ', x)
            # print('y: ', y)

         
      # if(x==s_x and y==s_y):
      #    trajectory_file = f
   trajectory_file = l_files[0]
   if(x is None or y is None):
      raise Exception("Invalid Initial positions",s_x,s_y)
   trajectory = None
   # if(trajectory_file):
   other_trajectories = []
   if(True):
      
      df = pandas.read_csv(os.path.join(args.i,trajectory_file),delimiter=",",index_col="index")
        # print(df)
      trajectory = df.to_numpy()
      trajectory[:,2] *= -1
      for ff in l_files[1:]:
         df = pandas.read_csv(os.path.join(args.i,ff),delimiter=",",index_col="index")
         # print(df)
         o_t = df.to_numpy()
         o_t[:,2] *= -1
         other_trajectories.append( o_t )

   else:
      print("Invalid trajectory")
      sys.exit(0)
   
   trajectory = np.array( trajs_utils.fix_traj([list(trajectory)])[0] )
   trajectory_vecs = [utils.list_to_position(x) for x in trajectory]
   print("FOLLOWING trajectory:",trajectory_file)
   print("\t traj_sum",trajectory[:4],"...",trajectory[-4:])
   print("\t num. of points:", np.shape(trajectory)[0] )
   
   # Create AirSim client
   asClient = NewMyAirSimClient(trajColFlag=False,
            canDrawTrajectories=True,crabMode=True,thickness = 140,trajs2draw=other_trajectories,traj2follow=[])
   
   asClient.enable_trace_lines()
   pointer = asClient.moveToZAsync(-50,20)
   pointer.join()
   pointer = asClient.moveOnPathAsync(
      trajectory_vecs,
      args.velocity,
      adaptive_lookahead=1,vehicle_name="Drone0")

   print("UAV following trajectory...")
   pointer.join()
   print("UAV completed its mission.")
   


                                       