import argparse
import datetime
import logging
import math
import sys
from time import sleep
from airsim.types import Pose
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


configYml = utils.read_yaml("inputData/config.yaml")
c_paths = configYml["layer1"]["paths"]
c_settings = configYml["layer1"]["settings"]
c_verSep= configYml["layer1"]["vertical_separation"]

# if(args.debug):
logging.basicConfig(filename=c_paths["LOG_FOLDER"]+"L3log(AIRSIM)"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


logger = logging.getLogger('RL Layer3')
logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Layer 3')
   
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
      trajectory = trajectory.tolist()
      for ff in l_files[1:]:
         df = pandas.read_csv(os.path.join(args.i,ff),delimiter=",",index_col="index")
         # print(df)
         o_t = df.to_numpy()
         o_t[:,2] *= -1
         other_trajectories.append( o_t )

   else:
      print("Invalid trajectory")
      sys.exit(0)
   
   trajs_utils.plot_xy([trajectory],cell_size=c_settings["SCALE_SIZE"])
   trajectory_vecs = [utils.list_to_position(x) for i,x in enumerate(trajectory) if i%10==0]
   np_trajectory = np.array( trajectory)
   print("FOLLOWING trajectory:",trajectory_file)
   print("\t traj_sum",trajectory[:4],"...",trajectory[-4:])
   print("\t num. of points:", np.shape(np_trajectory)[0] )
   
   # Create AirSim client
   asClient = NewMyAirSimClient(trajColFlag=False,
            canDrawTrajectories=True,crabMode=True,thickness = 140,trajs2draw=other_trajectories,traj2follow=trajectory)
   

   asClient.disable_trace_lines()
   sleep(0.5)
   pose = Pose(utils.list_to_position(trajectory[0]), to_quaternion(0, 0, 0) ) 
   asClient.simSetVehiclePose(pose,True,"Drone0")        
   print("Drone set at start position.")

   gps_coo = asClient.nedToGps(*trajs_utils.rotate_point_2d(-math.pi/2, trajectory[0][0],trajectory[0][1]),trajectory[0][2] ) 
   print("GPS STARTING COO (lon,lat,alt):",gps_coo)
   
   # asClient.enable_trace_lines()

   print("Positioning at std altitude...")
   pointer = asClient.moveToZAsync(-50,20)
   pointer.join()
   
   print("Altitude reached.\n Starting following path...")
   pointer = asClient.moveOnPathAsync(
      trajectory_vecs,
      args.velocity,
      adaptive_lookahead=1,vehicle_name="Drone0")

   
   for i in range(0,500):
      sleep(1)
      pos = utils.position_to_list( asClient.getPosition("Drone0" ) ) 
      gps_pos = asClient.nedToGps(*trajs_utils.rotate_point_2d(-utils.O_THETA,pos[0],pos[1]),pos[2] )
      toPrint = ", ".join( [ str(x) for x in list(pos) + list(gps_pos) ] )
      logger.info( ","+ str(i)+","+ toPrint )

   pointer.join()
   print("UAV completed its mission.")