import argparse
import datetime
import logging
import math
import sys
from time import sleep
from airsim.types import DrivetrainType, Pose, Vector3r, YawMode
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



def main(trajectories_folder,velocity):

   # Create AirSim client
   asClient = NewMyAirSimClient(trajColFlag=False,
            canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])

   # Read trajectories
   exp_folders = [os.path.join(trajectories_folder,d) for d in os.listdir(trajectories_folder)]
   latest_mod_folder = max(exp_folders , key=os.path.getmtime)

   print("Detected ",len(latest_mod_folder)," files.")
   for f in sorted(os.listdir(latest_mod_folder)):
      asClient.disable_trace_lines()
      if( f[-4:]==".csv" ):
         print("Reading trajectory from:",f)
         with open(os.path.join(latest_mod_folder,f),"r") as fin:
            lr= fin.readlines()
            trajectory = []
            for idx,line in enumerate(lr):
               if(idx==0):
                  continue
               values = line.split(",")
               x = float( values[1] )
               y = float( values[2] )
               z = float( values[3] )
               z = z if z < 0 else -z

               if(idx==1):
                  pose = Pose(Vector3r(x,y,z), to_quaternion(0, 0, 0) ) 
                  asClient.simSetVehiclePose(pose,True,"Drone0")       
                  asClient.enable_trace_lines()
               else:
                  trajectory.append( Vector3r(x,y,z) )

            trajs_utils.plot_xy([trajs_utils.vec2d_list_to_tuple_list(trajectory)],20,doScatter=True)
            print("POSE: ",asClient.simGetVehiclePose( vehicle_name="Drone0").position)

            pointer = asClient.moveOnPathAsync(
               trajectory,
                 velocity, 120,
                        DrivetrainType.ForwardOnly , YawMode(False,0),lookahead=20,adaptive_lookahead=1,vehicle_name="Drone0")
   
                  
            print("UAV following trajectory", f,"...")
            pointer.join()
            print("UAV completed its mission.")

   # regex="Init"
   # for j in range(0,3):
   #    trajectory=[] 
   #    if(j==1):
   #       for i in [5,18,6,7,17,46] : 
   #          bn = asClient.simListSceneObjects(regex+str(i))
   #          pose = asClient.simGetObjectPose(bn[0])
   #          trajectory.append (trajs_utils.vec2d_to_numpy_array(pose.position))
   #    elif(j==0):
   #       for i in [8,13,42,50,54,49,24] : 
   #          bn = asClient.simListSceneObjects(regex+str(i))
   #          print(bn)
   #          pose = asClient.simGetObjectPose(bn[0])
   #          trajectory.append (trajs_utils.vec2d_to_numpy_array(pose.position))
   #    elif(j==2):
   #       for i in [14,9,20,19,23,0,31,30,999,37] : 
   #          bn = asClient.simListSceneObjects(regex+str(i))
   #          pose = asClient.simGetObjectPose(bn[0])
   #          trajectory.append (trajs_utils.vec2d_to_numpy_array(pose.position))

      # pointer = asClient.moveToZAsync(trajectory[0][2],20)
      # pointer.join()

      # for i in range(0,500):
      #    sleep(1)
      #    logger.info( ","+ str(i)+","+", ".join([ str(x) for x in utils.position_to_list( asClient.getPosition("Drone0"))]) )

      

                                       



if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Layer 2')
   
   parser.add_argument("-i", type=str,required=False, default = c_paths["TRAJECTORIES_3D_FOLDER"],
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = 20, type=float,required=False,
                     help='Speed value')


   parser.add_argument( '--log',action='store_true',  default=False,
      help='Log into file (default: %(default)s)' )

   # parser.add_argument('--load-qtable', type=str, 
   #    help='qtable file (default: %(default)s)')

   args = parser.parse_args()

   
   if(args.log):
      logging.basicConfig(filename=c_paths["LOG_FOLDER"]+"L2log(AIRSIM)"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                              filemode='w',
                              format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                              datefmt='%H:%M:%S',
                              level=logging.INFO)


      logger = logging.getLogger('RL Layer2')
      logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )


   main(trajectories_folder=args.i,velocity=args.velocity)

