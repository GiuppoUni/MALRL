import argparse
import datetime
import logging
import math
import sys
from time import sleep

from numpy.lib.function_base import rot90
from airsim140.types import DrivetrainType, Pose, YawMode
from airsim140.utils import to_quaternion
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import pandas

import scipy.interpolate
import malrl_utils
from sklearn.neighbors import KDTree
import os
import re
from airsimgeo.newMyAirSimClient import NewMyAirSimClient
import trajs_utils

from layer2 import positionLoggerThread
import multiprocessing

configYml = malrl_utils.read_yaml("inputData/config.yaml")
c_paths = configYml["paths"]
c_settings = configYml["layer1"]["settings"]
c_verSep= configYml["layer1"]["vertical_separation"]

NEW_FOLDER = os.path.join(c_paths["LAYER3_OUTPUT_FOLDER"],malrl_utils.EXPERIMENT_DATE)    


def main(input_folder,velocity):

   os.makedirs(NEW_FOLDER)

   exp_folders = [os.path.join(input_folder,d) for d in os.listdir(input_folder)]
   latest_mod_folder = max(exp_folders , key=os.path.getmtime)

   l_files = os.listdir(latest_mod_folder)
   l_files=list(filter(lambda x: x[-4:]==".csv",l_files))
   print("Found:",len(l_files),"csv files inside '",latest_mod_folder,"'.")
   for f_num,f in enumerate(sorted(l_files,key=lambda x: int("".join( filter(str.isdigit, x))))):
      if( f[-4:]==".csv" ):
         print("Reading trajectory from file:",os.path.join(latest_mod_folder,f),". Reading ",f_num,"/",len(l_files), "different files.")
         trajectory=[]
         with open(os.path.join(latest_mod_folder,f),"r") as fin:
            lr = fin.readlines()
            for i in range(0,len(lr)):
               if(i==0): #(Heading)
                  continue
               values = lr[i].split(",")
               x = float( lr[i].split(",")[1] )
               y = float( lr[i].split(",")[2] )
               z = float( lr[i].split(",")[3]) if(len(values)>2) else configYml["layer3"]["FIXED_Z"]
   
               trajectory.append( (x,y,z) )
            # print('x: ', x)
            # print('y: ', y)

         plotProcess = multiprocessing.Process(target=trajs_utils.plot_xy , args=([trajectory], c_settings["SCALE_SIZE"] ))
         if(configYml["layer3"]["SHOW_XY_PLOT"]):
            plotProcess.start()

         trajectory_vecs = [malrl_utils.l3_pos_arr_to_airsim_vec(x) for i,x in enumerate(trajectory) if i%10==0]
         np_trajectory = np.array( trajectory)
         
         print("FOLLOWING trajectory:",f)
         print("\t traj_sum",trajectory[:4],"...",trajectory[-4:])
         print("\t num. of points:", np.shape(np_trajectory)[0] )

               # Create AirSim client
         asClient = NewMyAirSimClient(trajColFlag=False,
                  canDrawTrajectories=False,crabMode=False,thickness = 140,
                  trajs2draw=[],traj2follow=trajectory,ip="192.168.1.160")


         asClient.disable_trace_lines()
         sleep(0.5)
         pose = Pose(malrl_utils.pos_arr_to_airsim_vec(trajectory[0]), to_quaternion(0, 0, 0) ) 
         asClient.simSetVehiclePose(pose,True,"Drone0")        
         sleep(1)
         print("Drone set at start position.")
         asClient.enable_trace_lines()

         transformed_coo = *trajs_utils.rotate_point_2d(-math.pi/2, trajectory[0][0],trajectory[0][1]) , trajectory[0][2]
         print('transformed_coo: ', transformed_coo)
         gps_coo = asClient.nedToGps(*transformed_coo ) 
         print("GPS STARTING COO (lon,lat,alt):",gps_coo)
         
         # asClient.enable_trace_lines()

         # print("Positioning at std altitude...")
         # pointer = asClient.moveToZAsync(-50,20)
         # pointer.join()
         # print("Altitude reached.\n Starting following path...")
         
         pointer = asClient.moveOnPathAsync(
            trajectory_vecs,
                 velocity, 120,DrivetrainType.ForwardOnly , YawMode(False,0),
               lookahead=20,adaptive_lookahead=1,vehicle_name="Drone0")

         fidx = f.split(".csv")[0].split("traj")[1]
         if(not fidx.isdigit()):
            raise Exception("Error in format of filename:"+f+", (it should be ...traj<id>.csv)")

         positionThread = positionLoggerThread("Drone0",fidx,f,gpsOn=True,layer=3)
         positionThread.start()

            # pos = utils.position_to_list( asClient.getPosition("Drone0" ) ) 
            # gps_pos = asClient.nedToGps(*trajs_utils.rotate_point_2d(-utils.O_THETA,pos[0],pos[1]),pos[2] )
            # toPrint = ", ".join( [ str(x) for x in list(pos) + list(gps_pos) ] )
            # logger.info( ","+ str(i)+","+ toPrint )

         pointer.join()
         if(configYml["layer3"]["SHOW_XY_PLOT"]):
            plotProcess.terminate()
         positionThread.stop()
         positionThread.join()
         
         print("UAV completed its mission.")


if __name__ == "__main__":
   

   parser = argparse.ArgumentParser(description='Layer 3')
   
   parser.add_argument("-i", type=str,required=False,default=c_paths["LAYER2_OUTPUT_FOLDER"],
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = configYml["layer3"]["VELOCITY"],
      type=float,required=False, help='Speed value')


   parser.add_argument( '--debug',action='store_true',  default=False,
      help='Log into file ' + malrl_utils.EXPERIMENT_DATE + '(default: %(default)s)' )

   # parser.add_argument('--load-qtable', type=str, 
   #    help='qtable file (default: %(default)s)')

   args = parser.parse_args()

   # Starting position of agent
   # s_x,s_y = args.sx, args.sy

   if(args.debug):
      logging.basicConfig(filename=c_paths["LOG_FOLDER"]+"L3log(AIRSIM)"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                              filemode='w',
                              format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                              datefmt='%H:%M:%S',
                              level=logging.INFO)


      logger = logging.getLogger('RL Layer3')
      logger.info('Experiment Date: {}'.format(malrl_utils.EXPERIMENT_DATE) )


   main(input_folder=args.i,velocity=args.velocity)