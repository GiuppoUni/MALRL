import argparse
import datetime
import logging
import math
import sys
import threading
from time import sleep
import time
from airsim140.client import MultirotorClient
from airsim140.types import DrivetrainType, Pose, Vector3r, YawMode
from airsim140.utils import to_quaternion
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

import nest_asyncio
nest_asyncio.apply()


configYml = utils.read_yaml("inputData/config.yaml")
c_paths = configYml["paths"]
c_settings = configYml["layer1"]["settings"]
c_settings2 = configYml["layer2"]
c_verSep= configYml["layer1"]["vertical_separation"]
EXPERIMENT_DATE = utils.get_experiment_date()
NEW_FOLDER = os.path.join(c_paths["LAYER2_OUTPUT_FOLDER"],EXPERIMENT_DATE)    

class positionLoggerThread(threading.Thread):

   def __init__(self,vehicle_name,idx,latest_mod_folder,traj_filename):
      
      print("IDX",idx)
      threading.Thread.__init__(self,daemon=True )
      
      trajectory_completed = False
      
      self.vehicle_name = vehicle_name
      self.traj_filename = traj_filename
      self.client = MultirotorClient()
      self.client.enableApiControl(True,vehicle_name=vehicle_name)

      self.latest_mod_folder = os.path.join( c_paths["LAYER2_OUTPUT_FOLDER"], latest_mod_folder.split("/")[-1])
      self.filename = NEW_FOLDER+ "/3dl2traj"+str(idx)+".csv"
      self._stop_event = threading.Event()

   def stop(self):
      self._stop_event.set()

   def is_stopped(self):
        return self._stop_event.is_set()

   def run(self): 
      print("[THREAD "+threading.current_thread().getName()+"] Started \n Position logging for trajectory: ", self.traj_filename)
      with open(self.filename,"w") as fout:
         fout.write("index,x_pos,y_pos,z_pos,w_or,x_or,y_or,"+
         "z_or,x_lin_vel,y_lin_vel,z_lin_vel,x_ang_vel,y_ang_vel,z_ang_vel,"+
         "x_lin_acc,y_lin_acc,z_lin_acc,x_ang_acc,y_ang_acc,z_ang_acc"+"\n") #HEADER
      with open(self.filename,"a") as fout:
         time_counter = 0.
         while(True):
            ks = self.client.simGetGroundTruthKinematics("Drone0")
            position = utils.vec_to_str(ks.position)
            orientation = utils.quat_to_str(ks.orientation,False)
            linear_velocity = utils.vec_to_str(ks.linear_velocity)
            angular_velocity = utils.vec_to_str(ks.angular_velocity)
            linear_acceleration = utils.vec_to_str(ks.linear_acceleration)
            angular_acceleration = utils.vec_to_str(ks.angular_acceleration)
                        
            line = ",".join([     str(time_counter),       position  ,
            orientation ,            linear_velocity ,            angular_velocity ,
            linear_acceleration , angular_acceleration ,])

            print("[THREAD "+threading.current_thread().getName()+"]",position)
            fout.write(line+"\n")
            time.sleep(configYml["layer2"]["AIRSIM_SAMPLING_INTERVAL"])
            time_counter += configYml["layer2"]["AIRSIM_SAMPLING_INTERVAL"]
            if(self.is_stopped()):
               print("[THREAD "+threading.current_thread().getName()+"] STOP for: ", self.traj_filename)
               break
            
def calibrate(vertices,scale=1):
   asClient = NewMyAirSimClient(trajColFlag=False,
         canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])
   for v in vertices:
      x,y,z = v[0]*scale,v[1]*scale,-50
      pose = Pose(utils.l3_pos_arr_to_airsim_vec((x,y,z),w_offset=c_settings2["W_OFFSET"],h_offset=c_settings2["H_OFFSET"]), 
      to_quaternion(0, 0, 0) ) 
      asClient.simSetVehiclePose(pose,True,"Drone0")     
      time.sleep(1.0)

def main(trajectories_folder,velocity):

   os.makedirs(NEW_FOLDER)

   # # Create AirSim client
   # asClient = NewMyAirSimClient(trajColFlag=False,
   #          canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])

   # Read trajectories
   exp_folders = [os.path.join(trajectories_folder,d) for d in os.listdir(trajectories_folder)]
   latest_mod_folder = max(exp_folders , key=os.path.getmtime)

   print("Detected ",len(latest_mod_folder)," files inside: "+latest_mod_folder+".")
   for f in   sorted(os.listdir(latest_mod_folder)) :
      asClient = NewMyAirSimClient(trajColFlag=False,
         canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])
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
                  trajectory.append( utils.l3_pos_arr_to_airsim_vec([x,y,z],w_offset=c_settings2["W_OFFSET"],h_offset=c_settings2["H_OFFSET"]) )

            # trajs_utils.plot_xy([trajs_utils.vec2d_list_to_tuple_list(trajectory)],20,doScatter=True)


            pointer = asClient.moveOnPathAsync(
               trajectory,
                 velocity, 120,DrivetrainType.ForwardOnly , YawMode(False,0),
               lookahead=20,adaptive_lookahead=1,vehicle_name="Drone0")

            fidx = f.split(".csv")[0].split("traj")[1]
            if(not fidx.isdigit()):
               raise Exception("Error in format of filename:"+f+", (it should be ...traj<id>.csv)")

            positionThread = positionLoggerThread("Drone0",fidx,latest_mod_folder,f)
            positionThread.start()

            print("UAV following trajectory", os.path.join(latest_mod_folder,f),"...")
            pointer.join()
            positionThread.stop()
            positionThread.join()
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
   
   parser.add_argument("-i", type=str,required=False, default = c_paths["LAYER1_3D_OUTPUT_FOLDER"],
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = configYml["layer2"]["VELOCITY"], type=float,required=False,
                     help='Speed value')


   parser.add_argument( '--log',action='store_true',  default=False,
      help='Log into file (default: %(default)s)' )

   parser.add_argument( '--calibrate',action='store_true',  default=False,
      help='Calibration flag to enable calibration (default: %(default)s)' )

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

   if(args.calibrate):
      while(True):
         print("Input scale (def: 1.0):")
         scale = float(input())
         print("Scale set at "+str(scale)+" .")
         vertices = [
         (0,0),
         (0,c_settings["NCOLS"]),
         (c_settings["NROWS"],c_settings["NCOLS"]),
         (c_settings["NROWS"],0),
         (0,0)] 
         calibrate(vertices,scale)
   else:
      main(trajectories_folder=args.i,velocity=args.velocity)

