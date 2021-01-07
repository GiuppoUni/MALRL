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
from newMyAirSimClient import NewMyAirSimClient
import trajs_utils

TRAJECTORIES_3D_FOLDER = "trajectories_3d/csv/"

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Layer 2')
   
   parser.add_argument('--lat', type=float,required=True,
                     help='Starting latitude coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--lon', type=float,required=True,
                     help='Starting longitude coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = 10, type=float,required=False,
                     help='Speed value')

   args = parser.parse_args()

   # Create AirSim client
   asClient = NewMyAirSimClient(trajColFlag=False,
            canDrawTrajectories=True,crabMode=True,thickness = 140)

   regex="CELL00"
   l = asClient.simListSceneObjects(regex)
   if(len(l)!=1):
      raise Exception("Problem in setting UAV initial position") 
   s0 = l[0]
   pose = asClient.simGetObjectPose(s0)
   asClient.simSetVehiclePose( pose, ignore_collison=True, vehicle_name = "Drone0")
                
   
   trajectory = np.array( trajs_utils.fix_traj([list(trajectory)])[0] )
   trajectory_vecs = [utils.list_to_position(x) for x in trajectory]
   
   asClient.moveOnPathAsync(
      trajectory_vecs,
      args.velocity,
      adaptive_lookahead=1,vehicle_name="Drone0")




