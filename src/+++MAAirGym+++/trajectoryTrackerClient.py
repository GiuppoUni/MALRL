from airsim.types import ImageRequest, Vector3r
import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
# Change the path below to point to the directoy where you installed the AirSim PythonClient
#sys.path.append('C:/Users/Kjell/Google Drive/MASTER-THESIS/AirSimpy')

from airsim import MultirotorClient
import airsim
import sys 
import utils

import datetime
import threading


class TrajectoryTrackerClient(MultirotorClient):

    def __init__(self):        

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.drones_names = [ v for v in utils.g_airsim_settings["Vehicles"] ]
        self.trajectory = []
        self.episode = 0
        self.isTracking = False
        self.folder_timestamp =str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))
        self.timestep = 0.1




    def _trackingLoop(self,episode,vName,doTimestamp=False):
        self.trajectory = []
        self.episode = episode
        starting_ts = time.time()
        while (self.isTracking):
            pos,ts = self.check_pos(vName,ret_list=True)
            ts -= starting_ts 
            if doTimestamp:
                data = [ts,pos]
            else: 
                data = pos
            self.trajectory.append(data )
            time.sleep(0.1)

    def start_tracking(self,episode,vName,doTimestamp=False):
        self.isTracking = True
        threading.Thread(target=self._trackingLoop, args=(episode,vName,doTimestamp)).start()
        
    
    def stop_tracking(self):
        self.isTracking = False
        # utils.pkl_save_obj(self.trajectory,"trajectory_" + str(self.episode))
        if(type(self.trajectory[0])==Vector3r):
            self.trajectory = [utils.position_to_list(pos) for pos in self.trajectory]
        utils.numpy_save(self.trajectory,self.folder_timestamp,"trajectory_"+ str(self.episode)+".npy")




    def check_pos(self,vName,ret_list=False):
        p = self.simGetGroundTruthKinematics(vehicle_name = vName).position
        ts = time.time()
        utils.set_offset_position(p)
        # print("[",vName,"]",(p.x_val,p.y_val,p.z_val) )
        if ret_list:
            return utils.position_to_list(p),ts
        else:
            return p,ts

    def simGetPosition(self,lock,vName):
        if(lock):
            lock.acquire()
            p = self.simGetGroundTruthKinematics(vehicle_name = vName).position
            lock.release()
        else:
            p = self.simGetGroundTruthKinematics(vehicle_name = vName).position
        
        pp=(p.x_val,p.y_val,p.z_val)
        print("[THREAD]",pp)
        return  pp


    
    def getPosition(self,vehicle_name = ""):
        kin_state = self.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
        return kin_state.position

    def getOrientation(self,vehicle_name = ""):
        kin_state = self.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
        return kin_state.orientation

    def getPitchRollYaw(self,vehicle_name=""):
        return self.toEulerianAngle(self.getOrientation(vehicle_name=vehicle_name))

    def rotateByYawRate(self, yaw_rate, duration,vehicle_name ):
        return super().rotateByYawRateAsync( yaw_rate, duration,vehicle_name )



    def goal_direction(self, goal, pos, vn):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name=vn)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    

    def _position_to_list(position_vector) -> list:
        return [position_vector.x_val, position_vector.y_val, position_vector.z_val]
    
    
