import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp

import sys
# Change the path below to point to the directoy where you installed the AirSim PythonClient
sys.path.append('C:/Users/gioca/Desktop/Repos/AirSim-PredictiveManteinance/src/AirGym')

from AirSimClient import *

vehicles = ["Drone1", "Drone2", "Drone3"]


class Drone:
    def __init__(self,vehicle_name = None ,img1=None,img2=None,
        home_pos=None,home_ori=None,z=None,client = None):
        
        self.vehicle_name = vehicle_name
        self.img1 = img1
        self.img2 = img2
        self.home_pos = home_pos
        self.home_ori = home_ori
        self.z = z
        self.client = client

    def straight(self, duration, speed):
        pitch, roll, yaw  = self.client.getPitchRollYaw(self.vehicle_name)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.client.moveByVelocityZ(vx, vy, self.z, duration, DrivetrainType.ForwardOnly,
            vehicle_name =self.vehicle_name)
        start = time.time()
        return start, duration
    
    def yaw_right(self, duration):
        self.client.rotateByYawRate(30, duration,vehicle_name=self.vehicle_name)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration):
        self.client.rotateByYawRate(-30, duration,vehicle_name = self.vehicle_name)
        start = time.time()
        return start, duration
    
    
    def take_action(self, action):
		
        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        while self.client.getPosition()["z_val"] < -7.0:
            print("["+self.vehicle_name+"]","Levelizing...")
            self.client.moveToZAsync(-6, 3,vehicle_name=self.vehicle_name)
            time.sleep(1)
            print(self.client.getPosition()["z_val"], "and", x)
            x = x + 1
            if x > 10:
                return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False

        if action == 0:

            start, duration = self.straight(1, 4)
        
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo(vehicle_name = self.vehicle_name).has_collided == True:
                    return True    
                
            self.client.moveByVelocity(0, 0, 0, 1,vehicle_name = self.vehicle_name)
            self.client.rotateByYawRate(0, 1, vehicle_name = self.vehicle_name)
            
            
        if action == 1:
         
            start, duration = self.yaw_right(0.8)
            
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo(vehicle_name = self.vehicle_name).has_collided == True:
                    return True
            
            self.client.moveByVelocity(0, 0, 0, 1,vehicle_name = self.vehicle_name)
            self.client.rotateByYawRate(0, 1,vehicle_name = self.vehicle_name)
            
        if action == 2:
            
            start, duration = self.yaw_left(1)
            
            while duration > time.time() - start:
                if self.client.simGetCollisionInfo(vehicle_name = self.vehicle_name).has_collided == True:
                    return True
                
            self.client.moveByVelocity(0, 0, 0, 1, vehicle_name = self.vehicle_name)
            self.client.rotateByYawRate(0, 1, vehicle_name = self.vehicle_name)
            
        return collided
    
    def goal_direction(self, goal, pos):
        
        pitch, roll, yaw  = self.client.getPitchRollYaw(vehicle_name = self.vehicle_name)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos["y_val"], goal[0]- pos["x_val"])
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)],
            vehicle_name = self.vehicle_name)
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        
        
        small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
                
        cut = small[20:40,:]
        
        info_section = np.zeros((10,cut.shape[1]),dtype=np.uint8) + 255
        info_section[9,:] = 0
        
        line = np.int((((track - -180) * (100 - 0)) / (180 - -180)) + 0)
        
        if line != (0 or 100):
            info_section[:,line-1:line+2]  = 0
        elif line == 0:
            info_section[:,0:3]  = 0
        elif line == 100:
            info_section[:,info_section.shape[1]-3:info_section.shape[1]]  = 0
            
        total = np.concatenate((info_section, cut), axis=0)
            
        # cv2.imshow("Test img "+self.vehicle_name, total)
        # cv2.waitKey(0)
        
        return total

    def getState(self):
        return self.client.getMultirotorState(vehicle_name=self.vehicle_name)[b"kinematics_estimated"]

    def enable_armDisarm(self):
        self.tagPrint("Enable Arm Disarm ...")
        self.client.enableApiControl(True,vehicle_name = self.vehicle_name)
        self.client.armDisarm(True,vehicle_name = self.vehicle_name)

    def reset_position(self):
        self.tagPrint("Resetting position (async) ...")
        self.enable_armDisarm()
        # time.sleep(1)
        self.client.moveToZAsync(self.z, 1.5,vehicle_name=self.vehicle_name) 
        # time.sleep(3)

    def tagPrint(self,s=""):
        sys.stdout.write(f"\t [{self.vehicle_name}] "+ s + "\n")
        sys.stdout.flush()

class MyAirSimClient(MultirotorClient):

    def __init__(self):        
        
        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.drones = []
        for v in vehicles:
            uav = Drone(vehicle_name=v,client = self)
            uav.enable_armDisarm()
            
            kin_state = uav.getState()
            # print("STATE:",kin_state)

            uav.home_pos = kin_state[b"position"]
        
            uav.home_ori = kin_state[b"orientation"]
            
            uav.z = -6

            self.drones.append(uav)

    def AirSim_reset(self):
        
        self.reset()
        time.sleep(0.2)

        for d in self.drones:
            d.reset_position()
    
    def AirSim_reset_old(self):
        
        reset = False
        z = -6.0
        while reset != True:

            now = self.getPosition()
            self.simSetPose(Pose(Vector3r(now["x_val"], now["y_val"], -30),Quaternionr(self.home_ori["w_val"], self.home_ori["x_val"], self.home_ori["y_val"], self.home_ori["z_val"])), True) 
            now = self.getPosition()
            
            if (now["z_val"] - (-30)) == 0:
                self.simSetPose(Pose(Vector3r(self.home_pos["x_val"], self.home_pos["y_val"], -30),Quaternionr(self.home_ori["w_val"], self.home_ori["x_val"], self.home_ori["y_val"], self.home_ori["z_val"])), True)
                now = self.getPosition()
                
                if (now["x_val"] - self.home_pos["x_val"]) == 0 and (now["y_val"] - self.home_pos["y_val"]) == 0 and (now["z_val"] - (-30)) == 0 :
                    self.simSetPose(Pose(Vector3r(self.home_pos["x_val"], self.home_pos["y_val"], self.home_pos["z_val"]),Quaternionr(self.home_ori["w_val"], self.home_ori["x_val"], self.home_ori["y_val"], self.home_ori["z_val"])), True)
                    now = self.getPosition()
                    
                    if (now["x_val"] - self.home_pos["x_val"]) == 0 and (now["y_val"] - self.home_pos["y_val"]) == 0 and (now["z_val"] - self.home_pos["z_val"]) == 0:
                        reset = True
                        self.moveByVelocity(0, 0, 0, 1)
                        time.sleep(1)
                        
        self.moveToZAsync(z, 3,vehicle_name=VEHICLE_NAME)  
        time.sleep(3)



def position_to_list(position_vector) -> list:
    return [position_vector["x_val"], position_vector["y_val"], position_vector["z_val"]]

def fullDecodeDict(bDict):
    # TODO to avoid use bytes
        # for bk,bv in bDict.items():
        #     if(type(bv)!=dict):
        #         return {str(bk,'utf-8'):str(bv,'utf-8') }
            # else:
            #     k = str(bk,'utf-8')
            #     return { k :decodeDict(bDict[bk] ) }
    # d = dict()
    # for k,v in bDict.items():        
    #     if (type(k) == bytes):
    #         k = str(k,'utf-8')
    #     if (type(v) == bytes):
    #         v = str(v,'utf-8') 
    #     d[k]=v

    # return d
    pass 

