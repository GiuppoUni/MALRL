import time
import math

from numpy.lib import arraypad
from airsim.types import *
import cv2 
import sys
from pylab import array, arange, uint8 
from PIL import Image

import utils

class DroneAgent:
    def __init__(self,vehicle_name = None ,img1=None,img2=None,
        home_pos=None,home_ori=None,z=None,client = None):
        
        self.vehicle_name = vehicle_name
        self.img1 = img1
        self.img2 = img2
        self.home_pos = home_pos
        self.home_ori = home_ori
        self.z = z
        self.client = client
        self.targets = []
        self.track = None

    # FORWARD ONLY ACTIONS

    def straight(self, duration, speed):
        # pitch, roll, yaw  = self.client.getPitchRollYaw(self.vehicle_name)
        # vx = math.cos(yaw) * speed
        # vy = math.sin(yaw) * speed
        # vz = 0

        # self.client.wakeup_drone(self.vehicle_name)
        self.client.enableApiControl(True,"Drone0")
        self.client.armDisarm(True,"Drone0")
        vx,vy,vz = 3,0,0
        pointer = self.client.moveByVelocityAsync( vx=vx, vy=vy, vz=vz, duration = duration,vehicle_name = "Drone0")
        start = time.time()
        return start, duration, pointer
    
    def gotoGoal(self, duration, speed):
        
        self.client.moveToPositionAsync(5, -5, -20, duration, vehicle_name=self.vehicle_name)
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
    
    def stop_moving(self,duration):
        self.client.moveByVelocity(0, 0, 0, duration,vehicle_name = self.vehicle_name)
        self.client.rotateByYawRate(0, 1, vehicle_name = self.vehicle_name)
        start = time.time()
        return start, duration

    # CRAB ACTIONS

    def crab_straight(self, duration, speed):
        self.client.moveByVelocity(speed, 0, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = self.vehicle_name)
        start = time.time()
        return start, duration
    
    def crab_right(self, duration):
        self.client.moveByVelocityZ(0, 1, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = self.vehicle_name)
        start = time.time()
        return start, duration
    
    def crab_left(self, duration):
        self.client.moveByVelocityZ(0, -1, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = self.vehicle_name)        
        start = time.time()
        return start, duration


    # NOTE: ACTION EXECUTION
    def take_action(self, action):
		
        # Check if copter is on level cause sometimes he goes up without a reason
        min_z = float(utils.g_config["agent"]["min_z"])
        # self.z = self.client.getPosition(vehicle_name=self.vehicle_name).z_val
        # print("["+self.vehicle_name+"] Cur Z", self.z)
        
        # if  cur_z < min_z:
        #     print("["+self.vehicle_name+"]","Levelizing...")
        #     self.client.moveToZAsync(z = min_z, velocity = 3,vehicle_name=self.vehicle_name)
        
        # i = 0
        # while self.client.getPosition(vehicle_name=self.vehicle_name).z_val < min_z:
        #     print(self.client.getPosition(vehicle_name=self.vehicle_name).z_val, "and", i)
        #     i += 1
        #     time.sleep(1)
        #     if i > 10:
        #         return True        
        # time.sleep(5)
        # print(self.client.getPosition(vehicle_name=self.vehicle_name).z_val, "and", i)
        
    
        start = time.time()
        duration = 0 
        
        collided = False
        pointer = None
        if action == 0:
            start, duration, pointer = self.straight(5, 5)
            # start, duration = self.gotoGoal(10,5)            
        elif action == 1:
            start, duration = self.yaw_right(1)
        elif action == 2:
            start, duration = self.yaw_left(1)
        elif action == 3:
            start, duration = self.stop_moving(3)
        
        # self.tagPrint("CHECKING COLLISION...")
        # while duration > time.time() - start:
        #     if self.client.simGetCollisionInfo(vehicle_name = self.vehicle_name).has_collided == True:
        #         print("-"*40,"COLLISION!","-"*40)
        #         return True
        #     time.sleep(0.2)    
        # self.tagPrint("CHECK DONE.")

            

        # TO STOP CURRENT ACTION
        # self.client.moveByVelocity(0, 0, 0, 1,vehicle_name = self.vehicle_name)
        # time.sleep(1)
        # self.client.rotateByYawRate(0, 1, vehicle_name = self.vehicle_name)
        # time.sleep(1)

        return collided,pointer
    
    def goal_direction(self, goal, pos):
        
        # pitch, roll, yaw  = self.client.getPitchRollYaw(vehicle_name = self.vehicle_name)
        # yaw = math.degrees(yaw) 
        
        # pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        # pos_angle = math.degrees(pos_angle) % 360

        # track = math.radians(pos_angle - yaw)  
        
        # return ((math.degrees(track) - 180) % 360) - 180    
        return 0    
    
    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([ImageRequest(0, ImageType.DepthPerspective, True, False)],
            vehicle_name = self.vehicle_name)
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        if(img1d.size != 1):
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        else:
            img2d = (0,0)
        
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
        return self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated

    def enable_armDisarm(self):
        self.tagPrint("Enable Arm Disarm ...")
        self.client.enableApiControl(True,vehicle_name = self.vehicle_name)
        self.client.armDisarm(True,vehicle_name = self.vehicle_name)
    

    def reset_Zposition(self):
        self.tagPrint("Resetting Z position (async) ...")
        # self.enable_armDisarm()
        # time.sleep(1)
        return self.client.moveToZAsync(self.z, 0.1,vehicle_name=self.vehicle_name) 
        # time.sleep(3)
        # return self.client.takeoffAsync(vehicle_name = self.vehicle_name)

    def tagPrint(self,s=""):
        sys.stdout.write(f"\t [{self.vehicle_name}] "+ s + "\n")
        sys.stdout.flush()
