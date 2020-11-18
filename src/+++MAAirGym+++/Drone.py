import time
import math

from numpy.lib import arraypad
from airsim.types import *
import cv2 
import sys
from pylab import array, arange, uint8 
from PIL import Image


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
        while self.client.getPosition().z_val < -7.0:
            print("["+self.vehicle_name+"]","Levelizing...")
            self.client.moveToZAsync(-6, 3,vehicle_name=self.vehicle_name)
            time.sleep(1)
            print(self.client.getPosition().z_val, "and", x)
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
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
    def getScreenDepthVis(self, track):

        responses = self.client.simGetImages([ImageRequest(0, ImageType.DepthPerspective, True, False)],
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
        return self.client.getMultirotorState(vehicle_name=self.vehicle_name).kinematics_estimated

    def enable_armDisarm(self):
        self.tagPrint("Enable Arm Disarm ...")
        self.client.enableApiControl(True,vehicle_name = self.vehicle_name)
        self.client.armDisarm(True,vehicle_name = self.vehicle_name)
    

    def reset_Zposition(self):
        self.tagPrint("Resetting position (async) ...")
        self.enable_armDisarm()
        # time.sleep(1)
        self.client.moveToZAsync(self.z, 1.5,vehicle_name=self.vehicle_name) 
        # time.sleep(3)

    def tagPrint(self,s=""):
        sys.stdout.write(f"\t [{self.vehicle_name}] "+ s + "\n")
        sys.stdout.flush()
