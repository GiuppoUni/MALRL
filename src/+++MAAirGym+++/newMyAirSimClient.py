from airsim.types import ImageRequest
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

import threading

class DrivetrainType:
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1

class AirSimImageType:    
    Scene = 0
    DepthPlanner = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6

lock = threading.Lock()


class newMyAirSimClient(MultirotorClient):

    def __init__(self):        
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.drones_names = [ v for v in utils.g_airsim_settings["Vehicles"] ]
        def _colorize(idx): 

            if idx == 0:
                return utils.green_color
            elif idx==1: 
                return utils.blue_color
            else : 
                return utils.blue_color

        for i,dn in enumerate( self.drones_names ):
            self.enableApiControl(True,vehicle_name=dn)
            self.armDisarm(True,vehicle_name=dn)
            self.simSetTraceLine(_colorize(i)+[0.7],thickness=4.0,vehicle_name=dn)
            
        self.home_pos = self.getPosition(vehicle_name="Drone0")
    
        self.home_ori = self.getOrientation(vehicle_name="Drone0")
        
        self.z = -6


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


    def moveToPosition(self,x,y,z,velocity,duration,vName):
        now = self.getPosition(vName)
        distance = np.sqrt(np.power((x -now.x_val),2) + np.power((y -now.y_val),2))
        duration = distance / velocity
        super().moveToPositionAsync(x,y,z,velocity,vehicle_name=vName)
        start = time.time()
        return start,duration             

    def straight(self, duration, speed,vName):
        print('STRAIGHT: ', vName)
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name=vName)
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.moveByVelocityZAsync(vx, vy, self.z, duration, DrivetrainType.ForwardOnly, vehicle_name = vName )
        start = time.time()
        return start, duration
    
    def yaw_right(self, duration,vName):
        self.rotateByYawRate(30, duration,vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration,vName):
        self.rotateByYawRate(-30, duration,vehicle_name = vName)
        start = time.time()
        return start, duration
    
       # CRAB ACTIONS

    def crab_straight(self, duration, speed,vName):
        self.client.moveByVelocity(speed, 0, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def crab_right(self, duration,vName):
        self.client.moveByVelocityZ(0, 1, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def crab_left(self, duration,vName):
        self.client.moveByVelocityZ(0, -1, self.z, duration, DrivetrainType.MaxDegreeOfFreedom,
            vehicle_name = vName)        
        start = time.time()
        return start, duration
    
    def take_action(self, action,vName):

        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        while self.getPosition(vehicle_name=vName).z_val < -7.0:
            self.moveToZAsync(-6, 3,vName)
            time.sleep(1)
            print(self.getPosition(vehicle_name=vName).z_val, "and", x)
            x = x + 1
            if x > 10:
                return True        
        
    
        start = time.time()
        duration = 0 
        
        collided = False
        if action == 0:
            start, duration = self.straight(1, 4,vName)
        elif action == 1:         
            start, duration = self.yaw_right(0.8,vName)            
        elif action == 2:
            start, duration = self.yaw_left(1,vName)
            
        while duration > time.time() - start:
            if self.simGetCollisionInfo(vehicle_name=vName).has_collided == True:
                return True    

        self.moveByVelocityAsync(0, 0, 0, 1,vehicle_name=vName)
        self.rotateByYawRate(0, 1,vehicle_name=vName)            
        
        return collided
    

    def take_action_threaded(self, action,lock,vIdx,vName):

        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        levelized = False
        while not levelized:
            lock.acquire()
            z = self.getPosition(vehicle_name=vName).z_val 
            lock.release()
            if (z > -7.0):
                levelized = True
            lock.acquire()
            self.moveToZAsync(-6, 3,vName)
            lock.release()
            time.sleep(1)
            lock.acquire()
            print(self.getPosition(vehicle_name=vName).z_val, "and", x)
            lock.release()
            x = x + 1
            if x > 10:
                return [vIdx,True]        
        
    
        start = time.time()
        duration = 0 
        
        collided = False
        if action == 0:
            lock.acquire()
            start, duration = self.straight(3, 5,vName)
            lock.release()
        elif action == 1:         
            lock.acquire()
            start, duration = self.yaw_right(0.8,vName)            
            lock.release()
        elif action == 2:
            lock.acquire()
            start, duration = self.yaw_left(1,vName)
            lock.release()
        while duration > time.time() - start:
            lock.acquire()
            col = self.simGetCollisionInfo(vehicle_name=vName).has_collided
            lock.release()
            if  col == True:
                return [vIdx,True]    
        lock.acquire()
        self.moveByVelocityAsync(0, 0, 0, 1,vehicle_name=vName)
        lock.release()
        
        lock.acquire()
        self.rotateByYawRate(0, 1,vehicle_name=vName)            
        lock.release()
        
        return [vIdx,collided]

    def goal_direction(self, goal, pos, vn):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name=vn)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
    def getScreenDepthVis(self, track,vehicle_name):
        lock.acquire()
        responses = self.simGetImages([ImageRequest(0, AirSimImageType.DepthPerspective, True, False)],vehicle_name)
        lock.release()
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
            
        #cv2.imshow("Test", total)
        #cv2.waitKey(0)
        
        return total


    def AirSim_reset(self):

        self.reset()
            
        # TODO RESET ALL 
        time.sleep(0.2)
        for dn in self.drones_names:
            self.enableApiControl(True,vehicle_name=dn)
            self.armDisarm(True,vehicle_name=dn)
        time.sleep(1)
        for dn in self.drones_names:
            self.moveToZAsync(self.z, 3,vehicle_name=dn) 
            time.sleep(1)
        time.sleep(2)
    
    @staticmethod
    def toEulerianAngle(q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)


    def position_to_list(position_vector) -> list:
        return [position_vector.x_val, position_vector.y_val, position_vector.z_val]
    
    
