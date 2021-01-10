import os
from airsim.types import ImageRequest, Vector3r
from airsim.utils import to_eularian_angles
from matplotlib.pyplot import draw
import numpy as np
import time
import math
import cv2
from pylab import array, arange, uint8 
from PIL import Image
import eventlet
from eventlet import Timeout
import multiprocessing as mp
from sklearn.neighbors import KDTree
# Change the path below to point to the directoy where you installed the AirSim PythonClient
#sys.path.append('C:/Users/Kjell/Google Drive/MASTER-THESIS/AirSimpy')

from airsim import MultirotorClient
import airsim
import sys 
import utils
import gc 
import pandas
import threading
from scipy.interpolate import interp1d


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




class NewMyAirSimClient(MultirotorClient):

    deg_to_rad = lambda d_angle: d_angle * math.pi / 180.0

    def __init__(self,trajColFlag,canDrawTrajectories,crabMode,thickness,trajs2draw,traj2follow):        

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.drones_names = [ v for v in utils.g_airsim_settings["Vehicles"] ]

        for i,dn in enumerate( self.drones_names ):
            self.enableApiControl(True,vehicle_name=dn)
            self.armDisarm(True,vehicle_name=dn)

        self.trajColFlag = trajColFlag

        self.z_des = -10
        self.z_max = -20
        self.z_min =  -6
        
        self.kdtrees = [] 

        self.trajs2draw=trajs2draw
        self.traj2follow=traj2follow
        self.crabMode = crabMode
        self.canDrawTrajectories = canDrawTrajectories
        self.thickness = thickness
        if(self.canDrawTrajectories):
            self.drawTrajectories()



        # self.trajectories = self._loadPastTrajectories()

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
        self.moveByVelocityZAsync(vx, vy, self.z_des, duration, DrivetrainType.ForwardOnly, vehicle_name = vName )
        start = time.time()
        return start, duration
    

    def go_straight(self):
        if(to_eularian_angles(self.getOrientation("Drone0"))[2] != __class__.deg_to_rad( 0 )):
            rot_duration = 2
            self.moveByRollPitchYawThrottleAsync(0,0,  __class__.deg_to_rad(0),0.60,rot_duration, "Drone0")
            time.sleep(rot_duration)

        start, duration = self.straight(2, 6,"Drone0")
        return start,duration


    def go_right(self):
        if(to_eularian_angles(self.getOrientation("Drone0"))[2] != __class__.deg_to_rad( -90 )):
            rot_duration = 1
            self.moveByRollPitchYawThrottleAsync(0,0,  __class__.deg_to_rad(-90),0.6,rot_duration, "Drone0")
            time.sleep(rot_duration)

        start, duration = self.straight(2, 6,"Drone0")
        return start,duration

    def go_left(self):

        if(to_eularian_angles(self.getOrientation("Drone0"))[2] != __class__.deg_to_rad( 90 )):
            rot_duration = 1
            self.moveByRollPitchYawThrottleAsync(0,0, __class__.deg_to_rad( 90 ),0.6,rot_duration, "Drone0")
            time.sleep(rot_duration)

        start, duration = self.straight(2, 6,"Drone0")
        return start,duration


    def go_back(self):
        if(to_eularian_angles(self.getOrientation("Drone0"))[2] != __class__.deg_to_rad( 180 )):
            rot_duration = 2
            self.moveByRollPitchYawThrottleAsync(0,0, __class__.deg_to_rad( 180 ),0.60,rot_duration, "Drone0")
            time.sleep(rot_duration)

        start, duration = self.straight(2, 6,"Drone0")
        return start,duration



    def yaw_right(self, duration,vName,yawRate=-30):
        self.rotateByYawRate(yawRate, duration,vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def yaw_left(self, duration,vName,yawRate=30):
        self.rotateByYawRate(yawRate, duration,vehicle_name = vName)
        start = time.time()
        return start, duration
    
       # CRAB ACTIONS
    def crab_up(self, duration=12, speed=12,vName="Drone0"):
        self.moveByVelocityZAsync(0, -speed, self.z_des, duration, DrivetrainType.ForwardOnly,
            vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def crab_right(self, duration=12,speed=12,vName="Drone0"):
        self.moveByVelocityZAsync(speed, 0, self.z_des, duration, DrivetrainType.ForwardOnly,
            vehicle_name = vName)
        start = time.time()
        return start, duration
    
    def crab_left(self, duration=12,speed=12,vName="Drone0"):
        self.moveByVelocityZAsync(-speed, 0, self.z_des, duration, DrivetrainType.ForwardOnly,
            vehicle_name = vName)        
        start = time.time()
        return start, duration
    
    def crab_down(self, duration=12,speed=12,vName="Drone0"):
        self.moveByVelocityZAsync(0, speed, self.z_des, duration, DrivetrainType.ForwardOnly,
            vehicle_name = vName)        
        start = time.time()
        return start, duration
    
    def take_action(self, action,vName):

        #check if copter is on level cause sometimes he goes up without a reason
        x = 0
        cur_pos = self.getPosition(vehicle_name=vName)

        result = {"collisions_per_traj": None,"total_p":0, "obs":False, "zout":False}

        if(self.trajColFlag):
            total_p, p_per_traj = self.check_traj_collision(utils.position_to_list(cur_pos),
                radius = 10,count_only = True,specify_collision = True)
            print('traj_collisions: ', p_per_traj)
            
            if total_p > 0:
                print("*"*100,"\nPOINT COLLISION\n","*"*100)
                result["total_p"] = total_p
                result["collisions_per_traj"] = p_per_traj 
                return result
        
        while self.z_max > -cur_pos.z_val > self.z_min:
            print(cur_pos.z_val, "and", x)
            self.moveToZAsync(-6, 3,vName)
            time.sleep(1)
            x = x + 1
            if x > 10:
                print("LEVELEZING ATTEMPT TIMEOUT")
                result["zout"] = True  
                return result      
            cur_pos = self.getPosition(vehicle_name=vName)
        
    
        start = time.time()
        duration = 0 
        
        if(not self.crabMode):
            if action == 0:
                # start, duration = self.straight(1, 4,vName)
                start, duration = self.go_left()
            
            elif action == 1:         
                # start, duration = self.yaw_right(0.8,vName)            
                start, duration = self.go_straight()

            elif action == 2:
                # start, duration = self.yaw_left(0.8,vName)
                start, duration = self.go_right()

            elif action == 3:
                # start, duration = self.yaw_left(0.8,vName)
                start, duration = self.go_back()
        else:
            if action == 0:
                # start, duration = self.straight(1, 4,vName)
                start, duration = self.crab_left()
            
            elif action == 1:         
                # start, duration = self.yaw_right(0.8,vName)            
                start, duration = self.crab_up()

            elif action == 2:
                # start, duration = self.yaw_left(0.8,vName)
                start, duration = self.crab_right()

            elif action == 3:
                # start, duration = self.yaw_left(0.8,vName)
                start, duration = self.crab_down()

        while duration > time.time() - start:
            if self.simGetCollisionInfo(vehicle_name=vName).has_collided == True:
                print("OSBTACLE COLLISION")
                result["obs"] = True 
                return result
            
        self.moveByVelocityAsync(0, 0, 0, 1,vehicle_name=vName)
        self.rotateByYawRate(0, 1,vehicle_name=vName)

        
        return result
    

    # def take_action_threaded(self, action,lock,vIdx,vName):

    #     #check if copter is on level cause sometimes he goes up without a reason
    #     x = 0
    #     levelized = False
    #     while not levelized:
    #         lock.acquire()
    #         z = self.getPosition(vehicle_name=vName).z_val 
    #         lock.release()
    #         if (z > -7.0):
    #             levelized = True
    #         lock.acquire()
    #         self.moveToZAsync(-6, 3,vName)
    #         lock.release()
    #         time.sleep(1)
    #         lock.acquire()
    #         print(self.getPosition(vehicle_name=vName).z_val, "and", x)
    #         lock.release()
    #         x = x + 1
    #         if x > 10:
    #             return [vIdx,True]        
        
    
    #     start = time.time()
    #     duration = 0 
        
    #     collided = False
    #     if action == 0:
    #         lock.acquire()
    #         start, duration = self.straight(3, 5,vName)
    #         lock.release()
    #     elif action == 1:         
    #         lock.acquire()
    #         start, duration = self.yaw_right(0.8,vName)            
    #         lock.release()
    #     elif action == 2:
    #         lock.acquire()
    #         start, duration = self.yaw_left(1,vName)
    #         lock.release()
    #     while duration > time.time() - start:
    #         lock.acquire()
    #         col = self.simGetCollisionInfo(vehicle_name=vName).has_collided
    #         lock.release()
    #         if  col == True:
    #             return [vIdx,True]    
    #     lock.acquire()
    #     self.moveByVelocityAsync(0, 0, 0, 1,vehicle_name=vName)
    #     lock.release()
        
    #     lock.acquire()
    #     self.rotateByYawRate(0, 1,vehicle_name=vName)            
    #     lock.release()
        
    #     return [vIdx,collided]

    def goal_direction(self, goal, pos, vn):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name=vn)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180    
    
    
  

    def distanceFromTraj(self,pos: Vector3r):
        return 0

    def draw_numpy_trajectory(self,trajectory):
        # TODO replace for a specific trajectories file 
        # try:
        # trajectory =  np.load(filename)
        # if(filename[-3:]=="csv"):
        #     trajectory = np.array(pandas.read_csv(filename,delimiter=",",usecols=[1,2,3]) )
        #     # trajectory = utils.myInterpolate(trajectory,n_samples = 100)
        #     # trajectory = np.array(pandas.read_csv(filename,delimiter=",",index_col="index") )
        # elif filename[-3:]=="npy":
        #     trajectory = np.load(filename)
            
        print("Drawing trajectory:",trajectory)
        trajectory_vecs = [utils.list_to_position(x) for x in trajectory]
        self.simPlotLineStrip(trajectory_vecs,
            is_persistent= True, thickness = self.thickness)
        
        # _tree = KDTree(trajectory)
        # self.kdtrees.append(_tree)
        
        # Free some mem
        del trajectory
        del trajectory_vecs
        gc.collect()
        # except Exception as e:
        #     print(filename,"Exception in reading")
        #     raise Exception("Exception in reading",filename,e)
        return 


    def drawTrajectories(self):
        # traj_fold = os.path.join(utils.TRAJECTORIES_FOLDER,"csv")
        for t in self.trajs2draw:
            self.draw_numpy_trajectory(np.array(t))
        # print('self.kdtrees: ', self.kdtrees)


    def check_traj_collision(self,current_pos,radius,count_only,specify_collision):
        total_points = 0
        points_per_traj = None
        if count_only:
            if(specify_collision):
                points_per_traj = dict()
                for idx,_tree in enumerate(self.kdtrees):
                    res = _tree.query_radius( [current_pos],r=radius,count_only = count_only )
                    if res > 0:
                        print("Collisions with","Trajectory_"+str(idx))
                        points_per_traj[idx] = res
                        total_points += res
            else:
                total_points = np.sum([ _tree.query_radius( [current_pos],r=radius,count_only = count_only ) 
                for _tree in self.kdtrees ])
        else:
            for _tree in self.kdtrees:
                # Sono gli index
                res = _tree.query_radius( [current_pos],r=radius,count_only = count_only ) 
                total_points += res.shape()
                raise Exception("TODO")

        return total_points, points_per_traj 
            


    def AirSim_reset(self):

            
        # TODO RESET ALL 
        time.sleep(0.2)
        for dn in self.drones_names:
            self.enableApiControl(True,vehicle_name=dn)
            self.armDisarm(True,vehicle_name=dn)
        time.sleep(1)
        for dn in self.drones_names:
            self.moveToZAsync(self.z_des, 3,vehicle_name=dn) 
            time.sleep(1)


    def disable_trace_lines(self):
        for i,dn in enumerate(self.drones_names):
            self.simSetTraceLine([0,0,0,0],
                thickness=0.0,vehicle_name=dn)

    def enable_trace_lines(self):
        for i,dn in enumerate(self.drones_names):
            self.simSetTraceLine(utils.green_color+[0.7],
                thickness=self.thickness*2,vehicle_name=dn)


        
    
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
    
    
