import os
import airsim
import time
from airsim.client import MultirotorClient
from airsim.types import DrivetrainType, Vector3r, YawMode
import numpy as np
from pyproj import Proj
import sys
from Drone import DroneAgent
from gym_airsim.envs.Target import TargetManager,Target
# Change the path below to point to the directoy where you installed the AirSim PythonClient
# sys.path.append('../../')
import utils
import math
import cv2

#=======================================================#
# Wrapper class to handle airsim client and multi drones #
#            Developed on top of AirSim 1.3.1            #
#     Other versions of AirSim may be not compatible     #
#=======================================================#




vehicles = utils.g_airsim_settings["Vehicles"]
print('vehicles: ', [v for v in vehicles])


class MyAirSimClient2():
    
    """
    AirSim client that understands arbitrary projection systems
        Assumes that the simulation environment (unreal) is in the coordinate system specified
        by the srid but offset by the origin specified.
        Arguments:
            srid {str} -- EPSG SRID string. Example "EPSG:3857"
            origin {list} -- [Longitude, Latitude, Height]
            kwargs -- Any keyword arguments forwared to AirSim
    """
    
    def __init__(self,srid, origin,**kwargs):       
        print("Creating custom client ...")
        # super(MyAirSimClient, self).__init__(**kwargs)
        self.direct_client = airsim.MultirotorClient()

        # MultirotorClient.__init__(self)
        print("con conf ...")
        self.direct_client.confirmConnection()
        print("con confirmed.")
        self.drones = []
        self.drones_names = []
        self.pts= [] #List of pointers of future eleemnts for join

        # TODO replace with variable
        self.targetMg= TargetManager(6)

        self.srid = srid
        self.origin = origin

        self.proj = Proj(init=srid)
        self.origin_proj = self.proj(*self.origin[0:2]) + (self.origin[2],)
        
        

        for v in vehicles:
            print("Craete",v)
            uav = DroneAgent(vehicle_name=v, client = self.direct_client, 
                z = utils.g_vehicles[v]["Z"])
            uav.enable_armDisarm()
            
            kin_state = uav.getState()
            # print("STATE:",kin_state)

            uav.home_pos = kin_state.position
        
            uav.home_ori = kin_state.orientation
            
 
            self.drones.append(uav)
            self.drones_names.append(uav.vehicle_name)

        print("Custom client created.")
        self.wait_joins("reach init z")
        

    def wait_joins(self,msg=""):
        for i,p in enumerate(self.pts):
            p.join()
            utils.dronePrint(i,"Join completed "+msg)

    def place_drones(self,init_pose):
        poses = []
        for i,uav_name in enumerate(self.drone_names):
            status = self.place_one_drone( vehicle_name = uav_name, gps=init_pose[i] )
            poses.append(status[0])
        return poses



        
    # def AirSim_reset_old(self):
        
    #     reset = False
    #     z = -6.0
    #     while reset != True:

    #         now = self.getPosition()
    #         self.simSetPose(Pose(Vector3r(now.x_val, now.y_val, -30),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True) 
    #         now = self.getPosition()
            
    #         if (now.z_val - (-30)) == 0:
    #             self.simSetPose(Pose(Vector3r(self.home_pos.x_val, self.home_pos.y_val, -30),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True)
    #             now = self.getPosition()
                
    #             if (now.x_val - self.home_pos.x_val) == 0 and (now.y_val - self.home_pos.y_val) == 0 and (now.z_val - (-30)) == 0 :
    #                 self.simSetPose(Pose(Vector3r(self.home_pos.x_val, self.home_pos.y_val, self.home_pos.z_val),Quaternionr(self.home_ori.w_val, self.home_ori.x_val, self.home_ori.y_val, self.home_ori.z_val)), True)
    #                 now = self.getPosition()
                    
    #                 if (now.x_val - self.home_pos.x_val) == 0 and (now.y_val - self.home_pos.y_val) == 0 and (now.z_val - self.home_pos.z_val) == 0:
    #                     reset = True
    #                     self.moveByVelocity(0, 0, 0, 1)
    #                     time.sleep(1)
                        
    #     self.moveToZAsync(z, 3,vehicle_name="")  
    #     time.sleep(3)


    #==================  AIRSIM GEO APIs =====================
    
    def lonlatToProj(self, lon, lat, z, inverse=False):
        proj_coords = self.proj(lon, lat, inverse=inverse)
        return proj_coords + (z,)

    def projToAirSim(self, x, y, z):
        x_airsim = x - self.origin_proj[0]
        y_airsim = y - self.origin_proj[1]
        z_airsim = -z + self.origin_proj[2]
        return (x_airsim, -y_airsim, z_airsim)

    def lonlatToAirSim(self, lon, lat, z):
        return self.projToAirSim(*self.lonlatToProj(lon, lat, z))

    def nedToProj(self, x, y, z):
        """
        Converts NED coordinates to the projected map coordinates
        Takes care of offset origin, inverted z, as well as inverted y axis
        """
        x_proj = x + self.origin_proj[0]
        y_proj = -y + self.origin_proj[1]
        z_proj = -z + self.origin_proj[2]
        return (x_proj, y_proj, z_proj)

    def nedToGps(self, x, y, z):
        return self.lonlatToProj(*self.nedToProj(x, y, z), inverse=True)

    def getGpsLocation(self, vehicle_name = ""):
        """
        Gets GPS coordinates of the vehicle.
        """
        pos = self.direct_client.simGetGroundTruthKinematics(vehicle_name= vehicle_name).position
        gps = self.nedToGps(pos.x_val, pos.y_val, pos.z_val)
        return gps

    def moveToPositionAsyncGeo(self, gps=None, proj=None,vel=10, **kwargs):
        """
        Moves to the a position that is specified by gps (lon, lat, +z) or by the projected map 
        coordinates (x, y, +z).  +z represent height up.
        """
        coords = None
        if gps is not None:
            coords = self.lonlatToAirSim(*gps)
        elif proj is not None:
            coords = self.projToAirSim(*proj)
        if coords:
            return self.moveToPositionAsync(coords[0], coords[1], coords[2],velocity=vel, **kwargs)
        else:
            print('Please pass in GPS (lon,lat,z), or projected coordinates (x,y,z)!')

    def moveOnPathAsyncGeo(self, gps=None, proj=None, velocity=10, **kwargs):
        """
        Moves to the a path that is a list of points. The path points are either gps (lon, lat, +z) or by the projected map 
        coordinates (x, y, +z).  +z represent height is up.
        """
        path = None
        if gps is not None:
            path = [Vector3r(*self.lonlatToAirSim(*cds)) for cds in gps]
        elif proj is not None:
            path = [Vector3r(*self.projToAirSim(*cds)) for cds in proj]
        if path:
            # print(gps, path)
            return self.moveOnPathAsync(path, velocity=velocity, **kwargs)
        else:
            print(
                'Please pass in GPS [(lon,lat,z)], or projected coordinates [(x,y,z)]!')    

    #======================= CUSTOM API (WRAPPERS) =======================

    def getPosition(self,vehicle_name = ""):
        kin_state = self.direct_client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
        return kin_state.position

    def place_one_drone(self,vehicle_name, gps=None, proj=None,**kwargs ):
        """
            Place one drone using GPS/proj coords.
            Return: gps loc , orientation
        """
        coords = None
        if gps is not None:
            coords = self.lonlatToAirSim(*gps)
        elif proj is not None:
            coords = self.projToAirSim(*proj)
        if coords:
            ORIENTATION = airsim.Quaternionr(0, 0, 0, 0)
            pose = airsim.Pose(airsim.Vector3r(*coords), ORIENTATION)
            self.direct_client.simSetVehiclePose(pose, 
                ignore_collison=True,vehicle_name = vehicle_name,**kwargs)
            _gps = self.getGpsLocation(vehicle_name = vehicle_name)
            _or = self.getOrientation(vehicle_name=vehicle_name)
            return _gps,_or
        else:
            print('Please pass in GPS (lon,lat,z), or projected coordinates (x,y,z)!')
            



    
    def getOrientation(self,vehicle_name = ""):
        kin_state = self.direct_client.getMultirotorState(vehicle_name=vehicle_name).kinematics_estimated
        return kin_state.orientation

    def getPitchRollYaw(self,vehicle_name=""):
        return self.toEulerianAngle(self.getOrientation(vehicle_name=vehicle_name))

    def rotateByYawRate(self, yaw_rate, duration,vehicle_name ):
        return super().rotateByYawRateAsync( yaw_rate, duration,vehicle_name )

    def moveByVelocityZ(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(),vehicle_name = "" ):
        return super().moveByVelocityZAsync( vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name = vehicle_name)

    def moveByVelocity(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(),vehicle_name = ""):
        print("Moving:",vehicle_name)
        # self.drones[int(vehicle_name[-1])].moving = True
        return super().moveByVelocityAsync( vx, vy, vz, duration, drivetrain, yaw_mode,vehicle_name)


    def enable_control(self):
        for dn in self.drones_names :
            self.enableApiControl(True, dn)
            self.armDisarm(True, dn )

    def wakeup_drone(self,dn):
        self.enableApiControl(True,dn)
        self.armDisarm(True,dn)

    def takeoff_all_drones(self) -> None:
        """
        Make all vehicles takeoff, one at a time and return the
        pointer for the last vehicle takeoff to ensure we wait for
        all drones
        """
        vehicle_pointers = []
        for dn in self.drones_names:
            vehicle_pointers.append( self.takeoffAsync(vehicle_name= dn) )
        # All of this happens asynchronously. Hold the program until the last vehicle
        # finishes taking off.
        return vehicle_pointers[-1]

    def get_all_drone_positions(self) -> np.array:
        pos={"pos":[]}
        gps={"gps":[]}
        for i, drone_name in self.drones_names:
            state_data = self.getMultirotorState(vehicle_name=drone_name)
            # print(state_data)
            pos.append( position_to_list(state_data.kinematics_estimated.position) )
            gps.append( gps_position_to_list(state_data.gps_location) )
        return pos, gps

    def allocate_all_targets(self):
        for i,dn in enumerate(self.drones_names):
            self.targetMg.allocate_target(dn,i)
        return [  (self.targetMg.targets[t_id].x_val,self.targetMg.targets[t_id].y_val) for t_id in self.targetMg.targets]

    # ============================== HELPERS ==============================================
    
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



    def transform_input(responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final




    def snap_all_cams(client, vehicle_names):
        responses = dict()
        for v in vehicle_names:
            responses[v] = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("1", airsim.ImageType.DepthVis),  #depth visualization image
            airsim.ImageRequest("2", airsim.ImageType.DepthPlanner,True ),
            airsim.ImageRequest("3", airsim.ImageType.DepthPerspective),
            airsim.ImageRequest("4", airsim.ImageType.DisparityNormalized),
            # airsim.ImageRequest("5", airsim.ImageType.Segmentation),
            # airsim.ImageRequest("6", airsim.ImageType.SurfaceNormals ),
            # airsim.ImageRequest("7", airsim.ImageType.Infrared )
            ])  #scene vision image in uncompressed RGB array
            print(type(responses[v]))
            responses[v].append(client.simGetImage("0", airsim.ImageType.Segmentation))
            responses[v].append(client.simGetImage("0", airsim.ImageType.SurfaceNormals))
            responses[v].append(client.simGetImage("0", airsim.ImageType.Infrared))

            print(v+': Retrieved images in num: %d' % len(responses[v]))
        return responses



    def snap_cam(client,vehicles_names,cameraType):
        for v in vehicles_names:
            response = client.simGetImage("0", cameraType)
            tmp_dir = "single_shots"
            filename = os.path.join(tmp_dir, v +"_cam"+str(cameraType))

            try:
                os.makedirs(tmp_dir)
            except OSError:
                if not os.path.isdir(tmp_dir):
                    raise
        
            # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img_rgb = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
            
    def save_all_cams(responses,dir="."):
        tmp_dir = os.path.join(dir, "airsim_images"+time.strftime("%Y%m%d-%H%M%S"))
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        for drone_name in responses:
            for idx,response in enumerate(responses[drone_name]):
                filename = os.path.join(tmp_dir, drone_name +"_cam"+str(idx))

                if type(response) is bytes:
                    print("Type",type(response))
                    img_rgb = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
                    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
                elif response.pixels_as_float:
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                    airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
                elif response.compress: #png format
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
                else: #uncompressed array
                    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
                    cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


    def snap_and_save_all_cams(client,vehicle_names):
        pass
        