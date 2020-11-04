import airsim
import numpy as np
import json

# WINDOWS ONLY
SETTINGS_PATH = 'C:/Users/gioca/OneDrive/Documents/Airsim/'

ORIENTATION = airsim.Quaternionr(0, 0, 0, 0)
from configparser import ConfigParser

with open(SETTINGS_PATH + 'settings.json', 'r') as jsonFile:
    settings = json.load(jsonFile)



config = ConfigParser()
config.read('config.ini')
g_config =  config["uav_group_settings"]
num_uavs = int( g_config["num_uavs"] )


def enable_control_all(client,vehicles):
    print("Enabling drones...")
    for vehicle in vehicles:
        client.enableApiControl(True,vehicle)
        client.armDisarm(True,vehicle)

def disable_control_all(client,vehicles):
    print("Disabling drones...")
    for vehicle in vehicles:
        client.enableApiControl(False,vehicle)
        client.armDisarm(False,vehicle)


def takeoff_all(client,vehicles):
    """
       Make all vehicles takeoff, one at a time and return the
       pointer for the last vehicle takeoff to ensure we wait for
       all drones
    """
    print("Starting takeoff maneuver...")
    vehicle_pointers = []
    for drone_name in vehicles:
        vehicle_pointers.append(client.takeoffAsync(vehicle_name=drone_name))
    # All of this happens asynchronously. Hold the program until the last vehicle
    # finishes taking off.
    return vehicle_pointers[-1]


def generate_spawn_zones():

    _zones = {}
    _offset = int(g_config["spawn_empty_offset"])
    for i in range(0,num_uavs):
        _zones["zone"+str(i)]={}
        if(i ==0):
            _zones["zone"+str(i)]["x_low"] = 0
            _zones["zone"+str(i)]["x_high"] = 10
            
            _zones["zone"+str(i)]["y_low"] = 0
            _zones["zone"+str(i)]["y_high"] = 20
        elif(i==1):


            _zones["zone"+str(i)]["x_low"] = 30
            _zones["zone"+str(i)]["x_high"] = 40
            
            _zones["zone"+str(i)]["y_low"] = 0
            _zones["zone"+str(i)]["y_high"] = 20

        elif(i==2):

            _zones["zone"+str(i)]["x_low"] = 0
            _zones["zone"+str(i)]["x_high"] = 10
            
            _zones["zone"+str(i)]["y_low"] = 40
            _zones["zone"+str(i)]["y_high"] = 60



        """
        TODO Generate sequentially blocks
        """
        # _zones["zone"+str(i)]["x0"] = int(g_config["spawn_first_col_x0"])  if (i<2) \
        #     else int(g_config["spawn_second_col_x0"])
        # _zones["zone"+str(i)]["y0"] = int(g_config["spawn_first_col_y0"]) + \
        #     i * (int(g_config["spawn_zoneangle_height"])+_offset) if (i<2) \
        #     else int(g_config["spawn_second_col_y0"])
        
        # _zones["zone"+str(i)]["x1"] = _zones["zone"+str(i)]["x0"] + int(g_config["spawn_zoneangle_base"])
        # _zones["zone"+str(i)]["y1"] = _zones["zone"+str(i)]["y0"] + int(g_config["spawn_zoneangle_height"])
            
    print(_zones)

    
    return _zones


def place_drones(client,vehicles,zones):
    """
        Place each drone in a different random location inside a rectangle 
        lateral to the street to be patroled. More in the readme. 
    """
    for i,uav_name in enumerate(vehicles):
        x_random_position = np.random.uniform(low=zones["zone"+str(i)]["x_low"], high=zones["zone"+str(i)]["x_high"])
        y_random_position = np.random.uniform(low=zones["zone"+str(i)]["y_low"], high=zones["zone"+str(i)]["y_high"])
        z = 10 # TODO lookout to fixed z 
        place_one_drone(client,uav_name,(x_random_position,y_random_position,z))


def place_one_drone(client,vehicle,position):
        pose = airsim.Pose(airsim.Vector3r(*position), ORIENTATION)
        client.simSetVehiclePose(pose, vehicle_name=vehicle,ignore_collison=True)
