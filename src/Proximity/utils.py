import airsim
import numpy as np
import json
import math

# WINDOWS ONLY
SETTINGS_PATH = 'C:/Users/gioca/OneDrive/Documents/Airsim/'

ORIENTATION = airsim.Quaternionr(0, 0, 0, 0)
from configparser import ConfigParser

with open(SETTINGS_PATH + 'settings.json', 'r') as jsonFile:
    settings = json.load(jsonFile)



config = ConfigParser()
config.read('config.ini')

# Global vars from loaded config
g_config =  config["uav_group_settings"]
num_uavs = int( g_config["num_uavs"] ) # TODO make it uniform with enumerate(vehicles)
com_range = int( g_config["communication_coverage_range"] )

# We want a matrix to track who can communicate with who!

# It should be a nxn matrix, with each drone tracking itself and the matrix looks like
#            drone_1 drone_2 ... drone n
# drone_1    true    false   ... true
# drone_2    false   true    ... true
# drone_n    false   false   ... true
communications_matrix = np.zeros(
    (num_uavs, num_uavs), dtype=bool)

# It should be a nxn matrix, with each drone tracking itself and the matrix looks like
#            drone_1 drone_2 ... drone n
# drone_1    true    false   ... true
# drone_2    false   true    ... true
# drone_n    false   false   ... true


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
    for i in range(0,num_uavs):
        _zones["zone"+str(i)]={}
        _z = _zones["zone"+str(i)]    
        _offset =  0
        if(i>1):
            _offset = int(g_config["spawn_empty_offset_y"]) + _zones["zone"+str(i-2)]["y_high"] 
        if(i%2==0):
            # Assign x,y bounds of first col rect  
            _z["x_low"] = int(g_config["spawn_first_col_x_low"])
            _z["x_high"] = _z["x_low"] + int(g_config["spawn_rectangle_base"])
            _z["y_low"] = int(g_config["spawn_first_col_y_low"]) + _offset           
            _z["y_high"] = _z["y_low"]  + int(g_config["spawn_rectangle_height"])
        else:
            # Assign x,y bounds of second col rect  
            _z["x_low"] = int(g_config["spawn_second_col_x_low"])
            _z["x_high"] = _z["x_low"] + int(g_config["spawn_rectangle_base"]) 
            _z["y_low"] = int(g_config["spawn_second_col_y_low"]) + _offset        
            _z["y_high"] = _z["y_low"]  + int(g_config["spawn_rectangle_height"])
     
    print(_zones)

    
    return _zones


def place_drones(client,vehicles,zones):
    """
        Place each drone in a different random location inside a rectangle 
        lateral to the street to be patroled. More in the readme. 
    """
    poses = []
    for i,uav_name in enumerate(vehicles):
        x_random_position = np.random.uniform(low=zones["zone"+str(i)]["x_low"], high=zones["zone"+str(i)]["x_high"])
        y_random_position = np.random.uniform(low=zones["zone"+str(i)]["y_low"], high=zones["zone"+str(i)]["y_high"])
        z = 10 # TODO lookout to fixed z 
        pose = (x_random_position,y_random_position,z)
        poses.append( [uav_name,pose] )
        place_one_drone(client,uav_name, pose)
    return poses

def place_one_drone(client,vehicle,position):
        pose = airsim.Pose(airsim.Vector3r(*position), ORIENTATION)
        client.simSetVehiclePose(pose, vehicle_name=vehicle,ignore_collison=True)



def position_to_list(position_vector) -> list:
    return [position_vector.x_val, position_vector.y_val, position_vector.z_val]


def gps_position_to_list(gps_vector) -> list:
    return [gps_vector.latitude, gps_vector.longitude, gps_vector.altitude]



def get_all_drone_positions(client, vehicles) :
    _all_pos = dict()
    _all_pos["kin_pos_list"] = []
    _all_pos["gps_pos_list"] = []
    for i, drone_name in enumerate(vehicles):
        state_data = client.getMultirotorState(vehicle_name=drone_name)
        _all_pos["kin_pos_list"].append( position_to_list(state_data.kinematics_estimated.position) )
        _all_pos["gps_pos_list"].append( gps_position_to_list(state_data.gps_location))

    return _all_pos


def find_name(numb: int) -> str:
    name = "A"
    j = ord(name[0])
    j += numb
    return chr(j)


def haversine(lat1, lon1, lat2, lon2):
    # distance between latitudes 
    # and longitudes 
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
  
    # convert to radians 
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
  
    # apply formulae 
    a = (pow(math.sin(dLat / 2), 2) + 
         pow(math.sin(dLon / 2), 2) * 
             math.cos(lat1) * math.cos(lat2)); 
    rad = 6371 # kilometers
    c = 2 * math.asin(math.sqrt(a)) 
    return rad * c # kilometers

def build_vehicle_distance_matrix(drones_positions) :
    for i in range(0, num_uavs):
        for j in range(0, num_uavs):
            if i != j:
                first_drone = drones_positions["gps_pos_list"][i]
                second_drone = drones_positions["gps_pos_list"][j]
                # distance_matrix[i, j] = round(haversine(
                distance = round(haversine(
                    first_drone[0],
                    first_drone[1],
                    second_drone[0],
                    second_drone[1])*1000, 3)

            else:
                distance = 0
            communications_matrix[i,j] = (distance <= com_range )

    
