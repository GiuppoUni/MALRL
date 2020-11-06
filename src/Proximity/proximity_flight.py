#=========================================================================#
# # 3 (TODO more) drones start from set of random positions and comunicate #
#                # to other nearby using "proximity sensor"                #
#=========================================================================#


from utils import build_vehicle_distance_matrix
import utils
import setup_path
import airsim

import numpy as np
import pprint
import time
import copy
import traceback
import boto3
import sys
import json

 
    
def main():
    
    vehicles = [v for v in utils.settings["Vehicles"] ]
    print(vehicles)

    client = airsim.MultirotorClient()
    # client = None
    print(f"Client created: {client}")
    client.confirmConnection()
    print('Connection Confirmed')
    utils.enable_control_all(client,vehicles)
    print('UAVs Enabled')
    
    # takeoff_all(client,vehicles)

    zones = utils.generate_spawn_zones()
    places = utils.place_drones(client,vehicles,zones)
    print("UAVs positioned at:", places)
    
    last_vehicle_pointer = utils.take_off_all(client, vehicles)
    # We wait until the last drone is off the ground
    last_vehicle_pointer.join()

    # We mimic the memory bank of a drone, tracking the relative positions.
    # It should be a n-length vector, with each drone tracking itself and the matrix looks like

    #                                 drone_1 drone_2 ... drone n
    # all_positions["kin_pos_list"] = [x,y,z] [x,y,z] ... [x,y,z]
    # all_positions["gps_pos_list"] = [x,y,z] [x,y,z] ... [x,y,z]

    for _ in range(20):
        all_positions = utils.get_all_drone_positions(client,vehicles)
        build_vehicle_distance_matrix(all_positions)
        print(all_positions["gps_pos_list"])
        print(utils.communications_matrix)

        time.sleep(4)

main()