#=========================================================================#
# # 3 (TODO more) drones start from set of random positions and comunicate #
#                # to other nearby using "proximity sensor"                #
#=========================================================================#


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
    utils.place_one_drone(client,vehicles[0],(0,0,10))
    utils.place_one_drone(client,vehicles[1],(5,5,10))
    utils.place_one_drone(client,vehicles[2],(10,10,10))
    # takeoff_all(client,vehicles)

    zones = utils.generate_spawn_zones()
    utils.place_drones(client,vehicles,zones)



    time.sleep(10)

main()