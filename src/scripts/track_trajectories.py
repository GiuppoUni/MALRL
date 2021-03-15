from time import localtime
import airsim
import time
import threading
import signal
import utils
import time
import numpy as np
import datetime
import pickle
import signal 
import sys

# GLOBALS 
timestep = 0.1 # s
monitor_timeout =  20 * 60 # s
monitor_iterations = int(monitor_timeout // timestep)

doTimestamp = False

def check_pos(vName):
    p = client.simGetGroundTruthKinematics(vehicle_name = vName).position
    ts = time.time()
    print("[",vName,"]",(p.x_val,p.y_val,p.z_val) )
    return p,ts


def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting...')
    utils.pkl_save_obj(trajectories,"trajectory_",file_timestamp)
    sys.exit(0)






if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print('Press Ctrl+C TO STOP')

    client = airsim.MultirotorClient()

    # connect to the AirSim simulator
    print(client.confirmConnection() )


    vehicles_names = [ vn for vn in utils.g_airsim_settings["Vehicles"] ]
    file_timestamp =str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))
    trajectories = dict()
    starting_ts = dict()
    for vn in vehicles_names:
        trajectories[vn] = []
        starting_ts[vn] = time.time()

    # while(True):
    for i in range(0,monitor_iterations):
        for vn in vehicles_names:
            pos,ts = check_pos(vn)
            ts -= starting_ts[vn] 
            if doTimestamp:
                data = [ts,pos]
            else: 
                data = pos
            trajectories[vn].append(data )

        if(i % 10000 ==0):
            utils.pkl_save_obj(trajectories,"trajectory_",file_timestamp)
            
        time.sleep(timestep)



    utils.pkl_save_obj(trajectories,"trajectory_",file_timestamp)
    utils.pkl_load_obj("trajectory_",file_timestamp)


