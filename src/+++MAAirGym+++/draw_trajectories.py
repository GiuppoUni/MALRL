
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
for i in range(0,10):
    for vn in vehicles_names:
        pos,ts = check_pos(vn)
        ts -= starting_ts[vn] 
        trajectories[vn].append( [ts,pos])

    if(i % 10000 ==0):
        save_obj(trajectories,"trajectory_",file_timestamp)

        
    time.sleep(timestep)

save_obj(trajectories,"trajectory_",file_timestamp)
load_obj("trajectory_",file_timestamp)