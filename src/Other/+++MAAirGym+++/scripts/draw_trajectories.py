
from time import localtime
import airsim
import time
import threading
import signal

from airsim.types import Vector3r
import utils
import time
import numpy as np
import datetime
import os



if __name__ == "__main__":

    sample = [Vector3r(0,0,-10),Vector3r(0,12,-10),Vector3r(12,12,-10),Vector3r(12,0,-10),Vector3r(0,0,-10)]

    test_tra_file = os.listdir(utils.TRAJECTORIES_FOLDER)[-1]
    traj = utils.pkl_load_obj(filename=test_tra_file)
    print("Trajectory:",traj)

    for drone in traj:
        traj[drone]


    client = airsim.MultirotorClient()

    # connect to the AirSim simulator
    print("Connected to:",client.confirmConnection() )


    vehicles_names = [ vn for vn in utils.g_airsim_settings["Vehicles"] ]

    for d in traj:
        client.simPlotLineStrip(traj[d],is_persistent= True)


    time.sleep(120)