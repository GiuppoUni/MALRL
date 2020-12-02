
from time import localtime
import airsim
import time
import threading
import signal

from airsim.types import Vector3r
import time
import numpy as np
import datetime
import os
import sys
sys.path.append(os.path.abspath('../utils.py'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

if __name__ == "__main__":

    o_x,o_y,o_z = 3, -1, 0     
    
    sample_points_xy = [Vector3r(-10,-10,0),Vector3r(10,-10,0),Vector3r(10,10,0),Vector3r(-10,10,0),Vector3r(-10,-10,0)]
    sample_points_xz = [Vector3r(-10,0,-10),Vector3r(10,0,-10),Vector3r(10,0,10),Vector3r(-10,0,10),Vector3r(-10,0,-10)]
    sample_points_yz = [Vector3r(0,-10,-10),Vector3r(0,-10,10),Vector3r(0,10,10),Vector3r(0,10,-10),Vector3r(0,-10,-10)]

    # d_set = utils.g_airsim_settings["Vehicles"]["Drone0"]
    # o_x,o_y,o_z = d_set["X"],d_set["Y"],d_set["Z"]     

    fix_points = lambda points: [Vector3r(p.x_val+o_x, p.y_val+o_y, p.z_val+o_z)    for p in points] 
    sample_points_xy = fix_points(sample_points_xy)
    sample_points_xz = fix_points(sample_points_xz)
    sample_points_yz = fix_points(sample_points_yz)


    client = airsim.MultirotorClient()

    # connect to the AirSim simulator
    print("Connected to:",client.confirmConnection() )



    client.simPlotLineStrip(sample_points_xy,color_rgba=[1,0,0,1],thickness=10 ,is_persistent= True)
    client.simPlotLineStrip(sample_points_xz,color_rgba=[0,1,0,1],thickness=10 ,is_persistent= True)
    client.simPlotLineStrip(sample_points_yz,color_rgba=[0,0,1,1],thickness=10 ,is_persistent= True)


    time.sleep(120)