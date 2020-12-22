import argparse
import datetime
from gym_airsim.envs.collectEnv import CollectEnv


import gym_airsim
from gym_airsim.envs import AirSimEnv
import utils
import time
import numpy as np
import utils 
import os 
import pandas

TRAJECTORIES_FOLDER ="qtrajectories/csv/"

traj_files_list = os.listdir(TRAJECTORIES_FOLDER)
trajs = []

for tf in traj_files_list:
        df = pandas.read_csv(TRAJECTORIES_FOLDER+tf,delimiter=",",index_col="index")
        # print(df)
        traj = df.to_numpy()
        trajs.append(traj)

print(trajs)

trees = utils.build_trees(trajs)
trajs3d, zs = utils.avoid_collision(trajs,trees,300,0,20,3)
for idx,traj in enumerate(trajs3d):
        traj = np.array(traj)
        df = pandas.DataFrame({'x_pos': traj[:, 0], 'y_pos': traj[:, 1],
        'z_pos': traj[:, 2]})
        df.index.name = "index"
        df.to_csv("trajectories_3d/csv/"+traj_files_list[idx])

        