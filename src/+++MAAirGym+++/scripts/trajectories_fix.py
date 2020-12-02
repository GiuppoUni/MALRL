import os
import sys
sys.path.append("..")
import utils
import argparse
import numpy as np
import json 

TRAJECTORIES_FOLDER = "./trajectories/"

AIRSIM_SETTINGS_FOLDER = 'C:/Users/gioca/OneDrive/Documents/Airsim/'


if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description='Traj fixer')
    parser.add_argument('--folder', type=str, required = True,
        help='folder of trajectory (default: %(default)s)')
    # parser.add_argument('--debug', type=bool, default=False,
    # help='Log into file (default: %(default)s)')
    # parser.add_argument('--track-trajectories', type=bool, default=True,
    #     help='Track trajectories into file (default: %(default)s)')

    args = parser.parse_args()

    trajs = os.path.join(TRAJECTORIES_FOLDER,args.folder)
    traj_list = os.listdir(trajs)

    
    with open(AIRSIM_SETTINGS_FOLDER + 'settings.json', 'r') as jsonFile:
        g_airsim_settings = json.load(jsonFile)

    vsetk=[v for v in g_airsim_settings["Vehicles"]][0]
    vset=g_airsim_settings["Vehicles"][vsetk]

    for f in traj_list:
        fpath = os.path.join(trajs,f)
        traj = np.load(fpath)
        print(traj[:10],"\n")
        o_x = vset["X"]
        o_y = vset["Y"]
        o_z = vset["Z"]
        
        # new_traj = [ [pos[0] + o_x, pos[1] + o_y, pos[2] + o_z] 
        new_traj = [ [pos[0] , pos[1] , pos[2] ] 
        
        for pos in traj]
        new_fpath = os.path.join(TRAJECTORIES_FOLDER,"fixed_traj")
        if not os.path.exists(new_fpath):
            os.makedirs(new_fpath)
        new_fpath = os.path.join(new_fpath,f)
        np.save(file=new_fpath,arr=np.asarray(new_traj))
        print(new_traj[:10],"\n")