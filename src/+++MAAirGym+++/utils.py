import json
from os import O_EXCL
import os
from airsim.types import Vector3r
from dotmap import DotMap
from pyproj import Proj

from configparser import ConfigParser
import logging
import datetime
import numpy as np
import pickle

# CHANGE FOR FOLDER CONTAINING AIRSIM SETTINGS
AIRSIM_SETTINGS_FOLDER = 'C:/Users/gioca/OneDrive/Documents/Airsim/'
CONFIGS_FOLDER = "./configs/"
LOG_FOLDER = "./logs/"
TRAJECTORIES_FOLDER = "./trajectories/"

with open(AIRSIM_SETTINGS_FOLDER + 'settings.json', 'r') as jsonFile:
    g_airsim_settings = json.load(jsonFile)

g_vehicles = g_airsim_settings["Vehicles"]
g_config = ConfigParser()
g_config.read(CONFIGS_FOLDER + 'config.ini')

map_filename = "overlayMap.png"

SRID = "EPSG:5555"

ORIGIN = (
    12.457480,
    41.902243,
    0 )
DEST = (
    12.466382,
    41.902491,
    80) 





# GPS init position of uavs
init_gps = [
    (
        12.45727300643921,
        41.90169011784915,
        0
    ),

    (
        12.457227408885958,
        41.90276414312537,
        0
    ),
    (
        12.45587021112442,
        41.90220118045657,
        0
        ),
]

red_color = [1.0,0.0,0.0]
green_color = [0.0,1.0,0.0]
blue_color = [0.0,0.0,1.0]


def initiate_logger():
    logging.basicConfig(filename=LOG_FOLDER+"log"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)


    logger = logging.getLogger('multiAirGym')
    logger.debug('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )

    return logger


def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        true_array = ['True', 'TRUE', 'true', 'Yes', 'YES', 'yes']
        false_array = ['False', 'FALSE', 'false', 'No', 'NO', 'no']
        if input_string in true_array:
            input_string = True
        elif input_string in false_array:
            input_string = False

        return input_string


def read_cfg(config_filename='configs/map_config.cfg', verbose=False):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    cfg = DotMap()

    if verbose:
        hyphens = '-' * int((80 - len(config_filename))/2)
        print(hyphens + ' ' + config_filename + ' ' + hyphens)

    for section_name in parser.sections():
        if verbose:
            print('[' + section_name + ']')
        for name, value in parser.items(section_name):
            value = ConvertIfStringIsInt(value)
            cfg[name] = value
            spaces = ' ' * (30 - len(name))
            if verbose:
                print(name + ':' + spaces + str(cfg[name]))

    return cfg


env_cfg = read_cfg(config_filename = CONFIGS_FOLDER + 'map_config.cfg')

o_x = env_cfg["o_x"]
o_y = env_cfg["o_y"]
o_z = env_cfg["o_z"]

def projToAirSim( x, y, z):
    x_airsim = (x + o_x ) 
    y_airsim = (y - o_y) 
    z_airsim = (-z + o_z) 
    return (x_airsim, -y_airsim, z_airsim)

def lonlatToProj( lon, lat, z, inverse=False):
    proj_coords = Proj(init=SRID)(lon, lat, inverse=inverse)
    return proj_coords + (z,)

def lonlatToAirSim( lon, lat, z):
    return projToAirSim(*lonlatToProj(lon, lat, z)   )


def nedToProj( x, y, z):
    """
    Converts NED coordinates to the projected map coordinates
    Takes care of offset origin, inverted z, as well as inverted y axis
    """
    x_proj = x + o_x
    y_proj = -y + o_y
    z_proj = -z + o_z
    return (x_proj, y_proj, z_proj)

def nedToGps( x, y, z):
    return lonlatToProj(* nedToProj(x, y, z), inverse=True)

def dronePrint(idx,s):
    print("[Drone"+str(idx)+"]",s)

def addToDict(d: dict,k,v):
    if k not in d:
        d[k] = []
    d[k].append(v)


def pkl_save_obj(obj, name,file_timestamp ):
    with open(TRAJECTORIES_FOLDER + name + file_timestamp + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def pkl_load_obj(name=None,file_timestamp=None,filename=None):
    if filename:
        with open(TRAJECTORIES_FOLDER +filename, 'rb') as f:
            return pickle.load(f)
    elif name and file_timestamp:
        with open(TRAJECTORIES_FOLDER + name + file_timestamp+ '.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        raise Exception("Specify file name")


def numpy_save(arr,folder_timestamp,filename):
    file_path = TRAJECTORIES_FOLDER+"trajectories_"+folder_timestamp
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    data = np.asarray(arr)
    # save to npy file
    print("Saving",os.path.join(file_path, filename))
    np.save(os.path.join(file_path, filename) , data)


def position_to_list(position_vector) -> list:
    return [position_vector.x_val, position_vector.y_val, position_vector.z_val]

def list_to_position(l) -> Vector3r:
    if len(l) != 3:
        raise Exception("REQUIRED EXACTLY 3 elements")
    return Vector3r(l[0],l[1],l[2])


def set_offset_position(pos):
    _v = g_vehicles["Drone0"]
    _offset_x = _v["X"] 
    _offset_y = _v["Y"]
    _offset_z = _v["Z"]
    pos.x_val += _offset_x
    pos.y_val += _offset_y
    pos.z_val += _offset_z


def _colorize(idx): 

    if idx == 0:
        return green_color
    elif idx==1: 
        return blue_color
    else : 
        return blue_color



distance = lambda p1, p2: np.norm(p1-p2)
