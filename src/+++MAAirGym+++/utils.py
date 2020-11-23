import json
from dotmap import DotMap
from pyproj import Proj



# CHANGE FOR FOLDER CONTAINING AIRSIM SETTINGS
AIRSIM_SETTINGS_FOLDER = 'C:/Users/gioca/OneDrive/Documents/Airsim/'
CONFIGS_FOLDER = "./configs/"

from configparser import ConfigParser

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



def projToAirSim( x, y, z,o_x,o_y,o_z):
    x_airsim = (x/ 100000 + o_x ) 
    y_airsim = (y/ 100000 + o_y) 
    z_airsim = (-z + o_z) 
    return (x_airsim, -y_airsim, z_airsim)

def lonlatToProj( lon, lat, z, inverse=False):
    proj_coords = Proj(init=SRID)(lon, lat, inverse=inverse)
    return proj_coords + (z,)

def lonlatToAirSim( lon, lat, z,o_x,o_y,o_z):
    return projToAirSim(*lonlatToProj(lon, lat, z) ,o_x,o_y,o_z )


def addToDict(d: dict,k,v):
    if k not in d:
        d[k] = []
    d[k].append(v)

def dronePrint(idx,s):
    print("[Drone"+str(idx)+"]",s)