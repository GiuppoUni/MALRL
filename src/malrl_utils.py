import json
import math
import os
from typing import Tuple
from airsim.types import Vector3r
from dotmap import DotMap
from numpy.linalg import norm
from configparser import ConfigParser
import logging
import datetime
import numpy as np
import pickle
import time
#import winsound
import yaml



def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# CHANGE FOR FOLDER CONTAINING AIRSIM SETTINGS
configYml = read_yaml("./inputData/config.yaml")
c_paths = configYml["paths"]

AIRSIM_SETTINGS_FILE = configYml["paths"]["AIRSIM_SETTINGS_FILE"]
TRAJECTORIES_FOLDER = configYml["paths"]["Q_TRAJECTORIES"]

def get_experiment_date():
    return  str(datetime.datetime.now().strftime('-D-%d-%m-%Y-H-%H-%M-%S-') ) # To be used in log and prints

EXPERIMENT_DATE = get_experiment_date()

with open(AIRSIM_SETTINGS_FILE, 'r') as jsonFile:
    g_airsim_settings = json.load(jsonFile)

g_vehicles = g_airsim_settings["Vehicles"]

"""
       Assumes that the simulation environment (unreal) is in the coordinate system specified
        by the srid but offset by the origin specified.
        Arguments:
            srid {str} -- EPSG SRID string. Example "EPSG:3857"
            origin {list} -- [Longitude, Latitude, Height]
            kwargs -- Any keyword arguments forwared to AirSim
"""


map_filename = "overlayMap.png"

SRID = configYml["layer3"]["SRID"]
ORIGIN = (configYml["layer3"]["ORIGIN"][0]["x"],configYml["layer3"]["ORIGIN"][1]["y"],configYml["layer3"]["ORIGIN"][2]["z"])
O_THETA = configYml["layer3"]["O_THETA"]

NEW_TRAJ_PENALTY = 25 # negative reward for collision points of a new trajectory


red_color = [1.0,0.0,0.0]
green_color = [0.0,0.5,0.0]
blue_color = [0.0,0.0,1.0]
orange_color =[255/255, 102/255, 0]



#WINDOWS ONLY!
def play_audio_notification(n_beeps=3,frequency=2000,beep_duration=250):
    for _ in range(n_beeps):
        # try:
        #     winsound.Beep(frequency, beep_duration)
        # except:
        #     print("[WARNING] Bug in sound winsound")
        print("[BEEP]")
        time.sleep(0.1)



def initiate_logger():
    logging.basicConfig(filename=c_paths["LOG_FOLDER"]+"log"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
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



# def projToAirSim( x, y, z):
#     x_airsim = (x + ORIGIN_X ) 
#     y_airsim = (y - ORIGIN_Y) 
#     z_airsim = (-z + ORIGIN_Z) 
#     return (x_airsim, -y_airsim, z_airsim)

# def lonlatToProj( lon, lat, z, inverse=False):
#     proj_coords = Proj(init=SRID)(lon, lat, inverse=inverse)
#     return proj_coords + (z,)

# def lonlatToAirSim( lon, lat, z):
#     return projToAirSim(*lonlatToProj(lon, lat, z)   )


# def nedToProj( x, y, z):
#     """
#     Converts NED coordinates to the projected map coordinates
#     Takes care of offset origin, inverted z, as well as inverted y axis
#     """
#     x_proj = x + ORIGIN_X
#     y_proj = -y + ORIGIN_Y
#     z_proj = -z + ORIGIN_Z
#     return (x_proj, y_proj, z_proj)

# def nedToGps( x, y, z):
#     return lonlatToProj(* nedToProj(x, y, z), inverse=True)

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

def pos_arr_to_airsim_vec(l,wcell_in_meters=1,hcell_in_meters=1) -> Vector3r:
    # x = int(l[0]*wcell_in_meters)
    # y = int(l[1]*hcell_in_meters)
    x = l[0]*wcell_in_meters
    y = l[1]*hcell_in_meters
    if(len(l)>2):
        z = l[2]
    else:
        z = -50
    # if len(l) != 3:
    #     raise Exception("REQUIRED EXACTLY 3 elements")
    return Vector3r(x,y,z)

def vec_to_str(v,doNorm=False):
    if doNorm : return str( norm([v.x_val,v.y_val,v.z_val] ) )
    else: return ",".join( ( str(v.x_val),str(v.y_val),str(v.z_val)) )

def quat_to_str(q,doNorm=False): 
    if doNorm : return str( norm([q.w_val,q.x_val,q.y_val,q.z_val] ) )
    else: return ",".join( (str(q.w_val),str(q.x_val),str(q.y_val),str(q.z_val)) )
    

def l3_pos_arr_to_airsim_vec(l,wcell_in_meters=1,w_offset=0,hcell_in_meters=1,h_offset=0) -> Vector3r:
    # x = int(l[0]*wcell_in_meters)
    # y = int(l[1]*hcell_in_meters)
    x = l[0]*wcell_in_meters+w_offset
    y = l[1]*hcell_in_meters+h_offset
    z = l[2]

    if len(l) != 3:
        raise Exception("REQUIRED EXACTLY 3 elements")
    return Vector3r(x,y,z)

def getGoodCells(maze):
        
    rnd= np.where((maze.maze_cells!=0) & 
    ((maze.maze_cells == 7)|(maze.maze_cells == 15) |(maze.maze_cells  ==13 )) )
    
    rs= rnd[0]
    cs= rnd[1]
    arr = np.array( [[rs[i],cs[i]] for i in range(len(rs)) ] )
    
    # np.setdiff1d(arr, self.goals) 

    return arr

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

def xy_distance(point1, point2):
    if type(point1) == Vector3r:
        point1 = [point1.x_val,point1.y_val] 

    if type(point2) == Vector3r:
        point2 = [point2.x_val,point2.y_val] 
    
    return   np.linalg.norm(point1 - point2) 
    

def myInterpolate(arr, n_samples=10 ):
    res = []
    for i,p in enumerate(arr):
        if(i+1 >= len(arr)):
            break
        x1,y1,z1 = p[0], p[1], p[2]
        x2,y2,z2 = arr[i+1][0], arr[i+1][1], arr[i+1][2] 
    
        step_length = max(abs(x2-x1),abs(y2-y1)) / n_samples
        for i in range(n_samples):
            if(x2 > x1):
                # Moved on the right
                new_p = [x1 + i * step_length, y1,z1]
            elif (x1 > x2):
                # Moved left
                new_p = [x2 + i * step_length, y1,z1]
            elif (y2 > y1):
                # Moved left
                new_p = [x1, y1 + i * step_length,z1]
            elif (y1 > y2):
                # Moved left
                new_p = [x1, y2 + i * step_length,z1]
            else:
                raise Exception("Uncommmon points")
            res.append(new_p)

    return np.array(res)





def myInterpolate2D(trajs, n_samples=10,step_size=20 ):
    res = []
    for arr in trajs:
        res_t = []
        for i,p in enumerate(arr):
     
            if(i+1 >= len(arr)):
                break
            x1,y1 = p[0], p[1]
            x2,y2 = arr[i+1][0], arr[i+1][1] 
            # if(i==0):
            #     res_t.append([x1,y1])
            length = max(abs(x2-x1),abs(y2-y1))
            samples = math.floor(length/step_size)  
            print("|||")
            for i in range(samples):
                if(x2 > x1):
                    # Moved on the right
                    new_p = [x1 + i*step_size , y1]
                elif (x1 > x2):
                    # Moved left
                    new_p = [x1 - i*step_size  , y2]
                elif (y2 > y1):
                    # Moved left
                    new_p = [x1, y1 + i*step_size ]
                elif (y1 > y2):
                    # Moved left
                    new_p = [x2, y1 - i*step_size ]
                else:
                    raise Exception("Uncommmon points")
                print('new_p: ', new_p)
                res_t.append(new_p)
            if(length % step_size != 0):
                # last_step = length - step_size * samples
                print("last")
 
                if(x2 > x1):
                    # Moved on the right
                    new_p = [x2, y1]
                elif (x1 > x2):
                    # Moved left
                    new_p = [x1, y2]
                elif (y2 > y1):
                    # Moved left
                    new_p = [x1, y2]
                elif (y1 > y2):
                    # Moved left
                    new_p = [x2, y1]
                else:
                    raise Exception("Uncommmon points")
                print('new_pL: ', new_p)
                res_t.append(new_p)
            
            

        res.append(res_t)
    return res
            




def avoid_collision(trajectories,trees,min_height,max_height,
    sep_h,min_safe_points,radius=30,simpleMode=True):
    
    Tmax = max([len(traj) for traj in trajectories])
    drones = range(len(trajectories))
    points = {}
    zs=[[] for d in drones] 
    trajs_3d =[[] for d in drones] 
    colliding_trajs = dict()
    for d in drones:
        for t in range(len(trajectories[d])):
            point = tuple(trajectories[d][t])
            n_safe_points = 0
            res = 0
            for idx,_tree in enumerate(trees): 
                if(idx == d):
                    # E' quella attuale
                    continue
                res = _tree.query_radius( [point],r=radius,count_only = True )
                if res > 0:
                    print("Collisions with","Trajectory_"+str(idx))
                    print("\tcomputed from trajectory ",d,", point", point)
                    if(d not in colliding_trajs):
                        colliding_trajs[d]=[idx]
                    elif idx not in colliding_trajs[d]:
                        colliding_trajs[d].append(idx)
                  
                        
            if not simpleMode and res == 0:
                # TODO count and cooldown
                n_safe_points +=1
            if(n_safe_points >= min_safe_points):
                colliding_trajs[d] = []

            print("colliding_trajs",colliding_trajs)
            if(d not in colliding_trajs or colliding_trajs[d]==[]):
                new_z = max_height
            else:
                priorities = [d]+colliding_trajs[d]
                priorities.sort()
                offset = priorities.index(d)
                new_z = max_height - offset * sep_h 
                if new_z < min_height:
                    raise Exception("Out of height bounds")
            trajs_3d[d].append(list(point)+[new_z])
            zs[d].append(new_z)
    return trajs_3d,zs



def lonLatFromRotation(theta,phi,r_lon,r_lat):
    lon = math.atan2( math.sin(r_lon), math.tan(r_lat)* math.sin(theta) + math.cos(r_lon)* math.cos(theta)) - phi
    lat = math.asin( math.cos(theta) * math.sin(r_lat) - math.cos(r_lon) * math.sin(theta) * math.cos(r_lat) )
    return lon,lat

if __name__=="__main__":
    print("Test")
    lon,lat=2.1833298597595303, 41.409602234016496
    print(lonLatFromRotation(-5.41052,0,lon,lat))    

    res1,res2 = 2.179982, 41.403179 

