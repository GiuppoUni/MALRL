import numpy as np
import os

from numpy.core.shape_base import block

os.chdir(os.path.normpath("C:\\Users\\gioca\\Desktop\\Repos\\AirSim-PredictiveManteinance\\src\\+++MAAirGym+++\\gym_maze\\envs\\"))

a = np.load("maze_samples/maze2d_001.npy")


print("BEFORE")
print(a)

def actions_to_value(actions = None):
    value = 0x0
    if not actions:
        return value 

    if "N" in actions:
        value |= 0x1
    if "E" in actions:
        value |= 0x2
    if "S" in actions:
        value |= 0x4
    if "W" in actions:
        value |= 0x8
    
    return value
        
def add_action(value,action):
    if "N" in action:
        value |= 0x1
    if "E" in action:
        value |= 0x2
    if "S" in action:
        value |= 0x4
    if "W" in action:
        value |= 0x8
    return value

def remove_action(value,action):
    if "N" in action:
        value &= ~ 0x1
    if "E" in action:
        value &= ~ 0x2
    if "S" in action:
        value &= ~ 0x4
    if "W" in action:
        value &= ~ 0x8
    return value


for r in range(5):
    for c in range(5):
        a[r,c] = 15 

a[1,1] = 0
a[3,3] = 0
a[1,3] = 0
a[3,1] = 0



av = actions_to_value
a = [
    [av("ES"),av("WE"),av("WES"),av("WE"),av("WS")],
    [av("NS"),av(""),    av("NS"),  av(""), av("NS")],
    [av("NES"),av("WE"),av("NEWS"),av("WE"),av("NWS")],
    [av("NS"),av(""),av("NS"),av(""),av("NS")],
    [av("NE"),av("WE"),av("WEN"),av("WE"),av("WN")],
]



a = np.array(a)
print("AFTER")
print(a)

print(actions_to_value("NEWS"),actions_to_value(""))
np.save("maze_samples/maze2d_001.npy",a)


NROWS = 21
NCOLS = 21
OBS_BLOCKS = 3
# Each block can be considered 20m x 20m 

def cell_value(r,c,obs_blocks=1):
    if(obs_blocks ==1):
        if r %2 ==0 or c %2 ==0:
            return 15
        elif r != 0 and c !=0 and r != NROWS -1 and c != NCOLS -1 :
            return 0
        else:
            return 15
    else:
        if(r % ( obs_blocks+1)==0 or c % (obs_blocks +1 ) == 0 ):
            return 15
        else:
            return 0

aa = [ [cell_value(r,c,obs_blocks=OBS_BLOCKS) for c in range(NCOLS) ] for r in range(NROWS)]
aa = np.array(aa)

print('aa BEF: \n', aa)  


for r in range(len(aa)):
    for c in range( len(aa[0] )):
        
        cell = aa[r,c]
        if r -1 < 0 or  aa[r-1,c] == 0:
            cell = remove_action( cell,"N") 
        
        if r+1 >= len(aa) or aa[r+1,c] == 0:
            cell = remove_action( cell,"S") 
        
        if c -1 < 0 or  aa[r,c-1] == 0:
            cell = remove_action( cell,"W") 

        if c+1 >= len(aa[0]) or aa[r,c+1] == 0:
            cell = remove_action( cell,"E") 
        aa[r,c] = cell
          

print('aa AFT: \n', aa)
np.save("maze_samples/maze2d_002.npy",aa)
