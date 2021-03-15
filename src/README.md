# MALRL Framework
Multiple Abstraction Layers Reinforcement Learning (MALRL) is a Python based framework used to generate realistic 3D UAV trajectories, enforcing horizontal and vertical separation between UAVs flying in grid pattern. It is composed of three main layers, each one executable indipentently. The second and third layers (layer2.py, layer3.py) are based on custom AirSim environments free downloadable from the links at bottom page. 

## Install
Clone the repository

## Usage
To launch layer1:
python layer1.py

## Files
envs\my_maze_generator.py : let you generate custom mazes
layer1.py, layer2.py, layer3.py  : can be executed 
trajs_utils.py : contains useful API functions called in layer modules
gym_maze\envs\maze_env.py : Class of the custom maze environment

## Airsim instructions
Vehicle name used was "Drone0" so it is not guaranteed to function any other name.
It is strongly suggested to set it as so in the settings file of AirSim.
For Layer 3 you need to set an empty actor called: "CELL00", it will be used to pose the vehicle in that position


## Windowed env launch (Windows 10 AirSim)
To run Airsim windowed:
$ Blocks.exe -WINDOWED -ResX=640 -ResY=480

## Useful shortcuts (AirSim)
F8 to toggle Unreal free mode (very useful to explore the environment while simulating e.g. looking from above) 
' per debug
F2 to toggle rendering

## Conventions
<g_*> are global variables (present in utils)
LIM<letter><index> 
   are used as marker per bounds of  zone <letter> 
e.g. a quadratic shape zone B has four vertices 1,...4:
   LIMB1 [ 21.98999977 -69.79999542]
   LIMB2 [222.08999634 -69.69999695]
   LIMB3 [220.78999329 120.5       ]
   LIMB4 [ 22.09000015 120.79999542]


## Download links
Download link for layer2.py environment (simplified version of a grid plan city ):
#TODO
Download link for layer2.py environment (realistic 3D based on GPS open data):
#TODO

