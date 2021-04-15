# MALRL Framework
Multiple Abstraction Layers Reinforcement Learning (MALRL) is a Python based framework used to generate realistic 3D UAV trajectories, enforcing horizontal and vertical separation between UAVs flying in grid pattern. It is composed of three main layers, each one executable indipentently. The second and third layers (layer2.py, layer3.py) are based on custom AirSim environments free downloadable from the links at bottom page. 

## Install
Clone the repository

## Usage
**To launch layer1:**
```bash
python layer1.py
```
optional arguments:
``` bash
  -h, --help            show this help message and exit
  --nepisodes NEPISODES
                        episodes (default: 100)
  --ngoals NGOALS       n goals to collect (default: 1)
  --seed SEED           seed value (default: None)
  --ntrajs NTRAJS       num trajectories value (default: None)
  --nbuffer NBUFFER     size of buffer for past trajectories (default: 3)
  --nagents NAGENTS     num agents (supported 1 )(default: 1)
  --nsteps NSTEPS       enforce n-steps qlearning if 0 is standard qlearning
                        (default: 0)
  --debug               Log debug in file (default: False)
  --render-train        render maze while training/random (default: False)
  --render-test         render maze while testing (default: False)
  --quiet               render maze while testing (default: False)
  --random-mode         Agent takes random actions (default: False)
  --skip-train          Just assign altitude to 2D trajecories in folder
                        (default: False)
  --show-plot           Show generated trajectories each time (default: False)
  -v                    verbose (default: False)
  --randomInitGoal      Random init and goal per episode (default: False)
  --random-pos          Drones start from random positions exctrateced from
                        pool of 10 (default: False)
  --slow                Slow down training to observe behaviour (default:
                        False)
  --plot3d              Render 3d plots(default: False)
  --n-random-init N_RANDOM_INIT
                        n sample pool for random init (default: 5)
  --log-reward          log reward file in out (default: False)
  --load-qtable LOAD_QTABLE
                        qtable file (default: None)
  --load-maze LOAD_MAZE
                        maze file (default: None)
  --load-goals LOAD_GOALS
                        maze file (default: None)
  --load-start-pos LOAD_INIT_POS
                        maze file (default: None)
```
</br>

**To launch layer2:**
```bash
python layer2.py
```
</br>

**To launch layer3:**
```bash
python layer3.py
```
## Files
inputData : folder containing files to set (settings,initial positions for UAVs, final positions for UAVs)
envs\my_maze_generator.py : let you generate custom mazes
layer1.py, layer2.py, layer3.py  : Layers to be executed 
trajs_utils.py : contains useful API functions called in layer modules
utils.py : contains helper functions
gym_maze\envs\maze_env.py : Class of the custom maze environment
eurocontrolConverter.py : Python script to convert 3d trajectories into format usable for Bubbles collision measurements


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

