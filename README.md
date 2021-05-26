# MALRL
# **Repository in construction...**
[Multiple Abstraction Layers Reinforcement Learning (MALRL)](https://sites.google.com/view/malrl-framework/home-page) is a framework developed as final project for Master Thesis in Engineering in Computer Science, at Sapienza University of Rome, supervised by Prof. [Luca Iocchi](https://www.diag.uniroma1.it/users/luca_iocchi).

The scope of MALRL is to generate 3D trajectories realistic for a scenario of multiple UAVs flying from different locations inside a fixed portion of space, in a grid pattern. Horizontal and vertical separation is enforced to reduce collision risk. 

![Alt text](/relative/path/to/img.jpg?raw=true "Optional Title")

MALRL is composed of three main layers, each one executable indipentently:
1) Grid Layer (GL) -> layer1.py
   - 2D QLearning in a grid maze 
   - Maze based (scale 1:40) on abstraction from real world scenario
   - Horizontal separation enforced for UAV groups/total 
   - Vertical separation to generate 3D data 

2) Simplified 3D Layer (S3DL) -> layer2.py
   - 3D simulated data using AirSim
   - UAV following trajectories from previous layer
   - Simulation executed iteratively (one UAV/traj at the time)
   - Kinematic data in output (velocity, acc, orientation)
   - Real time collision detection (UAV-building, UAV-surface)

3) Georeferenced 3D Layer (G3DL) -> layer3.py
   - 3D simulated data using AirSim
   - Similar structure of previous layer
   - Plot of trajectory to be followed
   - Introduction of GPS (simulated) data exploiting OpenStreet map georeferenciation (case of study: Barcelona,Spain)

The output of each layer is used as input for the following.


The second and third layers (layer2.py, layer3.py) are based on **custom AirSim environments** free [downloadable](#Environments) from the links at bottom page. 



# Install
1. Clone the repository
2. Install requirements
    
Layer 2 and Layer 3 require Unreal Engine 4 (UE4) started and AirSim plugin installed.


## UE4 Install (Windows)
Use of Unreal Engine on Windows is recommended (editing and testing has been done on Windows 10).
 
1. [Download](https://www.unrealengine.com/download) the Epic Games Launcher. 
2. [Register Epic Games account](https://www.epicgames.com/id/register) (while the Unreal Engine is open source and free to download, registration is still required).
3. Run the Epic Games Launcher:
   - open the Unreal Engine tab on the left pane. 
   - Click on the Install button on the top right, which should show the option to download Unreal Engine >= 4.25. 
   - Choose the install location to suit your needs, as shown in the images below. If you have multiple versions of Unreal installed then make sure the version you are using is set to current by clicking down arrow next to the Launch button for the version.

Note: AirSim also works with UE >= 4.24, however, it's **recommended 4.25**. Note: If you have UE 4.16 or older projects, please see the upgrade guide to upgrade your projects

## UE4 Install (Ubuntu)
To install AirSim on Ubuntu follow the instructions at: 
https://www.unrealengine.com/en-US/ue4-on-github

In a nutshell you will need to:
1) Clone UA4 repo
2) Build
3) Run UA4 (Setup.sh)

## Environments download
Environments are available as open source, editable projects at the following links:
- [Layer2](https://drive.google.com/drive/folders/1754p_qQnguh83av8yTdc0yW9a_EnuZmJ?usp=sharing)
- [Layer3](https://drive.google.com/drive/folders/17a_t73zrxV6WmlYNmlWtj8eyTNHARFcs?usp=sharing)

# Usage

# Demo files 
The following is a list of demo python scripts and their explanation, taken from [AirSim](https://github.com/microsoft/AirSim/tree/master/PythonClient), ready to be run after installation to test airsim possibilities.

Multirotor demo files:
- arm.py -> No visual effect (simple arming drone)
- box.py -> Flying on a small square box path using moveByVelocityZ
- clock_speed.py -> See how to change clockspeed in settings
- disarm.py -> No visual effect (simple disarming drone)
- DQNdrone.py -> HOW TO DQN
- drone_lidar.py -> Get Lidar (simulated) data from a drone
- drone_stress_test.py -> Move order and reset reapeatedly in small time interval (stress test)
- gimbal.py -> Move camera like with a gimbal
- hello_drone.py -> Simple operations
- high_res_camera.py -> Use scene image request + settings to get high res images
- kinect_publisher.py -> Uses kinect in some way (more for ROS)
- land.py -> Launch land movement and then set land state  
- manual_mode_demo.py -> Get commands by radio-controller
- multi_agent_drone.py -> Snap photos by multi uav
- navigate.py -> Use open cv to show new images from AirSim 
- opencv_show.py -> test computer vision mode ( it requires settings mod in order to work) 
- orbit.py -> Make the drone fly in a circle
- params.txt -> PX4 parameters
- path.py -> This script is designed to fly on the streets of the Neighborhood environment follows a opath specifyed by 3d vectors
- pause_continue_drone.py -> Pause and start
- point_cloud.py -> To create point cloud using opencv
- reset_test_drone.py -> Fly, reset, fly
- set_trace_line.py -> Draw line and fly
- set_wind.py -> Set wind values
- state.py -> Check ready state, print ready message
- survey.py -> Fly over a box to survey it
- takeoff.py -> Useless
- teleport.py -> Teleport drones in another position


Environment demo files:
- set_wind.py -> Set wind with directions
- unreal_console_commands.py -> Run cmd onto unreal cmd
- weather.py -> Change weather conditions


# PC Requirements
To run smoothly Unreal Engine the use of a dedicated GPU is strongly recommended. 
For developing and testing a desktop computer with the following specs has been used:

- Intel(R) Core(TM) i5-6400 CPU @ 2.70GHz   2.71 GHz
- Nvidia GTX 970 GAMING 4 GB 
- RAM: 16 GB DDR4
- Samsung SSD 860 EVO 500GB

## Notes
Time to generate a trajectory depends on clockspeed value.

## Conventions

- All drone names **NEED** to be in this format:
"Drone<i>" where i is a number.
E.g.: 'Drone0', 'Drone1', 'Drone2', 'Drone3', ...


## Sources
- https://github.com/Kjell-K/AirGym
- https://github.com/koulanurag/ma-gym
- gym-airsim

Take a look at my watch/star  list




-------------------
------------------. 


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

