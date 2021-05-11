# **Repository in construction...**
# MALRL
Multiple Abstraction Layers Reinforcement Learning (MALRL) is a framework developed as a Master Thesis project in Engineering in Computer Science at Sapienza University of Rome, supervised by Prof. Luca Iocchi.



# Demo files 
The following are demo files, from [Airsim](https://github.com/microsoft/AirSim/tree/master/PythonClient) ready to be run after installation to test airsim possibilities.
Multirotor demo files:
- arm.py -> useless
- box.py -> Flying a small square box using moveByVelocityZ
- clock_speed.py -> See how to change clockspeed in settings
- disarm.py -> useless
- DQNdrone.py -> HOW TO DQN
- drone_lidar.py -> get Lidar data from a drone
- drone_stress_test.py -> drone while movement
- gimbal.py -> move camera like with a gimbal
- hello_drone.py -> simple operations
- high_res_camera.py -> use scene image request + settings to get high res images
- kinect_publisher.py -> ??? uses kinect in some way
- land.py -> launch land movement and then set land state  
- manual_mode_demo.py -> get commands by rc
- multi_agent_drone.py -> snap photos by multi uav
- navigate.py -> ??? use open cv to show new images from AirSim 
- opencv_show.py -> strano, va modificato settings per computer vision mode 
- orbit.py -> Make the drone fly in a circle.
- params.txt /// ??? PX4 parameters
- path.py -> This script is designed to fly on the streets of the Neighborhood environment follows a opath specifyed by 3d vectors
- pause_continue_drone.py -> pause and start
- point_cloud.py -> to create point cloud using opencv
- reset_test_drone.py -> fly, reset, fly
- set_trace_line.py -> draw line and fly
- set_wind.py -> set wind values
- state.py -> check state readt  ir ready message
- survey.py -> fly over a box to survey it
- takeoff.py -> useless
- teleport.py -> to teleport drones in another position


Environment demo files:
- set_wind.py -> with directions
- unreal_console_commands.py -> to run cmd onto unreal cmd
- weather.py -> change weather conditions


## Notes
Time to generate a trajectory: 4-5 sec or 10 sec

$ ls -l
total 160
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:46 -D-07-05-2021-H-12-46-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:47 -D-07-05-2021-H-12-47-19-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:49 -D-07-05-2021-H-12-49-01-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:51 -D-07-05-2021-H-12-51-01-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:51 -D-07-05-2021-H-12-51-52-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 15:55 -D-07-05-2021-H-15-55-51-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 15:56 -D-07-05-2021-H-15-56-45-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 15:57 -D-07-05-2021-H-15-57-38-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:09 -D-07-05-2021-H-16-09-16-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:27 -D-07-05-2021-H-16-27-37-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:28 -D-07-05-2021-H-16-28-31-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:29 -D-07-05-2021-H-16-29-21-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:31 -D-07-05-2021-H-16-31-34-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:38 -D-07-05-2021-H-16-38-27-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:39 -D-07-05-2021-H-16-39-18-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:40 -D-07-05-2021-H-16-40-09-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:42 -D-07-05-2021-H-16-42-06-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:43 -D-07-05-2021-H-16-43-08-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 16:43 -D-07-05-2021-H-16-43-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:23 -D-07-05-2021-H-17-23-04-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:23 -D-07-05-2021-H-17-23-43-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:24 -D-07-05-2021-H-17-24-30-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:25 -D-07-05-2021-H-17-25-01-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:26 -D-07-05-2021-H-17-26-25-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:26 -D-07-05-2021-H-17-26-27-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:26 -D-07-05-2021-H-17-26-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:27 -D-07-05-2021-H-17-27-32-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:27 -D-07-05-2021-H-17-27-56-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:32 -D-07-05-2021-H-17-32-16-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:32 -D-07-05-2021-H-17-32-56-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:33 -D-07-05-2021-H-17-33-41-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:34 -D-07-05-2021-H-17-34-12-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:36 -D-07-05-2021-H-17-36-24-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:36 -D-07-05-2021-H-17-36-32-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:36 -D-07-05-2021-H-17-36-42-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:37 -D-07-05-2021-H-17-37-43-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:37 -D-07-05-2021-H-17-37-46-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:38 -D-07-05-2021-H-17-38-02-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:38 -D-07-05-2021-H-17-38-20-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:38 -D-07-05-2021-H-17-38-23-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:38 -D-07-05-2021-H-17-38-47-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:39 -D-07-05-2021-H-17-39-33-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:39 -D-07-05-2021-H-17-39-48-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:42 -D-07-05-2021-H-17-42-00-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:42 -D-07-05-2021-H-17-42-53-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:45 -D-07-05-2021-H-17-45-05-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:45 -D-07-05-2021-H-17-45-33-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:46 -D-07-05-2021-H-17-46-02-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:48 -D-07-05-2021-H-17-48-15-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:48 -D-07-05-2021-H-17-48-31-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:48 -D-07-05-2021-H-17-48-48-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:49 -D-07-05-2021-H-17-49-09-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:51 -D-07-05-2021-H-17-51-21-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:51 -D-07-05-2021-H-17-51-50-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:53 -D-07-05-2021-H-17-53-01-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:53 -D-07-05-2021-H-17-53-09-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:53 -D-07-05-2021-H-17-53-32-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:53 -D-07-05-2021-H-17-53-48-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:54 -D-07-05-2021-H-17-54-56-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:56 -D-07-05-2021-H-17-56-03-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:56 -D-07-05-2021-H-17-56-50-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:57 -D-07-05-2021-H-17-57-52-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 17:58 -D-07-05-2021-H-17-58-09-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:00 -D-07-05-2021-H-18-00-11-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:00 -D-07-05-2021-H-18-00-41-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:02 -D-07-05-2021-H-18-02-54-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:05 -D-07-05-2021-H-18-05-06-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:07 -D-07-05-2021-H-18-07-18-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:09 -D-07-05-2021-H-18-09-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:10 -D-07-05-2021-H-18-10-54-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:11 -D-07-05-2021-H-18-11-22-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:13 -D-07-05-2021-H-18-13-49-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:14 -D-07-05-2021-H-18-14-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:15 -D-07-05-2021-H-18-15-00-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:15 -D-07-05-2021-H-18-15-49-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:16 -D-07-05-2021-H-18-16-17-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:17 -D-07-05-2021-H-18-17-03-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:17 -D-07-05-2021-H-18-17-36-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:19 -D-07-05-2021-H-18-19-20-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:20 -D-07-05-2021-H-18-20-05-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:21 -D-07-05-2021-H-18-21-42-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:23 -D-07-05-2021-H-18-23-54-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:24 -D-07-05-2021-H-18-24-03-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:24 -D-07-05-2021-H-18-24-22-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:25 -D-07-05-2021-H-18-25-05-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:25 -D-07-05-2021-H-18-25-20-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:25 -D-07-05-2021-H-18-25-43-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:25 -D-07-05-2021-H-18-25-53-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:26 -D-07-05-2021-H-18-26-26-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:26 -D-07-05-2021-H-18-26-51-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:27 -D-07-05-2021-H-18-27-21-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:27 -D-07-05-2021-H-18-27-48-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:30 -D-07-05-2021-H-18-30-00-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:30 -D-07-05-2021-H-18-30-23-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:31 -D-07-05-2021-H-18-31-38-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:31 -D-07-05-2021-H-18-31-46-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:33 -D-07-05-2021-H-18-33-35-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:34 -D-07-05-2021-H-18-34-05-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:34 -D-07-05-2021-H-18-34-07-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:34 -D-07-05-2021-H-18-34-44-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:35 -D-07-05-2021-H-18-35-07-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:35 -D-07-05-2021-H-18-35-29-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:36 -D-07-05-2021-H-18-36-07-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:37 -D-07-05-2021-H-18-37-52-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:39 -D-07-05-2021-H-18-39-05-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:39 -D-07-05-2021-H-18-39-24-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:39 -D-07-05-2021-H-18-39-52-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:41 -D-07-05-2021-H-18-41-44-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:41 -D-07-05-2021-H-18-41-59-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:42 -D-07-05-2021-H-18-42-22-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:42 -D-07-05-2021-H-18-42-33-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:43 -D-07-05-2021-H-18-43-00-/
drwxr-xr-x 1 Giuppo 197609     0 May  7 18:43 -D-07-05-2021-H-18-43-23-/
drwxr-xr-x 1 Giuppo 197609     0 Mar 16 10:14 Other/
-rw-r--r-- 1 Giuppo 197609  4404 Apr 14 11:08 README.md
drwxr-xr-x 1 Giuppo 197609     0 May  7 12:21 __pycache__/
drwxr-xr-x 1 Giuppo 197609     0 May  6 10:25 airsim140/
drwxr-xr-x 1 Giuppo 197609     0 Mar 15 19:19 airsimgeo/
-rw-r--r-- 1 Giuppo 197609  8352 May  4 11:24 eurocontrolConverter.py
drwxr-xr-x 1 Giuppo 197609     0 May  7 11:43 generatedData/
drwxr-xr-x 1 Giuppo 197609     0 May  6 09:44 generatedFigs/
drwxr-xr-x 1 Giuppo 197609     0 Mar 15 19:19 gym_maze/
drwxr-xr-x 1 Giuppo 197609     0 Apr 26 09:45 inputData/
-rw-r--r-- 1 Giuppo 197609 26685 May  7 12:08 layer1.py
-rw-r--r-- 1 Giuppo 197609  7515 May  8 18:13 layer2.py
-rw-r--r-- 1 Giuppo 197609  4963 May  7 11:10 layer3.py
drwxr-xr-x 1 Giuppo 197609     0 Mar 12 18:22 logs/
-rw-r--r-- 1 Giuppo 197609  6333 Apr 18 12:33 np.csv
-rw-r--r-- 1 Giuppo 197609 34848 May  6 18:01 trajs_utils.py
-rw-r--r-- 1 Giuppo 197609 13625 May  7 12:08 utils.py


## Conventions

All drone names **NEED** to be in this format:
"Drone<i>" where i is a number.
E.g.: Drone1, Drone2, Drone3, ...


## Sources
https://github.com/Kjell-K/AirGym
https://github.com/koulanurag/ma-gym
gym-airsim
TODO look to my watch/star  list
