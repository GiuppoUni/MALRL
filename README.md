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

## Conventions

All drone names **NEED** to be in this format:
"Drone<i>" where i is a number.
E.g.: Drone1, Drone2, Drone3, ...


## Sources
https://github.com/Kjell-K/AirGym
https://github.com/koulanurag/ma-gym
gym-airsim
TODO look to my watch/star  list
