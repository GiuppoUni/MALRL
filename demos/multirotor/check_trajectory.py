# Please add "EnableTrace": true to your setting.json as shown below

# {
#   "SettingsVersion": 1.2,
#   "SimMode": "Multirotor",
#   "Vehicles": {
#       "Drone": {
#           "VehicleType": "SimpleFlight",
#           "EnableTrace": true
#         }
#     }
# }

import setup_path
import airsim
import time
import threading
import signal

timestep = 0.1

def check_pos():
    p = client.simGetGroundTruthKinematics().position
    print((p.x_val,p.y_val,p.z_val) )




client = airsim.MultirotorClient()

# connect to the AirSim simulator
print(client.confirmConnection() )

while(True):
    check_pos()
    time.sleep(timestep)
# while(True):
#     print("Main execution flow")
#     time.sleep(1)

