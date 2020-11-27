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



def check_pos():
    p = client.simGetGroundTruthKinematics().position
    print((p.x_val,p.y_val,p.z_val) )



class Job(threading.Thread):
 
    def __init__(self):
        threading.Thread.__init__(self)
 
        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()
 
        # ... Other thread setup code here ...
 
    def run(self):
        print('Thread #%s started' % self.ident)
 
        while not self.shutdown_flag.is_set():
            # ... Job code here ...
            check_pos()
            time.sleep(0.5)
 
        # ... Clean shutdown code here ...
        print('Thread #%s stopped' % self.ident)
 
 
class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass
 
 
def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit





 



# -------------------------------------   # ------------------------------------------

# Register the signal handlers
signal.signal(signal.SIGTERM, service_shutdown)
signal.signal(signal.SIGINT, service_shutdown)
client = airsim.MultirotorClient()

j1 = Job()
try:
    # connect to the AirSim simulator
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.hoverAsync().join()

    client.simSetTraceLine([1.0, 0.0, 0.0, 1.0], 5)
    vehicleControl = client.moveByVelocityAsync(1, 4, 0, 50)




    j1.start()

    # while(True):
    #     p = client.simGetGroundTruthKinematics().position
    #     print((p.x_val,p.y_val,p.z_val) )
    #     time.sleep(1)
    # while(True):
    #     print("Main execution flow")
    #     time.sleep(1)

    vehicleControl.join()

    client.armDisarm(False)
    client.takeoffAsync().join()
    client.enableApiControl(False)

except ServiceExit:
    # Terminate the running threads.
    # Set the shutdown flag on each thread to trigger a clean shutdown of each thread.
    j1.shutdown_flag.set()
    # Wait for the threads to close...
    print("[INFO] Waiting threads to close...")
    j1.join()

print('Exiting main program')