import threading
import malrl_utils
import time 
import os 
import trajs_utils
from airsimgeo.newMyAirSimClient import NewMyAirSimClient

configYml = malrl_utils.read_yaml("inputData/config.yaml")
c_paths = configYml["paths"]


class positionLoggerThread(threading.Thread):

   def __init__(self,vehicle_name,idx,traj_filename,gpsOn,layer):
      print("IDX",idx)
      threading.Thread.__init__(self,daemon=True )
            
      self.vehicle_name = vehicle_name
      self.traj_filename = traj_filename
      self.client = NewMyAirSimClient(False,False,False,0,False,False)
      self.client.enableApiControl(True,vehicle_name=vehicle_name)
      self.gpsOn = gpsOn
      self.layer=layer

      if(layer==2):
         new_folder=os.path.join(c_paths["LAYER2_OUTPUT_FOLDER"],malrl_utils.EXPERIMENT_DATE)
         self.filename =  os.path.join(new_folder  , "3dl2traj"+str(idx)+".csv")
      elif (layer==3):
         new_folder=os.path.join(c_paths["LAYER3_OUTPUT_FOLDER"],malrl_utils.EXPERIMENT_DATE)
         self.filename =  os.path.join(new_folder  , "3dl3traj"+str(idx)+".csv")
      else:
         raise Exception("Specified layer is not correct")

 
      self._stop_event = threading.Event()

   def stop(self):
      self._stop_event.set()

   def is_stopped(self):
        return self._stop_event.is_set()

   def run(self): 
      print("[THREAD "+threading.current_thread().getName()+"] Started \n Position logging for trajectory: ", self.traj_filename)
      with open(self.filename,"w") as fout:
         fout.write("index,x_pos,y_pos,z_pos,w_or,x_or,y_or,"+
         "z_or,x_lin_vel,y_lin_vel,z_lin_vel,x_ang_vel,y_ang_vel,z_ang_vel,"+
         "x_lin_acc,y_lin_acc,z_lin_acc,x_ang_acc,y_ang_acc,z_ang_acc"+"\n") #HEADER
      with open(self.filename,"a") as fout:
         time_counter = 0.
         while(True):
            ks = self.client.simGetGroundTruthKinematics("Drone0")
            position = malrl_utils.vec_to_str(ks.position)
            if(self.layer==3):
               pos = malrl_utils.position_to_list( ks.position ) 
               gps_pos = self.client.nedToGps(*trajs_utils.rotate_point_2d(-malrl_utils.O_THETA,pos[0],pos[1]),pos[2] )
            else:
               gps_pos = None
            orientation = malrl_utils.quat_to_str(ks.orientation,False)
            linear_velocity = malrl_utils.vec_to_str(ks.linear_velocity)
            angular_velocity = malrl_utils.vec_to_str(ks.angular_velocity)
            linear_acceleration = malrl_utils.vec_to_str(ks.linear_acceleration)
            angular_acceleration = malrl_utils.vec_to_str(ks.angular_acceleration)
                        
            line = ",".join([     str(time_counter),       position  ,
            orientation ,            linear_velocity ,            angular_velocity ,
            linear_acceleration , angular_acceleration ,])
            if(self.layer==3):
               line += ","+ ",".join([str(x) for x in gps_pos])
               
            print("[THREAD "+threading.current_thread().getName()+"]",position)
            fout.write(line+"\n")
            time.sleep(configYml["layer2"]["AIRSIM_SAMPLING_INTERVAL"])
            time_counter += configYml["layer2"]["AIRSIM_SAMPLING_INTERVAL"]
            if(self.is_stopped()):
               print("[THREAD "+threading.current_thread().getName()+"] STOP for: ", self.traj_filename)
               break
                    