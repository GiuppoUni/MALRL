import argparse
import datetime
import logging
import threading
import time
from airsim140.client import MultirotorClient
from airsim140.types import DrivetrainType, Pose, Vector3r, YawMode
from airsim140.utils import to_quaternion
import malrl_utils
import os
from airsimgeo.newMyAirSimClient import NewMyAirSimClient
import nest_asyncio
nest_asyncio.apply()
from positionLoggerThread import positionLoggerThread

configYml = malrl_utils.read_yaml("inputData/config.yaml")
c_paths = configYml["paths"]
c_settings = configYml["layer1"]["settings"]
c_settings2 = configYml["layer2"]
c_verSep= configYml["layer1"]["vertical_separation"]

NEW_FOLDER = os.path.join(c_paths["LAYER2_OUTPUT_FOLDER"],malrl_utils.EXPERIMENT_DATE)    

def main(trajectories_folder,velocity):

   os.makedirs(NEW_FOLDER)

   # # Create AirSim client
   # asClient = NewMyAirSimClient(trajColFlag=False,
   #          canDrawTrajectories=False,crabMode=True,thickness = 100,trajs2draw=[],traj2follow=[])

   # Read trajectories
   exp_folders = [os.path.join(trajectories_folder,d) for d in os.listdir(trajectories_folder)]
   latest_mod_folder = max(exp_folders , key=os.path.getmtime)

   print("Detected ",len(latest_mod_folder)," files inside: "+latest_mod_folder+".")
   for f in   sorted(os.listdir(latest_mod_folder),key=lambda x: int("".join( filter(str.isdigit, x)))) :
      asClient = NewMyAirSimClient(trajColFlag=False,
         canDrawTrajectories=False,crabMode=True,
         thickness = 100,trajs2draw=[],traj2follow=[],
         ip="127.0.0.1")
      asClient.disable_trace_lines()

      if( f[-4:]==".csv" ):
         print("Reading trajectory from:",f)

         with open(os.path.join(latest_mod_folder,f),"r") as fin:
            lr= fin.readlines()
            trajectory = []
            for idx,line in enumerate(lr):
               if(idx==0):
                  continue #Skip header
               values = line.split(",")
               x = float( values[1] )
               y = float( values[2] )
               z = float( values[3] )
               z = z if z < 0 else -z

               if(idx==1):
                  pose = Pose(Vector3r(x,y,z), to_quaternion(0, 0, 0) ) 
                  asClient.simSetVehiclePose(pose,True,"Drone0")
                  time.sleep(1)
                  print("UAV start pose set.")       
                  asClient.enable_trace_lines()
               else:
                  trajectory.append( malrl_utils.l3_pos_arr_to_airsim_vec([x,y,z],w_offset=c_settings2["W_OFFSET"],h_offset=c_settings2["H_OFFSET"]) )

            # trajs_utils.plot_xy([trajs_utils.vec2d_list_to_tuple_list(trajectory)],20,doScatter=True)


            pointer = asClient.moveOnPathAsync(
               trajectory,
                 velocity, 120,DrivetrainType.ForwardOnly , YawMode(False,0),
               lookahead=20,adaptive_lookahead=1,vehicle_name="Drone0")

            fidx = f.split(".csv")[0].split("traj")[1]
            if(not fidx.isdigit()):
               raise Exception("Error in format of filename:"+f+", (it should be ...traj<id>.csv)")

            positionThread = positionLoggerThread("Drone0",fidx,f,gpsOn=False,layer=2)
            positionThread.start()

            print("UAV following trajectory", os.path.join(latest_mod_folder,f),"...")
            pointer.join()
            positionThread.stop()
            positionThread.join()
            print("UAV completed its mission.")



                                       

def calibrate(vertices,scale=1):
   asClient = NewMyAirSimClient(trajColFlag=False,
         canDrawTrajectories=False,crabMode=True,
         thickness = 100,trajs2draw=[],traj2follow=[])
   for v in vertices:
      x,y,z = v[0]*scale,v[1]*scale,-50
      pose = Pose(malrl_utils.l3_pos_arr_to_airsim_vec((x,y,z),w_offset=c_settings2["W_OFFSET"],h_offset=c_settings2["H_OFFSET"]), 
      to_quaternion(0, 0, 0) ) 
      asClient.simSetVehiclePose(pose,True,"Drone0")     
      time.sleep(1.0)

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Layer 2')
   
   parser.add_argument("-i", type=str,required=False, default = c_paths["LAYER1_3D_OUTPUT_FOLDER"],
                     help='input folder of trajs 3d')


   # parser.add_argument('-sx', type=int,required=True,
   #                   help='Starting x coordinate of agent to choose correct trajectory to follow')

   # parser.add_argument('-sy', type=int,required=True,
   #                   help='Starting y coordinate of agent to choose correct trajectory to follow')

   parser.add_argument('--velocity',default = configYml["layer2"]["VELOCITY"], type=float,required=False,
                     help='Speed value')


   parser.add_argument( '--log',action='store_true',  default=False,
      help='Log into file (default: %(default)s)' )

   parser.add_argument( '--calibrate',action='store_true',  default=False,
      help='Calibration flag to enable calibration (default: %(default)s)' )

   parser.add_argument( '--from-docker',action='store_true',  default=False,
      help='Set ip localhost for docker (default: %(default)s)' )


   # parser.add_argument('--load-qtable', type=str, 
   #    help='qtable file (default: %(default)s)')

   args = parser.parse_args()

   
   if(args.log):
      logging.basicConfig(filename=c_paths["LOG_FOLDER"]+"L2log(AIRSIM)"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".txt",
                              filemode='w',
                              format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                              datefmt='%H:%M:%S',
                              level=logging.INFO)


      logger = logging.getLogger('RL Layer2')
      logger.info('Experiment Date: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M') ) )

   if(args.calibrate):
      while(True):
         print("Input scale (def: 1.0):")
         scale = float(input())
         print("Scale set at "+str(scale)+" .")
         vertices = [
         (0,0),
         (0,c_settings["NCOLS"]),
         (c_settings["NROWS"],c_settings["NCOLS"]),
         (c_settings["NROWS"],0),
         (0,0)] 
         calibrate(vertices,scale)
   else:
      main(trajectories_folder=args.i,velocity=args.velocity)

