



import eventlet
import utils

import logging
import numpy as np
import random
from airsim import Vector3r,Pose, to_quaternion
import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from newMyAirSimClient import lock, newMyAirSimClient

import time
import utils
import sys
import concurrent.futures
import threading
import signal

# All coords
# this format -> (lon,lat,height)

class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]



 
class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass
 
 
def service_shutdown(signum, frame):
    print('Caught signal %d' % signum)
    raise ServiceExit


class CollectEnv(gym.Env):
    """
    Environment in which the agents have to collect the balls
    """
    
    ACTION = ["LEFT","FRONT","RIGHT","BACK",]

    def __init__(
        self,
        trajColFlag,
        size=10,
        width=None,
        height=None,
        num_targets=3,
        n_actions = 4, step_cost = -1,partial_obs = False,
        random_pos = False
        ):
        

        signal.signal(signal.SIGTERM, service_shutdown)
        signal.signal(signal.SIGINT, service_shutdown)
        self.lock = threading.Lock()
        self.threads = {}

            
        self.random_pos = random_pos
     

        # left depth, center depth, right depth, yaw
        # self.observation_space = [spaces.Box(low=0, high=255, shape=(30, 100)*n_agents)
        # NOTE Hardcoded
        self.width = 223 -21
        self.height = 121 - ( -70 )
        if partial_obs:
            agent_view_size = 20
            self.states_space = spaces.Box(
                low=0,
                high=255,
                shape=(agent_view_size, agent_view_size, 1),
                dtype='uint8'
            )

        else:
            # TODO get bounding box from tall obstacles at the margins
            
            self.states_space= spaces.Box(
              
                low = 0,
                high = 255,
                shape=(self.width, self.height, 1),
                dtype='uint8'
            )
        
        self.state = np.zeros(3, dtype=np.uint8) 
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
		

        self.episodeN = 0
        self.stepN = 0 
        self._step_cost = step_cost
        self._agent_done = False


        self._seed()
        
        # self.myClient = MyAirSimClient2(utils.SRID,utils.ORIGIN,ip="127.1.1.1")
        self.myClient = newMyAirSimClient(trajColFlag=trajColFlag)
        # TODO replace with  allocated targets
        

        

        self.target_zones = self._get_target_zone()

        self.goal = self.target_zones.pop() 

        self.targets = dict()
        self._get_targets()
        # self.myClient.direct_client.takeoffAsync(vehicle_name="Drone0")

        # TODO remove (it just prints em)
        self._get_limits()

        self.init_pool = []
        self.init_random_pos_pool()

        self.allLogs = dict()
        self.init_logs()

        # self.myClient.drawTrajectories()


    def _vec2r_to_numpy_array(self,vec):
        return np.array([vec.x_val, vec.y_val])

    def _get_limits(self,regex="LIMB.*"):
        limits_name = self.myClient.simListSceneObjects(regex)
        
        for l in limits_name: 
            pose = self.myClient.simGetObjectPose(l)
            print(l , self._vec2r_to_numpy_array(pose.position))


    def init_random_pos_pool(self,regex = "Init.*"):
        init_names = self.myClient.simListSceneObjects(regex)
        for name in init_names: 
            pose = self.myClient.simGetObjectPose(name)
            self.init_pool.append(pose.position)
            print(self._vec2r_to_numpy_array(pose.position))
        
        # Random shuffle the init positions
        self.np_random.shuffle(self.init_pool)


    def get_pos_from_pool(self):
        return self.init_pool[0]
        
    def printTargets(self):
        print('Env targets: ', self.targets)
        
    def _get_targets(self,regex="TargetB.*"):
        targets = self.myClient.simListSceneObjects(regex)
        targets.sort()
        self.targets_names = targets
        
        self.targets= dict()
        for wp in targets: 
            pose = self.myClient.simGetObjectPose(wp)
            self.targets[wp] = (self._vec2r_to_numpy_array(pose.position))
        
        return True 

    def _get_target_zone(self, regex = "TargetZone.*"):
        targets = self.myClient.simListSceneObjects(regex)
        targets.sort()
        
        targets_positions= []
        
        for t in targets: 
            pose = self.myClient.simGetObjectPose(t)
            targets_positions.append(  self._vec2r_to_numpy_array(pose.position) )
        
        return targets_positions 

    def dummy_allocate_targets(self):
        # for t in self.targets:
        pass


        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self,vehicle_name,now,goal,track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = utils.xy_distance(now,self.goal)
        
        distance_before = self.allLogs['distance'][-1]
              
        r = -1
        
        """
        if abs(distance_now - distance_before) < 0.0001:
            r = r - 2.0
            #Check if last 4 positions are the same. Is the copter actually moving?
            if self.stepN > 5 and len(set(self.allLogs['distance'][len(self.allLogs['distance']):len(self.allLogs['distance'])-5:-1])) == 1: 
                r = r - 50
        """  
            
        r = r + (distance_before - distance_now)
            
        return r, distance_now
		
    
    def step(self, agent_action):
        self.stepN += 1
        reward = self._step_cost 

        info = None 
        toPrint = ""

        #  single drone case:
        agent_name = "Drone0"

        print('STEPPING: ')
        if self._agent_done :
            print("No action (done): ")
            #agent_i has done with its task
            return self.state, reward, self._agent_done, info

        _current_log = self.allLogs

        assert self.action_space.contains(agent_action), "%r (%s) invalid"%(agent_action, type(agent_action))
        utils.addToDict(_current_log,"action", agent_action)
    
        # --- HERE EXECUTE DRONE ACTION ---
        result = self.myClient.take_action(agent_action,agent_name)
        collided = result["obs"]
        #---------------------------------------------
        
        now = self.myClient.getPosition(vehicle_name = agent_name)

        track = self.myClient.goal_direction(self.goal, now,agent_name) 
        

        # distanceTraj = self.myClient.distanceFromTraj(now)


        # distance = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2))
        goal_distance = utils.xy_distance(now,self.goal)

        if collided == True:
            reward = -100.0
            done = True 
        else: 
            reward, goal_distance = self.computeReward("Drone0",now,self.goal, track)
            done = False

        if result["total_p"] > 0:
            if result["collisions_per_traj"]:
                
                for idx,traj_key in enumerate( result["collisions_per_traj"] ) :
                    reward -= 5 * result["collisions_per_traj"][traj_key][0] +  utils.NEW_TRAJ_PENALTY * idx 
                    
            else:
                reward = -5 * result["total_p"] 
    
        # Youuuuu made it
        if goal_distance < 3:
            print( "TARGET ZONE REACHED")
            done = True
            reward = 100.0

        #Update reward for agent agent_i th
        # DEBUG ONLY 

        utils.addToDict(_current_log,"reward", reward)
        utils.addToDict(_current_log, 'distance', goal_distance) 
        utils.addToDict(_current_log, 'track', track)      
        
        rewardSum = np.sum(_current_log['reward'])
        
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -100:
            print("REWARD LOWER BOUND REACHED")
            done = True

        toPrint+=("\t uav"+": {:.1f}/{:.1f}, {:.0f}, {} \n".format( reward, rewardSum, track, CollectEnv.ACTION[agent_action]) )
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}

        # assert self.myClient.direct_client.ping()
        # self.states[agent_i] = drone.getScreenDepthVis(track)

        self.state = self._vec2r_to_numpy_array( self.myClient.getPosition("Drone0") )

        # self.myClient.wait_joins("STEP")
        self._agent_done = done 
        print(" Episode:{},Step:{}\n \t\t reward/r. sum, track, action: \n".format(self.episodeN, self.stepN) + toPrint )   

        return self.state, reward,self._agent_done , info


    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        (It's called also at the beginning)
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        self.init_random_pos_pool()

        self.stepN = 0
        self.episodeN += 1
        
        self._agent_done = False 
            
        print("Resetting...")
        # self.myClient.AirSim_reset()


        self.goals = [ [1,1,1] ] # global xy coordinates
        self.myClient.pts = []

        pose = Pose(self.get_pos_from_pool(), to_quaternion(0, 0, 0) ) 
        self.myClient.reset()    


        self.myClient.disable_trace_lines()
        if self.random_pos : 
            self.myClient.simSetVehiclePose( pose, ignore_collison=True, vehicle_name = "Drone0")
        
        self.myClient.AirSim_reset()
        self.myClient.enable_trace_lines()

        self.init_logs()


        
        return self.state





    def init_logs(self):
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [221]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]

