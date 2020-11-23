import logging
from myAirSimClient2 import MyAirSimClient2
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from myAirSimClient import *
from oldMyAirSimClient import oldMyAirSimClient

logger = logging.getLogger(__name__)

import utils
import sys

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


class AirSimEnv(gym.Env):


    def __init__(self,n_agents = 3,n_actions = 3, step_cost = -1):
        self.n_agents = n_agents
        # left depth, center depth, right depth, yaw
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100))
        self.states = [np.zeros((30, 100), dtype=np.uint8) for _ in range(n_agents)] 
        self.n_actions = n_actions
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(n_agents)])
		
        self.agent_names = [v for v in utils.g_airsim_settings["Vehicles"] ]
        
        self.episodeN = 0
        self.stepN = 0 
        self._step_cost = step_cost
        self._agents_dones = [ False for _ in range(n_agents)]

        

        self._seed()
        
        # self.myClient = MyAirSimClient2(utils.SRID,utils.ORIGIN,ip="127.1.1.1")
        self.myClient = oldMyAirSimClient()
        # TODO replace with  allocated targets
        self.goals = [ [221.0, -9.0 + (i*5)] for i in range(n_agents)] # global xy coordinates
        
        # self.myClient.direct_client.takeoffAsync(vehicle_name="Drone0")


        self.allLogs = dict()
        self.init_logs()


        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self,vehicle_name,now,goal,track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((goal[0]-now.x_val),2) + np.power((goal[1]-now.y_val),2))
        
        distance_before = self.allLogs[vehicle_name]['distance'][-1]
              
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
		
    
    def step(self, agents_actions):
        self.stepN += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]
        info = [None for _ in range(self.n_agents)]
        toPrint = ""

        self.myClient.pts = []
        for agent_i,action in enumerate(agents_actions):
            agent_name = 'Drone'+str(agent_i)
            if self._agents_dones[agent_i]: 
                print("[Drone"+str(agent_i)+"]"+"No action (done): ")
                continue    #agent_i has done with its task

            _current_log = self.allLogs[agent_name]

            assert self.action_space[agent_i].contains(action), "%r (%s) invalid"%(action, type(action))
            utils.addToDict(_current_log,"action", action)
        
            # assert self.myClient.direct_client.ping()    
            # Get current drone
            # drone = self.myClient.drones[agent_i]
            
            # --- HERE EXECUTE DRONE ACTION ---
            # collided,pt = drone.take_action(action)
            collided = self.myClient.take_action(action,agent_i)
            #---------------------------------------------
            # self.myClient.pts.append(pt)
            
            now = self.myClient.getPosition(vehicle_name = agent_name)
            goal = self.goals[agent_i]
            track = self.myClient.goal_direction(goal, now,agent_name) 
            
            done = True
            distance = np.sqrt(np.power((goal[0]-now.x_val),2) + np.power((goal[1]-now.y_val),2))
            reward = 0
            if collided == True:
                reward = -100.0
            elif collided == 99:
                reward = 0.0
            else: 
                done = False
                reward, distance = self.computeReward("Drone"+str(agent_i),
                                                    now,goal, track)
        
            # Youuuuu made it
            if distance < 3:
                done = True
                reward = 100.0

            #Update reward for agent agent_i th
            rewards[agent_i] = reward
            self._agents_dones[agent_i] = done 

            utils.addToDict(_current_log,"reward", reward)
            utils.addToDict(_current_log, 'distance', distance) 
            utils.addToDict(_current_log, 'track', track)      
            
            rewardSum = np.sum(_current_log['reward'])
            
            # Terminate the episode on large cumulative amount penalties, 
            # since drone probably got into an unexpected loop of some sort
            if rewardSum < -100:
                done = True

            toPrint+=("\t uav"+str(agent_i)+": {:.1f}/{:.1f}, {:.0f}, {:.0f} \n".format( reward, rewardSum, track, action))
            info[agent_i] = {"x_pos" : now.x_val, "y_pos" : now.y_val}

            # assert self.myClient.direct_client.ping()
            # self.states[agent_i] = drone.getScreenDepthVis(track)

        # self.myClient.wait_joins("STEP")

        sys.stdout.write(" Episode:{},Step:{}\n \t\t reward/r. sum, track, action: \n".format(self.episodeN, self.stepN) + toPrint )   
        sys.stdout.flush()

        return self.states, rewards, self._agents_dones, info

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

        
        self.stepN = 0
        self.episodeN += 1
        
            
        print("Resetting...")
        # self.myClient.AirSim_reset()
        self.local_reset()

        self.init_logs()
    
        
        return self.states

    def local_reset(self):
        # Reset targets
        # self.myClient.targetMg.reset_targets_status()
        # self.goals = self.myClient.allocate_all_targets()
        self.goals = [ [221.0, -9.0 + (i*5)] for i in range(self.n_agents)] # global xy coordinates
        self.myClient.pts = []
        # for i,d in enumerate(self.myClient.drones):
        #     self.myClient.place_one_drone(d.vehicle_name,
        #         gps = utils.init_gps[i])
        #     # _pt = d.reset_Zposition()
        #     d.z = -6
            
        #     pt = self.myClient.direct_client.moveToZAsync(d.z,3,
        #         vehicle_name=d.vehicle_name)
        #     self.myClient.pts.append(pt)
        #     d.tagPrint("move z reset Joining...")
        #     # _pt.join()
        #     # TODO SHOULD BE JOINED FOR Z BUT CRASH HAPPENS
        #     # now = self.myClient.getPosition(d.vehicle_name)
        #     # d.track = d.goal_direction( self.goals[i],now)
        #     # d.home_pos = now
        #     # d.home_ori = self.myClient.getOrientation(d.vehicle_name)

        # self.myClient.wait_joins()


    def init_logs(self):
        # return
        for vn in self.agent_names:
            self.allLogs[vn] = { 'reward':[0] }
            self.allLogs[vn]['distance'] = [221]
            self.allLogs[vn]['track'] = [-2]
            self.allLogs[vn]['action'] = [1]


