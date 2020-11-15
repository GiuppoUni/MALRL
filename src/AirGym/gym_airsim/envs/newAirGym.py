import logging
import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box

from gym_airsim.envs.myAirSimClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)



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

    myClient = None
        
    def __init__(self,n_agents = 3, step_cost = -1):
        self.n_agents = n_agents
        # left depth, center depth, right depth, yaw
        self.observation_space = spaces.Box(low=0, high=255, shape=(30, 100))
        self.states = [np.zeros((30, 100), dtype=np.uint8) for _ in range(n_agents)]  
        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(n_agents)])
		
        self.goals = [ [221.0, -9.0] for _ in range(n_agents)] # global xy coordinates
        
        
        self.episodeN = 0
        self.stepN = 0 
        self._step_cost = step_cost
        self._agents_dones = [ False for _ in range(n_agents)]


        self._seed()
        
        global myClient
        myClient = MyAirSimClient()

        self.allLogs = dict()
        self.init_logs()
        
        
        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def computeReward(self,vehicle_name,now,goal,track_now):
	
		# test if getPosition works here liek that
		# get exact coordiantes of the tip
      
        distance_now = np.sqrt(np.power((goal[0]-now["x_val"]),2) + np.power((goal[1]-now["y_val"]),2))
        
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
        for agent_i,action in enumerate(agents_actions):

            if self._agents_dones[agent_i]:
                continue    #agent_i has done with its task

            _current_log = self.allLogs['Drone'+str(agent_i+1)]

            assert self.action_space[agent_i].contains(action), "%r (%s) invalid"%(action, type(action))
            addToDict(_current_log,"action", action)
        
            assert myClient.ping()    
            drone = myClient.drones[agent_i]
            collided = drone.take_action(action)
        
            now = myClient.getPosition(vehicle_name = drone.vehicle_name)
            goal = self.goals[agent_i]
            track = drone.goal_direction(goal, now) 
            if collided == True:
                done = True
                reward = -100.0
                distance = np.sqrt(np.power((goal[0]-now["x_val"]),2) + np.power((goal[1]-now["y_val"]),2))
            elif collided == 99:
                done = True
                reward = 0.0
                distance = np.sqrt(np.power((goal[0]-now["x_val"]),2) + np.power((goal[1]-now["y_val"]),2))
            else: 
                done = False
                reward, distance = self.computeReward("Drone"+str(agent_i+1),
                                                    now,goal, track)
        
            # Youuuuu made it
            if distance < 3:
                done = True
                reward = 100.0

            #Update reward for agent agent_i th
            rewards[agent_i] = reward
            self._agents_dones[agent_i] = done 

            addToDict(_current_log,"reward", reward)
            addToDict(_current_log, 'distance', distance) 
            addToDict(_current_log, 'track', track)      
            
            rewardSum = np.sum(_current_log['reward'])
            
            # Terminate the episode on large cumulative amount penalties, 
            # since drone probably got into an unexpected loop of some sort
            if rewardSum < -100:
                done = True

            toPrint+=("\t uav"+str(agent_i)+": {:.1f}/{:.1f}, {:.0f}, {:.0f} \n".format( reward, rewardSum, track, action))
            info[agent_i] = {"x_pos" : now["x_val"], "y_pos" : now["y_val"]}

            assert myClient.ping()
            self.states[agent_i] = drone.getScreenDepthVis(track)
        
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
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        myClient.AirSim_reset()
        
        self.stepN = 0
        self.episodeN += 1
        
        
        self.init_logs()
            
        print("")
        for i,u in enumerate(myClient.drones) :
            now = myClient.getPosition(vehicle_name = u.vehicle_name)
            track = u.goal_direction(self.goals[i], now)
            self.states[i] = u.getScreenDepthVis(track)
        
        return self.states


    def init_logs(self):
        if myClient:
            for u in myClient.drones:
                self.allLogs[u.vehicle_name] = { 'reward':[0] }
                self.allLogs[u.vehicle_name]['distance'] = [221]
                self.allLogs[u.vehicle_name]['track'] = [-2]
                self.allLogs[u.vehicle_name]['action'] = [1]

def addToDict(d: dict,k,v):
    if k not in d:
        d[k] = []
    d[k].append(v)
