

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

from newMyAirSimClient import lock, NewMyAirSimClient

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

class Job(threading.Thread):
 
    def __init__(self,timer = 0.5,callback = None, **kwargs):
        threading.Thread.__init__(self)

        self.callback = callback
        self.timer = timer
        self.args_dict = kwargs
        # The shutdown_flag is a threading.Event object that
        # indicates whether the thread should be terminated.
        self.shutdown_flag = threading.Event()
 
        # ... Other thread setup code here ...
 
    def run(self):
        print('Thread #%s started' % self.ident)
 
        while not self.shutdown_flag.is_set():
            # ... Job code here ...
            if self.callback:
                self.callback(**self.args_dict)
            time.sleep(self.timer)
 
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


class CollectMTEnv(gym.Env):
    """
    Environment in which the agents have to collect the balls
    """
    def __init__(
        self,
        trajColFlag,
        size=10,
        width=None,
        height=None,
        num_targets=3,
        n_agents = int(utils.g_config["rl"]["n_agents"]),
        n_actions = 3, step_cost = -1,partial_obs = False,
        ):

        signal.signal(signal.SIGTERM, service_shutdown)
        signal.signal(signal.SIGINT, service_shutdown)
        self.lock = threading.Lock()
        self.threads = {}

            
        self.agent_names = [v for v in utils.g_airsim_settings["Vehicles"] ]

        # User Check
        if(n_agents > len(self.agent_names)):
            print("[WARNING] Inserted more agents than settings")
            n_agents = len(self.agent_names)
        self.n_agents = n_agents
        self.isThreaded = True if n_agents > 1 else False

        # left depth, center depth, right depth, yaw
        # self.observation_space = [spaces.Box(low=0, high=255, shape=(30, 100)*n_agents)
        # NOTE Hardcoded
        self.width = 223 -21
        self.height = 121 - ( -70 )
        if partial_obs:
            agent_view_size = 20
            self.states_space = spaces.Box(
                low=0,
                high=self.n_agents+len(self.targets),
                shape=(agent_view_size, agent_view_size, self.n_agents),
                dtype='uint8'
            )

        else:
            # TODO get bounding box from tall obstacles at the margins
            
            self.states_space= spaces.Box(
              
                low = 0,
                high = 255,
                shape=(self.width, self.height, self.n_agents),
                dtype='uint8'
            )
        
        self.states = [np.zeros(2, dtype=np.uint8) for _ in range(n_agents)] 
        self.n_actions = n_actions
        self.action_space = MultiAgentActionSpace([spaces.Discrete(n_actions) for _ in range(n_agents)])
		

        self.episodeN = 0
        self.stepN = 0 
        self._step_cost = step_cost
        self._agents_dones = [ False for _ in range(n_agents)]


        self._seed()
        
        # self.myClient = MyAirSimClient2(utils.SRID,utils.ORIGIN,ip="127.1.1.1")
        self.myClient = NewMyAirSimClient(trajColFlag=trajColFlag)
        # TODO replace with  allocated targets
        self.goals = [ [221.0, -9.0 + (i*5)] for i in range(n_agents)] # global xy coordinates
        
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
            print(self._vec2r_to_numpy_array(pose.position))


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

    def dummy_allocate_targets(self):
        # for t in self.targets:
        pass


        
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

        if(self.isThreaded):
            threadPts = [] 
            collideds= []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for agent_i,action in enumerate(agents_actions):
                    
                    agent_name = 'Drone'+str(agent_i)
                    print('STEPPING: ', agent_name)
                    if self._agents_dones[agent_i]: 
                        print("[Drone"+str(agent_i)+"]"+"No action (done): ")
                        continue    #agent_i has done with its task

                    _current_log = self.allLogs[agent_name]

                    assert self.action_space[agent_i].contains(action), "%r (%s) invalid"%(action, type(action))
                    utils.addToDict(_current_log,"action", action)

                    # --- HERE EXECUTE DRONE ACTION ---

                    future = executor.submit(self.myClient.take_action_threaded, action,self.lock,agent_i,agent_name)
                    threadPts.append(future)
                    #---------------------------------------------


                # NOTE: DISABLED  SIGINT 
                for th in concurrent.futures.as_completed(threadPts):
                    res = th.result()
                    collideds.append(res)
                    print("JOINED:", res[0])

            collideds.sort(key=lambda x: x[0])
            for agent_i,collided in collideds:
                agent_name = "Drone"+str(agent_i)
                now = self.myClient.getPosition(vehicle_name = agent_name)
                goal = list(self.targets.values())[agent_i]
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
                    reward, distance = self.computeReward(agent_name,
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
                # self.states[agent_i] = self.myClient.getScreenDepthVis(track,vehicle_name = agent_name)

            print(" Episode:{},Step:{}\n \t\t reward/r. sum, track, action: \n".format(self.episodeN, self.stepN) + toPrint )
        
        else: #(self.isThreaded == False) single drone case:
            agent_i = 0    
            action = agents_actions[agent_i]
            agent_name = 'Drone'+str(agent_i)
            print('STEPPING: ', agent_name)
            if self._agents_dones[agent_i]: 
                print("[Drone"+str(agent_i)+"]"+"No action (done): ")
                #agent_i has done with its task
                return self.states, rewards, self._agents_dones, info

            _current_log = self.allLogs[agent_name]

            assert self.action_space[agent_i].contains(action), "%r (%s) invalid"%(action, type(action))
            utils.addToDict(_current_log,"action", action)
        
            # --- HERE EXECUTE DRONE ACTION ---
            collided = self.myClient.take_action(action,agent_name)
            #---------------------------------------------
            
            now = self.myClient.getPosition(vehicle_name = agent_name)
            goal = self.goals[agent_i]
            track = self.myClient.goal_direction(goal, now,agent_name) 
            

            # distanceTraj = self.myClient.distanceFromTraj(now)


            
            distance = np.sqrt(np.power((goal[0]-now.x_val),2) + np.power((goal[1]-now.y_val),2))
            reward = 0
            if collided == True:
                reward = -100.0
                done = True 
            else: 
                reward, distance = self.computeReward("Drone"+str(agent_i),
                                                    now,goal, track)
                done = False
        
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

            print(" Episode:{},Step:{}\n \t\t reward/r. sum, track, action: \n".format(self.episodeN, self.stepN) + toPrint )   

        return self.states, rewards, self._agents_dones, info


    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def reset(self,random_pos=False):
        """
        Resets the state of the environment and returns an initial observation.
        (It's called also at the beginning)
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """

        self.init_random_pos_pool()

        self.stepN = 0
        self.episodeN += 1
        
        self._agents_dones = [False for _ in range(self.n_agents)]
            
        print("Resetting...")
        # self.myClient.AirSim_reset()


        self.goals = [ [221.0, -9.0 + (i*5)] for i in range(self.n_agents)] # global xy coordinates
        self.myClient.pts = []

        pose = Pose(self.get_pos_from_pool(), to_quaternion(0, 0, 0) ) 
        self.myClient.reset()    


        self.myClient.disable_trace_lines()
        if random_pos : 
            self.myClient.simSetVehiclePose( pose, ignore_collison=True, vehicle_name = "Drone0")
        
        self.myClient.AirSim_reset()
        self.myClient.enable_trace_lines()

        self.init_logs()


        
        return self.states





    def init_logs(self):
        # return
        for vn in self.agent_names:
            self.allLogs[vn] = { 'reward':[0] }
            self.allLogs[vn]['distance'] = [221]
            self.allLogs[vn]['track'] = [-2]
            self.allLogs[vn]['action'] = [1]

