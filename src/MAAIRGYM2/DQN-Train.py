# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:16:10 2017

@author: Kjell
"""
import numpy as np
import gym

import gym_airsim.envs
import gym_airsim


import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Concatenate
from keras.optimizers import Adam
import keras.backend as K

from PIL import Image

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from callbacks import *

from keras.callbacks import History

import utils
import threading
import time

from gym_airsim.envs.mDQNAgent import NamedDQNAgent



parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test',"multi-test","multi-train"], default='multi-test')
parser.add_argument('--env-name', type=str, default='MAGEnv-v1')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--weight-folder', type=str, default="./weights/")
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n


# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
INPUT_SHAPE = (30, 100)
WINDOW_LENGTH = 1
# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE


model = Sequential()
model.add(Conv2D(32, (4, 4), strides=(4, 4) ,activation='relu', input_shape=input_shape, data_format = "channels_first"))
model.add(Conv2D(64, (3, 3), strides=(2, 2),  activation='relu'))
model.add(Conv2D(64, (1, 1), strides=(1, 1),  activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())




# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)                        #reduce memmory


# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.0,
                              nb_steps=100000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=True, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])


  




multiDQNs = [ NamedDQNAgent(agent_name = vn,model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=3000, 
               enable_double_dqn=True, 
               enable_dueling_network=True, dueling_type='avg', 
               target_model_update=1e-2, policy=policy, gamma=.99) for vn in utils.g_airsim_settings["Vehicles"]  ] 

for mdqn in multiDQNs:
    mdqn.compile(Adam(lr=0.00025), metrics=['mae'])




threadLock = threading.Lock()

class myThread (threading.Thread):
   def __init__(self, threadID, name, counter,dqn):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.name = name
      self.counter = counter
      self.dqn = dqn
   def run(self):
        print ("Starting " + self.name)
        # Get lock to synchronize threads
        self.dqn.fit(env, callbacks=callbacks, nb_steps=251000, visualize=False, verbose=2, log_interval=100)
        
    
        # After training is done, we save the final weights.
        self.dqn.save_weights(args.weight_folder+'dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)
    
    #   TODO do client functions inside threadlock
    #   threadLock.acquire()
    #   print_time(self.name, self.counter, 3)
    #   # Free lock to release next thread
    #   threadLock.release()

def print_time(threadName, delay, counter):
   while counter:
      time.sleep(delay)
      print ("%s: %s" % (threadName, time.ctime(time.time())))
      counter -= 1


if args.mode == "train":
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=100)]
    
    dqn.fit(env, callbacks=callbacks, nb_steps=251000, visualize=False, verbose=2, log_interval=100)
    
    
    # After training is done, we save the final weights.
    dqn.save_weights(args.weight_folder+'dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)


elif args.mode == "test":

    dqn.load_weights(args.weight_folder+'dqn_{}_weights.h5f'.format(args.env_name))
    dqn.test(env, nb_episodes=10, visualize=False)

elif args.mode == "multi-train":
    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    
    
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [FileLogger(log_filename, interval=100)]
    
    threads=[]
    for i,mdqn in enumerate(multiDQNs):
        # dqn.fit(env, callbacks=callbacks, nb_steps=251000, visualize=False, verbose=2, log_interval=100)

        # # After training is done, we save the final weights.
        # dqn.save_weights(args.weight_folder+'dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)
        threads.append( myThread(i, "Thread-"+str(i), 1,mdqn) )
    for t in threads:
        t.start()

elif args.mode == "multi-test":
    for dqn in multiDQNs:
        dqn.load_weights(args.weight_folder+'dqn_{}_weights.h5f'.format(args.env_name))
        dqn.test(env, nb_episodes=10, visualize=False)



