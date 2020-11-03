import setup_path 
import airsim
import cv2
import numpy as np
import os
import pprint
import setup_path 
import tempfile

from argparse import ArgumentParser

import numpy as np
import time
import math

from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

import csv
import json
from pathlib import Path

from utils import *


# WINDOWS ONLY
SETTINGS_PATH = 'C:/Users/gioca/OneDrive/Documents/Airsim/'

initX = -.55265
initY = -31.9786
initZ = -19.0225

# RL agent params
NumBufferFrames = 4
SizeRows = 84
SizeCols = 84
NumActions = 7

# Train
epoch = 100
current_step = 0
max_steps = epoch * 250000

# Use below in settings.json with Blocks environment
"""
{ 
	"SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
	"SettingsVersion": 1.2,
	"SimMode": "Multirotor",
	"ClockSpeed": 1,
	
	"Vehicles": {
		"Drone1": {
		  "VehicleType": "SimpleFlight",
		  "X": 4, "Y": 0, "Z": -2
		},
		"Drone2": {
		  "VehicleType": "SimpleFlight",
		  "X": 8, "Y": 0, "Z": -2
		}

    }
}
"""


class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample mini-batches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #mini-batch() if you want to retrieve samples directly.

        Attributes:
            size (int): The mini-batch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold

    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=10000, train_interval=4, target_update_interval=10000,
                 monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # Action Value model (used by agent to interact with the environment)
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Convolution2D((8, 8), 16, strides=4),
                Convolution2D((4, 4), 32, strides=2),
                Convolution2D((3, 3), 32, strides=1),
                Dense(256, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])
        self._action_value_net.update_signature(Tensor[input_shape])

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Function computing Q-values targets as part of the computation graph
        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        # Define the loss, using Huber Loss (more robust to outliers)
        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            # Compute the q_targets
            q_targets = compute_q_targets(post_states, rewards, terminals)

            # actions is a 1-hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            # Define training criterion as the Huber Loss function
            return huber_loss(q_targets, q_acted, 1.0)

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)

                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                    filename = "models\model%d" % agent_step
                    self._trainer.save_checkpoint(filename)

    def _plot_metrics(self):
        """Plot current buffers accumulated values to visualize agent learning
        """
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)

def main():


    with open(SETTINGS_PATH + 'settings.json', 'r') as f:
        settings = json.load(f)
    

    veichle_names = settings["Vehicles"]


    print("*"*20," AIRSIM SIMULATION STARTED ","*"*20)
    # Make RL Agent
    agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols), NumActions, monitor=True)

    # connect to the AirSim simulator 
    client = airsim.MultirotorClient()
    print("\n",client,"\n")
    client.confirmConnection()
    print('Connection Confirmed')

    for v in veichle_names:
        client.enableApiControl(True,v)
        client.armDisarm(True,v)
    print("starting Take-Off")    
    last_vehicle_pointer = takeoff_all_drones(client,veichle_names)
    print("Waiting...")
    last_vehicle_pointer.join() 

    # client.moveToPositionAsync(initX, initY, initZ, 5).join()
    # client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
    # time.sleep(0.5)

    # responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
    # current_state = transform_input(responses)
    start_time = time.time()
    while True:

        # data_drone1 = client.getDistanceSensorData(vehicle_name="Drone1")
        # data_drone2 = client.getDistanceSensorData(vehicle_name="Drone2")
        # print(f"Distance sensor data: Drone1: {data_drone1.distance}, Drone2: {data_drone2.distance}")
    #     action = agent.act(current_state)
    #     quad_offset = interpret_action(action)
    #     quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    #     client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1], quad_vel.z_val+quad_offset[2], 5).join()
        time.sleep(0.5)
    
    #     quad_state = client.getMultirotorState().kinematics_estimated.position
    #     quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity
    #     collision_info = client.simGetCollisionInfo()
    #     reward = compute_reward(quad_state, quad_vel, collision_info)
    #     done = isDone(reward)
    #     print('Action, Reward, Done:', action, reward, done)

    #     agent.observe(current_state, action, reward, done)
    #     agent.train()

    #     if done:
    #         client.moveToPositionAsync(initX, initY, initZ, 5).join()
    #         client.moveByVelocityAsync(1, -0.67, -0.8, 5).join()
    #         time.sleep(0.5)
    #         current_step +=1

    #     responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.DepthPerspective, True, False)])
    #     current_state = transform_input(responses)
    
    end_time = time.time()
    total_time = end_time - start_time
    minutes = round(total_time / 60, 3)
    seconds = np.floor((total_time / 60) - minutes)
    print("Total Time: {mins} mins {secs} secs".format(
        mins=minutes, secs=seconds))

    client.reset()


main()


# def main2():

#     # connect to the AirSim simulator
#     client = airsim.MultirotorClient()
#     client.confirmConnection()
#     client.enableApiControl(True, "Drone1")
#     client.enableApiControl(True, "Drone2")
#     client.armDisarm(True, "Drone1")
#     client.armDisarm(True, "Drone2")

#     airsim.wait_key('Press any key to takeoff')
#     f1 = client.takeoffAsync(vehicle_name="Drone1")
#     f2 = client.takeoffAsync(vehicle_name="Drone2")
#     f1.join()
#     f2.join()

#     state1 = client.getMultirotorState(vehicle_name="Drone1")
#     s = pprint.pformat(state1)
#     print("state: %s" % s)
#     state2 = client.getMultirotorState(vehicle_name="Drone2")
#     s = pprint.pformat(state2)
#     print("state: %s" % s)

#     airsim.wait_key('Press any key to move vehicles')
#     f1 = client.moveToPositionAsync(-5, 5, -10, 5, vehicle_name="Drone1")
#     f2 = client.moveToPositionAsync(5, -5, -10, 5, vehicle_name="Drone2")
#     f1.join()
#     f2.join()

#     airsim.wait_key('Press any key to take images')
#     # get camera images from the car
#     responses1 = client.simGetImages([
#         airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#         airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)], vehicle_name="Drone1")  #scene vision image in uncompressed RGB array
#     print('Drone1: Retrieved images: %d' % len(responses1))
#     responses2 = client.simGetImages([
#         airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
#         airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)], vehicle_name="Drone2")  #scene vision image in uncompressed RGB array
#     print('Drone2: Retrieved images: %d' % len(responses2))

#     tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
#     print ("Saving images to %s" % tmp_dir)
#     try:
#         os.makedirs(tmp_dir)
#     except OSError:
#         if not os.path.isdir(tmp_dir):
#             raise

#     for idx, response in enumerate(responses1 + responses2):

#         filename = os.path.join(tmp_dir, str(idx))

#         if response.pixels_as_float:
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
#             airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
#         elif response.compress: #png format
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#             airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
#         else: #uncompressed array
#             print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
#             img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
#             img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
#             cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png

#     airsim.wait_key('Press any key to reset to original state')

#     client.armDisarm(False, "Drone1")
#     client.armDisarm(False, "Drone2")
#     client.reset()

#     # that's enough fun for now. let's quit cleanly
#     client.enableApiControl(False, "Drone1")
#     client.enableApiControl(False, "Drone2")


