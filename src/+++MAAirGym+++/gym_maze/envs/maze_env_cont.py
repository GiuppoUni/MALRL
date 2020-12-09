import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math


from operator import mul
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import itertools
from sklearn.neighbors import KDTree
from gym_maze.envs.maze_view_2d_cont import MazeView2DCont

class MazeEnvCont(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 640

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,do_track_trajectories=False,num_goals = 1):
        self.debug = False
        self.max_steps = 100
        self.max_step_size = 10
        self.viewer = None
        
        self.enable_render = enable_render
        self.num_goals = num_goals
        
        if maze_file:
            print("maze from file")
            self.maze_view = MazeView2DCont(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(MazeEnvCont.SCREEN_WIDTH, MazeEnvCont.SCREEN_HEIGHT), 
                                        enable_render=enable_render,
                                        num_goals = self.num_goals)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                print("not plus")
                has_loops = False
                num_portals = 0
            print("maze_sized")
            self.maze_view = MazeView2DCont(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render,num_goals=self.num_goals)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size
        self.len_court_x = self.maze_size[0]              # the size of the environment
        self.len_court_y = self.maze_size[1]              # the size of the environment

        self.action_angle_low = -1
        self.action_angle_high = 1
        self.action_step_low = -1
        self.action_step_high = 1
        self.action_space = spaces.Box(np.array([self.action_angle_low, self.action_step_low]),
                                       np.array([self.action_angle_high, self.action_step_high]), dtype=np.float32)

        # observation is the x, y coordinate of the grid
        self.obs_low_state = np.array([-1, -1, -1, -1, 0]) # x_agent,y_agent, x_goal, y_goal, distance
        self.obs_high_state = np.array([1, 1, 1, 1, 1])
        self.observation_space =  spaces.Box(self.obs_low_state, self.obs_high_state, dtype=np.float32)


        self.count_actions = 0  # count actions for rewarding
        self.eps = 5  # distance to goal, that has to be reached to solve env
        self.np_random = None  # random generator

        # agent
        self.agent_x = 0
        self.agent_y = 0
        self.positions = []                 # track agent positions for drawing

        # the goal
        self.goal_x = 0
        self.goal_y = 0

        # rendering
        self.screen_height = MazeEnvCont.SCREEN_HEIGHT
        self.screen_width = MazeEnvCont.SCREEN_WIDTH
        self.viewer = None                  # viewer for render()
        self.agent_trans = None             # Transform-object of the moving agent
        self.track_way = None               # polyline object to draw the tracked way
        self.scale = self.screen_width/self.maze_size[0] # o 1?

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Past trajectories variables
        self.trajColFlag = do_track_trajectories
        self.kdtrees = []
        self.trajectory = []

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()
        


    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.count_actions += 1
        self._calculate_position(action)
        
        self.maze_view.update_pos(self.agent_x,
            self.agent_y)

        obs = self._observation()

        done = bool(obs[4] <= self.eps)
        rew = 0
        if not done:
            rew += self._step_reward()
        else:
            rew += self._reward_goal_reached()

        # if(self.maze_view.num_goals==1):
        #     if np.array_equal(self.maze_view.robot, self.maze_view.goal):
        #         reward = 1
        #         done = True
        #     else:
        #         reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        #         done = False
        # else:
        #     # Multiple goals
        #     for _goal in self.maze_view.goals:
        #         # print('self.maze_view.robot, _goal: ', self.maze_view.robot, _goal)
        #         if np.array_equal(self.maze_view.robot, _goal):
        #             reward = 1
        #             self.maze_view.decolor(_goal)
        #             self.maze_view.goals.remove(_goal)
        #             if(self.maze_view.goals==[]):
        #                 done = True
        #             else:
        #                 done = False
        # if reward < 1 :
        #     reward = -0.1/(self.maze_size[0]*self.maze_size[1])
        #     done = False
            
        # self.state = self.maze_view.robot
        # moved = any(self.trajectory[-1] != self.state) if len(self.trajectory)>0 else True
        # if(moved):
        #     self.maze_view.color_visited_cell(self.state[0],self.state[1])
            
        #     if(self.trajColFlag):
        #         self.trajectory.append(list(self.state))
        
        
        # print('trajectory: ', self.trajectory)
        # print('self.kdtrees: ', self.kdtrees)




        # break if more than max_steps actions taken
        done = bool(obs[4] <= self.eps or self.count_actions >= self.max_steps)

        # track, where agent was
        self.positions.append([self.agent_x, self.agent_y])

        normalized_obs = self._normalize_observation(obs)

        info = "Debug:" + "actions performed:" + str(self.count_actions) + ", act:" + str(action[0]) + "," + str(action[1]) + ", dist:" + str(normalized_obs[4]) + ", rew:" + str(
            rew) + ", agent pos: (" + str(self.agent_x) + "," + str(self.agent_y) + ")", "goal pos: (" + str(
            self.goal_x) + "," + str(self.goal_y) + "), done: " + str(done)


        info = {}

        return normalized_obs, rew, done, info

    def reset(self):
        # self.maze_view.goals = self.maze_view.saved_goals
        
        # self.maze_view.reset_robot()
        # self.state = np.zeros(2)
        # self.steps_beyond_done = None
        # self.done = False
        # if(self.trajColFlag and self.trajectory!=[]):
        #     # Add points of one trajectory removing also duplicates
        #     self.trajectory = list(num for num,_ in itertools.groupby(self.trajectory))
        #     _tree = KDTree(np.array(self.trajectory))
        #     self.kdtrees.append(_tree)
        
        self.count_actions = 0
        self.positions = []
        # set initial state randomly
        # self.agent_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.agent_y = self.np_random.uniform(low=0, high=self.len_court_y)
        self.agent_x = 10
        self.agent_y = 240
        # self.goal_x = self.np_random.uniform(low=0, high=self.len_court_x)
        # self.goal_y = self.np_random.uniform(low=0, high=self.len_court_x)
        self.goal_x = 125
        self.goal_y = 125
        if self.goal_y == self.agent_y or self.goal_x == self.agent_x:
            self.reset()
        self.positions.append([self.agent_x, self.agent_y])
        if self.debug:
            print("x/y  - x/y", self.agent_x, self.agent_y, self.goal_x, self.goal_y)
            print("scale x/y  - x/y", self.agent_x*self.scale, self.agent_y*self.scale, self.goal_x*self.scale, self.goal_y*self.scale)

        obs = self._observation()
        return self._normalize_observation(obs)


    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        # if close:
        #     self.maze_view.quit_game()

        self.maze_view.update(mode)

        if mode == 'ansi':
            return self._observation()
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            #track the way, the agent has gone
            self.track_way = rendering.make_polyline(np.dot(self.positions, self.scale))
            self.track_way.set_linewidth(4)
            self.viewer.add_geom(self.track_way)

            # draw the agent
            car = rendering.make_circle(5)
            self.agent_trans = rendering.Transform()
            car.add_attr(self.agent_trans)
            car.set_color(0, 0, 255)
            self.viewer.add_geom(car)

            goal = rendering.make_circle(5)
            goal.add_attr(rendering.Transform(translation=(self.goal_x*self.scale, self.goal_y*self.scale)))
            goal.set_color(255, 0, 0)
            self.viewer.add_geom(goal)

            self.agent_trans.set_translation(self.agent_x * self.scale, self.agent_y * self.scale)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        
        else:
            super(MazeEnvCont, self).render(mode=mode)





# ============================================================================================================================================================================================
# ============================================================================================================================================================================================
# ============================================================================================================================================================================================
# ============================================================================================================================================================================================
# =============================================================================================================================================

    def _distance(self):
        return math.sqrt(pow((self.goal_x - self.agent_x), 2) + pow(self.goal_y - self.agent_y, 2))

    # todo: think about a good reward fct that lets the agents learn to go to the goal by
    #  extra rewarding reaching the goal and learning to do this by few steps as possible

    def _step_reward(self):
        return - self._distance()/10 - 1

    def _observation(self):
        return np.array([self.agent_x, self.agent_y, self.goal_x, self.goal_y, self._distance()])

    def _normalize_observation(self, obs):
        normalized_obs = []
        for i in range(0, 4):
            normalized_obs.append(obs[i]/255*2-1)
        normalized_obs.append(obs[-1]/360.62)
        return np.array(normalized_obs)

    def _calculate_position(self, action):
        angle = (action[0] + 1) * math.pi + math.pi / 2
        if angle > 2 * math.pi:
            angle -= 2 * math.pi
        step_size = (action[1] + 1) / 2 * self.max_step_size
        # calculate new agent state
        _agent_x = self.agent_x + math.cos(angle) * step_size
        _agent_y = self.agent_y + math.sin(angle) * step_size
        
        if not self.maze_view.is_in_obstacle(_agent_x,_agent_y):
            self.agent_x = _agent_x
            self.agent_y = _agent_y
            # borders
            if self.agent_x < 0:
                self.agent_x = 0
            if self.agent_x > self.len_court_x:
                self.agent_x = self.len_court_x
            if self.agent_y < 0:
                self.agent_y = 0
            if self.agent_y > self.len_court_y:
                self.agent_y = self.len_court_y

    



        

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None






if __name__ == "__main__" :

    env = MazeEnvCont( maze_file = "maze_samples/maze2d_001.npy",                  
            # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                        # maze_size=(640, 640), 
                                        enable_render=True)

    env.render()
    input("Enter any key to quit.")


