from operator import mul
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D
import itertools
from sklearn.neighbors import KDTree

class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N","S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,do_track_trajectories=False,num_goals = 1):

        self.viewer = None
        self.enable_render = enable_render
        self.num_goals = num_goals

        if maze_file:
            print("maze from file")
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
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
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render,num_goals=self.num_goals)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

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
        reward = 0
        if isinstance(action, int):
            self.maze_view.move_robot(self.ACTION[action])
        else:
            self.maze_view.move_robot(action)

        if(self.maze_view.num_goals==1):
            if np.array_equal(self.maze_view.robot, self.maze_view.goal):
                reward = 1
                done = True
            else:
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False
        else:
            # Multiple goals
            for _goal in self.maze_view.goals:
                # print('self.maze_view.robot, _goal: ', self.maze_view.robot, _goal)
                if np.array_equal(self.maze_view.robot, _goal):
                    reward = 1
                    self.maze_view.decolor(_goal)
                    self.maze_view.goals.remove(_goal)
                    if(self.maze_view.goals==[]):
                        done = True
                    else:
                        done = False
        if reward < 1 :
            reward = -0.1/(self.maze_size[0]*self.maze_size[1])
            done = False

            
        self.state = self.maze_view.robot
        moved = any(self.trajectory[-1] != self.state) if len(self.trajectory)>0 else True
        if(moved):
            self.maze_view.color_visited_cell(self.state[0],self.state[1])
            
            if(self.trajColFlag):
                self.trajectory.append(list(self.state))
        
        
        # print('trajectory: ', self.trajectory)
        # print('self.kdtrees: ', self.kdtrees)

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.maze_view.goals = self.maze_view.saved_goals
        
        self.maze_view.reset_robot()
        self.state = np.zeros(2)
        self.steps_beyond_done = None
        self.done = False
        if(self.trajColFlag and self.trajectory!=[]):
            # Add points of one trajectory removing also duplicates
            self.trajectory = list(num for num,_ in itertools.groupby(self.trajectory))
            _tree = KDTree(np.array(self.trajectory))
            self.kdtrees.append(_tree)
        

        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy", enable_render=enable_render)


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), enable_render=enable_render)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy", enable_render=enable_render)


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10), enable_render=enable_render)


class MazeEnvSample3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy", enable_render=enable_render)


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3), enable_render=enable_render)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy", enable_render=enable_render)


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100), enable_render=enable_render)


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self, enable_render=True):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus", enable_render=enable_render)


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self, enable_render=True):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus", enable_render=enable_render)


if __name__ == "__main__" :

    env = MazeEnv( maze_file = "maze_samples/maze2d_001.npy",                  
            # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                        # maze_size=(640, 640), 
                                        enable_render=True)

    env.render()
    import time
    input("Enter any key to quit.")