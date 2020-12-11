from operator import mul
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from tensorflow.python.ops.gen_linalg_ops import Qr
from gym_maze.envs.maze_view_2d import MazeView2D
import itertools
from sklearn.neighbors import KDTree
import time

# class myBox(spaces.Box):
    
#     def __init__(self, low, high, shape=None, dtype=np.float32):
    
#         super(myBox, self).__init__(low, high, shape, dtype)

    
#     def sample(self):
#         """
#         Generates a single random sample inside of the Box.

#         In creating a sample of the box, each coordinate is sampled according to
#         the form of the interval:

#         * [a, b] : uniform distribution
#         * [a, oo) : shifted exponential distribution
#         * (-oo, b] : shifted negative exponential distribution
#         * (-oo, oo) : normal distribution
#         """
#         high = self.high if self.dtype.kind == 'f' \
#                 else self.high.astype('int64') + 1
#         sample = np.empty(self.shape)

#         # Masking arrays which classify the coordinates according to interval
#         # type
#         unbounded   = ~self.bounded_below & ~self.bounded_above
#         upp_bounded = ~self.bounded_below &  self.bounded_above
#         low_bounded =  self.bounded_below & ~self.bounded_above
#         bounded     =  self.bounded_below &  self.bounded_above


#         # Vectorized sampling by interval type
#         sample[unbounded] = self.np_random.normal(
#                 size=unbounded[unbounded].shape)

#         sample[low_bounded] = self.np_random.exponential(
#             size=low_bounded[low_bounded].shape) + self.low[low_bounded]

#         sample[upp_bounded] = -self.np_random.exponential(
#             size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

#         sample[bounded] = self.np_random.uniform(low=self.low[bounded],
#                                             high=high[bounded],
#                                             size=bounded[bounded].shape)
#         if self.dtype.kind == 'i':
#             sample = np.floor(sample)

#         return sample.astype(self.dtype)


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N","S", "E", "W"]
    VISTED_TO_IDX = {"visited":16}

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,
        do_track_trajectories=False,num_goals = 1,verbose = True,human_mode=False, 
        measure_distance = False,n_trajs = None,random_pos = False,seed_num = None):
        
        self.measure_distance = measure_distance
        self.verbose = verbose
        self.viewer = None
        self.enable_render = enable_render
        self.num_goals = num_goals
        self.human_mode = human_mode
        self.chosen_goal = None
        self.random_pos = random_pos
        self.n_trajs = n_trajs

        self.seed(seed_num)


        if maze_file:
            print("maze from file")
            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render,
                                        num_goals = self.num_goals,random_pos = random_pos,
                                        verbose = self.verbose,np_random=self.np_random,
                                        n_trajs = n_trajs)
        # elif maze_size:
        #     if mode == "plus":
        #         has_loops = True
        #         num_portals = int(round(min(maze_size)/3))
        #     else:
        #         print("not plus")
        #         has_loops = False
        #         num_portals = 0
        #     print("maze_sized")
        #     self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
        #                                 maze_size=maze_size, screen_size=(640, 640),
        #                                 has_loops=has_loops, num_portals=num_portals,
        #                                 enable_render=enable_render,num_goals=self.num_goals,
        #                                 verbose = self.verbose,random_pos = random_pos,np_random=self.np_random)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        # Set random

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


        # if(self.measure_distance):
        #     self.assign_goal()
        self.covered_cells = self.maze_view.maze.maze_cells.copy()
        # print('self.covered_cells = self.maze_view.copy()', self.covered_cells )
        
        # Remove goals from init
        # print('self.random_init_pool: ', self.random_init_pool)

        # Simulation related variables.
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()
      

    # def assign_goal(self):
    #     self.maze_view.robot
  

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(int(seed))
        return [seed]

    def get_n_to_be_covered(self):
        n_total_cells = np.count_nonzero(self.covered_cells != 0)
        n_covered_cells = np.count_nonzero(self.covered_cells == MazeEnv.VISTED_TO_IDX["visited"])
        return n_total_cells - n_covered_cells

    def compute_reward(self,moved):
        reward = 0
        if(self.num_goals==1):
            # Single goal mode
            if np.array_equal(self.maze_view.robot, self.maze_view.goal):
                reward = 1
                done = True
            else:
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False
            if not moved:
                reward -= 1.5

            
        elif self.num_goals > 1:
            # Multiple goals mode
            for _goal in self.maze_view.goals:
                # print('self.maze_view.robot, _goal: ', self.maze_view.robot, _goal)
                if np.array_equal(self.maze_view.robot, _goal):
                    reward = 1
                    if(self.enable_render):
                        self.maze_view.color_visited_cell(_goal[0],_goal[1])
                    self.maze_view.goals.remove(_goal)
                    
            if reward < 1 : # Not found any goal
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False
            if(self.maze_view.goals==[]):

                done = True
            else:

                done = False
        else:
            # I am doing covering 
            reward = 0
            if moved and self.covered_cells[self.maze_view.robot[0],self.maze_view.robot[1]] == MazeEnv.VISTED_TO_IDX["visited"]:
                reward = -0.5
                # cell already covered
            elif moved:
                reward = 1
                # New cell covered 
                self.covered_cells[self.maze_view.robot[0],self.maze_view.robot[1]] = MazeEnv.VISTED_TO_IDX["visited"]
            
            elif not moved: # Not found any goal
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False

            if(self.verbose):
                print('self.get_n_to_be_covered(): ', self.get_n_to_be_covered())
            if self.get_n_to_be_covered() == 0:                
                done = True
            else: 
                done = False

        return reward,done

    def step(self, action):
        # print('self.maze_view.goals: ', self.maze_view.goals)
        if isinstance(action, int) or isinstance(action, np.int64):
            if(self.human_mode):
                my_action = int(input("Action (0 N,1 S,2 E,3 W, -1 random):"))
                action = my_action if my_action != -1 else action
            _moved = self.maze_view.move_robot(self.ACTION[action])
        else:
            print('type(action): ', type(action))
            _moved = self.maze_view.move_robot(action)

        reward, done = self.compute_reward(moved=_moved)

        self.state = self.maze_view.robot
        
        moved = any(self.trajectory[-1] != self.state) if len(self.trajectory)>0 else True

        if(moved):
            if(self.enable_render):
                self.maze_view.color_visited_cell(self.state[0],self.state[1])
            
            if(self.trajColFlag):
                self.trajectory.append(list(self.state))
        
        
        # print('trajectory: ', self.trajectory)
        # print('self.kdtrees: ', self.kdtrees)

        info = {}

        return self.state, reward, done, info

    def reset(self):
        if(self.num_goals > 1):
            self.maze_view.goals = self.maze_view.saved_goals.copy()
            


        self.steps_beyond_done = None
        self.done = False
        if(self.trajColFlag and self.trajectory!=[]):
            # Add points of one trajectory removing also duplicates
            self.trajectory = list(num for num,_ in itertools.groupby(self.trajectory))
            _tree = KDTree(np.array(self.trajectory))
            self.kdtrees.append(_tree)
        if(self.num_goals < 1):
            # reset covered_cells
            self.covered_cells = self.maze_view.maze.maze_cells.copy()
        
        #Redraw goals
        if(self.random_pos):
            self.state = self.maze_view.entrance 
            self.maze_view.resetEntrance()
        else:
            self.state = np.zeros(2)

        if(self.enable_render):
            if(self.num_goals < 1 ):
                # Reset all cells
                self.maze_view.maze_layer.fill((0, 0, 0, 0,))


        self.maze_view.reset_robot(rend=self.enable_render)
        
        return self.state

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self, mode="human", close=False):
        if close:
            self.maze_view.quit_game()

        return self.maze_view.update(mode)



if __name__ == "__main__" :

    env = MazeEnv( maze_file = "maze_samples/maze2d_001.npy",                  
            # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                        # maze_size=(640, 640), 
                                        enable_render=True)

    env.render()
    import time
    input("Enter any key to quit.")