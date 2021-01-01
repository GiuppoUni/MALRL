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



class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N","S", "E", "W"]
    VISTED_TO_IDX = {"visited":16}

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,
        do_track_trajectories=False,num_goals = 1,verbose = True,human_mode=False, 
        measure_distance = False,n_trajs = None,random_pos = False,seed_num = None,
        fixed_goals = None, fixed_init_pos = None,visited_cells = []):
        
        self.visited_cells = visited_cells
        self.measure_distance = measure_distance
        self.verbose = verbose
        self.viewer = None
        self.enable_render = enable_render
        self.num_goals = num_goals
        
        if(fixed_goals is not None):
            self.num_goals = len(fixed_goals)
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
                                        n_trajs = n_trajs, fixed_goals = fixed_goals,
                                        fixed_init_pos =fixed_init_pos)
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
        # print('self.action_space: ', self.action_space)
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

        self.final_points = []
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
            # Single goal modeù
            
            if not moved:
                return -0.1,False
            if np.array_equal(self.maze_view.robot, self.maze_view.goal):
                # Found goal
                reward = 1000
                done = True
            else:
                #  Not found goal
                if self.maze_view.robot.tolist() in self.visited_cells:
                    reward = -5
                else:
                    reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False

            
        elif self.num_goals > 1:
            # Multiple goals mode
            if not moved:
                return -5,False
            
            for _goal in self.maze_view.goals:
                # print('self.maze_view.robot, _goal: ', self.maze_view.robot, _goal)
                if np.array_equal(self.maze_view.robot, _goal):
                    if(self.enable_render):
                        self.maze_view.color_visited_cell(_goal[0],_goal[1])
                    reward = 300
                    self.maze_view.goals.remove( _goal )
                    break

            if reward < 1 : # Not found any goal
                reward = -1/(self.maze_size[0]*self.maze_size[1])
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
            
            # print('action: ', action)
            _moved = self.maze_view.move_robot(self.ACTION[action])
        else:
            print('type(action): ', type(action))
            _moved = self.maze_view.move_robot(action)


        reward, done = self.compute_reward(moved=_moved)

        self.state = [ int(x) for x in self.maze_view.robot]
        
        
        if(_moved):
            if(self.enable_render):
                self.maze_view.color_visited_cell(self.state[0],self.state[1])
            
            if(self.trajColFlag):
                self.trajectory.append(list(self.state))
                self.final_points.append(list(self.state))
        
        # print('trajectory: ', self.trajectory)
        # print('self.kdtrees: ', self.kdtrees)

        info = {"moved": _moved }

        # print('self.state: ', self.state,_moved)
        return self.maze_view.robot, reward, done, info

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

    def set_render(self):
        self.enable_render = True


if __name__ == "__main__" :

    env = MazeEnv( maze_file = "maze_samples/maze2d_001.npy",                  
            # maze_file="maze"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M') ),
                                        # maze_size=(640, 640), 
                                        enable_render=True)

    env.render()
    import time
    input("Enter any key to quit.")