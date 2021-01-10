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

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,num_goals = 1,verbose = True,human_mode=False, 
        n_trajs = None,random_pos = False,seed_num = None,
        fixed_goals = None, fixed_init_pos = None,visited_cells = []):
        
        self.visited_cells = visited_cells
        self.verbose = verbose
        self.viewer = None
        self.enable_render = enable_render
        self.num_goals = num_goals
        self.fixed_init_pos = fixed_init_pos
        self.fixed_goals = fixed_goals

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
                                        n_trajs = None, fixed_goals = fixed_goals,
                                        fixed_init_pos =fixed_init_pos)

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
        
        self.final_points = []
 
        self.covered_cells = self.maze_view.maze.maze_cells.copy()
   
        self.reset()

        self.configure()
      

  


    

    def setNewEntrance(self,entrance):
        self.fixed_init_pos = entrance
        self.maze_view.fixed_init_pos = entrance
        self.maze_view.setEntrance ( np.array(entrance) ) 
        print('self.visited_cells: ', len(self.visited_cells))
    
    def setNewGoals(self,goals):
        # TODO make multi
        self.fixed_goals = goals
        self.maze_view.fixed_goals = goals
        self.maze_view.setGoal(  np.array(goals[0]) ) 
        
    def setVisitedCells(self,cells):
        self.visited_cells = cells


    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        # self.np_random, seed = seeding.np_random(int(seed))
        np.random.seed(seed=seed)
        self.np_random = np.random
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
                return -0.5,False
            if np.array_equal(self.maze_view.robot, self.maze_view.goal):
                # Found goal
                reward = 1000
                done = True
            else:
                #  Not found goal
                if self.maze_view.robot.tolist() in self.visited_cells:
                    reward = -1
                else:
                    # Should be lower than not moved case
                    reward = -0.5/(self.maze_size[0]*self.maze_size[1])
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
            
        
        
        # print('trajectory: ', self.trajectory)
        # print('self.kdtrees: ', self.kdtrees)

        info = {"moved": _moved }

        # print('self.state: ', self.state,_moved)
        return self.maze_view.robot, reward, done, info

    def reset(self):
        if(self.num_goals > 1):
            self.maze_view.goals = self.maze_view.saved_goals.copy()
        elif(self.num_goals < 1):
            # reset covered_cells
            self.covered_cells = self.maze_view.maze.maze_cells.copy()

        self.steps_beyond_done = None
        self.done = False

        
        #Redraw entrance
        if(self.random_pos):
            self.state = self.maze_view.entrance 
            self.maze_view.resetEntrance()
        elif (self.fixed_init_pos is not None):
            self.state = self.maze_view.entrance         
        else:
            self.state = np.zeros(2)

        if(self.enable_render and self.num_goals < 1 ):
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
