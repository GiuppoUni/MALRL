from gym_maze.envs.multi_maze_view_2d import MultiMazeView2D
from operator import mul
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze.envs.maze_view_2d import MazeView2D
import itertools
from sklearn.neighbors import KDTree

import time

class MultiAgentActionSpace(list):
    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]


class MultiMazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION = ["N","S", "E", "W"]
    VISTED_TO_IDX = {"visited":16}

    def __init__(self, maze_file=None, maze_size=None, mode=None, enable_render=True,
        do_track_trajectories=False,num_goals = 1,verbose = True,human_mode=False, measure_distance = False,
        n_agents=1,partial_obs = False,sleep_secs=None):
        
        self.n_agents= n_agents
        self.partial_obs = partial_obs 
        self.measure_distance = measure_distance
        self.verbose = verbose
        self.viewer = None
        self.enable_render = enable_render
        self.num_goals = num_goals
        self.human_mode = human_mode
        self.chosen_goal = None

        self.sleep_secs = sleep_secs

        self.radius = 3


        if maze_file:
            print("maze from file")
            self.maze_view = MultiMazeView2D(n_agents = self.n_agents,maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640), 
                                        enable_render=enable_render,
                                        num_goals = self.num_goals,verbose = self.verbose)
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size)/3))
            else:
                print("not plus")
                has_loops = False
                num_portals = 0
            print("maze_sized")
            self.maze_view = MultiMazeView2D(n_agents = self.n_agents,maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals,
                                        enable_render=enable_render,num_goals=self.num_goals,verbose = self.verbose)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        # forward or backward in each dimension
        # self.action_space = spaces.Discrete(2*len(self.maze_size))

        # observation is the x, y coordinate of the grid
        low = np.zeros(len(self.maze_size), dtype=int)
        high =  np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        # self.observation_space = spaces.Box(low, high, dtype=np.int64)


        if partial_obs:
            agent_view_size = 20
            self.states_space = spaces.Box(
                low=low,
                high=high,
                shape=(agent_view_size, agent_view_size, self.n_agents),
                dtype='uint8'
            )

        else:
            # TODO get bounding box from tall obstacles at the margins
            
            self.states_space= spaces.Box(
              
                low = 0,
                high = 255,
                shape=(self.maze_size[0], self.maze_size[1], self.n_agents),
                dtype='uint8'
            )
        
        self.states = [np.zeros(2, dtype=np.uint8) for _ in range(n_agents)] 
        
        # forward or backward in each dimension
        self.action_space = MultiAgentActionSpace([spaces.Discrete(2*len(self.maze_size)) for _ in range(n_agents)])
		

        self.episodeN = 0
        self.stepN = 0 
        self._agents_dones = [ False for _ in range(n_agents)]


        # Past trajectories variables
        self.trajColFlag = do_track_trajectories
        self.kdtrees = []
        self.trajectories = {}

        # Simulation related variables.
        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self.configure()
        # if(self.measure_distance):
        #     self.assign_goal()

        self.covered_cells = self.maze_view.maze.maze_cells.copy()
        self.init_states()
        self.old_pos = self.states.copy()
        print('self.covered_cells = self.maze_view.copy()', self.covered_cells )

    # def assign_goal(self):
    #     self.maze_view.robot

    def init_states(self):
        # TODO make it random
        for i,s in enumerate(self.states):
            if i == 0: s = [0,0]
            elif i==1: s= [self.maze_size[0]-1,0]
            elif i==2: s = [0,self.maze_size[1]-1]
            elif i==3: s = [self.maze_size[0]-1,self.maze_size[1]-1]

    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def configure(self, display=None):
        self.display = display

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_n_to_be_covered(self):
        n_total_cells = np.count_nonzero(self.covered_cells != 0)
        n_covered_cells = np.count_nonzero(self.covered_cells == MultiMazeEnv.VISTED_TO_IDX["visited"])
        return n_total_cells - n_covered_cells

    def compute_reward(self,moved,agent_i):
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
                        self.maze_view.decolor(_goal)
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
            
            if moved and self.covered_cells[self.maze_view.robots[agent_i][0],self.maze_view.robots[agent_i][1]] == MultiMazeEnv.VISTED_TO_IDX["visited"]:
                reward = -0.1
                # cell already covered
            elif moved:
                reward = 0.5
                # New cell covered 
                self.covered_cells[self.maze_view.robots[agent_i][0],self.maze_view.robots[agent_i][1]] = MultiMazeEnv.VISTED_TO_IDX["visited"]
                done = False
            elif not moved:
                reward = -0.1/(self.maze_size[0]*self.maze_size[1])
                done = False
            if(moved and int(self.tree.query_radius([self.maze_view.robots[agent_i]], r=self.radius, count_only=True)) > 0):
                print("COLLISION",agent_i)
                reward = -2


            if(self.verbose):
                print('self.get_n_to_be_covered(): ', self.get_n_to_be_covered())
            if self.get_n_to_be_covered() == 0:                
                done = True
            else: 
                done = False

        return reward,done

    def step(self, actions):
        rewards = np.zeros(self.n_agents)
        dones = [False for _ in range(self.n_agents)]
        for agent_i,action in enumerate(actions):
            other_robots = [s for s in self.maze_view.robots if any(np.not_equal(s,self.maze_view.robots[agent_i])) ]
            self.tree = KDTree(np.array(other_robots))
            # Save old pos of each agent
            self.old_pos[agent_i] = self.maze_view.robots[agent_i].copy()
            if isinstance(action, int) or isinstance(action, np.int64):
                if(self.human_mode):
                    my_action = int(input("Action (0 N,1 S,2 E,3 W, -1 random):"))
                    action = my_action if my_action != -1 else action
                _moved = self.maze_view.move_robot(self.ACTION[action],agent_i)
            else:
                print('type(action): ', type(action))
                _moved = self.maze_view.move_robot(action,agent_i)

            rewards[agent_i], dones[agent_i] = self.compute_reward(moved=_moved,agent_i=agent_i)
            
            self.states[agent_i] = self.maze_view.robots[agent_i]
   
   
            
            moved = any(np.not_equal(self.old_pos[agent_i], self.states[agent_i])) 

            if(moved):
                if(self.enable_render):
                    self.maze_view.color_visited_cell(self.states[agent_i][0],self.states[agent_i][1])
                
                if(self.trajColFlag):
                    self.trajectories[agent_i].append(list(self.states))
            
            
            # print('trajectory: ', self.trajectory)
            # print('self.kdtrees: ', self.kdtrees)

        info = {}
        if(self.sleep_secs):
            time.sleep(self.sleep_secs)
        return self.states, rewards, dones, info

    def reset(self):
        if(self.num_goals > 1):
            self.maze_view.goals = self.maze_view.saved_goals.copy()
            print('self.maze_view.saved_goals: ', self.maze_view.saved_goals)
            

        
        self.maze_view.reset_robot()
        self.init_states()
        self.old_pos = self.states.copy() 

        self.steps_beyond_done = None
        self.done = False
        for agent_i in range(self.n_agents):
            if(self.trajColFlag and self.trajectories[agent_i]!=[]):
                # Add points of one trajectory removing also duplicates
                self.trajectories[agent_i] = list(num for num,_ in itertools.groupby(self.trajectories[agent_i]))
                _tree = KDTree(np.array(self.trajectories[agent_i]))
                self.kdtrees.append(_tree)
        if(self.num_goals < 1):
            # reset covered_cells
            self.covered_cells = self.maze_view.maze.maze_cells.copy()
        
        #Redraw goals
        if(self.render):
            # if(self.num_goals < 1 ):
            #     # Reset all cells
            #     self.maze_view.maze_layer.fill((0, 0, 0, 0,))

            self.maze_view.update()
            
        return self.states

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