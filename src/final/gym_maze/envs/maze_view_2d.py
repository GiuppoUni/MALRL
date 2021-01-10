import pygame
import random
import numpy as np
import os

from collections.abc import Iterable

class MazeView2D:

    def __init__(self, maze_name="Maze2D", maze_file_path=None,
                 maze_size=(30, 30), screen_size=(600, 600),
                 has_loops=False, num_portals=0, enable_render=True,num_goals = 1,verbose = True,
                 random_pos=False,np_random=None,n_trajs=None, fixed_goals = None,
                 fixed_init_pos = None):

        # if(num_goals<=0 ):
        #     raise ValueError("Error in num_goals parameter")
        self.random_pos = random_pos
        self.fixed_goals = fixed_goals 
        self.fixed_init_pos = fixed_init_pos
        
        self.num_goals = num_goals
        self.verbose = verbose
        self.np_random = np_random
        self.n_trajs = n_trajs

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(maze_name)
        self.clock = pygame.time.Clock()
        self.__game_over = False
        self.__enable_render = enable_render
        
        # Load a maze
        if maze_file_path is None:
            print(" no maze filepath")
            self.__maze = Maze(maze_size=maze_size, has_loops=has_loops, num_portals=num_portals,verbose = self.verbose)
        else:
            if not os.path.exists(maze_file_path):
                print("New maze")
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "maze_samples", maze_file_path)
                if os.path.exists(rel_path):
                    maze_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % maze_file_path)
            self.__maze = Maze(maze_cells=Maze.load_maze(maze_file_path),verbose = self.verbose)

        self.maze_size = self.__maze.maze_size
        if self.__enable_render is True:
            # to show the right and bottom border
            self.screen = pygame.display.set_mode(screen_size)
            self.__screen_size = tuple(map(sum, zip(screen_size, (-1, -1))))

       

        # Set the Goal
        print('num_goals: ', num_goals)
        self.__goal = None
        if self.num_goals == 1:        
            print('=========================================')
            if(self.fixed_goals is None):
                self.__goal = np.array(self.maze_size) - np.array((1, 1))
            else:
                if( len(self.fixed_goals) == 1):
                    self.__goal = np.array(self.fixed_goals)[0]
                else:
                    idx = self.np_random.choice(range(len(fixed_goals)) )
                    self.__goal =  np.array(self.fixed_goals[idx]) 
            self.goals = [self.__goal]

            
        # Set multiple random goals
        elif self.num_goals > 1:
            self.goals = self.init_goals()
            self.saved_goals = self.goals.copy()


        if self.random_pos:
            # Set the starting point
            _arr= self.get_init_pool()
            self.np_random.shuffle( _arr )
            if(self.n_trajs and len(_arr)-self.n_trajs>0):
                self.random_init_pool = _arr[0:self.n_trajs]  
                print('self.random_init_pool: ', self.random_init_pool)
            else:
                self.random_init_pool = _arr 
            self.__entrance = self.random_init_pool[self.np_random.choice(self.random_init_pool.shape[0])]

            # it s set on reset
        elif self.fixed_init_pos is not None:
            self.__entrance = self.fixed_init_pos
        else:
            self.__entrance = np.zeros(2)

        # Create the Robot
        self.__robot = self.entrance.copy()


        if self.__enable_render is True:
            # Create a background
            self.background = pygame.Surface(self.screen.get_size()).convert()
            self.background.fill((255, 255, 255))

            # Create a layer for the maze
            self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
            self.maze_layer.fill((0, 0, 0, 0,))

            # show the maze
            self.__draw_maze()

            # # show the portals
            # self.__draw_portals()

            # # show the robot
            # self.__draw_robot()

            # # show the entrance
            # self.__draw_entrance()

            # # show the goal
            # self.__draw_goal()
    
    def get_init_pool(self):
        rnd= np.where((self.maze.maze_cells!=0) & 
        ((self.maze.maze_cells == 7)|(self.maze.maze_cells == 15) |(self.maze.maze_cells  ==13 )) )
        
        rs= rnd[0]
        cs= rnd[1]
        arr = np.array( [[rs[i],cs[i]] for i in range(len(rs)) ] )
                
        np.setdiff1d(arr, self.goals) 
        return arr

    def _get_random_xy(self):
        r = self.np_random.choice( np.arange(1,self.maze_size[0]),1 )
        c = self.np_random.choice( np.arange(1,self.maze_size[1]),1 )
        while not any(self.maze.get_walls_status(self.maze.maze_cells[c, r]).values() ):
            r = self.np_random.choice( np.arange(1,self.maze_size[0]),1 )
            c = self.np_random.choice( np.arange(1,self.maze_size[1]),1 )
        return [int(r),int(c)]

    def init_goals(self):
        if self.num_goals > 0:
            # Not covering
            if(self.fixed_goals is None):
                return [self._get_random_xy() for _ in range(self.num_goals) ]  
            else:
                return [ [cell[0],cell[1]] for cell  in self.fixed_goals ]  

        else:
            return []

    def resetEntrance(self):
        if self.__enable_render: self.decolor(self.__entrance)
        self.__entrance =  self.random_init_pool[self.np_random.choice(self.random_init_pool.shape[0])]

  

    def update(self, mode="human"):
        
        try:
            img_output = self.__view_update(mode)
            self.__controller_update()
        except Exception as e:
            self.__game_over = True
            self.quit_game()
            raise e
        else:
            return img_output

    def quit_game(self):
        try:
            self.__game_over = True
            if self.__enable_render is True:
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    def tr(self,dir):
        if dir=="N": return "UP"
        elif dir=="S": return "DOWN"
        elif dir=="E": return "RIGHT"
        elif dir=="W": return "LEFT"
        else: raise ValueError("Not acceptable dir") 


    # TODO
    # def setGoals(self,goals):
    #     self.goals = goals

    def setGoal(self,goal):
        if self.__enable_render: self.decolor(self.__goal)
        self.__goal = goal
        self.goals = [goal]

    def setEntrance(self,ent):
        if self.__enable_render: self.decolor(self.__entrance)
        self.__entrance = ent
        


    def move_robot(self, dir):
        if dir not in self.__maze.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.COMPASS.keys())))

        # print('self.entrance: ', self.entrance,self.goals,self.goal)

        moved = False
        if self.__maze.is_open(self.__robot, dir):
            if(self.verbose):
                print("MOVING:", self.tr(dir),"\n")    
            # update the drawing
            if self.__enable_render:
                self.__draw_robot(transparency=0)

            # move the robot
            self.__robot += np.array(self.__maze.COMPASS[dir])
            # if it's in a portal afterward
            if self.maze.is_portal(self.robot):
                self.__robot = np.array(self.maze.get_portal(tuple(self.robot)).teleport(tuple(self.robot)))
            
            if self.__enable_render:
                self.__draw_robot(transparency=255)
            moved = True
            
        return moved
    def reset_robot(self,rend):
        
        if rend: self.__draw_robot(transparency=0)
        self.__robot = self.entrance.copy()
        # if rend: self.__draw_robot(transparency=255)

    def __controller_update(self):
        if not self.__game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.__game_over = True
                    self.quit_game()

    def __view_update(self, mode="human"):
        if not self.__game_over:
            # update the robot's position
            self.screen.blit(self.background, (0, 0))
            
            self.__draw_entrance()
            # print('__draw_entrance: ', self.__entrance)
            # print('__draw_entrance: ', self.robot)
            
            self.__draw_robot()
            self.__draw_goal()
            self.__draw_portals()


            # update the screen
            self.screen.blit(self.maze_layer,(0, 0))

            if mode == "human":
                pygame.display.flip()

            return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def __draw_maze(self):
        
        if self.__enable_render is False:
            return
        
        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

        # breaking the walls
        for x in range(len(self.maze.maze_cells)):
            for y in range (len(self.maze.maze_cells[x])):
                # check the which walls are open in each cell

                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x, y])
                # print('walls_status: ', walls_status, "| cell:",x,y)
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, y, x, dirs, colour=(0, 0, 255, 15)):

        if self.__enable_render is False:
            return
        
        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")
        
        # print("DIRS",dirs,"| cell:",x,y)
        
        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_robot(self, colour=(0, 0, 150), transparency=255):

        if self.__enable_render is False:
            return
        
        x = int(self.__robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self.__robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5)

        pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):
        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)

    def __draw_goal(self, colour=(255, 255, 0), transparency=235):
        if(self.num_goals == 1):
            self.__colour_cell(self.goal, colour=colour, transparency=transparency)
        elif self.num_goals > 1:
            # print("goals",self.goals)
            [self.__colour_cell( _g,  colour=colour, transparency=transparency) for _g in self.goals]

    def __draw_portals(self, transparency=160):

        if self.__enable_render is False:
            return
        
        colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
        colour_i = 0
        for portal in self.maze.portals:
            colour = ((100 - colour_range[colour_i])% 255, colour_range[colour_i], 0)
            colour_i += 1
            for location in portal.locations:
                self.__colour_cell(location, colour=colour, transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if self.__enable_render is False:
            return

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))
        
    def decolor(self,cell):
        r = cell[0]
        c = cell[1]
        
        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(r * self.CELL_W + 0.5 + 1)
        y = int(c * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)

        cc = self.maze_layer.get_at((x, y))
        # print('cc: ', cc)

        rgba_colour = (255,255,255,255)  
        
        pygame.draw.rect(self.maze_layer, rgba_colour , (x, y, w, h))


    up_till_255 = lambda c : c+1 if c+1 <255 else 255
    def color_visited_cell(self,r,c):
        cell = [r,c]
        
        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(r * self.CELL_W + 0.5 + 1)
        y = int(c * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)

        cc = self.maze_layer.get_at((x, y))
        # print('cc: ', cc) 

        rgba_colour = (120,0,0,30) if cc[0] == 0 or cc[0:2]== (255,255)  else (120,0,0, MazeView2D.up_till_255(cc[3]) ) 
        
        pygame.draw.rect(self.maze_layer, rgba_colour , (x, y, w, h))

    @property
    def maze(self):
        return self.__maze

    @property
    def robot(self):
        return self.__robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def game_over(self):
        return self.__game_over

    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)


class Maze:

    COMPASS = {
        "N": (0, -1),
        "E": (1, 0),
        "S": (0, 1),
        "W": (-1, 0)
    }

    def __init__(self, maze_cells=None, maze_size=(10,10), has_loops=True, num_portals=0,verbose = True):

        # maze member variables
        self.maze_cells = maze_cells
        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals
        self.verbose = verbose

        # Use existing one if exists
        if self.maze_cells is not None:
            print("not none cells")
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                print("corret 2d array")
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.maze_size = maze_size

            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)

    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # Initializing constants and variables needed for maze generation
        current_cell = (random.randint(0, self.MAZE_W-1), random.randint(0, self.MAZE_H-1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                    #if self.num_walls_broken(self.maze_cells[x1, y1]) <= 1:
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(num_portal_sets=self.num_portals, set_size=2)

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H*self.MAZE_W*percent))
        cell_ids = random.sample(range(self.MAZE_W*self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id/self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x, y] = self.__break_walls(self.maze_cells[x, y], dir)
                    break

    def __set_random_portals(self, num_portal_sets, set_size=2):
        # find some random cells to break
        num_portal_sets = int(num_portal_sets)
        set_size = int(set_size)

        # limit the maximum number of portal sets to the number of cells available.
        max_portal_sets = int(self.MAZE_W * self.MAZE_H / set_size)
        num_portal_sets = min(max_portal_sets, num_portal_sets)

        # the first and last cells are reserved
        cell_ids = random.sample(range(1, self.MAZE_W * self.MAZE_H - 1), num_portal_sets*set_size)

        for i in range(num_portal_sets):
            # sample the set_size number of sell
            portal_cell_ids = random.sample(cell_ids, set_size)
            portal_locations = []
            for portal_cell_id in portal_cell_ids:
                # remove the cell from the set of potential cell_ids
                cell_ids.pop(cell_ids.index(portal_cell_id))
                # convert portal ids to location
                x = portal_cell_id % self.MAZE_H
                y = int(portal_cell_id / self.MAZE_H)
                portal_locations.append((x,y))
            # append the new portal to the maze
            portal = Portal(*portal_locations)
            self.__portals.append(portal)

            # create a dictionary of portals
            for portal_location in portal_locations:
                self.__portals_dict[portal_location] = portal

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        
        x1 = int(cell_id[0] + self.COMPASS[dir][0])
        y1 = int(cell_id[1] + self.COMPASS[dir][1])


        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            row = int(cell_id[1])
            col = int(cell_id[0])
            # check if the wall is opened

            # print('self.get_walls_status(self.maze_cells[row, col])[dir]: ', row, col,self.maze_cells )
            this_wall = bool(self.get_walls_status(self.maze_cells[row, col])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1, y1])[self.__get_opposite_wall(dir)])
            if(self.verbose):
                print('col: ', col, 'row: ', row )
                print('self.maze_cell: ', self.maze_cells[row, col])
                print('this_wall: ', this_wall)
                print('other_wall: ', other_wall)
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    def is_portal(self, cell):
        return tuple(cell) in self.__portals_dict

    @property
    def portals(self):
        return tuple(self.__portals)

    def get_portal(self, cell):
        if cell in self.__portals_dict:
            return self.__portals_dict[cell]
        return None

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "N" : (cell & 0x1) >> 0, #/1
            "E" : (cell & 0x2) >> 1, #/2
            "S" : (cell & 0x4) >> 2, #/4
            "W" : (cell & 0x8) >> 3, #/8
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs

class Portal:

    def __init__(self, *locations):

        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError("location must be a list or a tuple.")

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations


if __name__ == "__main__":

    maze = MazeView2D(screen_size= (500, 500), maze_size=(10,10))
    maze.update()
    input("Enter any key to quit.")

    


