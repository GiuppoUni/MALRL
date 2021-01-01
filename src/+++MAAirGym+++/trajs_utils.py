import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

import scipy.interpolate
import utils
from sklearn.neighbors import KDTree





SEED = 668
N_DRONES=5
N_POINTS = 100
STEP_SIZE = 20
ACTION = ["N","S", "E", "W"]
AINDEX = {"N":0,"S":1, "E":2, "W":3}


random.seed(SEED)
np.random.seed(seed=SEED)




def myInterpolate2D(trajs, n_samples=10,step_size=20 ):
    n_collisions = []
    for arr in trajs:
        res_t = []
        for i,p in enumerate(arr):
     
            if(i+1 >= len(arr)):
                break
            x1,y1 = p[0], p[1]
            x2,y2 = arr[i+1][0], arr[i+1][1] 
            # if(i==0):
            #     res_t.append([x1,y1])
            length = max(abs(x2-x1),abs(y2-y1))
            samples = math.floor(length/step_size)  
            print("|||")
            for i in range(samples):
                if(x2 > x1):
                    # Moved on the right
                    new_p = [x1 + i*step_size , y1]
                elif (x1 > x2):
                    # Moved left
                    new_p = [x1 - i*step_size  , y2]
                elif (y2 > y1):
                    # Moved left
                    new_p = [x1, y1 + i*step_size ]
                elif (y1 > y2):
                    # Moved left
                    new_p = [x2, y1 - i*step_size ]
                else:
                    raise Exception("Uncommmon points")
                print('new_p: ', new_p)
                res_t.append(new_p)
            if(length % step_size != 0):
                # last_step = length - step_size * samples
                print("last")
 
                if(x2 > x1):
                    # Moved on the right
                    new_p = [x2, y1]
                elif (x1 > x2):
                    # Moved left
                    new_p = [x1, y2]
                elif (y2 > y1):
                    # Moved left
                    new_p = [x1, y2]
                elif (y1 > y2):
                    # Moved left
                    new_p = [x2, y1]
                else:
                    raise Exception("Uncommmon points")
                print('new_pL: ', new_p)
                res_t.append(new_p)
            
            

        n_collisions.append(res_t)
    return n_collisions
            

def allocate_height(trajectories,max_height,min_height,sep_h):
    Tmax = max([len(traj) for traj in trajectories])
    drones = range(len(trajectories))
    points = {}
    trajs_3d =[[] for d in drones] 
    for d in drones:
        for t in range(len(trajectories[d])):
                            
            point = tuple(trajectories[d][t])

            if point not in points:
                new_z = max_height
                points[point] = [max_height]
            else:
                new_z = points[point][-1] - sep_h
                points[point].append(new_z)
                if new_z < min_height:
                    raise Exception("Out of height bounds")

            trajs_3d[d].append(list(point)+[new_z])
            # print("d:",d,'point: ', point,"z",new_z)
    return trajs_3d


def build_trees(trajectories):
    _trees = []
    for traj in trajectories:
        _trees.append(KDTree(np.array(traj)))
    return _trees

def check_trees_collision(fId,point,trees,radius):
    n_collisions = 0
    for idx,_tree in enumerate(trees): 
        if(idx == fId):
            # E' quella attuale
            continue
        n_collisions = _tree.query_radius( [point],r=radius,count_only = True )
        if(n_collisions > 0):
            return n_collisions
    return n_collisions

def complex_avoid_collision_in_busy_space(trajs_2d,assigned_trajs,min_height,max_height,
    sep_h,min_safe_points,radius=30,simpleMode=True):
    print("Started col avoidance in busy space...")
    # First remove the points not interesting by height constraints
    assigned_points = []
    for traj in assigned_trajs:
        for p in traj:
            if min_height<=p[2]<=max_height : assigned_points.append(p)
    
    zs = [[] for t in range(len(trajs_2d))]
    trajs_3d =   []
    for traj in trajs_2d:
        traj_3d=[]
        for p in traj:
            traj_3d.append(p+[max_height] )
        trajs_3d.append(traj_3d)
    
    static_tree = build_trees([assigned_points])[0]
    dynamic_trees = build_trees(trajs_3d)

    for fId, traj in enumerate(trajs_3d):
        for point in traj:
            n_collisions = static_tree.query_radius( [point],r=radius,count_only = True )
            if(n_collisions > 0): point[2]-= sep_h

            for idx,_tree in enumerate(dynamic_trees): 
                if(idx == fId):
                    # E' quella attuale
                    continue
                n_collisions = _tree.query_radius( [point],r=radius,count_only = True )
                if(n_collisions > 0 ):
                    point[2] -= sep_h
                    _tree = KDTree(np.array(traj))
                n_collisions = check_trees_collision(fId=fId,point=point,trees=dynamic_trees,radius=radius)
                if(n_collisions > 0 ):
                    point[2] -= sep_h
                    _tree = KDTree(np.array(traj))
                while(check_trees_collision(fId=fId,point=point,trees=dynamic_trees,radius=radius)>0):
                    point[2] -= sep_h
                    _tree = KDTree(np.array(traj))

                zs[fId].append(point[2])
                # TODO check on min heigth
    print("Avoidance completed.")
    return trajs_3d,zs


def _d_insert(dic,k,v):
    if k not in dic:
        dic[k] = [v]
    else:
        dic[k].append(v)

def _vertical_allocate(a_heights:dict,p_trajs,p_heights:dict,trees,mh):
    pass

def avoid_collision_in_busy_space(trajs_2d,assigned_trajs,min_height,max_height,
    sep_h,min_safe_points,radius=30,simpleMode=True):
    
    '''
    - trajectories are [[p11,p12,...],[p21,p22,...],...]
      pij is a point j-th in 2d: [x,y]  for i-th flight
    - read only assigned_trajs are [[p11,p12,...],[p21,p22,...],...]
      pij is a point j-th in 3d: [x,y,z]  for i-th flight (they are assumed to be generated compatible with this algo)
    - trees are built using build_trees function, they are k-d trees 
    - max_height,min_height are bounds for allocation, it starts from max and allocates towards min
    - sep_h is the amount of height separating two trajectories with same x,y
    - min_safe_points is used in simpleMode False, it counts amount of contiguos point out of collision space
    - radius for collision check using trees
    - simpleMode is a flag to allocate height as one height for to a flight till the end in case of collision
        or if its set to False it assign the heigth only near the collision.
    
    '''
    if(trajs_2d == []): return []

    assigned_heights = dict()
    static_trees = build_trees(assigned_trajs)
    for ffid in assigned_trajs:
        zref = ffid[0][2] #first point [0] third coo. [2]
        _d_insert(assigned_heights,zref,ffid)
    
    trees_2d = build_trees(trajs_2d)

    trajs_3d = [[] for t in range(len(trajs_2d))]
   
    colliding_trajs = dict()
    for ti,traj in enumerate(trajs_2d):
        traj_3d=[]
        for p in traj:
            for idx,_tree in enumerate(trees_2d): 
                if(idx == ti):
                # E' quella attuale
                    continue
                n_collisions = _tree.query_radius( [ p ],r=radius,count_only = True )
                if n_collisions > 0:
                    print("Collisions with","Trajectory_"+str(idx))
                    print("\tcomputed from trajectory ",ti,", point", p)
                    _d_insert(colliding_trajs,ti,idx)

 
    proposed_heights = dict() # Dict with k = height value, v = array of fligths assigned to that height
   
    # Allocate 2D trajs colliding 
    priorities = list(colliding_trajs.keys())
    priorities.sort()
    for fId in priorities:
        offset = priorities.index(fId)
        new_z = max_height - offset * sep_h 
        if new_z < min_height:
            raise Exception("Traj ",fId," out of height bounds")
    
        trajs_3d[fId] = [ p+[new_z] for p in trajs_2d[fId] ]
        _d_insert(proposed_heights,new_z,fId)        

    # Allocate remaining trajs (not colliding)
    not_colliding_fids= set( range(len(trajs_2d))) - set(priorities) 
    for fId in not_colliding_fids:
        safe_z = max_height - len(priorities) * sep_h 
        trajs_3d[fId] = [ p+[safe_z] for p in trajs_2d[fId] ]
        _d_insert(proposed_heights,safe_z,fId)        

    # Check on same height collision bwin 2d in input and 3d preexisting
    for z in proposed_heights:
        for busy_z in assigned_heights:
            if(z == busy_z):
                for fid in proposed_heights[z]:
                    is_colliding = False
                    for point in trajs_2d[fid]:
                        for ffid in assigned_heights[busy_z]:
                            n_collisions = static_trees[ffid].query_radius( [ point ],r=radius,count_only = True )
                            if(n_collisions > 0):
                                # Need to move down to new z the traj 2d
                                is_colliding = True
                                break
                        if(is_colliding):
                            break
                    if(is_colliding):
                        # Ho trovato almeno una collisione con traiettorie fisse su stesso z
                        proposed_heights[z].remove(fid)
                        new_z = z - sep_h
                        if(new_z < min_height):
                            raise Exception("Out of min height bound")
                        proposed_heights[z].append(fid)
                        # Ora devo controllare sia che non collida con le busy sia che non collida con le proposed su new_z    
                        _vertical_allocate(assigned_heights,trajs_2d,proposed_heights,static_trees,min_height)
            
    # # Now we need to distanciate considering also 3d pre existing trajectories
    # trees_3d = build_trees(trajs_3d)
    # new_trajs_3d = [[] for _ in range(len(trajs_3d))]
    # for ti,traj in enumerate(trajs_2d):
    #     traj_3d=[]
    #     for p in traj:
    #         for idx,_tree in enumerate(static_tree): 
    #             if(idx == ti):
    #             # E' quella attuale
    #                 continue
    #             n_collisions = _tree.query_radius( [ p ],r=radius,count_only = True )
    #             if n_collisions > 0:
    #                 print("Collisions with","Trajectory_"+str(idx))
    #                 print("\tcomputed from trajectory ",ti,", point", p) 
    #                 if(ti not in colliding_trajs):
    #                     colliding_trajs[ti]=[idx]
    #                 elif idx not in colliding_trajs[ti]:
    #                     colliding_trajs[ti].append(idx)

    return trajs_3d

def avoid_collision_in_empty_space(trajectories,min_height,max_height,
    sep_h,min_safe_points,radius=30,simpleMode=True):
    """
    - trajectories are [[p11,p12,...],[p21,p22,...],...]
    - pij is a point j-th in 2d: [x,y] or 3d: [x,y,z] for i-th flight
    - trees are built using build_trees are k-d trees 
    - max_height,min_height are bounds for allocation, it starts from max and allocates towards min
    - sep_h is the amount of height separating two trajectories with same x,y
    - min_safe_points is used in simpleMode False, it counts amount of contiguos point out of collision space
    - radius for collision check using trees
    - simpleMode is a flag to allocate height as one height for to a flight till the end in case of collision
        or if its set to False it assign the heigth only near the collision.
    """
    trees = build_trees(trajectories)

    Tmax = max([len(traj) for traj in trajectories])
    drones = range(len(trajectories))
    points = {}
    zs=[[] for d in drones] 
    trajs_3d =[[] for d in drones] 
    colliding_trajs = dict()
    for d in drones:
        for t in range(len(trajectories[d])):
            point = tuple(trajectories[d][t])
            n_safe_points = 0
            n_collisions = 0
            for idx,_tree in enumerate(trees): 
                if(idx == d):
                    # E' quella attuale
                    continue
                n_collisions = _tree.query_radius( [point],r=radius,count_only = True )
                if n_collisions > 0:
                    print("Collisions with","Trajectory_"+str(idx))
                    print("\tcomputed from trajectory ",d,", point", point)
                    if(d not in colliding_trajs):
                        colliding_trajs[d]=[idx]
                    elif idx not in colliding_trajs[d]:
                        colliding_trajs[d].append(idx)
                  
                        
            if not simpleMode and n_collisions == 0:
                # TODO count and cooldown
                n_safe_points +=1
            if(n_safe_points >= min_safe_points):
                colliding_trajs[d] = []

            print("colliding_trajs",colliding_trajs)
            if(d not in colliding_trajs or colliding_trajs[d]==[]):
                new_z = max_height
            else:
                priorities = [d]+colliding_trajs[d]
                priorities.sort()
                offset = priorities.index(d)
                new_z = max_height - offset * sep_h 
                if new_z < min_height:
                    raise Exception("Out of height bounds")
            trajs_3d[d].append(list(point)+[new_z])
            zs[d].append(new_z)
    return trajs_3d,zs

    #         if point not in points:
    #             new_z = max_height
    #             points[point] = [max_height]
    #         else:
    #             new_z = points[point][-1] - sep_h
    #             points[point].append(new_z)
    #             if new_z < min_height:
    #                 raise Exception("Out of height bounds")

    #         trajs_3d[d].append(list(point)+[new_z])
    #         # print("d:",d,'point: ', point,"z",new_z)






def random_trajs():
    n_drones = N_DRONES
    step = 120
    trajs = [[] for i in range(n_drones)]
    for i in range(n_drones):
        n_points=N_POINTS
        xs = []
        ys = []
        for j in range(n_points):
            if xs == []:
                xs.append(random.randrange(200,320,1))
                ys.append(random.randrange(200, 320,1 ))
            else:
                coin = np.random.randint(0,2)
                if coin %2 ==0:
                    xs.append(xs[-1])
                    new_y = random.randrange(ys[-1]-step,ys[-1]+step,step)
                    while(new_y in ys):
                        new_y = random.randrange(ys[-1]-step,ys[-1]+step,step)
                    ys.append(new_y)
                else:
                    new_x = random.randrange(xs[-1]-step,xs[-1]+step,step)
                    while(new_x in xs):
                        new_x = random.randrange(xs[-1]-step,xs[-1]+step,step)
                    xs.append(new_x)
                    ys.append(ys[-1])
        
        trajs[i] = list(zip(xs,ys)) 
    return trajs

def plot_trajs(trajs):
    for i in range(len(trajs)):
        plt.plot(*zip(*trajs[i]),"-o")
        # for j in range(len(trajs[i])):
        #     if(j+1>=len(trajs[i])): break
        #     plt.arrow(*trajs[i][j],*trajs[i][j+1],  head_width = 0.2, )
    
    print('trajs: ', trajs)
    # plt.grid()
    plt.title("not interpolated" )
    plt.show()

def interpolate_trajs(trajs):
    trajs = myInterpolate2D(trajs,step_size=STEP_SIZE)
    fig = plt.figure()
    for i in range(len(trajs)):
        plt.plot(*zip(*trajs[i]),"-o")
    plt.title("interpolated" )
    plt.show()
    return trajs

def height_algo(trajs):

    # trajs_3d = allocate_height(trajs,10,0,2)
    trajs_3d,zs = avoid_collision_in_empty_space(trajs,0,300,sep_h=20,
        min_safe_points=3,radius = 20,simpleMode=False)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(len(trajs)):
        ax.plot(*zip(*trajs_3d[i]))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    fig = plt.figure()
    plt.title("height" )

    for i in range(len(trajs)):
        z_t = [ [t,z] for t,z in enumerate(zs[i])]
        print(z_t)
        plt.plot(*zip(*z_t),"-o")


    plt.show()
    return trajs_3d,zs

def get_action(s0,s1):
    x0,y0 = s0[0],s0[1]
    x1,y1 = s1[0],s1[1]
    if(x1 > x0): return AINDEX["E"]
    elif(x0 > x1): return AINDEX["W"]
    elif(y1 > y0): return AINDEX["N"]
    elif(y0 > y1): return AINDEX["S"]
    # else: raise Exception("NOT MOVED NOT ACCEPTABLE")
    else: return -1

def are_opposite_actions(a1,a2):
    return a1!=a2 \
        and ( (ACTION[a1]=="N" and ACTION[a2] =="S") or  
            (ACTION[a1]=="S" and ACTION[a2] =="N") or 
            (ACTION[a1]=="E" and ACTION[a2] =="W") or  
            (ACTION[a1]=="W" and ACTION[a2] =="E") )  


def fix_traj(trajs):
    """
    Remove states going back (indecisions in agent)
    """
    for i in range(len(trajs)):
        last_action = None
        last_state = None
        
        history = dict()
        j=0
        while(j < len(trajs[i]) ):
            # Current trajectory is trajs[i]
            # Controllo se fuori dal bound quindi sarebbe primo stato e vado avanti
            s = tuple(trajs[i][j])
            if(j-1) < 0 : 
                history[s] = 1
                j += 1
                continue
            if(last_state is None):
                last_state = tuple(trajs[i][j-1])
            action = get_action(last_state,s)
            if(j+1) >= len(trajs[i]) : break
            next_s= tuple(trajs[i][j+1])
            next_action = get_action(s,trajs[i][j+1])
            # print("s",s,s in history,'next_s: ', next_s,next_s in history,"actions",action,last_action)
            print("s",s,'last_s: ', last_state,s not in history and (next_s in history or last_action == action),"actions",action,next_action)
            
            if s not in history:
                # stato nuovo
                history[s] = 1
            else:
                # Rimuovo duplicato solo se non porta a obliquità (next_state anche lui duplicato)
                if(next_s in history or next_action == action):
                    trajs[i].pop(j)
                    j-=1
            last_action = action
            last_state = s 
            j += 1

    return trajs




def plot_3d(trajs):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(len(trajs)):
        ax.plot(*zip(*trajs[i]))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.title("3d plot" )
    plt.show()


def plot_z(zs):

    fig = plt.figure()
    for i in range(len(zs)):
        z_t = [ [t,z] for t,z in enumerate(zs[i])]
        print(z_t)
        plt.plot(*zip(*z_t),"-o")

    plt.title("height" )
    plt.show()


if __name__ == "__main__":
    d1 = [[0,0],[1,0],[2,0],[1,0],[2,0],[1,0],[0,0],[0,1],[0,2],[0,1],[0,0],[0,1],[0,2],[0,3],[0,4]]
    
    d2 = [[1,0],[2,0],[2,1],[2,2],[2,3],[1,3],[0,3],[0,4]]
    
    d3 = [[4,0],[4,1],[4,2],[4,3],[3,3],[2,3],[1,3],[0,3]]
    
    d4 = [[5,0],[5,1],[5,2],[5,3],[4,3],[3,3],[3,2],[3,1],[3,0]]
    trajectories = [d1,d2]
    trajs3d = [d3,d4]

    # Caso complesso dove le traiettorie non le ho generate io con avoid in empty
    # d3d = [ [ p+[random.randint(0,10)] for p in [ traj for traj in d1]] ]
    d3d,zds = avoid_collision_in_empty_space(trajs3d,0,100,10,3)
    print('d3d: ', d3d)

    # trajs  = random_trajs()
    # plot_trajs(trajectories)
    # trajs = fix_traj(trajectories)
    # trajs = fix_traj(trajectories)
    # print('fixed trajs: ', trajs)
    print('trajectories: ', trajectories)
    utils.myInterpolate2D(trajectories)
    trajs_3d = avoid_collision_in_busy_space(trajectories,[],min_height=0,max_height=100,sep_h=20,radius=5,min_safe_points=3)
    print('trajs_3d,zs: ', trajs_3d)
    
    plot_3d(trajs_3d)
    # plot_z(zs)

    # trajs = interpolate_trajs(trajs)
    # height_algo(trajs)

                                                                                                                                                                                                                                                                                                                                                                                                                                           