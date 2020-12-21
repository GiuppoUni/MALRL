import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

import scipy.interpolate
import utils
from sklearn.neighbors import KDTree







def myInterpolate2D(trajs, n_samples=10,step_size=20 ):
    res = []
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
            
            

        res.append(res_t)
    return res
            

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


def avoid_collision(trajectories,max_height,min_height,sep_h,radius=30):
    Tmax = max([len(traj) for traj in trajectories])
    drones = range(len(trajectories))
    points = {}
    trajs_3d =[[] for d in drones] 
    colliding_trajs = []
    for d in drones:
        for t in range(len(trajectories[d])):
            point = tuple(trajectories[d][t])
            
            for idx,_tree in enumerate(TREES): 
                if(idx == d):
                    # E' quella attuale
                    continue
                res = _tree.query_radius( [point],r=radius,count_only = True )
                if res > 0:
                    print("Collisions with","Trajectory_"+str(idx))
                    print("\tcomputed from trajectory ",d,", point", point)
                    if(d not in colliding_trajs):
                        colliding_trajs.append(d)

            if(d not in colliding_trajs):
                new_z = max_height
            else:
                offset = colliding_trajs.index(d)
                new_z = max_height - offset * sep_h
                if new_z < min_height:
                    raise Exception("Out of height bounds")
            trajs_3d[d].append(list(point)+[new_z])

    return trajs_3d

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






d1 = [[0,0],[0,1],[0,2],[0,3],[0,4]]
d2 = [[1,0],[2,0],[2,1],[2,2],[2,3],[1,3],[0,3],[0,4]]
trajectories = [d1,d2]

# # plt.scatter([p[0] for p in d1 ], [p[1] for p in d1 ])
# plt.plot(*zip(*d1),"-o")
# plt.plot([p[0] for p in d2 ], [p[1] for p in d2 ],"-o")
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# d1_3d = [[p[0],p[1],10] for p in d1]
# d2_3d = [[p[0],p[1],10] for p in d2]

# ax.plot(*zip(*d1_3d))
# ax.plot(*zip(*d2_3d))

# plt.show()
# zs = np.randrange(n, 2, 10)


# trajs = allocate_height(trajectories,10,0,2)

# fig = plt.figure()
# ax = plt.axes(projection='3d')


# ax.plot(*zip(*trajs[0]))
# ax.plot(*zip(*trajs[1]))

# plt.show()
SEED = 668
random.seed(SEED)
np.random.seed(seed=SEED)

n_drones = 5
step = 120
trajs = [[] for i in range(n_drones)]
for i in range(n_drones):
    n_points=100
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


for i in range(n_drones):
    plt.plot(*zip(*trajs[i]),"-o")
print('trajs: ', trajs)
plt.title("not interpolated" )
# plt.show()

trajs = myInterpolate2D(trajs)
fig = plt.figure()
for i in range(n_drones):
    plt.plot(*zip(*trajs[i]),"-o")
plt.title("interpolated" )

TREES = build_trees(trajs)

# trajs_3d = allocate_height(trajs,10,0,2)
trajs_3d = avoid_collision(trajs,10,0,2,50)

fig = plt.figure()
ax = plt.axes(projection='3d')

for i in range(n_drones):
    ax.plot(*zip(*trajs_3d[i]))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()