import pandas
from scipy.interpolate import interp1d
import numpy as np


def myInterpolate(arr, n_samples=10 ):
    res = []
    for i,p in enumerate(arr):
        if(i+1 >= len(arr)):
            break
        x1,y1,z1 = p[0],p[1],p[2]
        x2,y2,z2 = arr[i+1][0],arr[i+1][1],arr[i+1][2]
        step_length = max(abs(x2-x1),abs(y2-y1)) / n_samples
        for i in range(n_samples):
            if(x2 > x1):
                # Moved on the right
                new_p = [x1 + i * step_length, y1,z1]
            elif (x1 > x2):
                # Moved left
                new_p = [x2 + i * step_length, y1,z1]
            elif (y2 > y1):
                # Moved left
                new_p = [x1, y1 + i * step_length,z1]
            elif (y1 > y2):
                # Moved left
                new_p = [x1, y2 + i * step_length,z1]
            else:
                raise Exception("Uncommmon points")
            res.append(new_p)
    
    return res


def simplify_traj(arr):
    arr = [tuple(x) for x in arr]
    print("ARR",arr[:20])
    print("\n\n")
    print("UNI:",np.unique(arr[:20], axis=0) )
    return np.unique(arr, axis=0)

filename = "q_traj_20202020-12-17--13-36.csv"
df=pandas.read_csv("qtrajectories\csv\\"+filename,delimiter=",",usecols=[1,2,3])
print(df)
arr = df.to_numpy(dtype=int)
# print('arr: ', arr)

# st = simplify_traj(arr)
it = myInterpolate(arr)
# print(it)
np.save("qtrajectories\interpolated\\"+ filename[:-4],it)

# print("myInterpolate", myInterpolate(arr))

# x = [p[0] for p in arr]
# print('x: ', x)

# y = [p[1] for p in arr]

# print( interp1d(x, y, kind='cubic'))

# pathXY = [0 0; 1 1; 10 2; 12 3]
# stepLengths = sqrt(sum(diff(pathXY,[],1).^2,2))
# stepLengths = [0; stepLengths] % add the starting point
# cumulativeLen = cumsum(stepLengths)
# finalStepLocs = linspace(0,cumulativeLen(end), 100)
# finalPathXY = interp1(cumulativeLen, pathXY, finalStepLocs)