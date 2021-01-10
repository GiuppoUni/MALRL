import os
import numpy as np
import pandas as pd
import trajs_utils

from pathlib import Path


# 2d not int
df = pd.read_csv( "C:/Users/gioca/Desktop/Repos/AirSim-PredictiveManteinance/src/final/generatedData/2dL1/qTrajs2D-D-10-01-2021-H-11-44-22-/traj2d_0.csv" ,
delimiter=",",index_col="index")
            # print(df)
t=df.to_numpy() 
tt=t.tolist()

trajs_utils.plot_xy([ [ [p[0],p[1],10]for p in tt] ],cell_size=1,doScatter=True,isCell=True)


tt2=trajs_utils.fix_traj([tt])
print('tt2: ', tt2)
trajs_utils.plot_xy([     [ [p[0],p[1],10] for p in tt2[0]] ] ,cell_size=1,doScatter=True  ,isCell=True )




# 2d int
df = pd.read_csv( "generatedData/3dL2/csv/trajs3D-D-10-01-2021-H-11-44-22-/3dtraj0.csv" ,
delimiter=",",index_col="index")
            # print(df)
t=df.to_numpy() 
tt=t.tolist()

trajs_utils.plot_xy([tt],cell_size=20,dotSize=0.5,doScatter=True)


# trajs_utils.plot_3d([ [ [p[0],p[1],10]for p in tt] ])

