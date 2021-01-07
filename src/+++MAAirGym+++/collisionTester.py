import numpy as np
import pylab
from trajs_utils import setSeed
import time 

import os, sys
import logging
import contextlib
import datetime

import trajs_utils

a_logger = logging.getLogger("collisionTesterLog")
a_logger.setLevel(logging.DEBUG)

output_file_handler = logging.FileHandler("finalCollisionTimeTester.log")
stdout_handler = logging.StreamHandler(sys.stdout)

a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)




a_logger.debug(
   "\nRUNNING EXP at: " + str(datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S'))+"\n"
)

# RANDOM SEED
# a_logger.debug("SEED: "+ str(trajs_utils.setRandomSeed())  )
a_logger.debug("SEED: "+ str(trajs_utils.setSeed(999))  )

tolerances = [0,0.05]
n_iteration = [10,10,50,50,100,100]


for j in range(6):
   
   assigned_trajs3d = []
   outs=0
   zeroStart=time.time()

   for i in range(n_iteration[j]):
      # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
      start = time.time()
      trajs2d = trajs_utils.random_gen_2d(0,1000,0,1000,
      step=120,n_trajs=10)
      end = time.time()

      a_logger.debug("2D generated for:"+str(i)+",in time:"+str(end-start))

      # trajs2d = trajs_utils.interpolate_trajs(trajs2d)
      trajs2d = [ [ list(p) for p in t]  for t in trajs2d]
      with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
         start = time.time()
      res, i_outs =  trajs_utils.avoid_collision_complex(trajs2d,assigned_trajs=assigned_trajs3d,min_height=50,max_height=300,
      sep_h=10,radius=200, tolerance=tolerances[j%2])
      outs+=i_outs
      for t in res:
         assigned_trajs3d.append(t)
      end = time.time()
      a_logger.debug("3D generated for:"+ str(i)+",in time:"+ str(end-start))
         
   finalEnd=time.time()
   # for i,a in enumerate(assigned_trajs3d):
   #    a_logger.debug("traj "+str(i)+","+ str(len(a))+","+str(a[-1]))


   a = np.array(assigned_trajs3d)
   
   trajs_utils.plot_3d(assigned_trajs3d,also2d=False,doSave=True,name="test"+str(j)+"3d")
   trajs_utils.plot_z(assigned_trajs3d,second_axis=0,doSave=True,name="test"+str(j)+"xz")
   trajs_utils.plot_z(assigned_trajs3d,second_axis=1,doSave=True,name="test"+str(j)+"yz")

   a_logger.debug("Ass:"+ str(a.shape))
   a_logger.debug("Ref:"+ str(outs))
   
   print('a.shape: ', a.shape)
   print('res: ', outs)
   print('time: ', finalEnd-zeroStart)

