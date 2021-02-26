import numpy as np
from numpy.lib.format import BUFFER_SIZE
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


def strDate():
    return str(datetime.datetime.now().strftime('-D-%d-%m-%Y-H-%H-%M-%S-') )

EXPERIMENT_DATE =  strDate()



a_logger.debug(
   "\nRUNNING EXP at: " + EXPERIMENT_DATE +"\n"
)

# RANDOM SEED
# a_logger.debug("SEED: "+ str(trajs_utils.setRandomSeed())  )
a_logger.debug("SEED: "+ str(trajs_utils.setSeed(999))  )

tolerances = [0,0.0]
n_trajectories = [30,300,10,10,50,50,100,100]

NUM_TESTS = 1
BUFFER_SIZE=3
OUT_FOLDER = "generatedData/randTrajs/"

exp_path = OUT_FOLDER + "rand"+EXPERIMENT_DATE
created = False
i=1

while(not created ):
   try:
      exp_path = exp_path[:-1]+ str(i) +"/"
      os.mkdir(exp_path)
      print ("Successfully created the directory %s " % exp_path)
      created = True
   except OSError:
      print ("Creation of the directory %s failed" % exp_path)
   i+=1

for j in range(0,NUM_TESTS):
   
   assigned_trajs3d = []
   outs=0
   zeroStart=time.time()
   fids=[]
   for i in range(n_trajectories[j]):
      # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
      start = time.time()
      trajs2d = trajs_utils.random_gen_2d(0,860,0,860,
      step=120,n_trajs=BUFFER_SIZE)
      end = time.time()
      
      # a_logger.debug("2D generated for:"+str(i)+",in time:"+str(end-start))

      # DO INTERPOLATION
      trajs2d = trajs_utils.interpolate_trajs(trajs2d)
      
      trajs2d = [ [ list(p) for p in t]  for t in trajs2d]
      with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
         start = time.time()
      res, i_outs,lfids =  trajs_utils.avoid_collision_complex(trajs2d,fids= [ n+i*BUFFER_SIZE for n in range(0,BUFFER_SIZE) ],
         assigned_trajs=assigned_trajs3d,min_height=50,max_height=300,
         sep_h=10,radius=100, tolerance=tolerances[j%2])
   
      print (len(trajs2d),i_outs)
   
      for fid in lfids:
         fids.append(fid)
      
      outs+=i_outs

      for t in res:
         assigned_trajs3d.append(t)
      end = time.time()
      # a_logger.debug("3D generated for:"+ str(i)+",in time:"+ str(end-start))
         
   finalEnd=time.time()
   # for i,a in enumerate(assigned_trajs3d):
   #    a_logger.debug("traj "+str(i)+","+ str(len(a))+","+str(a[-1]))

   a = np.array(assigned_trajs3d)
   
   np.save(exp_path+"trajs"+str(j)+".npy",a)
   
   trajs_utils.plot_3d(assigned_trajs3d,ids=fids,also2d=False,doSave=True,name="test"+str(j)+"3d")
   trajs_utils.plot_3d(assigned_trajs3d,ids=fids,also2d=False,doSave=False,name="test"+str(j)+"3d")
   
   # trajs_utils.plot_z(assigned_trajs3d,second_axis=0,doSave=True,name="test"+str(j)+"xz")
   # trajs_utils.plot_z(assigned_trajs3d,second_axis=1,doSave=True,name="test"+str(j)+"yz")

   
   a_logger.debug("Ass:"+ str(a.shape))
   a_logger.debug("Ref:"+ str(outs))
   
   print('a.shape: ', len(assigned_trajs3d))
   print('outs: ', outs)
   print('time: ', finalEnd-zeroStart)

