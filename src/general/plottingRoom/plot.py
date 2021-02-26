import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')


parser.add_argument('-i', type=str,
                  help='episodes (default: %(default)s)')

args=  parser.parse_args()


ys = []

for ff in os.listdir(args.i):
   print(ff)
   if(not ".txt" "" in ff):
      continue

   x = []
   y = []

   with open(os.path.join(args.i,ff),'r') as csvfile:
      plots = csv.reader(csvfile, delimiter=',')
      for i,row in enumerate(plots):
         print(row[0])
         if(i==0 or row[0][0]=="E"):
               continue
         x.append(int(row[0]))
         y.append( float(row[1]))
      ys.append(y)

meaned = np.mean(ys,axis=0)
print(len(meaned),meaned)
# plt.plot(x,y, label='Loaded from file!')

ax = plt.gca()
ax.set_ylim([0, 8000])

plt.plot(x,meaned,color="blue")
plt.xlabel("Episodes")
plt.ylabel('Number of steps')

plt.title('N-steps QLearning (N=20): Number of steps per episode')
# plt.title('QLearning: Number of steps per episode')

plt.legend()
# plt.show()
plt.savefig(os.path.join(args.i,"steps.png") )