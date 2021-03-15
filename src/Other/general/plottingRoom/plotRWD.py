import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='')


parser.add_argument('-i', type=str,
                  help='episodes (default: %(default)s)')

args=  parser.parse_args()


def plot():
   for ff in os.listdir(args.i):
      if(ff[-4:] !=".txt"):
         continue
      
      print(ff)
      fig = plt.figure()
      ax = plt.gca()
      ax.set_ylim([800, 1000])
      
      ys = []
      x = []
      y = []
      count=0
      with open(os.path.join(args.i,ff),'r',errors="ignore") as csvfile:
         for row in csvfile:
            if(row[0]=="-"):
               if(count ==3):
                  ys.append(y)
                  y_mean = []
                  for i in range(0,len(y)):
                     y_mean.append( np.mean([   ys[0][i],ys[1][i],ys[2][i]   ]  ) )
                  plt.plot(x,y_mean)
                  break
               else:
                  if(count !=0):
                     ys.append(y)
                     
                     # plt.plot(x,[ np.mean(y) for y in range(ys[-1]              ])
                     x=[]
                     y=[]
                  count+=1
            elif "," in row and not row[0].isalpha() and row[0]!="999":
               fields = row.split(",")
               if(fields[0]=="300"):
                  continue
               print(row)
               x.append(int(fields[0]))
               y.append( float(fields[2]))
         print(count)      

      # meaned = np.mean(ys,axis=0)
      # print(len(meaned),meaned)



      # plt.plot(x,y, label='Loaded from file!')
      # plt.plot(x,meaned,color="blue")
      plt.xlabel("Episodes")
      plt.ylabel('Reward')

      plt.title('QLearning (N=90, cell memory buffer size = 3): Reward per episode')
      # plt.title('QLearning: Reward per episode')

      plt.legend()
      # plt.show()
      ax.legend()
      plt.savefig(os.path.join(args.i,ff+"rwd.png") )


if __name__ == "__main__":
   plot()