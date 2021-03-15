import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Layer 1')


parser.add_argument('-d', type=str,
                  help='episodes (default: %(default)s)')

args = parser.parse_args()

if args.d:
   dir= args.d
else:
   dir="./"

times =[]
for ff in os.listdir(dir):
   print(ff)
   if(not ".txt" in ff):
      continue

   with open(dir+ff,'r') as csvfile:
      plots = csv.reader(csvfile, delimiter=',')
      for i,row in enumerate(plots):
         print(row[0])
         if(i!=0 and row[0][0]=="E"):
            time = row[0].split(":")[-1]
            times.append(float(time))

print('times: ', times)
meaned = np.mean(times)
print('len(meaned),meaned: ', meaned)
