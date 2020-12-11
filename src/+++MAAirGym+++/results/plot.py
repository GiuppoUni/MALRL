import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('rewards.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(plots):
        if(i==0):
            continue
        x.append(int(row[0]))
        y.append( float(row[1]))

# plt.plot(x,y, label='Loaded from file!')
plt.plot(x,y)
plt.xlabel("Episodes")
plt.ylabel('Time (s)')
plt.title('Time in sec to catch t=4 targets ')
plt.legend()
plt.show()