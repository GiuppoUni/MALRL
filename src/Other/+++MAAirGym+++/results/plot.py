import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('rewards.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for i,row in enumerate(plots):
        if(i==0 or row[0]=="E"):
            continue
        x.append(int(row[0]))
        y.append( float(row[1]))

# plt.plot(x,y, label='Loaded from file!')
plt.plot(x,y)
plt.xlabel("Episodes")
plt.ylabel('Number of steps')
plt.title('Number of steps - Episodes')
plt.legend()
plt.show()