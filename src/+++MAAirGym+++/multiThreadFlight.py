import airsim,time,threading
import numpy as np

# connect to the AirSim simulator

global client
client = airsim.MultirotorClient()
client.confirmConnection()

client.reset()

for count in range(0,2):
    (client.enableApiControl(True,vehicle_name='Drone' + str(count + 1)))
    (client.armDisarm(True,vehicle_name='Drone' + str(count + 1)))

global V, Vt, eps

Vt=1
V=2
eps=0.1

#################################################################### TAKE-OFF

def myjoint_TAKEOFF(goalx, goaly, goalz, Vel,vehicle_name):

    dist=(abs(goalz) - abs(client.simGetGroundTruthKinematics(vehicle_name).position.z_val))
    time.sleep(abs(dist)/Vel)  

#################################################################### NAVIGATE

def myjoint(goalx,goaly,goalz,Vel,vehicle_name):
    global lk
    
    lk.acquire()
    pos = client.simGetGroundTruthKinematics(vehicle_name)
    lk.release()
        
    dist = ((np.sqrt(np.power((goalx -pos.position.x_val),2) + np.power((goaly - pos.position.y_val),2)+ np.power((goalz - pos.position.z_val),2))))
    print ("waiting ",int(abs(dist)/Vel), vehicle_name)
    time.sleep(int(abs(dist)/Vel))
    
    lk.acquire()
    pos = client.simGetGroundTruthKinematics(vehicle_name)
    lk.release()
    while ((np.sqrt(np.power((goalx -pos.position.x_val),2) + np.power((goaly - pos.position.y_val),2)+ np.power((goalz - pos.position.z_val),2))))>eps:
        lk.acquire()
        pos = client.simGetGroundTruthKinematics(vehicle_name)
        lk.release()
		
lk=threading.Lock()


def move_drone(n):
    global lk
    name=["Drone1", "Drone2"]
    
    positions=[[[15,0,-4],[15,5,-4],[0,5,-4],[0,0,-4]],
               [[12,0,-3],[12,5,-3],[0,5,-3],[0,0,-3]]]
    
    for x,y,z in positions[n]:
        lk.acquire()
        client.moveToPositionAsync(x,y,z, V, vehicle_name=name[n])
        lk.release()
        print("goint to", x,y,z,name[n])
        myjoint(x,y,z,V,vehicle_name=name[n])
        print("reached to", x,y,z,name[n])
            
                

client.moveToPositionAsync(0,0,-2, Vt, vehicle_name="Drone1")
while (abs(-2) - abs(client.simGetGroundTruthKinematics("Drone1").position.z_val))>eps:
    myjoint_TAKEOFF(0,0,-2,Vt,"Drone1")
    
client.moveToPositionAsync(0,0,-2, Vt, vehicle_name="Drone2")
while (abs(-2) - abs(client.simGetGroundTruthKinematics("Drone2").position.z_val))>eps:
    myjoint_TAKEOFF(0,0,-2,Vt,"Drone2")

#move_drone
global d
d=[]

for i in [0,1]:
    d.append(threading.Thread(target=move_drone, args=[i]))    
    d[i].start()
    
for i in [0,1]:
    d[i].join()
    
print("Done!")