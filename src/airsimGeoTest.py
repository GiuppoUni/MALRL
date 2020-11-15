import time
from time import sleep
from airsimgeo import AirSimGeoClient
import utils
srid = 'EPSG:3857'
origin = (
    12.457480,
    41.902243,
    0 )
dest = (
    12.466382,
    41.902491,
    80) #(lat,long,height)

client = AirSimGeoClient(srid=srid, origin=origin)
client.confirmConnection()
print('Connection Confirmed with',client)
client.enableApiControl(True, "Drone1")
pointer = client.takeoffAsync(vehicle_name="Drone1")
pointer.join()

pos = client.getGpsLocation()
print("{:.4f},{:.4f},{:.4f}".format(pos[0], pos[1], pos[2]))

# Move to GPS position
client.moveToPositionAsyncGeo(gps=dest,vehicle_name="Drone1")
while(True):
    sleep(1)
    print(client.getGpsLocation())