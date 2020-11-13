import time
from airsimgeo import AirSimGeoClient
import utils
srid = 'EPSG:3857'
origin = (41.9022357,12.4572837, 0 )
dest = (41.9030354,12.4663601, 60) #(lat,long,height)

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
