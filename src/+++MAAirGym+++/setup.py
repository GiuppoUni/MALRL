import json
import utils
from airsimgeo import AirSimGeoClient

# Get settings file
sfp = utils.AIRSIM_SETTINGS_FOLDER + "settings.json"
with open(sfp, "r") as jsonFile:
    data = json.load(jsonFile)
print(data["Vehicles"]) 
    
#  Set uavs positions based on gps positions specified inside utils module
for i,v in enumerate(data["Vehicles"]):
    gps = utils.init_gps[i]
    ned = utils.lonlatToAirSim(*gps)
    v["X"] = ned[0]
    v["Y"] =  ned[1]
    # v["Z"] = v["Z"] 

# Write changes into setting file
with open(utils.AIRSIM_SETTINGS_FOLDER + "settings.json", "w") as jsonFile:
    json.dump(data, jsonFile)
