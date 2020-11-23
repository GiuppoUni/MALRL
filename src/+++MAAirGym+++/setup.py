import json
import utils
from airsimgeo import AirSimGeoClient

sfp = utils.AIRSIM_SETTINGS_FOLDER + "settings.json"


with open(sfp, "r") as jsonFile:
    data = json.load(jsonFile)


print(data["Vehicles"]) 
    
    
for i,v in enumerate(data["Vehicles"]):
    gps = utils.init_gps[i]
    ned = utils.lonlatToAirSim(*gps)
    v["X"] = ned[0]
    v["Y"] =  ned[1]
    # v["Z"] = v["Z"] 



# with open("replayScript.json", "w") as jsonFile:
#     json.dump(data, jsonFile)
