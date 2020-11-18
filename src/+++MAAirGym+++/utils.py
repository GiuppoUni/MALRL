import json
from dotmap import DotMap

# CHANGE FOR FOLDER CONTAINING AIRSIM SETTINGS
SETTINGS_PATH = 'C:/Users/gioca/OneDrive/Documents/Airsim/'

from configparser import ConfigParser

with open(SETTINGS_PATH + 'settings.json', 'r') as jsonFile:
    g_settings = json.load(jsonFile)

g_config = ConfigParser()
g_config.read('config.ini')

map_filename = "./overlayMap.png"
# GPS init position of uavs
init_gps = [
    (
        12.45727300643921,
        41.90169011784915,
        0
    ),

    (
        12.457227408885958,
        41.90276414312537,
        0
    ),
    (
        12.45587021112442,
        41.90220118045657,
        0
        ),
]



def read_cfg(config_filename='configs/main.cfg', verbose=False):
    parser = ConfigParser()
    parser.optionxform = str
    parser.read(config_filename)
    cfg = DotMap()

    if verbose:
        hyphens = '-' * int((80 - len(config_filename))/2)
        print(hyphens + ' ' + config_filename + ' ' + hyphens)

    for section_name in parser.sections():
        if verbose:
            print('[' + section_name + ']')
        for name, value in parser.items(section_name):
            value = ConvertIfStringIsInt(value)
            cfg[name] = value
            spaces = ' ' * (30 - len(name))
            if verbose:
                print(name + ':' + spaces + str(cfg[name]))

    return cfg




def ConvertIfStringIsInt(input_string):
    try:
        float(input_string)

        try:
            if int(input_string) == float(input_string):
                return int(input_string)
            else:
                return float(input_string)
        except ValueError:
            return float(input_string)

    except ValueError:
        true_array = ['True', 'TRUE', 'true', 'Yes', 'YES', 'yes']
        false_array = ['False', 'FALSE', 'false', 'No', 'NO', 'no']
        if input_string in true_array:
            input_string = True
        elif input_string in false_array:
            input_string = False

        return input_string