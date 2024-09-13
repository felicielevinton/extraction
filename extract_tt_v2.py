from sequences import *
from extraction_utils import extract
from get_data_v2 import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json


#analog_in et digital_in
#ANALOG_TRIGGERS_MAPPING = {"MAIN": 1, "PLAYBACK": 0, "MOCK": 3, "TARGET": 2} # ca c'est le mapping trigger normal
ANALOG_TRIGGERS_MAPPING = {"MAIN": 1, "PLAYBACK": 0} # pour 2 sessions cheloues on a eu ce mapping (triggers mock ont sauté)

#hje checke que je suis en v2 dans le json
path='//mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240503_SESSION_02/'

json_file_path = path+'session_BURRATA_SESSION_02_20240503.json'

with open(json_file_path, 'r') as f:
        json_file = json.load(f)


dig_trig_mapping = get_digital_mapping(json_file)


# for le nombre de blocks
n_blocks = get_n_iter(json_file) #ne marche pas encore
# pour un block :
n_blocks = 1

# charger les positions

#extraire les canaux analogiques et digitaux avec le bon mapping des canaux digitaux
output = extract(path)

#extraire positions et tons block par block
for block in range(n_blocks):
    if check_if_block_complete(json_file):
        condition = "tracking"
        # extraire les positions et les tons
        positions = extract_positions_path(json_file, block, condition)
        tones = extract_tones_path(json_file, block, condition)
        if condition=='playback':
            #en playback : convertir les positions en tones pour avoir les MOCK
            # mock = get_tones_from_position (nom donné au pif)
            print('yo')
        else:
            pass

    else:
        break
analog_channels = list(output['ANALOG'])
digital_channels = list(output['DIGITAL'])
#print('digital_output = ', digital_channels)
an_trig = extract_analog_lines(path, analog_channels, ANALOG_TRIGGERS_MAPPING)
dig_trig = extract_digital_lines(path, digital_channels, dig_trig_mapping )

#extract_data(path, analog_channels=None, digital_channels=None, compatibility=False)
triggers = output
print('triggers = ',  triggers)
print(triggers["ANALOG"]["MAIN"])
print('exp_type = ', get_session_type(path))



