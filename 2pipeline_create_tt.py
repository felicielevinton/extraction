from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *

# ARGUMENTS
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)

path = 'Z:/eTheremin/ALTAI/ALTAI_20240708_SESSION_00/'
#session_type = get_session_type_final(path)
session_type = 'Playback' #TrackingOnly ou PbOnly
features = np.load(path + 'headstage_0/' +f'features_{bin_width}.npy', allow_pickle=True)
unique_conditions = set(d['Condition'] for d in features)

print(unique_conditions)

#create_tones_triggers_and_condition_V3(path, session_type)
if session_type == 'TrackingOnly' or session_type == 'PbOnly' or session_type == 'Playback':
    create_tt_no_mock(path, mock=False) # peut etre ici que ca va beuguer
elif session_type =='MappingChange':
    create_tt_mc(path)

print(f'tt.pkl created,   for {path}')