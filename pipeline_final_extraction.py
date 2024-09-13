from utils_extraction import *
from create_tt import *
bin_width = 0.02


path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240423_SESSION_01/headstage_1/'
# Déterminer le type  de session : 
session_type = get_session_type_final(path)
#print(session_type)

# gérer quand on a 1 ou 2 headstages

ANALOG_TRIGGERS_MAPPING = get_triggers_mapping(path)
print(ANALOG_TRIGGERS_MAPPING)


if session_type=='PureTones':
    print('tonotopy')
    n_blocs = 1
    
elif session_type=='PlaybackOnly':
    print('PlaybackOnly')
    n_blocs = 1
    create_tt_playback_only(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False)
    create_data_condition(path, bin_width, n_blocs, 'playback')
    

elif session_type=='Tracking':
    print('TrackingOnly')
    n_blocs = 1
    create_tt_tracking_only(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False)
    create_data_condition(path, bin_width, n_blocs, 'tracking')

elif session_type=='Playback': # ca dit quoi quand c'est 'playback' cad alternance tracking playback??
    print('Tracking')
    blocs = get_block_numbers(path)
    n_blocs = blocs[1]-blocs[0]+1
    print('n_blocs = ', n_blocs)