from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final

# ARGUMENTS
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)

path = '/mnt/working2/felicie/data2/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240725_SESSION_00/headstage_0/'
#session_type = get_session_type_final(path)
session_type = 'Playback' #TrackingOnly ou PbOnly



# vérifier qu'il n existe pas de tt.pkl, s'il n''existe pas alors on le créée, sinon c'est pas la peine.
# get_session_type pour le session_type

# 1. Creér le tt.pkl

create_tones_triggers_and_condition_V3(path, session_type)


#2. Créer le data.npy et features.npy
create_data_features(path, bin_width, sr)