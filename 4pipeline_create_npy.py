from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *
from spike_sorting import *

# ARGUMENTS
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
channel = 15

#path = 'Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/headstage_0'
path = 'Z:/eTheremin/ALTAI/ALTAI_20240712_SESSION_00/headstage_0'

mock=False
#session_type = get_session_type_final(path)
#print(session_type)
#session_type = 'Playback' #TrackingOnly ou PbOnly



# vérifier qu'il n existe pas de tt.pkl, s'il n''existe pas alors on le créée, sinon c'est pas la peine.
# get_session_type pour le session_type

 
#2. Créer le data.npy et features.npy
#create_data_features_mock(path+'headstage_0/', bin_width, sr, mock=False)

# version test de spike_sorting
# mat_file = 'Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/spike_sorting/times_C' + str(channel) + '.mat'
# npy_file = 'Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/spike_sorting/times_C' + str(channel) + '.npy'

mat_file = 'Z:/eTheremin/ALTAI/ALTAI_20240712_SESSION_00/spike_sorting/times_C' + str(channel) + '.mat'
npy_file = 'Z:/eTheremin/ALTAI/ALTAI_20240712_SESSION_00/spike_sorting/times_C' + str(channel) + '.npy'
create_spikes_clusters(path + '/spike_sorting',channel, mat_file, npy_file) 
create_data_features_ss(path ,channel,  bin_width, fs, mock=False)