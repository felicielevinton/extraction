from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_tt import *
from spike_sorting import *

# ARGUMENTS
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.05
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)  


session = 'ALTAI_20240724_SESSION_02'
path = '/Volumes/data2/eTheremin/ALTAI/'+ session + '/'
#path = '/auto/data2/eTheremin/MUROLS/MUROLS_20230218/MUROLS_20230218_SESSION_01/'
mock=False
#session_type = get_session_type_final(path)
#print(session_type)
#session_type = 'Playback' #TrackingOnly ou PbOnly



# vérifier qu'il n existe pas de tt.pkl, s'il n''existe pas alors on le créée, sinon c'est pas la peine.
# get_session_type pour le session_type

 
#2. Créer le data.npy et features.npy
create_data_features_mock(path+'headstage_0/', bin_width, sr, mock=mock)

# version test de spike_sorting

#create_data_features_ss(path+'headstage_0/', bin_width, fs, mock=False)




