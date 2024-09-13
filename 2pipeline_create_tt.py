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

path = '/auto/data2/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240722_SESSION_03/'
#session_type = get_session_type_final(path)
session_type = 'PbOnly' #TrackingOnly ou PbOnly

#create_tones_triggers_and_condition_V3(path, session_type)

create_tt_no_mock(path, mock=False) # peut etre ici que ca va beuguer

print(f'tt.pkl created,   for {path}')