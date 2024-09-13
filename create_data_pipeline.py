from create_data import *

#path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240521_SESSION_02/headstage_0'
path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240419_SESSION_04/headstage_0/'
fs = 30e3
bin_width = 0.02
n_blocs = 1
#create_data_v2_no_mock(path, bin_width, n_blocs)
#create_data_v2(path, bin_width, n_blocs)
#create_data_v2(path, bin_width, n_blocs)
create_data_condition(path, bin_width, n_blocs, 'playback')
