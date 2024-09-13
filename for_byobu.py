from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
from tonotopy import *
from matplotlib.colors import ListedColormap, Normalize
from format_data import *
from skimage import measure
import matplotlib.colors as colors
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.30#0.300
bin_width = 0.005
#bin_width = 0.02
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
max_freq = 2
min_freq=2 #3 for A1
threshold = 2 #threshold for contour detection 3.2 is good

n_jobs = -1
#path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/FRINAULT/'
#path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/MUROLS/'

path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240503_SESSION_02/'
#folders = get_folders(path)
#create_data(path, bin_width)
#for i in range(1,len(folders)):
    #folder = folders[i]
    #create_data(path+folder, bin_width)
#del folders[1] # only for Frinault

folders = ['headstage_0', 'headstage_1']

condition = 'playback'

for folder in folders:
    print(folder + ' en cours de traitement')
    #create_data_v2(path+folder, bin_width)
    data = np.load(path+folder+'/data.npy')
    features = np.load(path+folder+'/features.npy', allow_pickle=True)
    #gc = np.load(path+folder+f'/good_clusters_playback.npy', allow_pickle=True)
    gc = np.arange(32)  
    unique_tones = np.load(path+folder+'/unique_tones.npy', allow_pickle=True) 
    save_name_tono = path+folder+f'/heatmaps_{condition}.npy'
    heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, max_freq, min_freq, condition, save_name_tono)
    
    plot_heatmap_bandwidth(heatmaps,threshold, gc,unique_tones, min_freq, max_freq, bin_width, psth_bins, t_pre,path, folder, condition)
    bd = np.load(path+folder+f'/heatmap_bandwidth_{condition}.npy')
    plotted_heatmaps = np.load(path+folder+f'/heatmap_plot_{condition}.npy')
    plot_inout(heatmaps, bd, gc,unique_tones, min_freq, max_freq, psth_bins, path, folder, condition)

#test pour swiped tonotopy
#for folder in folders : 
  #  data = np.load(path+folder+'/data.npy')
   # features = np.load(path+folder+'/features.npy', allow_pickle=True)
  #  gc = np.load(path+folder+'/good_clusters_playback.npy', allow_pickle=True)  
   # unique_#
   # psth_tracking = get_psth(data, features, t_pre, t_post, bin_width, gc, 'tracking')
   # f_tracking = get_played_frequency(features, 'tracking')

   # psth_playback = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback')
   # f_playback = get_played_frequency(features, 'playback')
    
    #all_tr, all_pb = get_baseline_per_tones(psth_tracking, psth_playback, f_tracking, f_playback, gc, unique_tones, t_pre, t_post, bin_width)
    
   # np.save(path+folder+'/mean_psth_per_tones_tracking.npy', all_tr)
   # np.save(path+folder+'/mean_psth_per_tones_playback.npy', all_pb)
    
print('all izz well')