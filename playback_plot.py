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
from scipy.signal import find_peaks
import numpy as np
from utils import *
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.30#0.300
bin_width = 0.005
#bin_width = 0.02
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
max_freq = 2
min_freq=2 #3 for A1
threshold = 2 #threshold for contour detection 3.2 is good

#path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/FRINAULT/'
path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/MUROLS/'
folders = get_folders(path)

for i in range(1,len(folders)):
    
    all_baseline, all_peak, all_unique_tones = [], [], []
    
    folder = folders[i]
    features = np.load(path+folder+'/features.npy', allow_pickle=True)
    data = np.load(path+folder+'/data.npy')
    gc = np.load(path+folder+'/good_clusters_playback.npy', allow_pickle=True)  
    unique_tones = np.load(path+folder+'/unique_tones.npy', allow_pickle=True)
    heatmaps = np.load(path+folder+'/heatmaps_playback.npy', allow_pickle=True)
    
    psth = get_psth(data, features, t_pre, t_post, bin_width, gc, 'playback')
    played_f = get_played_frequency(features, 'playback')
    mock_f = get_mock_frequency(features)
    best_f= np.load(path+folder+'/best_frequency_playback.npy')
    bd = np.load(path+folder+'/heatmap_bandwidth_playback.npy')

    for cluster, ax in enumerate(gc):
        bf=best_f[cluster]
        #index = indices_valeurs_egales(played_f, bf) # je prends quand la frequence jouée est la BF du neurone
        fmin, fmax = bd[cluster][0], bd[cluster][1]
        #je ne garde que les clusters dont la bandwidth est inférieure à 2 octaves:
        if np.log2(fmax/unique_tones[0])-np.log2(fmin/unique_tones[0])<2:
            index=indices_valeurs_comprises(played_f, fmin, fmax)
            psth_bf = [psth[cluster][i] for i in index] #psth quand les sons joués sont à la BF
                
            position_bf = [mock_f[i] for i in index] #mock frequencies qui aurait dû etre jouées alors que c'est la BF --> c'est la position du museau
            psth_per_f, n_presentations = moyenne_psth_par_frequence(psth_bf, position_bf, unique_tones,10)
            #calculer la baseline moyenne par mock
            baseline = [get_sustained_activity_nan(psth, t_pre, t_post, bin_width) for psth in psth_per_f]
            # calculer le peak moyen par mock
            peak = [mean_maxima_nan(psth, 1, 40, 60)[0] for psth in psth_per_f] 
            plt.scatter(np.log2(unique_tones/unique_tones[0]), peak, label = 'peak', markers = 'Square')
            plt.scatter(np.log2(unique_tones/unique_tones[0]), baseline, label = 'baseline', marker = 'o')
            plt.axvline(x=np.log2(bf/unique_tones[0]), color='g', linestyle='--')
            plt.axvline(x=np.log2(fmin/unique_tones[0]), color='r', linestyle='--')
            plt.axvline(x=np.log2(fmax/unique_tones[0]), color='r', linestyle='--')
            plt.legend()
            plt.title(f'cluster {bf}')
            plt.savefig(f'/mnt/working2/felicie/Python_theremin/Analyse/Analyse/outputs/{folder}_cluster_{ax}.png')
        else:
            pass
            
            
           