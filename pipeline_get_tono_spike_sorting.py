from kneed import DataGenerator, KneeLocator
#import tkinter
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
from extract_data_total import *
import PostProcessing.tools.utils as ut
from PostProcessing.tools.extraction import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json
import pickle
from functions_get_data import *
import matplotlib
matplotlib.use('Agg')
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
#bin_width = 0.02
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
max_freq = 3
min_freq=1 #3 for A1
threshold = 3 #3.2 #threshold for contour detection 3.2 is good


freq_min = 3 #Hz


#sessions = ['ALTAI_20240710_SESSION_00']
sessions = ['MUROLS_20230218_SESSION_01']
for session in sessions:

    # path = 'Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/headstage_0' 
    #chemin  = 'Z:/eTheremin/ALTAI/'  + session + '/'
    chemin  = 'Z:/eTheremin/MUROLS/MUROLS_20230218/' + session + '/' 

    #num_channel = [31, 30, 16, 14,  1, 23, 29,  5, 17, 27, 25,  8, 28, 26,3,  2,  6, 20]
    if os.path.exists(chemin + 'headstage_0' + '/good_clusters.npy'):
        good_cluster = np.load(chemin + 'headstage_0' + '/good_clusters.npy', allow_pickle = True)
    else : 
        good_cluster = np.arange(32)
    print(good_cluster)
    save_path = 'Y:/eTheremin/clara/' + session + '/'#+ 'filtered/std.min =5 bis/'


    print(bin_width)


    data = np.load(save_path + f'data_ss_{bin_width}.npy', allow_pickle=True)
    features = np.load(save_path +f'features_{bin_width}.npy', allow_pickle=True)
    unique_conditions = set(d['Condition'] for d in features)
    nbr_spikes_min = len(data[0])*freq_min*bin_width
    print("nbr-spikesmin:",nbr_spikes_min)

    spk_clusters = np.load(save_path +'ss_spike_clusters.npy', allow_pickle=True)
    k, counts = np.unique(spk_clusters, axis=0,return_counts=True)
    count_dict = {tuple(row): count for row, count in zip(k, counts)}

    #enlève les clusters avec peu de spikes
    # filtered_rows = [row for row in k if count_dict[tuple(row)] > n_spikes_min]
    # filtered_count_dict = {tuple(row): count_dict[tuple(row)] for row in filtered_rows}

    # mettre une condition si good_clusters.npy n'existe pas alors gc = 32
    gc = np.arange(len(k))

    #récupérer les tones joués
    tones = get_played_frequency(features, t_pre, t_post, bin_width, 'playback')
    # prendre les valeurs uniques de tones
    unique_tones = sorted(np.unique(tones))
    #récupérer les heatmaps
    heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, max_freq, min_freq, 'playback', 'spike_sorting_heatmaps')
    print(f"heatmap : {heatmaps}")
    cluster_order = np.load(save_path + 'ss_spike_clusters.npy', allow_pickle = True)
    plot_heatmap_bandwidth(heatmaps,threshold,good_cluster,cluster_order,count_dict, k, nbr_spikes_min,unique_tones, min_freq, max_freq, bin_width, psth_bins, t_pre,save_path, '', 'spike_sorting_playback')

