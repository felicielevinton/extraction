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

print(bin_width)
path = '/Volumes/data2/eTheremin/ALTAI/ALTAI_20240822_SESSION_00/headstage_0'

data = np.load(path+f'/data_{bin_width}.npy', allow_pickle=True)
features = np.load(path+f'/features_{bin_width}.npy', allow_pickle=True)

# mettre une condition si good_clusters.npy n'existe pas alors gc = 32
#gc = np.arange(32)
gc = np.load(path+'/good_clusters.npy', allow_pickle=True)

#récupérer les tones joués
tones = get_played_frequency(features, t_pre, t_post, bin_width, 'playback')
# prendre les valeurs uniques de tones
unique_tones = sorted(np.unique(tones))
#récupérer les heatmaps
heatmaps = get_tonotopy(data, features, t_pre, t_post, bin_width, gc, unique_tones, max_freq, min_freq, 'playback', 'heatmaps')

plot_heatmap_bandwidth(heatmaps,threshold, gc,unique_tones, min_freq, max_freq, bin_width, psth_bins, t_pre,path, '', 'playback')
