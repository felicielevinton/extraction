
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
from format_data import *
from utils import *
from utils_tonotopy import *
sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.002
#bin_width = 0.02
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
max_freq = 1
min_freq=0 #3 for A1
threshold = 3.2 #threshold for contour detection 3.2 is good
gc = np.arange(0,32)

path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240417_SESSION_00/hdstg2/'

neural = np.load('/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240417_SESSION_00/filtered_neural_data.npy')

filter_and_cmr_chunked(neural[32:], sr, path, int(len(neural)/10))
quick_extract(path+'/refiltered_neural_data.npy')