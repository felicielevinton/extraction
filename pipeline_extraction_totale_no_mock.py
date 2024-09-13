# fichier pour tout extraure totalement les données depuis le rhd

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
from extract_data_total import *
from utils_extraction import *
from functions_get_data import *
import numpy as np
from utils_extraction import get_session_type_final
from utils_detect_spikes import *
import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit
import spikeinterface.widgets as sw
import numpy as np
from convert_positions_in_tones import *
from utils_extraction import *
import numpy as np
import spikeinterface
import zarr as zr
import os
from  pathlib import Path
import tqdm
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from probeinterface import Probe, ProbeGroup
import matplotlib
import pickle
matplotlib.use('Agg')


sr = 30e3
t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.005
psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
n_headstages = 1 # nombre de headstages utilisés pendant la manip

#pour oscypek : /mnt/working2/felicie/data2/eTheremin/OSCYPEK/OSCYPEK
path = '/mnt/working2/felicie/data2/eTheremin/ALTAI/ALTAI_20240814_SESSION_00/'
session_type = 'Playback' #TrackingOnly ou PbOnly
# 1.  extract from ephys.rhd

#extract_from_rhd(path, sampling_rate=sr, n_headstages=n_headstages, channels_to_remove=None)

#copy_files(n_headstages, path)

#session_type = get_session_type_final(path)

# 2. Create tt.pkl

create_tones_triggers_and_condition_V3(path+'headstage_0/', session_type)

print('tt.pkl created')

# 3. Detect spikes

neural_data = np.load(path +'headstage_0/neural_data.npy')
sig = neural_data

def get_fma_probe():
    """ Implements the FMA probe using Probeinterface.
        Particular care has to be taken on how the headstage is connected on the probe.
    """
    ### The following distances are in mm:
    inter_hole_spacing = 0.4  # along one row
    inter_row_spacing = np.sqrt(0.4**2-0.2**2)  # between row/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240419_SESSION_01/headstage_1/psth_figure_spikeinterface.png

    # We need to remove the ground from these positions:
    mask = np.zeros((16, 2), dtype=bool)
    mask[[0, 15], [0, 1]] = True
    positions = positions[np.logical_not(mask.reshape(-1))]

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions,
                       shapes='circle', shape_params={'radius': 100})
    polygon = [(0, 0), (0, 16000), (800, 16000), (800, 0)]
    probe.set_planar_contour(polygon)

    ## This mapping is ordered from 32 to 1, so we use[::-1] to invert it and make it 1 to 32
    mapping = np.arange(32).reshape(16, 2)[::-1].reshape(-1)
    mapping = mapping[np.logical_not(mask.reshape(-1))]

    probe.set_device_channel_indices(mapping)

    return probe


n_cpus = os.cpu_count()

full_raw_rec = se.NumpyRecording(traces_list=np.transpose(sig), sampling_frequency=sr)
#raw_rec = full_raw_rec.set_probe(probe)
#raw_rec = raw_rec.remove_channels(["CH2"])
recording_cmr = si.common_reference(full_raw_rec, reference='global', operator='median')
recording_f = si.bandpass_filter(full_raw_rec, freq_min=300, freq_max=9000)

n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
job_kwargs = dict(chunk_duration='5s', n_jobs=n_jobs, progress_bar=True)

peaks = detect_peaks(
        recording_f,
        method='by_channel',
        gather_mode="memory",
        peak_sign='neg',
        detect_threshold=2,  # thresh = 3.32 for burrata # 3.2 sinon c'est bien
        exclude_sweep_ms=0.1,
        noise_levels=None,
        random_chunk_kwargs={},

        **job_kwargs,
)

peaks_array = np.array(peaks)
spk_times = peaks_array['sample_index'].tolist()
spk_clusters = peaks_array['channel_index'].tolist()
np.save(path+'/spike_times.npy',spk_times )
np.save(path+'/spike_clusters.npy',spk_clusters)

clusters = {}
for value, cluster in zip(spk_times, spk_clusters):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(value)
for cluster, values in clusters.items():
    print(f"Cluster {cluster}: {len(values)}")
    
#triggers
pkl_path = path+'headstage_0/'+'tt.pkl'

with open(pkl_path, 'rb') as file:
    triggers_data = pickle.load(file)
an_times =  triggers_data['triggers']


#PLOT
# Créer les bins de temps"
window = [-t_pre, t_post]

import numpy as np
import matplotlib.pyplot as plt

def compute_psth(spike_times, stimulus_times, bin_size, window):
    # Combine all spikes relative to stimulus times
    stimulus_times = stimulus_times/sr
    spike_times = spike_times/sr
    all_spikes = []
    for stim_time in stimulus_times:
        relative_spikes = spike_times - stim_time
        all_spikes.extend(relative_spikes[(relative_spikes >= window[0]) & (relative_spikes <= window[1])])
    
    # Create histogram
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    hist, bin_edges = np.histogram(all_spikes, bins=bins)
    
    # Normalize to get the firing rate
    psth = hist / (len(stimulus_times) * bin_size)
    #psth=hist
    return psth, bin_edges

gc = np.arange(0,32)
num_plots, num_rows, num_columns = get_better_plot_geometry(gc)


fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Heatmaps clusters', y=1.02)
plt.subplots_adjust() 


stim_times = an_times
    #
for cluster in range(num_plots):
    if cluster < num_plots:
        row, col = get_plot_coords(cluster)
        spike_times = np.array(clusters[cluster])
        psth,edges = compute_psth(spike_times, stim_times, bin_width, window)
        axes[row, col].plot(psth_bins, psth)
        axes[row, col].axvline(0, c = 'black', linestyle='--')
        axes[row, col].set_title(f'Cluster {cluster}')
fig.tight_layout()
fig.savefig(path+'psth_figure_spikeinterface.png') 


# 4. Create data.npy et features. npy
create_data_features(path+'headstage_0/', bin_width, sr)

print('all is well')
