import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit
import spikeinterface.widgets as sw
import numpy as np
from convert_positions_in_tones import *
from utils_extraction import *
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sr=30e3

root = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/OSCYPEK/OSCYPEK_20240708_SESSION_02/'
path = root+'headstage_0/'



def highpass_filter(data, cutoff, fs, order=5):
    """
    Applique un filtre passe-haut à chaque signal dans une matrice.

    Parameters:
    data (2D array): La matrice de signaux, chaque ligne étant un signal.
    cutoff (float): La fréquence de coupure du filtre.
    fs (float): La fréquence d'échantillonnage du signal.
    order (int): L'ordre du filtre.

    Returns:
    2D array: La matrice de signaux filtrés.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    # Appliquer le filtre à chaque ligne de la matrice
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, data[i, :])
    
    return filtered_data

def get_triggers(path):
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    return an_times

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



# load le rhd
load_rhd(root+'ephys.rhd', root, digital=True, analog=True, accelerometer=True, filtered=False, export_to_dat=False)
neural_data = np.load(root +'/neural_data.npy')

sig = neural_data[32:] 

#Passe haut

cutoff = 300.0  # Fréquence de coupure en Hz

# Appliquer le filtre passe-haut à chaque signal dans la matrice
filtered_signals = highpass_filter(sig, cutoff, sr, order=5)
np.save(path+'/high_pass_neural_data.npy', filtered_signals)

# Extraire les spikes
quick_extract(path+'/high_pass_neural_data.npy', mode="absolute", threshold=-70)

spk_times = np.load(path+'/spike_times.npy')
spk_clusters = np.load(path+'/spike_clusters.npy')



# regrouper les spikes par cluster
clusters = {}
for value, cluster in zip(spk_times, spk_clusters):
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(value)
for cluster, values in clusters.items():
    print(f"Cluster {cluster}: {len(values)}")

# Extraire les triggers
an_times = get_triggers(path)

#PLOT

t_pre = 0.2#0.2
t_post = 0.50#0.300
bin_width = 0.02
# Créer les bins de temps"
psth_bins = np.arange(-t_pre, t_post, bin_width)
window = [-t_pre, t_post]




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
fig.savefig(path+'psth_figure.png')     