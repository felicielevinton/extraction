from load_rhd import *
from quick_extract import *
from get_data import *
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np
path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA'
save_path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA/BURRATA_20240404'

#load_rhd(path + '/ephys.rhd', save_path, digital=True, analog=True, accelerometer=True, filtered=True, export_to_dat=False)

#quick_extract(path+'/BURRATA_20240404/filtered_neural_data.npy', mode="relative", threshold=-3.7)

folder = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA/BURRATA_20240404'
#extract_data(folder, analog_channels=None, digital_channels=None, compatibility=False)
trigs = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA/BURRATA_20240404/analog_in.npy'
def __tonotopy(n_tones, trigs, sequence):
    f = np.unique(sequence)
    if n_tones is not None:
        trigs = np.hstack((trigs[:n_tones], trigs[-n_tones:]))
    if len(trigs) < len(sequence):
        sequence = sequence[:len(trigs)]
    return f, sequence, trigs, "_tonotopy"


def get_tonotopy_flavien(folder, trigs):
    tonotopy_seq = np.empty(0)
    file = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA/BURRATA_20240404/tones/tones_pt_01_BURRATA_SESSION_04_20240404.bin'
    tonotopy_seq = np.hstack((tonotopy_seq, np.fromfile(file, dtype=np.double)))
    print(tonotopy_seq)
    if len(tonotopy_seq) > 0:
        n_tones = int(len(tonotopy_seq) / 2)
        return tonotopy_seq
    else:
        return 0, 0, 0
    
    
sequence = get_tonotopy_flavien(folder, trigs)



spk = ut.Spikes(folder)
an_triggers = np.load(os.path.join(folder, "analog_in.npy"))

an_times = ut.extract_analog_triggers_compat(an_triggers[0])
tables = list()

frequencies, tones_total, triggers_spe, tag = get_data(folder, trigs=an_times)

l_spikes = list()
hm_tonotopy = hm.Heatmap()
#hm_tonotopy.compute_heatmap(trigs=an_times, tone_sequence=tones_total, spikes=spk, t_pre=0.100, t_post=0.450,bin_size=0.002)
    # hm_tonotopy.plot("Tonotopy", folder=opt.folder, ext="png")
#hm_tonotopy.plot_smooth_2d("Tonotopy", folder=folder, ext="png")
#hm_tonotopy.save(folder=folder, typeof="tonotopy")
print('times', an_times)
print('trigs =', an_triggers)
print('seq =' , sequence)
print('spk =', spk)