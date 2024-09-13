import argparse
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np



folder = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/data/BURRATA/BURRATA_20240404'

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
print('tones =' ,tones_total)
print('spk =', spk)