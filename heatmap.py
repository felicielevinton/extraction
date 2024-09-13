import PostProcessing.tools.utils as ut
import PostProcessing.tools.plot_mp as pmp
import matplotlib.pyplot as plt
import argparse
from scipy import signal
from get_data import *
import numpy as np
import glob


def parse_args():
    parser = argparse.ArgumentParser(prog="HeatMap")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    parser.add_argument("--tonotopy", type=bool, help="Tonotopy only", default=False)
    parser.add_argument("--tracking", type=bool, help="Use tracking only", default=False)
    parser.add_argument("--all", type=bool, help="Everything!", default=False)
    parser.add_argument("--playback", type=bool, help="Use tracking only", default=False)
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    opt = parse_args()
    spk = ut.Spikes(opt.folder)
    triggers = np.load(os.path.join(opt.folder, "dig_in.npy"))[1]
    triggers = triggers.astype(np.int8)
    dig_times, _ = signal.find_peaks(triggers, 1.0, distance=150)
    foo = np.copy(triggers)
    triggers = ut.extract_digital_triggers(triggers)
    an_triggers = np.load(os.path.join(opt.folder, "analog_in.npy"))
    an_times = ut.extract_analog_triggers(an_triggers[0])
    an_times_pb = ut.extract_analog_triggers(an_triggers[1])
    tables = list()
    new_an_times = an_times[np.logical_and(an_times >= dig_times[0], an_times <= dig_times[-1] + 500)]
    plt.plot(dig_times[:200])
    plt.plot(new_an_times[:200])
    print(len(new_an_times))
    print(len(dig_times))
    plt.show()

    if opt.all:
        tables = [[False, False], [True, False], [False, True]]
    elif opt.tonotopy:
        tables = [[True, False]]
    elif opt.tracking:
        tables = [[False, True]]
    for table in tables:
        frequencies, tones_total, triggers_spe, tag = get_data(opt.folder, trigs=new_an_times,
                                                               tonotopy_only=table[0], tracking_only=table[1])

        l_spikes = list()
        print(len(frequencies))
        for cluster in range(spk.get_n_clusters()):
            xx = spk.get_spike_times(cluster=cluster)
            hm, bin_edges, freqs = ut.heatmap(tone_sequence=tones_total, trigs=triggers_spe, spikes=xx,
                                              t_post=0.4,
                                              t_pre=0.1,
                                              bin_size=0.005)
            l_spikes.append(hm)
        l_spikes = np.array(l_spikes)
        pmp.plot_mp_fma_32_sd(l_spikes, "tag", opt.folder)
