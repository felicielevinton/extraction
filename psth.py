import PostProcessing.tools.utils as ut
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
from get_data import *
import glob


def parse_args():
    parser = argparse.ArgumentParser(prog="PSTH")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    parser.add_argument("--cells", type=int, nargs="+")
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    opt = parse_args()
    spk = ut.Spikes(opt.folder)
    triggers = np.load(os.path.join(opt.folder, "dig_in.npy"))[1]
    triggers = triggers.astype(np.int8)
    triggers = ut.extract_trigger_time(triggers)
    _, seq, triggers_tono, tag_tono = get_data(opt.folder, trigs=triggers,
                                               tonotopy_only=True, tracking_only=False)
    tones, seq_track, triggers_track, tag_track = get_data(opt.folder, trigs=triggers,
                                                           tonotopy_only=False, tracking_only=True)
    clusters = opt.cells
    t_pre = 0.2
    t_post = 0.75
    bin_size = 0.01
    bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    centers = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        centers[i] = (bins[i + 1] + bins[i]) / 2
    # print(centers)
    for cluster in clusters:
        tonotopy = list()
        tracking = list()
        for tone in tones:
            idx_tono = np.where(seq == tone)[0]
            psth_tono, _ = ut.psth(spikes=spk.get_spike_times(cluster), triggers=triggers_tono[idx_tono],
                                   bins=bins, t_0=t_pre, t_1=t_post, bin_size=bin_size)
            idx_tracking = np.where(seq_track == tone)[0]
            psth_tracking, _ = ut.psth(spikes=spk.get_spike_times(cluster), triggers=triggers_track[idx_tracking],
                                       bins=bins, t_0=t_pre, t_1=t_post, bin_size=bin_size)
            tracking.append(psth_tracking)
            tonotopy.append(psth_tono)
            # plt.plot(psth_tracking, c="g", linewidth=0.5)
            # plt.plot(psth_tono, c="r", linewidth=0.5)
            # plt.show()
        for i in range(len(tonotopy)):
            plt.plot(centers, ut.gaussian_smoothing(tonotopy[i]), c="k", linewidth=0.5)
            plt.plot(centers, ut.gaussian_smoothing(tracking[i]), c="orange", linewidth=0.5)
            plt.axvline(0, c="r")
            plt.show()
