import argparse
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(prog="Tonotopy")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    opt = parse_args()
    spk = ut.Spikes(opt.folder)
    an_triggers = np.load(os.path.join(opt.folder, "analog_in.npy"))

    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    tables = list()

    frequencies, tones_total, triggers_spe, tag = get_data(opt.folder, trigs=an_times)

    l_spikes = list()
    hm_tonotopy = hm.Heatmap()
    hm_tonotopy.compute_heatmap(trigs=an_times, tone_sequence=tones_total, spikes=spk, t_pre=0.100, t_post=0.450,
                                bin_size=0.002)
    # hm_tonotopy.plot("Tonotopy", folder=opt.folder, ext="png")
    hm_tonotopy.plot_smooth_2d("Tonotopy", folder=opt.folder, ext="png")
    hm_tonotopy.save(folder=opt.folder, typeof="tonotopy")
