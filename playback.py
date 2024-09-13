import matplotlib.pyplot as plt
import argparse
from get_data import *
import numpy as np
import os
import PostProcessing.tools.utils as ut
from scipy import signal


def plot_average_psth(c, hm_list, path):
    psths = dict()
    best_tone = hm_list["playback"].get_best_frequency(c)
    relevant_tones = [2**-0.5 * best_tone, 2**0.5 * best_tone]
    for key in hm_list.keys():
        ix = np.logical_and(hm_list[key].get_tones() >= relevant_tones[0], hm_list[key].get_tones() <= relevant_tones[1])
        psths[key] = hm_list[key].get_hm_1_cluster(c)[ix].mean(0)
        plt.plot(hm_list[key].get_hm_1_cluster(c)[ix].mean(0), label=key)

    plt.title(f"Average PSTH on 1 oct around best responding tone for cluster {c}")
    plt.legend()
    plt.savefig(os.path.join(path, f"cluster_{c}.png"), dpi=240)
    plt.close()
    return psths


def plot_tc(c, hm_list, path):
    for key in hm_list.keys():
        conv_hm = hm_list[key].get_hm_1_cluster(c)
        smooth, bf_positions = hm_list[key].detect_peak(c)
        best_tone = bf_positions[1]
        for i in range(conv_hm.shape[0]):
            conv_hm[i] = (conv_hm[i] - conv_hm[i].mean()) / conv_hm[i].std()
        kernel = np.outer(signal.windows.gaussian(3, 3), signal.windows.gaussian(3, 3))
        conv_hm = signal.convolve(conv_hm, kernel, "same")
        conv_hm -= conv_hm.mean()
        conv_hm /= conv_hm.std()
        tones = hm_list[key].get_tones()
        plt.plot(tones, conv_hm[:, best_tone], label=key)
    plt.title(f"Tuning curve cluster {c}")
    plt.legend()
    plt.savefig(os.path.join(path, f"tc_cluster_{c}.png"), dpi=240)
    plt.close()


def plot_activity(*args):
    return


def get_triggers(list_snippets):
    l_triggers = list()
    for snippet in list_snippets:
        l_triggers.append(snippet[1])
    return l_triggers


def parse_args():
    parser = argparse.ArgumentParser(prog="Playback")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    _opt = parser.parse_args()
    return _opt


# if __name__ == "__main__":
#     options = parse_args()
#     plot_path = os.path.join(options.folder, "plots")
#     plot_hm = True
#     if not os.path.exists(plot_path):
#         os.mkdir(plot_path)
#
#     # PARAMS
#     t_pre = 0.01
#     t_post = 0.1
#     bin_size = 0.002
#     sr = 30e3
#
#     trig_and_tones, recording_length = main(options)
#     spk = ut.Spikes(options.folder, recording_length=recording_length)
#
#     pb_pattern = "pb_[0-9]"
#     tracking_pattern = "tracking_[0-9]"
#     mock_pattern = "mock_[0-9]"
#     warmup_pattern = "warmup_[0-9]"
#     if "warmup_0" in trig_and_tones.keys():
#         pass
#     warmup = trig_and_tones["warmup_0"]
#
#     pb = merge_pattern_3(trig_and_tones, pb_pattern)
#     tr = merge_pattern(trig_and_tones, tracking_pattern)
#     mck = merge_pattern(trig_and_tones, mock_pattern)
#     cluster_i_want_to_see = [2, 3, 4, 13, 20, 21, 22, 24, 25, 26, 27, 30]
#
#     hm_warmup = hm.Heatmap()
#     hm_warmup.compute_heatmap(tone_sequence=warmup[0], trigs=warmup[1],
#                               spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
#
#     if plot_hm:
#         hm_warmup.plot("warmup", plot_path, l_ex=2, r_ex=1, ext="png")
#         hm_warmup.plot_bf("warmup", plot_path, ext="png")
#         hm_warmup.plot_smooth("warmup", plot_path, l_ex=2, r_ex=1, ext="png")
#     if "warmup_1" in trig_and_tones.keys():
#         warmout = trig_and_tones["warmup_1"]
#         hm_warmout = hm.Heatmap()
#         hm_warmout.compute_heatmap(tone_sequence=warmout[0], trigs=warmout[1],
#                                    spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
#         if plot_hm:
#             hm_warmout.plot("warmout", plot_path, l_ex=2, r_ex=1, ext="png")
#             hm_warmout.plot_bf("warmout", plot_path, ext="png")
#             hm_warmout.plot_smooth("warmout", plot_path, l_ex=2, r_ex=1, ext="png")
#
#     hm_tracking = hm.Heatmap()
#     hm_tracking.compute_heatmap(tone_sequence=tr[0], trigs=tr[1],
#                                 spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
#     if plot_hm:
#         hm_tracking.plot("tracking_total", plot_path, l_ex=2, r_ex=2, ext="png")
#         hm_tracking.plot_smooth("tracking_total", plot_path, l_ex=2, r_ex=2, ext="png")
#         hm_tracking.plot_bf("tracking_total", plot_path, ext="png")
#     hm_playback = hm.Heatmap()
#     hm_playback.compute_heatmap(tone_sequence=pb[0][1:], trigs=pb[1][1:],
#                                 spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
#     if plot_hm:
#         hm_playback.plot("playback", plot_path, l_ex=2, r_ex=2, ext="png")
#         hm_playback.plot_smooth("playback", plot_path, l_ex=2, r_ex=2, ext="png")
#         hm_playback.plot_bf("playback", plot_path, ext="png")
#     hm_conc = hm.concatenate(hm_tracking, hm_playback)
#     if plot_hm:
#         hm_conc.plot("conc_all", plot_path, l_ex=2, r_ex=1, ext="pdf")
#
#     hm_diff = hm_tracking - hm_playback
#     if plot_hm:
#         hm_diff.plot("diff", plot_path, l_ex=2, r_ex=2, ext="pdf")
#         hm_diff.plot_smooth("diff", plot_path, l_ex=2, r_ex=2, ext="pdf")
#     if options.folder == "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221107/MANIGODINE_20221107_SESSION_00":
#         clusters = [2, 3, 4, 11, 12, 13, 20, 21, 22, 23, 24, 25, 27]
#     elif options.folder == "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221117/MANIGODINE_20221117_SESSION_00":
#         clusters = [3, 4, 20, 22, 24, 25, 27]
#     elif options.folder == "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221118/MANIGODINE_20221118_SESSION_00":
#         clusters = [4, 24, 27]
#     elif options.folder == "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221121/MANIGODINE_20221121_SESSION_00":
#         clusters = [20, 21, 25, 27, 30]
#     elif options.folder == "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221129/MANIGODINE_20221129_SESSION_00":
#         clusters = [2, 4, 19, 20, 21, 24, 25, 26, 27]
#     else:
#         clusters = [0]
#     # l_hm = [hm_warmup, hm_tracking, hm_playback, hm_warmout]
#     l_hm_lab = {
#         # "warmup": hm_warmup,
#         # "warmout": hm_warmout,
#         "tracking": hm_tracking,
#         "playback": hm_playback
#         }
#     playback_save = list()
#     tracking_save = list()
#     for cluster in clusters:
#         psths_out = plot_average_psth(cluster, l_hm_lab, plot_path)
#         playback_save.append(psths_out["playback"])
#         tracking_save.append(psths_out["tracking"])
#         plot_tc(cluster, l_hm_lab, plot_path)
#     path_data = os.path.join(options.folder, "data_stats")
#     if not os.path.exists(path_data):
#         os.mkdir(path_data)
#     playback_save = np.vstack(playback_save)
#     tracking_save = np.vstack(tracking_save)
#     np.save(os.path.join(path_data, "playback.npy"), playback_save)
#     np.save(os.path.join(path_data, "tracking.npy"), tracking_save)
#     pb_l = merge_pattern_2(trig_and_tones, pb_pattern)
#     tr_l = merge_pattern_2(trig_and_tones, tracking_pattern)
#     mock_l = merge_pattern_2(trig_and_tones, mock_pattern)
#
#     # GARBAGE COLLECTOR
#     # plot activity for cell
#     cl = [4, 20, 21, 24, 25, 27]
#     activity_warmup = ut.activity_baseline(spk, cl, warmup[1], 5)
#     activity_warmout = ut.activity_baseline(spk, cl, warmout[1], 5)
#     activity_tracking_interleaved = ut.activity_snippet(spk, cl, get_triggers(tr_l))
#     activity_playback = ut.activity_snippet(spk, cl, get_triggers(pb_l))
#
#     ll = {
#         "warmup": activity_warmup,
#         "tracking": activity_tracking_interleaved,
#         "playback": activity_playback,
#         "warmout": activity_warmout
#           }
#
#     std_warmup = activity_warmup.std(1)
#     mean_warmup = activity_warmup.mean(1)
#     ticks = np.arange(1, 6)
#     plt.plot(activity_warmup.mean(1))
#     plt.plot(activity_tracking_interleaved.mean(1))
#     plt.plot(activity_playback.mean(1))
#     plt.plot(activity_warmout.mean(1))
#     plt.savefig(os.path.join(plot_path, "activity_cell_interest.png"), dpi=240)
#     plt.close()
