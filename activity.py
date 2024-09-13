import matplotlib.pyplot as plt
import argparse
from get_data import *
import numpy as np
import os
import PostProcessing.tools.utils as ut
import PostProcessing.tools.accelerometer as acu
from scipy import signal


def parse_args():
    parser = argparse.ArgumentParser(prog="Activity")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    options = parse_args()
    plot_path = os.path.join(options.folder, "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    # PARAMS
    t_pre = 0.1
    t_post = 0.35
    bin_size = 0.02
    sr = 30e3
    bin_duration = 0.025
    cluster = 25
    window = 50000

    spk = ut.Spikes(options.folder, recording_length=recording_length)
    binned_activity, _ = spk.get_binned_activity(cluster, bin_duration)
    acc = acu.Accelerometer(options.folder)
    ax = acc.get_x_std(bin_duration)
    pb_pattern = "pb_[0-9]"
    tracking_pattern = "tracking_[0-9]"
    mock_pattern = "mock_[0-9]"
    warmup_pattern = "warmup_[0-9]"
    warmup = trig_and_tones["warmup_0"]
    warmout = trig_and_tones["warmup_1"]

    pb = merge_pattern_3(trig_and_tones, pb_pattern)
    tr = merge_pattern(trig_and_tones, tracking_pattern)
    cluster_i_want_to_see = [2, 3, 4, 13, 20, 21, 22, 24, 25, 26, 27, 30]
    activity_warmup = ut.activity_baseline(spk, cluster_i_want_to_see, warmup[1], 5)
    activity_warmout = ut.activity_baseline(spk, cluster_i_want_to_see, warmout[1], 5)
    binned_activity_warmup, _ = spk.get_binned_activity_between(cluster, warmup[1][0], warmup[1][-1], bin_duration)
    warmup_acc_x = acc.get_x_binned_between(warmup[1][0], warmup[1][-1], bin_duration)
    binned_activity_warmout, _ = spk.get_binned_activity_between(cluster, warmout[1][0], warmout[1][-1], bin_duration)
    warmout_acc_x = acc.get_x_binned_between(warmout[1][0], warmout[1][-1], bin_duration)

    corr_wt = signal.correlate(warmout_acc_x, binned_activity_warmout, mode="full")
    lags_wt = signal.correlation_lags(len(binned_activity_warmout), len(warmout_acc_x))
    idx_wt = np.logical_and(lags_wt > -window, lags_wt < window)
    # corr_wt = ut.gaussian_smoothing(corr_wt, sigma=3, size=6)
    # corr_wt /= np.amax(corr_wt)

    corr = signal.correlate(binned_activity_warmup, warmup_acc_x, mode="full")
    lags = signal.correlation_lags(len(binned_activity_warmup), len(warmup_acc_x))
    idx = np.logical_and(lags > -window, lags < window)
    # corr = ut.gaussian_smoothing(corr, sigma=3, size=6)
    # corr /= np.amax(corr)
    index = np.argmax(corr)
    print(f"Warmup max at: {lags[index] * bin_duration} s")
    index = np.argmax(corr_wt)
    print(f"Warmout max at: {lags_wt[index] * bin_duration} s")
    # plt.plot(lags, corr)
    # plt.plot(lags_wt, corr_wt)
    # plt.axvline(0, c="r")
    # plt.show()

    # faire ça pour les trackings.
    tr_l = merge_pattern_2(trig_and_tones, tracking_pattern)
    l_corr = list()
    for i, elt in enumerate(tr_l):
        binned_activity_tr, _ = spk.get_binned_activity_between(cluster, elt[1][0], elt[1][-1], bin_duration)
        binned_acc_tr = acc.get_x_binned_between(elt[1][0], elt[1][-1], bin_duration)
        corr_tr = signal.correlate(binned_acc_tr, binned_activity_tr, mode="full")
        lags_tr = signal.correlation_lags(len(binned_activity_tr), len(binned_acc_tr))
        idx_tr = np.logical_and(lags_tr > -window, lags_tr < window)
        # corr_tr = ut.gaussian_smoothing(corr_tr, sigma=5, size=6)
        # corr_tr /= np.amax(corr_tr)
        index = np.argmax(corr_tr)
        print(f"{i} max at: {lags_tr[index] * bin_duration} s")
        corr_tr = corr_tr[idx_tr]
        lags_tr = lags_tr[idx_tr]
        l_corr.append(np.vstack((corr_tr, lags_tr)))
    # for i, elt in enumerate(l_corr):
    #     plt.plot(elt[1], elt[0], label=f"{i}")
    # plt.plot(lags[idx], corr[idx])
    # plt.plot(lags_wt[idx_wt], corr_wt[idx_wt])
    # plt.axvline(0, c="r")
    #plt.legend()
    #plt.title("Tracking Correlation")
    #plt.show()

    #l_corr = list()
    #for i, elt in enumerate(pb_l):
    #    binned_activity_tr, _ = spk.get_binned_activity_between(cluster, elt[1][0], elt[1][-1], bin_duration)
    #    binned_acc_tr = acc.get_x_binned_between(elt[1][0], elt[1][-1], bin_duration)
    #    corr_tr = signal.correlate(binned_acc_tr, binned_activity_tr, mode="full")
    #    lags_tr = signal.correlation_lags(len(binned_activity_tr), len(binned_acc_tr))
    #    idx_tr = np.logical_and(lags_tr > -window, lags_tr < window)
    #    # corr_tr = ut.gaussian_smoothing(corr_tr, sigma=5, size=5)
    #    # corr_tr /= np.amax(corr_tr)
    #    index = np.argmax(corr_tr)
    #    print(f"{i} max at: {lags_tr[index] * bin_duration} s")
#
    #    corr_tr = corr_tr[idx_tr]
    #    lags_tr = lags_tr[idx_tr]
    #    l_corr.append(np.vstack((corr_tr, lags_tr)))
    # for elt in l_corr:
    #    plt.plot(elt[1], elt[0])
    # plt.axvline(0, c="r")
    # plt.title("PB Correlation")
    # plt.show()
    pb_l = merge_pattern_2(trig_and_tones, pb_pattern)
    tr2 = tr_l[2]
    pb2 = pb_l[2]

    # s'intéresser à une fréquence.
    i_f = 8701.
    time_around = 2
    l_corr_total = list()
    for i, elt in enumerate(tr_l):
        tg_tr = elt[1][np.where(elt[0] == i_f)[0]]
        l_corr_tr = list()
        for j, trigger in enumerate(tg_tr):
            ax_around = acc.get_x_binned_around(trigger, time_around, bin_duration)
            act_around, _ = spk.get_spikes_activity_around(cluster, trigger, time_around, bin_duration)
            corr = signal.correlate(act_around, ax_around, mode="full")
            l_corr_tr.append(corr)
        l_corr_total.append(np.vstack(l_corr_tr))

    for xcorr in l_corr_total:
        plt.plot(xcorr.mean(0))
    plt.title("fs")
    plt.show()

    tg_tr = tr2[1][np.where(tr2[0] == i_f)[0]]

    l_corr_test = list()
    for i, elt in enumerate(tg_tr):
        ax_around = acc.get_x_binned_around(elt, time_around, bin_duration)
        act_around, _ = spk.get_spikes_activity_around(cluster, elt, time_around, bin_duration)
        corr = signal.correlate(act_around, ax_around, mode="full")
        lags_tr = signal.correlation_lags(len(ax_around), len(act_around))

        # idx_tr = np.logical_and(lags_tr > -window, lags_tr < window)
        l_corr_test.append(corr)

    tg_pb = pb2[1][np.where(pb2[0] == i_f)[0]]
    l_corr_pb = list()
    for i, elt in enumerate(tg_pb):
        ax_around = acc.get_x_binned_around(elt, time_around, bin_duration)
        act_around, _ = spk.get_spikes_activity_around(cluster, elt, time_around, bin_duration)
        corr = signal.correlate(act_around, ax_around, mode="full")
        lags_tr = signal.correlation_lags(len(ax_around), len(act_around))
        l_corr_pb.append(corr)

    mu_corr = np.vstack(l_corr_test)
    plt.plot(mu_corr.mean(0))
    mu_corr_pb = np.vstack(l_corr_pb)
    plt.plot(mu_corr_pb.mean(0))
    plt.title("fs tr and pb")
    plt.show()

    # essayer de retirer les bins d'activité aux onsets.
    # extraire les bonnes bins d'accéléromètre
    # convertir les triggers en numéro de bins.
