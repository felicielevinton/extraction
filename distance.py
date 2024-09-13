import numpy as np
import PostProcessing.tools.utils as ut
import PostProcessing.tools.heatmap as hm
from PostProcessing.tools.extraction import Pair, TT
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import hankel
import os


def virtual_tones(seq_tt):
    """

    """
    p_to_all = list()
    p_tr_all = list()
    t_to_all = list()
    t_tr_all = list()
    m_to_all = list()
    m_tr_all = list()
    p_tr = list()

    for k in range(seq_tt.get_n_iter()):
        pb = seq_tt.get_from_type_and_number("playback", k)
        mck = seq_tt.get_from_type_and_number("mock", k)
        tr = seq_tt.get_from_type_and_number("tracking", k)
        mck_pair = TT(tones=np.hstack((tr.get_tones()[-1], mck.get_tones())),
                      triggers=np.hstack((tr.get_triggers()[-1], mck.get_triggers())))

        pb_tones, pb_trigs = pb.get_tones(), pb.get_triggers()
        mck_tones, mck_trigs = mck_pair.tones, mck_pair.triggers
        tr_tones, tr_trigs = tr.get_tones(), tr.get_triggers()
        d = np.zeros(len(pb_trigs))

        h = hankel(mck_trigs)[:, :2]

        n = h.shape[0]

        for j in range(n):
            borders = h[j]
            if j == n - 1:
                idx = np.greater(pb_trigs, borders[0])
            else:
                idx = np.logical_and(pb_trigs >= borders[0], pb_trigs <= borders[1])
            d[idx] = mck_tones[j]

        print(np.unique(mck_tones))

        p_tr.append(d)
        p_to_all.append(pb_tones)
        p_tr_all.append(pb_trigs)
        m_to_all.append(mck_tones)
        m_tr_all.append(mck_trigs)
        t_to_all.append(tr_tones)
        t_tr_all.append(tr_trigs)

    p_tr = np.hstack(p_tr)
    p_to_all = np.hstack(p_to_all)
    p_tr_all = np.hstack(p_tr_all)
    t_to_all = np.hstack(t_to_all)
    t_tr_all = np.hstack(t_tr_all)
    m_to_all = np.hstack(m_to_all)

    return p_tr, p_to_all, p_tr_all, t_to_all, t_tr_all, m_to_all


def compute_activity_with_distance(spikes, cluster, heatmap, pb_tones, pb_triggers, virtual, bin_size, first_trigger, absolute=False):
    x = spikes.get_spike_times(cluster)
    mean_activity, std_activity = spikes.get_mean_std_activity(interval=[0, first_trigger],
                                                               cluster=cluster,
                                                               bin_duration=bin_size)
    # On attrape la meilleure fréquence pour un cluster donné. On prend une octave autour.
    bt = heatmap.get_best_frequency_for(cluster)

    out = heatmap.detect_peak_and_contours(cluster, contour_std=2)
    if out[1] is None:
        return None

    else:
        temporal = out[1]
    # max_peak_response = heatmap.get_activity_at_peak(cluster)
    # relevant_tones = [2 ** -0.5 * bt, 2 ** 0.5 * bt]
    # On cherche où sont les meilleures fréquences dans la liste du playback
    # ix = np.logical_and(pb_tones >= relevant_tones[0], pb_tones <= relevant_tones[1])
    ix = np.equal(pb_tones, bt)
    virtual = virtual[ix]
    pb_tones = pb_tones[ix]
    pb_triggers = pb_triggers[ix]
    tones = np.unique(np.hstack((virtual, pb_tones)))
    index_mck = np.zeros(len(virtual))
    index_pb = np.zeros(len(pb_tones))
    for tone in tones:
        tp = np.where(tone == tones)[0]

        if len(tp) > 0:
            index_mck[np.where(tone == virtual)[0]] = tp[0]
            index_pb[np.where(tone == pb_tones)[0]] = tp[0]
        else:
            continue
    clean_delta = index_mck - index_pb
    best_time, _ = heatmap.get_best_time_for(cluster)  # +0.001
    count = ut.get_activity(x, pb_triggers, t_0=temporal[0], t_1=temporal[1])
    # clean_delta = clean_delta[np.logical_and(clean_delta > -9, clean_delta < 9)]
    if absolute:
        clean_delta = np.abs(clean_delta)
        counts_per_delta = pd.Series(index=np.arange(33, dtype=int))
    else:
        counts_per_delta = pd.Series(index=np.arange(-33, 33, dtype=int))

    unique_delta = np.unique(clean_delta).astype(int)

    activity = list()
    delta_list = list()
    for i, delta in enumerate(unique_delta):
        idx = np.where(delta == clean_delta)[0]
        idx = np.equal(clean_delta, delta)
        if sum(idx) > 0:  # retrouver ce que je cherchais à exclure.
            a = count[idx]
            # a = np.array([count[index] for index in idx])
            if len(a) != 0:
                idx_zeros = np.greater(a, 0)
                if sum(idx_zeros) == 0:
                    continue
                else:
                    a = a[idx_zeros]
                    length = len(a)
                    if np.isnan(counts_per_delta[delta]):
                        counts_per_delta[delta] = 0
                    counts_per_delta[delta] += length
                    delta_list.append(delta)
                    activity.append(a.mean())
    activity = np.array(activity)
    activity = (activity - mean_activity) / mean_activity
    delta_list = np.array(delta_list)
    return activity, delta_list, counts_per_delta


def build_heatmap(tr_tones, pb_tones, tr_triggers, pb_triggers, spikes):
    """
    Utilisée dans le cadre de l'obtention de la meilleure fréquence / meilleur temps.
    """
    if pb_tones is not None and pb_triggers is not None:
        triggers = np.hstack([tr_triggers, pb_triggers])
        tones = np.hstack([tr_tones, pb_tones])
    else:
        tones = tr_tones
        triggers = tr_triggers
    hm_total = hm.Heatmap()
    hm_total.compute_heatmap(tone_sequence=tones, trigs=triggers, spikes=spikes,
                             t_pre=0.1, t_post=0.1, bin_size=0.003)
    return hm_total, hm_total.get_tones()


def plot_distance(df_sessions, bin_size, folder=None, df_counts=None, absolute=False, error_bars=False):
    a = df_sessions.mean(1, skipna=True)
    a_std = df_sessions.sem(1, skipna=True)
    ax = a[a.notnull()].index
    a_std = a_std[a.notnull()].values
    a = a[a.notnull()].values
    a = ut.mean_smoothing(a, size=4, pad_size=50)
    if absolute:
        fn = "absolute_distance"
    else:
        fn = "distance"
    if df_counts is None:
        if error_bars:
            plt.errorbar(ax, a, yerr=a_std, c="purple")
            fn += "_errors"
        else:
            plt.plot(ax, a, c="purple", marker="D")
        if absolute:
            plt.title(f"activity = f(|Δ|), extreme positions excluded. Δt = {bin_size} s around peak.")
            plt.xlabel("|Δ|")
        else:
            plt.title(f"activity = f(Δ), extreme positions excluded. Δt = {bin_size} s around peak.")
            plt.xlabel("Δ")
        plt.ylabel("Activity (fraction of baseline)")

    else:
        fig, axes = plt.subplots(2)
        if error_bars:
            axes[0].errorbar(ax, a, yerr=a_std, c="purple")
            fn += "_errors"
        else:
            axes[0].plot(ax, a, c="purple")
        axes[0].set_ylabel("Activity (fraction of baseline)")
        axes[1].bar(ax, df_counts.loc[ax].sum(1, skipna=True).values)
        if absolute:
            axes[0].set_title(f"activity = f(|Δ|), extreme positions excluded. Δt = {bin_size} s around peak.")
            axes[0].set_xlabel("|Δ|")
            axes[1].set_xlabel("|Δ|")
        else:
            axes[0].set_title(f"activity = f(Δ), extreme positions excluded. Δt = {bin_size} s around peak.")
            axes[0].set_xlabel("Δ")
            axes[1].set_xlabel("Δ")

        axes[1].set_ylabel("Counts")
        axes[1].set_title("Counts per Δ")
        fn += "_counts"
    fn += ".jpg"
    if folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(folder, fn), dpi=300)
        plt.close()

