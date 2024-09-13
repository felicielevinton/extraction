import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from get_all_good_sessions import get_good_sessions
import PostProcessing.tools.utils as ut
import PostProcessing.tools.heatmap as hm
import get_data as gd
from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser(prog="TemporalEvolution")
    parser.add_argument("--name", type=str, help="Ferret Name.")
    parser.add_argument("--compatibility", type=bool, help="Old data.")
    opt = parser.parse_args()
    return opt


def temporal_evolution(folders, save_folder):
    """
    On cherche à regarder comment évolue au cours du temps la réponse d'un
    neurone aux sons pendant la condition tracking
    """
    sr = 30e3
    t_pre = 0.1
    t_post = 0.500
    bin_size = 0.01
    bin_duration_minute = 5  # minute
    duration_bin = bin_duration_minute * sr * 60
    psth_bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    plot_path = gd.check_plot_folder_exists(save_folder)

    total = list()

    for sess_num, folder in enumerate(folders):

        plot_path_session = gd.check_plot_folder_exists(folder)

        sequence = gd.extract_data(folder)

        recording_length = sequence.get_recording_length()

        spk = ut.Spikes(folder, recording_length=recording_length)

        wp = sequence.merge("warmup")

        tr_0 = sequence.get_xp_number("tracking", 0)

        if os.path.exists(os.path.join(folder, "durations.json")):
            import json
            fn = os.path.join(folder, "durations.json")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    d = json.load(f)
                wp_duration = d["warmup"]
        else:
            wp_duration = 10

        total_duration = wp_duration + 5

        n_bins = total_duration / bin_duration_minute

        good_clusters = np.load(os.path.join(folder, "good_clusters_playback.npy"))

        fn = os.path.join(folder, "heatmap_playback.npz")

        if os.path.exists(fn):
            hm_total = hm.load_heatmap(fn)

        psths = list()

        triggers = np.hstack((wp.get_triggers(), tr_0.get_triggers()))

        start = triggers[0]

        # bins = np.arange(0, total_duration + bin_duration_minute, step=bin_duration_minute)[1:] * duration_bin + start

        bins = np.arange(0, n_bins + 1) * duration_bin

        bins += + np.full_like(bins, start)

        bins = bins[1:]

        tones = np.hstack((wp.get_tones(), tr_0.get_tones()))

        for cluster in good_clusters:

            out = hm_total.detect_peak_and_contours(cluster)

            if out[1] is None:
                continue

            else:
                bandwidth = out[-1]

            x = spk.get_spike_times(cluster)

            # on veut les triggers.
            idx = np.logical_and(tones >= bandwidth[0], tones <= bandwidth[1])

            cp_triggers = deepcopy(triggers)

            cp_triggers = cp_triggers[idx]

            _p = list()

            for i, elt in enumerate(bins):
                idx = np.less(cp_triggers, elt)
                h, _ = ut.psth(x, cp_triggers[idx], bins=psth_bins)
                cp_triggers = cp_triggers[~idx]
                if h is not None:
                    h = ut.mean_smoothing(h, size=5, pad_size=20)
                    _p.append(h)
                    plt.plot(psth_bins[:-1], h, label=f"After {bin_duration_minute * i} min")
                else:
                    _p.append(np.full_like(psth_bins[:-1], np.nan))

            plt.ylabel("Firing rate [Hz]")

            plt.xlabel("Time [s]")

            plt.title("Evolution of response at bandwidth tones")

            plt.savefig(os.path.join(plot_path_session, f"evolution_cluster_{cluster}.png"), dpi=300)

            plt.close()

            psths.append(np.vstack(_p))

        psths = np.dstack(psths)

        for i in range(len(bins)):
            plt.plot(psth_bins[:-1], np.nanmean(psths[i, :, :], axis=1))

        plt.ylabel("Firing rate [Hz]")

        plt.xlabel("Time [s]")

        plt.title("Evolution of response at bandwidth tones")

        plt.savefig(os.path.join(plot_path_session, "evolution_mean.png"), dpi=300)

        plt.close()

        total.append(np.nanmean(psths, axis=2))

    max_n_bin = 0

    for i, elt in enumerate(total):

        if i == 0:
            max_n_bin = elt.shape[0]

        else:
            if elt.shape[0] > max_n_bin:
                max_n_bin = elt.shape[0]

    print(max_n_bin)
    for i, elt in enumerate(total):
        if elt.shape[0] < max_n_bin:
            total[i] = np.vstack((elt, np.full((max_n_bin - elt.shape[0], len(psth_bins[:-1])), np.nan)))
    for elt in total:
        print(elt.shape)

    total = np.dstack(total)

    for i in range(max_n_bin):
        plt.plot(psth_bins[:-1], np.nanmean(total[i], axis=1))

    plt.ylabel("Firing rate [Hz]")

    plt.xlabel("Time [s]")

    plt.title("Evolution of response at bandwidth tones")

    plt.savefig(os.path.join(plot_path, "total_evolution_mean.png"), dpi=300)

    plt.close()


if __name__ == "__main__":
    options = parse_args()
    print("Orbis te Saluto!")
    good_dirs, save_dir = get_good_sessions(options.name)
    good_dirs.sort()
    temporal_evolution(good_dirs, save_folder=save_dir)