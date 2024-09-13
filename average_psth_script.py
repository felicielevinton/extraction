import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import PostProcessing.tools.utils as ut
import PostProcessing.tools.heatmap as hm
from get_all_good_sessions import get_good_sessions
import get_data as gd
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(prog="AveragePSTH")
    parser.add_argument("--name", type=str, help="Ferret Name.")
    parser.add_argument("--compatibility", type=bool, help="Old data.")
    opt = parser.parse_args()
    return opt


def average_psth(folders, save_folder):
    sr = 30e3
    t_pre = 0.05
    t_post = 0.300
    bin_size = 0.005
    psth_bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    plot_path = gd.check_plot_folder_exists(save_folder)

    big_mean_tr = list()

    big_mean_pb = list()

    spectral_bw_tr = list()

    spectral_bw_pb = list()

    for sess_num, folder in enumerate(folders):

        plot_path_session = gd.check_plot_folder_exists(folder)

        sequence = gd.extract_data(folder)

        recording_length = sequence.get_recording_length()

        spk = ut.Spikes(folder, recording_length=recording_length)

        good_clusters = np.load(os.path.join(folder, "good_clusters_playback.npy"))

        fn = os.path.join(folder, "heatmap_playback.npz")

        if os.path.exists(fn):
            hm_playback = hm.load_heatmap(fn)

        else:
            pb = sequence.merge("playback")
            hm_playback = hm.Heatmap()
            hm_playback.compute_heatmap(tone_sequence=pb.tones, trigs=pb.triggers,
                                        spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)

        fn = os.path.join(folder, "heatmap_tracking.npz")

        if os.path.exists(fn):
            hm_tracking = hm.load_heatmap(fn)

        else:
            tr = sequence.merge("tracking")
            hm_tracking = hm.Heatmap()
            hm_tracking.compute_heatmap(tone_sequence=tr.tones, trigs=tr.triggers,
                                        spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
        psths_tr, psths_pb = list(), list()

        n_iter = sequence.get_n_iter()

        mean_psth_tr = list()

        mean_psth_pb = list()

        df_pb = pd.DataFrame(columns=[f"block_{i}" for i in range(n_iter)], index=good_clusters)

        df_tr = pd.DataFrame(columns=[f"block_{i}" for i in range(n_iter)], index=good_clusters)

        for cluster in good_clusters:

            out = hm_playback.detect_peak_and_contours(cluster)

            if out[1] is None:
                continue
            else:
                _, temporal_span, bandwidth = out[0], out[1], out[2]

            out = hm_tracking.detect_peak_and_contours(cluster)
            if out[1] is None:
                continue
            else:
                _, temporal_span_tr, bandwidth_tr = out[0], out[1], out[2]

            spectral_bw_tr.append(np.log2(bandwidth_tr[1] / bandwidth_tr[0]))

            spectral_bw_pb.append(np.log2(bandwidth[1] / bandwidth[0]))

            mean_psth_tr.append(hm_tracking.get_bf_psth_for(cluster=cluster, position=bandwidth_tr))

            mean_psth_pb.append(hm_playback.get_bf_psth_for(cluster=cluster, position=bandwidth))

            x = spk.get_spike_times(cluster)

            for it in range(n_iter):

                tr = sequence.get_xp_number("tracking", it)

                pb = sequence.get_xp_number("playback", it)

                idx_tr = np.logical_and(tr.get_tones() >= bandwidth_tr[0], tr.get_tones() <= bandwidth_tr[1])

                idx_pb = np.logical_and(pb.get_tones() >= bandwidth[0], pb.get_tones() <= bandwidth[1])

                trt = tr.get_triggers()[idx_tr]

                pbt = pb.get_triggers()[idx_pb]

                htr, _ = ut.psth(x, trt, bins=psth_bins)

                hpb, _ = ut.psth(x, pbt, bins=psth_bins)

                if htr is None or hpb is None:
                    continue
                df_tr.loc[cluster][f"block_{it}"] = htr[
                    np.logical_and(psth_bins[:-1] >= temporal_span_tr[0], psth_bins[:-1] <= temporal_span_tr[1])].max()
                df_pb.loc[cluster][f"block_{it}"] = hpb[
                    np.logical_and(psth_bins[:-1] >= temporal_span[0], psth_bins[:-1] <= temporal_span[1])].max()

        df_tr["C"] = "tracking"

        df_pb["C"] = "playback"

        df_concat = pd.concat([df_tr, df_pb])

        dfm = pd.melt(df_concat, id_vars="C", value_vars=[f"block_{i}" for i in range(n_iter)], var_name="block", value_name="firing rate")

        sns.swarmplot(x="block", y="firing rate", hue="C", data=dfm)

        plt.show()

        mean_psth_tr = np.vstack(mean_psth_tr)

        mean_psth_pb = np.vstack(mean_psth_pb)

        plt.plot(psth_bins[:-1], mean_psth_tr.mean(0))

        plt.plot(psth_bins[:-1], mean_psth_pb.mean(0))

        plt.title(f"Mean PSTH {mean_psth_pb.shape[0]} units.")

        plt.ylabel("Firing rate [Hz]")

        plt.xlabel("Time [s]")

        plt.show()

        # plt.savefig(os.path.join(plot_path_session, "mean_psth.png"), dpi=300)

        # plt.close()

        big_mean_tr.append(mean_psth_tr)

        big_mean_pb.append(mean_psth_pb)

    big_mean_tr = np.vstack(big_mean_tr)

    big_mean_pb = np.vstack(big_mean_pb)

    plt.plot(psth_bins[:-1], big_mean_tr.mean(0), label="tracking")

    plt.plot(psth_bins[:-1], big_mean_pb.mean(0), label="playback")

    plt.title(f"Mean PSTH {big_mean_tr.shape[0]} units. {len(folders)} sessions.")

    plt.ylabel("Firing rate [Hz]")

    plt.xlabel("Time [s]")

    plt.savefig(os.path.join(plot_path, "mean_psth.png"), dpi=300)

    plt.close()

    spectral_bw_tr = np.array(spectral_bw_tr)

    spectral_bw_pb = np.array(spectral_bw_pb)

    print(spectral_bw_tr.mean())

    print(spectral_bw_pb.mean())

    print("orbis te saluto.")


if __name__ == "__main__":
    options = parse_args()
    good_dirs, save_dir = get_good_sessions(options.name)
    average_psth(good_dirs, save_folder=save_dir)

