import os.path
from distance import *
import argparse
import pandas as pd
import get_data as gd
import PostProcessing.tools.utils as ut


def parse_args():
    parser = argparse.ArgumentParser(prog="distance")
    parser.add_argument("--folders", type=str, nargs="+", help="Path to folder having data.")
    parser.add_argument("--save", type=str, help="Folder to save data.")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    bin_duration = 5
    options = parse_args()
    folders = options.folders
    save_folder = options.save
    plot_path = gd.check_plot_folder_exists(save_folder)
    for sess_num, folder in enumerate(folders):
        sequence = gd.extract_data(folder)

        # wup_t = sequence.get_xp_number("warmup").triggers[0] / 30000
        tr = [sequence.get_xp_number("tracking", i).triggers[0] / (30000 * bin_duration) for i in range(10)]
        pb = [sequence.get_xp_number("playback", i).triggers[0] / (30000 * bin_duration) for i in range(10)]

        good_clusters = np.load(os.path.join(folder, "good_clusters_playback.npy"))

        recording_length = sequence.get_recording_length()

        spk = ut.Spikes(folder, recording_length=recording_length)
        c = list()
        for cluster in range(spk.get_n_clusters()):
            if cluster not in good_clusters:
                continue
            h, _ = spk.get_binned_activity(cluster, bin_duration)
            c.append(h)

        c = np.vstack(c)
        b = np.arange(c.shape[1]) * bin_duration
        plt.plot(b[:-1], ut.mean_smoothing(c.mean(0)[:-1], size=1000, pad_size=10000))
        for elt in tr:
            plt.axvline(elt, c="g")
        for elt in pb:
            plt.axvline(elt, c="r")
        plt.title(f"Cluster: mean activity")
        plt.show()

