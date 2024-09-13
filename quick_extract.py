import numpy as np
from tqdm import tqdm
import os
from ExtractRecordings.manual import simple_sort as ss
#from sorting.spike_sorter import filtering_script
import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog="QuickExtract")
    parser.add_argument("--path", type=str, help="Chemin d'accès vers le fichier npy.")
    parser.add_argument("--mode", type=str, help="Méthode d'extraction.", default="relative")
    parser.add_argument("--threshold", type=float, default=-3.7)
    return parser.parse_args()


def quick_extract(path, mode="relative", threshold=-3.7):
    root_dir = os.path.split(path)[0]
    data = ss.load_spike_data(path)
    print(data)
    channels = np.arange(data.shape[0])
    spike_times = np.empty(0, dtype=np.uint64)
    spike_clusters = np.empty(0, dtype=np.int32)
    assert mode in ("relative", "absolute"), "Mode is relative (from RMS calculation) or absolute (threshold in µV)."
    to_float = False
    if data.dtype == np.uint16:
        to_float = True
    if mode == "absolute":
        threshold = -60
    else:
        threshold = threshold
    for i, channel in tqdm(enumerate(channels)):
        print(i, channel)
        if to_float:
            chan = np.multiply(0.195, (data[channel].astype(np.int32) - 32768))
            spk, _ = ss.thresholder(chan, mode, threshold=threshold)
        else:
            spk, _ = ss.thresholder(data[channel], mode, threshold=threshold)
        cluster = np.full_like(spk, i)
        print(i)
        spike_times = np.hstack((spike_times, spk))
        spike_clusters = np.hstack((spike_clusters, cluster))
    idx = np.argsort(spike_times)
    spike_times = spike_times[idx]
    spike_clusters = spike_clusters[idx]
    print(spike_clusters)
    np.save(os.path.join(root_dir, "spike_times.npy"), spike_times)
    np.save(os.path.join(root_dir, "spike_clusters.npy"), spike_clusters)
    pass


if __name__ == "__main__":
    options = parse_args()
    quick_extract(options.path, options.mode, options.threshold)

