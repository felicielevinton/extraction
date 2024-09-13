import matplotlib.pyplot as plt
import argparse
import get_data as gd
import numpy as np
import os
import PostProcessing.tools.utils as ut
import PostProcessing.tools.heatmap as hm


def parse_args():
    parser = argparse.ArgumentParser(prog="FindGoodClusters")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    parser.add_argument("--playback", type=bool, help="Search playback only for responsive cells. FASTER.")
    _opt = parser.parse_args()
    return _opt


def find_good_clusters(folder, playback=True):
    """
    Utilise la librairie zetapy pour chercher les clusters qui répondent statistiquement
    aux stimuli de l'expérience.
    Passer l'option playback à True permet de se concentrer sur les triggers du playback
    permettant une exécution plus rapide de l'algorithme.
    """
    sequence = gd.extract_data(folder)
    if playback:
        tag = "playback"
        triggers = sequence.get_all_triggers_for_type("playback")
    else:
        tag = None
        triggers = sequence.get_all_triggers()
    recording_length = sequence.get_recording_length()
    spk = ut.Spikes(folder, recording_length=recording_length)

    if os.path.exists(os.path.join(folder, "good_clusters.npy")):
        gc = np.load(os.path.join(folder, "good_clusters.npy"))
    else:
        gc = None

    ut.check_responsiveness(triggers=triggers, spikes=spk, folder=folder, clusters=gc, tag=tag)


if __name__ == "__main__":
    options = parse_args()
    f = options.folder
    pb = options.playback
    find_good_clusters(f, playback=pb)


