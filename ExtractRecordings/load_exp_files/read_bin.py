"""
fonctions pour la lecture des fichiers .bin de position et de fréquence
"""
import numpy as np
import os


def read_positions_file(path):
    return np.fromfile(path, dtype=np.int32)


def read_tones_file(path):
    return np.fromfile(path, dtype=np.double)


def read_dig_in(folder):
    triggers = np.load(os.path.join(folder, "dig_in.npy"))[1]
    triggers = triggers.astype(np.int8)
    triggers = np.where(np.diff(triggers) == 1)[0] + 1
    return triggers
