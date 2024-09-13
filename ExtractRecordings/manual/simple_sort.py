import numpy as np
import os
from scipy import signal


def load_spike_data(path):
    ext = os.path.splitext(path)[-1]
    if ext == ".dat":
        x = np.memmap(path, dtype=np.uint16, mode="r", offset=0, order="C")
        return x.reshape((32, len(x) // 32))
    else:
        return np.lib.format.open_memmap(path, mode="r")


def find_peaks(chan, threshold, distance=0.001, fs=30e3):
    """

    """
    threshold = np.abs(threshold)
    distance *= fs
    opp_chan = chan * -1
    _spk_times, _height = signal.find_peaks(opp_chan, threshold, distance=distance)
    return _spk_times, _height["peak_heights"] * -1


def compute_rms_threshold(neural_channel, rms_level):
    rms = np.power(neural_channel, 2).mean()
    rms = np.sqrt(rms)
    return rms * rms_level


def relative_thresholding(chan, threshold_rms):
    rms = compute_rms_threshold(chan, threshold_rms)
    return find_peaks(chan, rms)


def absolute_thresholding(chan, threshold):
    _spk_times, _height = find_peaks(chan, threshold)
    return _spk_times, _height


def thresholder(chan, mode, threshold):
    if mode == "absolute":
        return absolute_thresholding(chan, threshold)
    else:
        return relative_thresholding(chan, threshold)
