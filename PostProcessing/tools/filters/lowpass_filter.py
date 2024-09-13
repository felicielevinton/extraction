import numpy as np
from scipy import signal


def _butter_lowpass(cutoff, fs, order=5):
    return signal.butter(order, cutoff, fs=fs, btype="low", analog=True, output="ba")


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_lowpass(cutoff, fs, order)  # too many values to unpack.
    return signal.lfilter(b, a, data)
