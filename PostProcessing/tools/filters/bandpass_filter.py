import numpy as np
from scipy import signal


def _butter_bandpass(low_cut, high_cut, fs, order=5):
    return signal.butter(order, [low_cut, high_cut], btype="band", analog=True)


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = _butter_bandpass(low_cut, high_cut, fs, order)
    return signal.lfilter(b, a, data)
