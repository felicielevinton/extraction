import numpy as np
from simple_sort import find_peaks
from scipy import signal


def common_average(x):
    """

    """
    global_median = np.median(x, axis=0)
    x -= global_median
    return


def filtering(x, order=3, low=300., high=5000., sr=30e3):
    """

    """
    b, a = signal.butter(order, np.array([low, high]) / sr / 2., "pass")
    x = signal.filtfilt(b, a, x, axis=0)
    return


def extract_templates(x_filtered, rms_level=5, n_points=64, fs=30e3):
    """

    """
    templates, _ = find_peaks(x_filtered, threshold=rms_level, fs=fs)
    wv_templates = list()
    for elt in templates:
        wv_templates.append(x_filtered[elt-n_points/2:elt+n_points/2])
    wv_templates = np.vstack(wv_templates)
    template = wv_templates.mean(0)
    return template


def convolution_on_trace(x_filtered, template):
    """

    """
    x_filtered -= x_filtered.mean()
    x_filtered /= x_filtered.std()
    conv = signal.convolve(x_filtered, template)
    # todo: sur quel critère dire qu'une trace est un spike ou non?

    pass

