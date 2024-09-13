import numpy as np
from scipy import signal
import os
from tqdm import tqdm
import cv2 as cv
from findpeaks import findpeaks
from contourpy import contour_generator
import re
from copy import deepcopy
from zetapy import getZeta


class Spikes(object):
    def __init__(self, path, recording_length=None):
        self.spike_times, self.spike_clusters = load_data(path)
        self.n_clusters = len(np.unique(self.spike_clusters))
        self.spikes = dict()
        self.fs = 30e3
        self.recording_length = recording_length
        for cluster in np.arange(self.n_clusters):
            idx = np.where(self.spike_clusters == cluster)[0]
            self.spikes[cluster] = self.spike_times[idx]

    def get_spike_times(self, cluster):
        if cluster in self.spikes.keys():
            return self.spikes[cluster]
        else:
            return -1

    def get_spike_times_between_(self, cluster, t_0, t_1, zero=False):
        if cluster in self.spikes.keys():
            spikes = self.spikes[cluster]
            x = spikes[np.logical_and(spikes > t_0, spikes < t_1)]
            if zero:
                x -= t_0
            return x

    def get_n_clusters(self):
        return self.n_clusters

    def get_binned_activity(self, cluster, bin_duration, recording_length=None):
        bin_size = int(bin_duration * self.fs)
        if self.recording_length is None:
            assert(recording_length is not None), "No recording length has been set"
            self.recording_length = recording_length
        n_bins, remainder = self.recording_length // bin_size, self.recording_length % bin_size
        bins = np.arange(0, self.recording_length - remainder + bin_size, bin_size)
        h, bins = np.histogram(self.get_spike_times(cluster=cluster), bins)
        h = h.astype(np.double)
        h /= bin_duration
        return h, bins

    def get_binned_activity_between(self, cluster, t0, t1, bin_duration):
        delta = t1 - t0
        bin_size = int(bin_duration * self.fs)
        n_bins, remainder = delta // bin_size, delta % bin_size
        bins = np.arange(t0, t1 - remainder + bin_size, bin_size)
        x = self.get_spike_times(cluster=cluster)
        x = x[np.logical_and(x > t0, x < t1)]
        h, bins = np.histogram(x, bins)
        h = h.astype(np.double)
        h /= bin_duration
        return h, bins

    def get_mean_std_activity(self, bin_duration, cluster, interval=None, recording_length=None):
        if interval is None:
            assert (recording_length is not None), "Must give a recording length."
            x, _ = self.get_binned_activity(cluster, bin_duration, recording_length)
        else:
            assert (type(interval) == list)
            assert (len(interval) == 2)
            interval.sort()
            x, _ = self.get_binned_activity_between(cluster, t0=interval[0], t1=interval[1], bin_duration=bin_duration)
        return x.mean(), x.std()

    def get_spikes_activity_around(self, cluster, t, time_around, bin_duration):
        time_around = int(time_around * self.fs)
        left, right = t - time_around, t + time_around
        return self.get_binned_activity_between(cluster, left, right, bin_duration)


def load_data(path):
    """

    """
    spike_clusters = np.load(os.path.join(path, "spike_clusters.npy"))
    spike_times = np.load(os.path.join(path, "spike_times.npy"))
    return spike_times, spike_clusters


def find_spikes(spikes, t_0, t_1, trig=None, trigger_unit="seconds", fs=30e3):
    """
    Prend en arguments des temps exprimés en nb de samples
    N'est pas sensible aux valeurs d'intervalles négatives.
    """
    assert(trigger_unit in ("seconds", "samples")), "Trigger unit available are seconds or samples"
    if trigger_unit == "seconds":
        t_0 = int(t_0 * fs)
        t_1 = int(t_1 * fs)

    if trig is not None:
        t_0 = trig - abs(t_0)
        t_1 = trig + t_1
    else:
        trig = t_0

    # LOGICAL_AND.non_zero() plus rapide peut-être?
    x = spikes[np.logical_and(spikes > t_0, spikes < t_1)]
    x = x.astype(np.double)
    x -= trig
    x /= fs
    return x


def check_responsiveness(triggers, spikes, folder, clusters=None, tag=None):
    """
    Vérifie qu'une unité répond aux stimuli.
    """
    good_clusters = list()
    if clusters is not None:
        iterator = clusters
    else:
        iterator = list(range(spikes.get_n_clusters()))
    for cluster in tqdm(iterator):
        x = spikes.get_spike_times(cluster=cluster)
        a, b = getZeta(x * (1 / 30000), triggers * (1 / 30000))
        if a < 0.001:
            good_clusters.append(cluster)
    good_clusters = np.array(good_clusters)
    if tag is not None:
        filename = f"good_clusters_{tag}.npy"
    else:
        filename = "good_clusters.npy"
    np.save(os.path.join(folder, filename), good_clusters)

    return good_clusters


def psth(spikes, triggers, t_0=0.2, t_1=0.5, bin_size=0.01, bins=None, trigger_unit="seconds", fs=30e3):
    """

    """
    x = raster(spikes, triggers, t_0, t_1, trigger_unit, fs)
    if len(x) == 0:
        return None, None
    x = np.hstack(x)
    if bins is None:
        bins = np.arange(t_0, t_1 + bin_size, bin_size)
    h, b = np.histogram(x, bins)
    h = h.astype(dtype=np.float64)
    h /= (len(triggers) * bin_size)  # donne l'activité
    return h, b


def raster(spikes, triggers, t_0=0.2, t_1=0.5, trigger_unit="seconds", fs=30e3):
    """

    """
    x = list()
    for trigger in triggers:
        x.append(find_spikes(spikes, t_0, t_1, trigger, trigger_unit, fs))
    return x


def count_spikes(spikes, t_0, t_1, trig=None, trigger_unit="seconds", fs=30e3):
    assert (trigger_unit in ("seconds", "samples")), "Trigger unit available are seconds or samples"
    if trigger_unit == "seconds":
        t_0 = int(t_0 * fs)
        t_1 = int(t_1 * fs)

    if trig is not None:
        t_0 = trig + abs(t_0)
        t_1 = trig + t_1

    # LOGICAL_AND.non_zero() plus rapide peut-être?
    x = spikes[np.logical_and(spikes > t_0, spikes < t_1)]
    x = x.astype(np.double)
    activity = len(x)
    activity /= ((t_1 - t_0) / fs)
    return activity


def get_activity(spikes, triggers, t_0=0.2, t_1=0.5, trigger_unit="seconds", fs=30e3):
    x = list()
    for trigger in triggers:
        x.append(count_spikes(spikes, t_0, t_1, trigger, trigger_unit, fs))
    return np.array(x)


def isi(spikes, bin_size=0.001, fs=30e3):
    """
    Jamais testée...
    """
    x_diff = np.diff(spikes)
    x_diff = x_diff.astype(dtype=np.float64)
    x_diff /= fs
    bins = np.arange(0, 1 + bin_size, bin_size)
    h, _ = np.histogram(x_diff, bins=bins)
    return h, bins


def heatmap(tone_sequence, trigs, spikes, t_pre=0.1, t_post=1, bin_size=0.01):
    bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    n_bin = len(bins) - 1
    tones = np.unique(tone_sequence)
    hist = np.empty((0, n_bin))
    for tone in tones:
        tone_idx = np.where(tone_sequence == tone)[0]
        trigger_time = trigs[tone_idx]
        h, _ = psth(spikes, trigger_time, t_0=t_pre, t_1=t_post, bins=bins)
        hist = np.vstack((hist, h))
    return hist, bins, tones


def z_score_heatmap(hm):
    for i in range(hm.shape[0]):
        hm[i] -= hm[i].mean()
        hm[i] /= hm[i].std()
    return hm


def z_score_hm_2(hm, means, stds):
    for i in range(hm.shape[0]):
        hm[i] -= means[i]
        hm[i] /= stds[i]
    return hm


def z_score(x):
    zx = (x - x.mean()) / x.std()
    return zx


def norm_mean(x):
    min_x = (x - np.mean(x)) / np.mean(x)
    return min_x


def get_mu_sig(hm):
    mu_list = list()
    std_list = list()
    for i in range(hm.shape[0]):
        mu_list.append(hm[i].mean())
        std_list.append(hm[i].std())
    return mu_list, std_list


def mean_firing_rate(spike_times, fs=30e3):
    t_begin, t_end = spike_times[0], spike_times[-1]
    d = t_end - t_begin  # ?
    count = len(spike_times)
    d /= fs
    mean_fr = count / d
    return mean_fr


def extract_trigger_time(dig_in_channel):
    return np.where(np.diff(dig_in_channel) == 1)[0] + 1


def extract_digital_triggers(digital_channel, min_time_between=0.005, fs=30e3):
    distance = int(min_time_between * fs)
    return signal.find_peaks(digital_channel, height=1, distance=distance, plateau_size=[10, 1000])[1]["left_edges"]


def extract_analog_triggers(analog_channel, min_time_between=0.005, fs=30e3, playback=False):
    tracking_triggers = extract_tracking_triggers(analog_channel, min_time_between=min_time_between,
                                                  fs=fs)

    if not playback:
        mock_triggers = extract_mock_triggers(analog_channel, min_time_between=min_time_between, fs=fs)
        return tracking_triggers, mock_triggers

    else:
        return tracking_triggers


def extract_analog_triggers_compat(analog_channel, min_time_between=0.005, fs=30e3):
    distance = int(min_time_between * fs)
    analog_channel = np.square(analog_channel)
    analog_channel = np.where(analog_channel <= 2, 0, 1)
    fp_out = signal.find_peaks(analog_channel, height=1, distance=distance, plateau_size=[10, 1000])
    return fp_out[1]["left_edges"]


def extract_tracking_triggers(analog_channel, min_time_between=0.005, fs=30e3):
    distance = int(min_time_between * fs)
    analog_channel = np.where(analog_channel <= 2, 0, 1)
    fp_out = signal.find_peaks(analog_channel, height=1, distance=distance, plateau_size=[10, 1000])
    return fp_out[1]["left_edges"]


def extract_mock_triggers(analog_channel, min_time_between=0.005, fs=30e3):
    distance = int(min_time_between * fs)
    analog_channel *= -1
    analog_channel = np.where(analog_channel <= 2, 0, 1)
    fp_out = signal.find_peaks(analog_channel, height=1, distance=distance, plateau_size=[10, 1000])
    return fp_out[1]["left_edges"]


def gaussian_smoothing(x, sigma=5, size=10, pad_size=None):
    """
    Créer un kernel gaussien
    M = taille de la fenêtre.
    std = distribution de la gaussienne.
    """
    kernel = signal.windows.gaussian(M=size, std=sigma)
    return smooth(kernel, x, pad_size)


def mean_smoothing(x, size=10, pad_size=None):
    """
    Créer un noyau gaussien
    M = taille de la fenêtre.
    std = distribution de la gaussienne.
    """
    kernel = np.ones(size) / size
    return smooth(kernel, x, pad_size)


def smooth(kernel, x, pad_size):
    if pad_size is not None:
        x = np.hstack([np.full(pad_size, x[0]), x, np.full(pad_size, x[-1])])
        x_conv = signal.fftconvolve(x, kernel, "same")[pad_size:-pad_size]
    else:
        x_conv = signal.fftconvolve(x, kernel, "same")

    return x_conv


def find_common_tones(t1, t2):
    assert(len(t1) > len(t2)), "Array in position 0 must have greater length"
    found = False
    front = 0
    back = len(t2) - len(t1)
    while found is False:
        bit_t1 = t1[front:back]
        if np.array_equal(bit_t1, t2):
            found = True
        else:
            front += 1
            back += 1
    return front, back


def z_score(x):
    x -= x.mean()
    x /= x.std()
    return x


def activity_snippet(spikes, cells, list_snippets, fs=30e3):
    activity = np.zeros((len(list_snippets), len(cells)))
    for n, snippet in enumerate(list_snippets):
        for ii, cell in enumerate(cells):
            x = spikes.get_spike_times_between_(cluster=cell, t_0=snippet[0], t_1=snippet[-1])
            activity[n][ii] = mean_firing_rate(x, fs=fs)
    return activity


def activity_baseline(spikes, cells, trigs, duration_tracking, fs=30e3):
    list_snippets = bin_experiment(trigs, duration_tracking, fs)
    activity = activity_snippet(spikes, cells, list_snippets, fs=fs)
    return activity


def bin_experiment(trigs, duration_tracking, fs=30e3):
    t = duration_tracking * 60
    dts = int(t * fs)
    b, e = trigs[0], trigs[-1]
    n_bins = int((e - b) // dts)
    left_limit, right_limit = b, b + dts
    list_snippets = list()
    for _ in range(n_bins):
        list_snippets.append(([int(left_limit), int(right_limit)]))
        left_limit = right_limit
        right_limit += dts
    return list_snippets


def q10(tuning_curve):
    # -10 = 10 * log(x / max) <=> x = 10**-1 * max.
    max_tc = np.amax(tuning_curve)
    q_10 = 0.1 * max_tc
    return q_10


def q3(tuning_curve):
    # -3 = 10 * log(x / max) <=> -0.3 = log(x / max) <=> x = 10**-0.3 * max.
    max_tc = np.amax(tuning_curve)
    q_3 = 10**-0.3 * max_tc
    return q_3


def qx(tuning_curve, x):
    max_tc = np.amax(tuning_curve)
    q = 10**(x / 10) * max_tc
    return q


def peak_and_contour_finding(hm, contour_std=2):
    """
    On passe une heatmap en argument.
    """
    # trouver min vs max
    n = 5
    gb = cv.GaussianBlur(hm, (n, n), 0)
    gb -= gb.mean()
    gb /= gb.std()
    max_val, min_val = gb.max(), np.abs(gb.min())
    is_valley = False
    if min_val > max_val:
        gb = np.square(gb)
        is_valley = True
    fp = findpeaks(method='topology', scale=True, denoise="lee_enhanced", togray=True, imsize=gb.shape[::-1], verbose=0)
    res = fp.fit(gb)
    persistence = res["persistence"]
    # keys in pd.DataFrame are: x, y, birth_level, death_level, score, peak, valley.
    best_score_arg = persistence["score"].argmax()
    series = persistence.iloc[best_score_arg]
    assert (series["peak"])
    x, y = series["x"], series["y"]
    xx = np.arange(gb.shape[1])
    yy = np.arange(gb.shape[0])
    xx, yy = np.meshgrid(xx, yy)
    contour = contour_generator(xx, yy, gb)
    lines = contour.lines(contour_std)  # on veut supérieur à deux
    line = None
    for i in range(len(lines)):
        elt = deepcopy(lines[i])
        elt = np.transpose(elt)
        elt.sort()
        x_axis, y_axis = elt[0], elt[1]
        if x_axis[0] <= x <= x_axis[-1]:
            if y_axis[0] <= y <= y_axis[-1]:
                line = lines[i]
                break
    if line is None:
        pass

    return x, y, line, is_valley


def find_spectral_span(line):
    """
    C'est l'étalement dans le temps de la réponse d'un neurone. Retourne?
    """
    y_axis = np.round(line[1])
    y_axis = np.unique(y_axis)
    low, high = y_axis[0], y_axis[-1]
    return int(low), int(high)


def find_temporal_span(line):
    """
    C'est l'étalement dans le spectre de la réponse d'un neurone. Retourne?
    """
    x_axis = np.round(line[0])
    x_axis = np.unique(x_axis)
    low, high = x_axis[0], x_axis[-1]
    return int(low), int(high)
