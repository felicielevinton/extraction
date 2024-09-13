import os
import numpy as np
from .utils import z_score
from scipy import signal
from Analyse.GLM.glm import build_dm

SAMPLING_RATE_ACCELEROMETER = 7.5e3


def load_accelerometer_data(folder):
    if os.path.exists(os.path.join(folder, "recording_length.bin")):
        with open(os.path.join(folder, "recording_length.bin"), "r") as f:
            length = int(f.read())
    return np.load(os.path.join(folder, "accelerometer.npy")), length


class Accelerometer(object):
    def __init__(self, folder, fs=30e3):
        data, self.recording_length = load_accelerometer_data(folder)
        self.FSA = SAMPLING_RATE_ACCELEROMETER
        self.fs = fs
        self.ratio = int(self.fs / self.FSA)
        self.x = z_score(data[0])
        self.y = z_score(data[1])
        self.z = z_score(data[2])
        self.scaled_to_ephys = np.arange(1, self.recording_length + 1) * self.ratio

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def _get_binned(self, channel, bin_duration):
        """
        Remainder will be dropped.
        """
        bin_size = int(bin_duration * self.FSA)
        n_bins, remainder = self.recording_length // bin_size, self.recording_length % bin_size
        return np.vstack(np.hsplit(channel[:-remainder], n_bins))

    def _get_binned_std(self, channel, bin_duration):
        binned = self._get_binned(channel, bin_duration)
        return binned.std(1)

    def _get_between(self, channel, t0, t1):
        return channel[np.logical_and(self.scaled_to_ephys >= t0, self.scaled_to_ephys <= t1)]

    def _get_binned_std_between(self, channel, t0, t1, bin_duration):
        bin_size = int(bin_duration * self.FSA)
        delta = t1 - t0
        delta //= self.ratio
        n_bins, remainder = delta // bin_size, delta % bin_size
        between = self._get_between(channel, t0, t1)
        binned = np.array_split(between[:-remainder], n_bins)
        std = np.zeros(len(binned))
        for i, elt in enumerate(binned):
            std[i] = elt.std()
        return std

    def _get_binned_around(self, channel, event, time_around, bin_duration):
        time_around = int(time_around * self.fs)
        left, right = event - time_around, event + time_around
        return self._get_binned_std_between(channel, left, right, bin_duration)

    def get_x_binned(self, bin_duration):
        return self._get_binned(self.x, bin_duration=bin_duration)

    def get_x_std(self, bin_duration):
        return self._get_binned_std(self.x, bin_duration=bin_duration)

    def get_y_binned(self, bin_duration):
        return self._get_binned(self.y, bin_duration=bin_duration)

    def get_y_std(self, bin_duration):
        return self._get_binned_std(self.y, bin_duration=bin_duration)

    def get_z_binned(self, bin_duration):
        return self._get_binned(self.z, bin_duration=bin_duration)

    def get_z_std(self, bin_duration):
        return self._get_binned_std(self.z, bin_duration=bin_duration)

    def get_x_between(self, t0, t1):
        return self._get_between(self.x, t0, t1)

    def get_x_binned_around(self, event, time_around, bin_duration):
        return self._get_binned_around(self.x, event, time_around, bin_duration)

    def get_y_between(self, t0, t1):
        return self._get_between(self.y, t0, t1)

    def get_y_binned_around(self, event, time_around, bin_duration):
        return self._get_binned_around(self.y, event, time_around, bin_duration)

    def get_z_between(self, t0, t1):
        return self._get_between(self.y, t0, t1)

    def get_z_binned_around(self, event, time_around, bin_duration):
        return self._get_binned_around(self.z, event, time_around, bin_duration)

    def get_x_binned_between(self, t0, t1, bin_duration):
        return self._get_binned_std_between(self.x, t0, t1, bin_duration)

    def get_y_binned_between(self, t0, t1, bin_duration):
        return self._get_binned_std_between(self.y, t0, t1, bin_duration)

    def get_z_binned_between(self, t0, t1, bin_duration):
        return self._get_binned_std_between(self.z, t0, t1, bin_duration)

    def get_samples_at_ephys_rate(self):
        """
        Comme électrophysiologie et accéléromètre sont samplés à deux fréquences différentes,
        on convertit les timestamps de l'accéléromètre vers celle de l'électrophysiologie.
        """
        return self.scaled_to_ephys

    def get_design_matrix(self, t0, t1, bin_duration, len_pad):
        x = self.get_x_binned_between(t0, t1, bin_duration)
        y = self.get_y_binned_between(t0, t1, bin_duration)
        z = self.get_y_binned_between(t0, t1, bin_duration)
        # fonction de dm dans glm.py
        dm_ax_x = build_dm(x, len_pad, norm=True)
        dm_ax_y = build_dm(x, len_pad, norm=True)
        dm_ax_z = build_dm(x, len_pad, norm=True)
        dm_ax = np.concatenate((dm_ax_x, dm_ax_y, dm_ax_z), axis=1)
        return dm_ax


