import numpy as np
from scipy import signal
import os


def cwt_lfp(lfp_trace):
    # comment choisir width?
    # return signal.cwt(lfp_trace, signal.morlet)
    return 0


def normalized_psd(x, fs=2500):
    f, pxx = signal.welch(x, fs, window="hann", nperseg=8192)
    norm_pxx = 10 * np.log10(pxx / np.amax(pxx))
    return f, norm_pxx


def psd(x, fs=2500):
    f, pxx = signal.welch(x, fs, window="hann", nperseg=1024)
    return f, pxx


def spectrogram(x, fs=2500, duration=None):
    f, t, sxx = signal.spectrogram(x, fs=fs)
    return f, t, sxx


class IntanLFP(object):
    """

    """
    def __init__(self, folder, fs=30e3, fs_lfp=2500.):
        self.data = np.load(os.path.join(folder, "neural_data.npy"), mmap_mode="r")
        self.channels = np.arange(self.data.shape[0])
        if self.data.dtype == np.uint16:
            self.dtype_is_uint = True
        else:
            self.dtype_is_uint = False
        self.mul_scalar = 0.195
        self.convert_type = np.int32
        self.add_scalar = 32768
        self.fs = fs
        self.fs_lfp = fs_lfp
        self.downsampling_factor = int(self.fs / self.fs_lfp)
        if self.downsampling_factor >= 13:
            self.decimated_factors = np.array([10, self.downsampling_factor / 10])
        else:
            self.decimated_factors = np.array([self.downsampling_factor])
        self.cutoff = 250  # Hz
        self.b_high, self.a_high = signal.butter(N=2, Wn=1.0, fs=self.fs, btype="high", output="ba")
        self.b, self.a = signal.butter(N=8, Wn=self.cutoff, fs=self.fs, btype="low", output="ba")
        # self.transfer_function = signal.TransferFunction(b, a)

    def get_channel(self, channel):
        x = self.data[channel]
        return self._filtering_and_downsampling(x)

    def get_lfp_between(self, channel, t0, t1):
        """
        Utile pour extraire des bouts d'expérience.
        """
        x = self.data[channel][t0:t1]
        return self._filtering_and_downsampling(x)

    def get_lfp_around(self, channel, event, time_around):
        """
        Utile pour obtenir le LFP autour d'un trigger.
        """
        time_around = int(time_around * self.fs)
        t0, t1 = event - time_around, event + time_around
        return self.get_lfp_between(channel, t0, t1)

    def _filtering_and_downsampling(self, x):
        if self.dtype_is_uint:
            x = self._convert_to_float(x)
        x = signal.filtfilt(self.b, self.a, x)
        # x = signal.filtfilt(self.b_high, self.a_high, x)
        q = int(len(x) / self.downsampling_factor)
        x = signal.resample(x, num=q)
        # if self.downsampling_factor >= 13. ?
        # for factor in self.decimated_factors:
        #     # pour l'instant Chebyshev ordre 8.
        #     x = signal.decimate(x, q=factor, ftype="iir", zero_phase=True)
        return x

    def _convert_to_float(self, x):
        return np.multiply(self.mul_scalar, (x.astype(self.convert_type) - self.add_scalar))



