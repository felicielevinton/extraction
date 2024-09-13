import numpy as np
import matplotlib.pyplot as plt


class Record:
    def __init__(self):
        self.path = None
        self.data = None
        # self.aux_chan = False

    def set_path(self, path):
        self.path = path

    def load_data(self):
        self.data = open_dat(self.data)

    def extract_aux_channels(self):
        """
        En assumant que les canaux auxilliaires sont à la fin de la matrice. Et au nombre de 8.
        :return:
        """
        if self.data is not None:
            aux_chan = self.data[-8:]
            return aux_chan


def open_dat(file_name):
    data = np.fromfile(file_name, np.int16)
    data = np.reshape(data, [104, int(len(data) / 104)], order="F")
    return data


def spectrum(x, file_name, plot=False, fs=30000):
    xfft = 20 * np.log10(np.abs(np.fft.rfft(x)))
    f = np.linspace(0, fs / 2, len(xfft))
    if plot:
        plt.plot(f, xfft)
        # plt.savefig(file_name)
        # plt.figure()
    return xfft


if __name__ == "__main__":
    n_chan = 96
    path_recordings = "/home/feral/PycharmProjects/Electrophysiology/recordings_cottage/continuous.dat"
    d = open_dat(path_recordings)
    fft = list()
    for elt in range(96):
        fft.append(spectrum(d[elt][10000000:10060000], "", plot=False))
    np.save("recordings_cottage/fft.npy", np.vstack(fft))





