import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse


def extract_trigger_time(dig_in_channel):
    return np.where(np.diff(dig_in_channel) == 1)[0]


def find_peaks(chan):
    opp_chan = chan * -1
    return signal.find_peaks(opp_chan, 70)


def absolute_thresholding(chan, threshold=-50):
    opp_chan = chan * -1
    return signal.find_peaks(opp_chan, threshold * -1)[0]


def relative_thresholding(chan, threshold_rms=-4.0):
    pass


def get_spikes(spikes, triggers, fs=30000., t_pre=0.100, t_post=0.500):
    found = np.empty(0, dtype=np.double)
    for trig in triggers:
        x = np.where(np.logical_and(spikes > trig - t_pre * fs, spikes < trig + t_post * fs))[0]
        x = spikes[x]
        x -= trig
        found = np.hstack((found, x))
    return found / fs


def tuning_curve(spikes, trigger_array, tone_sequence):
    """
    Calcul des tuning curves.

    On recupere les temps de spike dans un intervalle donne. On somme le nombre d'unites.

    On retourne la moyenne et l'ecart type.

    spikes: temps de spike
    trigger_array: tableau contenant les valeurs des temps de triggers
    tone_sequence: frequences diffusees lors de la tonotopie.
    """
    list_found = list()
    tones = np.unique(tone_sequence)
    for i in range(len(tones)):
        list_found.append(list())
    for j, tone in enumerate(tones):
        idx = np.where(tones == tone)[0]
        spk_bin = sum(get_spikes(spikes, trigger_array[idx], t_pre=0))
        list_found[j].append(spk_bin)
    # pour chaque frequence, calculer moyenne / std
    m_spk_count = np.zeros(len(tones))
    std_spk_count = np.zeros(len(tones))
    for j, found in enumerate(list_found):
        found = np.array(found)
        m_spk_count[j] = found.mean()
        std_spk_count[j] = found.std()
    return m_spk_count, std_spk_count


def psth(spikes, trigger_array, tone_sequence, t_pre=0.100, t_post=0.500, t_bin=0.002):
    list_founds = list()
    tones = np.unique(tone_sequence)
    for tone in tones:
        idx = np.where(tones == tone)[0]
        list_founds.append(get_spikes(spikes, trigger_array[idx]))
    n_bin = int((t_pre + t_post) / t_bin)
    hist = np.empty((0, n_bin))
    for found in list_founds:
        _h, _b = np.histogram(found, n_bin)
        hist = np.vstack((hist, np.histogram(found, n_bin)[0]))
    bins = np.zeros(len(_b) - 1)
    for i in range(len(bins)):
        bins[i] = (_b[i] + _b[i + 1]) / 2
    return hist, bins


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spikes", type=str)
    parser.add_argument("--triggers", type=str)
    parser.add_argument("--tones", type=str)
    parser.add_argument("--channel", type=int)
    opt = parser.parse_args()
    return opt


def main(opt):
    spikes = np.load(opt.spikes)
    spikes = spikes[opt.channel]
    triggers = np.load(opt.triggers)
    triggers = extract_trigger_time(triggers)
    tones = np.fromfile(opt.tones, dtype=np.double)
    spk_times = absolute_thresholding(spikes)
    _tuning_curve = tuning_curve(spk_times, triggers, tones)
    plt.plot(_tuning_curve[0])
    plt.show()
    _psth, _bins = psth(spk_times, triggers, tones)
    for i in range(_psth.shape[0]):
        plt.plot(_psth[i])
    plt.show()


if __name__ == "__main__":
    #  "C:\Users\Flavi\Desktop\filtered.npy" --channel 23 --triggers "C:\Users\Flavi\Desktop\au
    # dio_trigger.npy" --tones "C:\Users\Flavi\Desktop\tones_00_MANIGODINE_SESSION_01_20220705.bin"
    opt = parse_args()
    main(opt)
