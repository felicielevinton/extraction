import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import argparse





def mean_fr_std(spikes, t_end=None, t_begin=0, bin_size=1, fs=30000.):
    if t_end is None and t_begin is None:
        x = spikes
    else:
        if t_end is None:
            t_end = spikes[-1]
        if t_begin is None:
            t_begin = 0
        x = np.where(np.logical_and(spikes > t_begin * fs, spikes < t_end))[0]
        x = spikes[x]
    t = t_end - t_begin
    step = bin_size * fs
    window = t_begin
    n_bin = int(t // (bin_size * fs))
    fr = np.zeros(n_bin - 1)
    for _ii in range(n_bin - 1):
        y = np.where(np.logical_and(x > window, x < window + step))[0]
        window += step
        fr[_ii] = len(y) / bin_size
    return fr, fr.mean(), fr.std()


def get_spikes_tc(spikes, triggers, fs=30000., t_pre=0.0, t_post=0.6):
    found = np.zeros(len(triggers))
    for i, trig in enumerate(triggers):
        x = len(np.where(np.logical_and(spikes > trig - t_pre * fs, spikes < trig + t_post * fs))[0])
        found[i] = x
    return found


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
        idx = np.where(tone_sequence == tone)[0]
        spk_bin = get_spikes_tc(spikes, trigger_array[idx])
        list_found[j].append(spk_bin)
    m_spk_count = np.zeros(len(tones))
    std_spk_count = np.zeros(len(tones))
    for j, found in enumerate(list_found):
        found = np.array(found)
        m_spk_count[j] = found.mean()
        std_spk_count[j] = found.std()
    return m_spk_count, std_spk_count


def get_spikes(spikes, triggers, fs=30000., t_pre=0.100, t_post=0.700):
    found = np.empty(0, dtype=np.double)
    for trig in triggers:
        x = np.where(np.logical_and(spikes > trig - t_pre * fs, spikes < trig + t_post * fs))[0]
        x = spikes[x]
        x -= trig
        x = x.astype(np.double)
        x /= fs
        found = np.hstack((found, x))
    return found


def psth(spikes, trigger_array, tone_sequence, t_pre=0.150, t_post=0.600, t_bin=0.005):
    list_founds = list()
    tones = np.unique(tone_sequence)
    for tone in tones:
        idx = np.where(tone_sequence == tone)[0]
        list_founds.append(get_spikes(spikes, trigger_array[idx], t_pre=t_pre, t_post=t_post))
    n_bin = int((abs(t_pre) + t_post) / t_bin)
    hist = np.empty((0, n_bin))
    for found in list_founds:
        _h, _b = np.histogram(found, n_bin)
        hist = np.vstack((hist, _h))
    bins = np.zeros(len(_b) - 1)
    for i in range(len(bins)):
        bins[i] = (_b[i] + _b[i + 1]) / 2
    return hist, bins


def get_spikes_raster(spikes, triggers, fs=30000., t_pre=0.100, t_post=0.6):
    list_trial = list()
    for trig in triggers:
        x = np.where(np.logical_and(spikes > trig - t_pre * fs, spikes < trig + t_post * fs))[0]
        x = spikes[x]
        x -= trig
        x = x.astype(np.double)
        x /= fs
        list_trial.append(x)
    return list_trial


def raster(spikes, trigger_array, tone_sequence, t_pre=0.100, t_post=0.700):
    list_found = list()
    tones = np.unique(tone_sequence)
    for j, tone in enumerate(tones):
        idx = np.where(tone_sequence == tone)[0]
        spk_bin = get_spikes_raster(spikes, trigger_array[idx])
        list_found.append(spk_bin)
    return list_found


def parse_args():
    parser = argparse.ArgumentParser(prog="fast_analysis")
    parser.add_argument("--spikes", type=str, help="neural recordings in .npy format and double fp precision.")
    parser.add_argument("--triggers", type=str, help="triggers in .npy format.")
    parser.add_argument("--tones", nargs="+", type=str, help="tones played during recording session."
                                                             "Those are .bin files.")
    parser.add_argument("--channel", nargs="+", type=int, help="channels you want to threshold.")
    parser.add_argument("--threshold", type=float, help="threshold to use, in µV.")
    parser.add_argument("--detection_mode", type=str, help="Thresholding mode: absolute or rms")
    _opt = parser.parse_args()
    return _opt


def load_spike_data(path):
    return np.lib.format.open_memmap(path, mode="r")


def extract_trigger_time(dig_in_channel):
    return np.where(np.diff(dig_in_channel) == 1)[0] + 1


def thresholder(chan, mode, threshold):
    if mode == "absolute":
        return absolute_thresholding(chan, threshold)
    else:
        return relative_thresholding(chan, threshold)


def find_peaks(chan, threshold):
    if threshold < 0:
        threshold *= -1
    opp_chan = chan * -1
    _spk_times, _height = signal.find_peaks(opp_chan, threshold)
    return _spk_times, _height["peak_heights"] * -1


def compute_rms_threshold(neural_channel, rms_level=4.):
    print("Computing floor noise...")
    rms = np.power(neural_channel, 2).mean()
    rms = np.sqrt(rms)
    return rms * rms_level


def relative_thresholding(chan, threshold_rms=-4.0):
    rms = compute_rms_threshold(chan, threshold_rms)
    return find_peaks(chan, rms)


def absolute_thresholding(chan, threshold=70):
    opp_chan = chan * -1
    _spk_times, _height = signal.find_peaks(opp_chan, threshold)
    return _spk_times, _height["peak_heights"] * -1


def main(opt):
    spikes = load_spike_data(opt.spikes)
    channels = opt.channel
    triggers = np.load(opt.triggers)
    triggers = triggers.astype(np.int8)
    triggers = extract_trigger_time(triggers)
    triggers = np.hstack((triggers[:660], triggers[-660:]))
    tones = np.empty(0)
    for _t in opt.tones:
        tones = np.hstack((tones, np.fromfile(_t, dtype=np.double)))
    b_sup = 90000
    b_inf = 60000
    u = np.unique(tones)
    for i, channel in enumerate(channels):
        spike = spikes[channel]
        duration_recording = len(spike) / 30000.
        spk_times, height = thresholder(spike, opt.detection_mode, opt.threshold)
        _a, mu_fr, sigma = mean_fr_std(spk_times, t_end=triggers[0])
        ticks = np.where(np.logical_and(spk_times > b_inf, spk_times < b_sup))[0]
        plt.plot(np.arange(b_inf, b_sup), spike[b_inf:b_sup])
        plt.plot(spk_times[ticks], height[ticks], "x")
        plt.show()
        _tuning_curve = tuning_curve(spk_times, triggers, tones)
        tune_for = u[np.where(_tuning_curve[0] == _tuning_curve[0].max())]
        fr = round(len(spk_times) / duration_recording, 1)
        print(f"Channel #{opt.channel[i]}: fd = {fr} Hz, tune for: {tune_for[0]}Hz")
        plt.semilogx(u, _tuning_curve[0])
        plt.title(f"TC for Channel #{opt.channel[i]}: tuned for {tune_for}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Spike count")
        plt.show()
        _raster = raster(spk_times, triggers, tones)
        _psth, _bins = psth(spk_times, triggers, tones)
        __raster = list()
        for elt in _raster:
            for ii in elt:
                __raster.append(ii)
        plt.eventplot(__raster, linestyles="solid", color="k", linewidths=0.7, lineoffsets=0.5, linelengths=0.5)
        plt.show()
        for j, elt in enumerate(_raster):
            figs, axes = plt.subplots(2, 1)
            axes[0].eventplot(elt, linestyles="solid", color="k", linewidths=0.5, lineoffsets=0.5, linelengths=0.5)
            # "dotted"
            axes[0].axvline(0, color="r", linewidth=0.5)
            axes[0].set_ylabel("Trial #")
            axes[0].set_xlabel("Time [s]")
            axes[0].set_title(f"Raster. Channel #{opt.channel[i]}, Frequency = {u[j]}, fr = {fr}")
            axes[1].plot(_bins, _psth[j], color="k")
            axes[1].axvline(0, color="r", linewidth=0.5)
            mu_psth = _psth[j].mean()
            sig_psth = _psth[j].std()
            axes[1].axhline(mu_psth, color="g", linewidth=0.5)
            axes[1].axhline(mu_psth + sig_psth * 3, color="g", linewidth=0.3, linestyle="dotted")
            axes[1].axhline(mu_psth - sig_psth * 3, color="g", linewidth=0.3, linestyle="dotted")
            axes[1].set_title(f"PSTH. Frequency = {u[j]}")
            plt.show()


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
