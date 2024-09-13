from PostProcessing.tools import lfp
import PostProcessing.tools.utils as ut
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def main(opt):
    folder = opt.folder
    triggers = dict()
    if os.path.exists(os.path.join(folder, "recording_length.bin")):
        with open(os.path.join(opt.folder, "recording_length.bin"), "r") as f:
            length = int(f.read())

    else:
        d_trigs = np.load(os.path.join(opt.folder, "dig_in.npy"))
        length = d_trigs.shape[1]
        with open(os.path.join(opt.folder, "recording_length.bin"), "w") as f:
            f.write('{:03d}\n'.format(length))

    if os.path.exists(os.path.join(folder, "trig_dig_chan1.npy")):
        triggers["dig"] = np.load(os.path.join(folder, "trig_dig_chan1.npy"))
    else:
        d_trigs = np.load(os.path.join(opt.folder, "dig_in.npy"))
        triggers["dig"] = ut.extract_digital_triggers(d_trigs[1])
        np.save(os.path.join(folder, "trig_dig_chan0.npy"), ut.extract_digital_triggers(d_trigs[0]))
        np.save(os.path.join(folder, "trig_dig_chan1.npy"), triggers["dig"])

    if os.path.exists(os.path.join(folder, "trig_analog_chan1.npy")):
        triggers["tracking"] = np.load(os.path.join(folder, "trig_analog_chan0.npy"))
        triggers["playback"] = np.load(os.path.join(folder, "trig_analog_chan1.npy"))

    else:
        a_trigs = np.load(os.path.join(opt.folder, "analog_in.npy"))
        triggers["tracking"] = ut.extract_analog_triggers(a_trigs[0])
        triggers["playback"] = ut.extract_analog_triggers(a_trigs[1])
        np.save(os.path.join(folder, "trig_analog_chan0.npy"), triggers["tracking"])
        np.save(os.path.join(folder, "trig_analog_chan1.npy"), triggers["playback"])

    d_out = get_data_2(folder, triggers)
    return d_out, length


def parse_args():
    parser = argparse.ArgumentParser(prog="LFP")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    options = parse_args()
    plot_path = os.path.join(options.folder, "plots")
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    trig_and_tones, recording_length = main(options)
    pb_pattern = "pb_[0-9]"
    tracking_pattern = "tracking_[0-9]"
    warmup_pattern = "warmup_[0-9]"
    warmup = trig_and_tones["warmup_0"]
    warmout = trig_and_tones["warmup_1"]
    pb_l = merge_pattern_2(trig_and_tones, pb_pattern)
    f_int = 10079.
    data = lfp.IntanLFP(options.folder)
    idx = np.where(warmup[0] == f_int)[0]
    trigs = warmup[1][idx]
    channel = 21
    lfp_list = list()
    for trig in trigs:
        x = data.get_lfp_around(channel, trig, 0.5)
        lfp_list.append(x)
    lfp_list = np.vstack(lfp_list)
    trigs = list()
    for elt in pb_l:
        trigs.append(elt[1][np.where(elt[0] == f_int)[0]])
    trigs = np.hstack(trigs)
    lfp_list_pb = list()
    for trig in trigs:
        x = data.get_lfp_around(channel, trig, 0.5)
        lfp_list_pb.append(x)
    lfp_list_pb = np.vstack(lfp_list_pb)
    from scipy import signal
    f, t, sxx_wp = signal.spectrogram(lfp_list.mean(0), fs=2500, nperseg=256, window="hann")
    _, _, sxx_pb = signal.spectrogram(lfp_list_pb.mean(0), fs=2500, nperseg=256, window="hann")
    sxx = np.hstack((sxx_wp, sxx_pb))
    idx = np.where(f < 175)[0]
    plt.pcolormesh(np.hstack((t, t + t)), f[idx], sxx[idx])
    plt.show()
    plt.plot(lfp_list.mean(0))
    plt.plot(lfp_list_pb.mean(0))
    plt.show()
    tr_l = merge_pattern_2(trig_and_tones, tracking_pattern)

    wp_l = ut.bin_experiment(warmup[1], 5)
    shift = 0
    for wp in wp_l:
        x = data.get_lfp_between(channel, wp[0], wp[1])
        f, psd = lfp.normalized_psd(x)
        psd += shift
        shift -= 10
        idx = np.where(f < 250)[0]
        plt.plot(f[idx], psd[idx], linewidth=0.25)
    plt.show()
    wo_l = ut.bin_experiment(warmout[1], 5)
    shift = 0
    for i, wo in enumerate(wo_l):
        x = data.get_lfp_between(channel, wo[0], wo[1])
        f, psd = lfp.normalized_psd(x)
        psd += shift
        np.save(os.path.join(options.folder, f"warmout_{i}_lfp.npy"), x)
        shift -= 10
        idx = np.where(f < 250)[0]
        plt.plot(f[idx], psd[idx], linewidth=0.25)
    plt.show()
    shift = 0
    for tr in tr_l:
        x = data.get_lfp_between(channel, tr[1][0], tr[1][-1])
        f, psd = lfp.normalized_psd(x)
        psd += shift
        shift -= 10
        idx = np.where(f < 250)[0]
        plt.plot(f[idx], psd[idx], linewidth=0.25)
    plt.title("Tracking PSD")
    plt.show()

    shift = 0
    for i, pb in enumerate(pb_l):
        x = data.get_lfp_between(channel, pb[1][0], pb[1][-1])
        if i == 4:
            np.save(os.path.join(options.folder, "playback_4_lfp.npy"), x)
        f, psd = lfp.normalized_psd(x)
        psd += shift
        shift -= 10
        idx = np.where(f < 250)[0]
        plt.plot(f[idx], psd[idx], linewidth=0.25)
    plt.title("Playback PSD")
    plt.show()

    x = data.get_lfp_between(channel, warmup[1][0], warmup[1][-1])
    f, psd = lfp.normalized_psd(x)
    idx = np.where(f < 250)[0]
    plt.plot(f[idx], psd[idx])
    plt.title("Warmup PSD")
    plt.show()

    x = data.get_lfp_between(channel, warmout[1][0], warmout[1][-1])
    f, psd = lfp.normalized_psd(x)
    idx = np.where(f < 250)[0]
    plt.plot(f[idx], psd[idx])
    plt.title("Warmout PSD")
    plt.show()

    # demain, regarder onset par onset la différence des lfp dans
    # les deux conditions.
