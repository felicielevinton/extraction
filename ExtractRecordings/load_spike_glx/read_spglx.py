import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def read_meta_file(bin_full_path):
    """

    :param bin_full_path:
    :return:
    """
    meta_name = bin_full_path.stem + ".meta"
    meta_path = Path(bin_full_path.parent / meta_name)
    meta_dict = dict()
    if meta_path.exists():
        with meta_path.open() as f:
            m_dat_list = f.read().splitlines()
            for m in m_dat_list:
                cs_list = m.split(sep="=")
                if cs_list[0][0] == "~":
                    current_key = cs_list[0][1:len(cs_list[0])]
                else:
                    current_key = cs_list[0]
                meta_dict.update({current_key: cs_list[1]})
    else:
        print("No meta file.")
    return meta_dict


def sample_rate(meta):
    """
    Donne la fréquence d'échantillonage de l'enregistrement, que ce soit le .meta d'imec
    ou le .meta de NI.
    :param meta:
    :return:
    """
    if meta["typeThis"] == "imec":
        s_rate = float(meta["imSampRate"])
    else:
        s_rate = float(meta["niSampRate"])
    return s_rate


def integer_to_volts(meta):
    """
    Conversion des entiers vers des volts. NI et IMEC.
    :param meta:
    :return:
    """
    if meta["typeThis"] == "imec":
        if "imMaxInt" in meta:
            max_int = int(meta["imMaxInt"])
        else:
            max_int = 512
        fi2v = float(meta["niAiRangeMax"]) / max_int
    else:
        fi2v = float(meta["niAiRangeMax"]) / 32768
    return fi2v


def original_channels(meta):
    if["snsSaveChanSubset"] == "all":
        channels = np.arange(0, int(meta["nSavedChans"]))
    else:
        channels = np.zeros(1)
    return channels


def channel_count_ni(meta):
    """
    Traitement des fichiers .meta NI: donne les canaux Analogues et digitaux.
    :param meta:
    :return:
    """
    chan_count_list = meta["snsMnMaXaDw"].split(sep=',')
    mn = int(chan_count_list[0])
    ma = int(chan_count_list[1])
    xa = int(chan_count_list[2])
    dw = int(chan_count_list[3])
    return mn, ma, xa, dw


def channel_count_imec(meta):
    chan_count_list = meta["snsApLfSy"].split(sep=',')
    ap = int(chan_count_list[0])
    lf = int(chan_count_list[1])
    sy = int(chan_count_list[2])
    return ap, lf, sy


def chan_gain_ni(ichan, saved_mn, saved_ma, meta):
    if ichan < saved_mn:
        gain = float(meta["niMNGain"])
    elif ichan < (saved_mn + saved_ma):
        gain = float(meta["niMAGain"])
    else:
        gain = 1
    return gain


def chan_gain_im(meta):
    imro_list = meta["imroTb1"].split(sep=")")
    n_chan = len(imro_list) - 2
    ap_gain = np.zeros(n_chan)
    lf_gain = np.zeros(n_chan)
    if "imDatPrb_type" in meta:
        probe_type = meta["imDatPrb_type"]
    else:
        probe_type = 0
    if probe_type == 21 or probe_type == 24:
        ap_gain = ap_gain + 80
    else:
        for i in range(n_chan):
            current_list = imro_list[i + 1].split(" ")
            ap_gain[i] = current_list[3]
            lf_gain[i] = current_list[4]
    return ap_gain, lf_gain


def gain_correct_ni(array, channel_list, meta):
    mn, ma, xa, dw = channel_count_ni(meta)
    fi2v = integer_to_volts(meta)
    conv_array = np.zeros(array.shape, dtype=float)
    for i in range(0, len(channel_list)):
        j = channel_list[i]  # index into timepoint
        conv = fi2v / chan_gain_ni(j, mn, ma, meta)
        # dataArray contains only the channels in chanList
        conv_array[i, :] = array[i, :] * conv
    return conv_array


def gain_correct_im(array, chan_list, meta):
    chans = original_channels(meta)
    ap_gain, lf_gain = chan_gain_im(meta)
    n_ap = len(ap_gain)
    n_nu = n_ap * 2
    fI2V = integer_to_volts(meta)
    conv_array = np.zeros(array.shape, dtype="float")
    for i in range(chan_list):
        j = chan_list[i]
        k = chans[j]
        if k < n_ap:
            conv = fI2V / ap_gain[k]
        elif k < n_nu:
            conv = fI2V / lf_gain[k - n_ap]
        else:
            conv = 1
        conv_array[i, :] = array[i, :] * conv
    return conv_array


def memmap_raw(bin_full_path, meta):
    n_chan = int(meta["nSavedChans"])
    n_file_samp = int(int(meta["fileSizeBytes"]) / (2 * n_chan))
    raw_data = np.memmap(bin_full_path, dtype="int16", mode="r", shape=(n_chan, n_file_samp), offset=0, order="F")
    return raw_data


def extract_digital(raw, dw_req, d_line_list, meta):
    """

    """
    if meta["typeThis"] == "imec":
        ap, lf, sy = channel_count_imec(meta)
        if sy == 0:
            print("No SYNC channel.")
            dig_array = np.zeros(0, 'uint8')
            return dig_array
        else:
            dig_ch = ap + lf + dw_req
    else:
        mn, ma, xa, dw = channel_count_ni(meta)
        if dw_req > dw - 1:
            dig_array = np.zeros(0, "uint8")
            return dig_array
        else:
            dig_ch = mn + ma + xa + dw_req

    select_data = np.ascontiguousarray(raw[dig_ch], "int16")
    n_samp = len(select_data)
    bitwise_data = np.unpackbits(select_data.view(dtype="uint8"))
    bitwise_data = np.transpose(np.reshape(bitwise_data, (n_samp, 16)))
    n_line = len(d_line_list)
    dig_array = np.zeros((n_line, n_samp), "uint8")
    for i in range(n_line):
        byte_n, bit_n = np.divmod(d_line_list[i], 8)
        targ_i = byte_n * 8 + (7 - bit_n)
        dig_array[i, :] = bitwise_data[targ_i, :]
    return dig_array





