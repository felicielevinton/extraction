import numpy as np
import load_spike_glx as io
import load_exp_files as io_bin
import matplotlib.pyplot as plt


def extract_barcodes_from_times(sync, inter_barcode_interval=400000, bar_duration=850,
                                barcode_duration_ceiling=30000, nbits=32):
    """Read barcodes from timestamped rising and falling edges.
    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimum duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode
    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.
    Notes
    -----
    ignores first code in prod (ok, but not intended)
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)
    """
    on_times = np.where(np.diff(sync) == 1)[0]
    # h, bins = np.histogram(np.diff(on_times), bins=2)
    # print(h)
    # print(bins)
    off_times = np.where(np.diff(sync) == -1)[0]
    start_indices = np.diff(on_times)  # prendre le premier indice aussi, indiquant le premier code barre.

    _foo = np.where(start_indices > inter_barcode_interval)[0] + 1
    a = np.zeros(1, dtype=_foo.dtype)
    a = np.hstack((a, _foo))
    barcode_start_times = on_times[_foo]
    barcodes = np.zeros_like(barcode_start_times, dtype=np.int64)
    for i, t in enumerate(barcode_start_times):

        # oncode = on_times[np.logical_and(on_times > t, on_times < t + barcode_duration_ceiling)]
        oncode = on_times[np.logical_and(on_times > t - 100, on_times < t + barcode_duration_ceiling)]
        plt.plot(sync[oncode[0]:oncode[0]+barcode_duration_ceiling])
        plt.show()
        bits = np.zeros(nbits, dtype=np.bool)
        offcode = off_times[np.logical_and(off_times > t - 100, off_times < t + barcode_duration_ceiling)]
        start = 0
        oncode = oncode[1:]
        _s = np.hstack((oncode, offcode))
        vals = np.hstack((np.ones_like(oncode), np.zeros_like(offcode)))
        args = np.argsort(_s)
        _s = _s[args]
        vals = vals[args]
        for h, elt in _s:
            y = (oncode[h] - elt) % bar_duration
            bits[start:start+y] = vals[h]
        bits = np.zeros(nbits, dtype=np.bool)
        diff = offcode[1:] - oncode[1:]
        _s = np.hstack((oncode, offcode))
        vals = np.hstack((np.ones_like(oncode), np.ones_like(offcode) * -1))
        args = np.argsort(_s)
        _s = _s[args]
        vals = vals[args]
        plt.plot(_s, vals)
        plt.show()
        # _s = np.sort(_s)

        if oncode[1] < offcode[1]:
            bit = True
        else:
            bit = False

        truth = diff // bar_duration
        # print(truth)
        # if oncode[0] < offcode[0]:
        #     curr_time = oncode[0]
        # else:
        #     curr_time = offcode[0]


        start = 0
        for elt in truth:
            bits[start:start+elt] = bit
            start += elt
            bit = not bit
        print(bits)
        # for bit in range(0, nbits):
#
        #     next_on_arr = np.where(oncode > curr_time)[0]
        #     next_off_arr = np.where(offcode > curr_time)[0]
#
        #     if next_on_arr.size > 0:
        #         next_on = oncode[next_on_arr[0]]
        #     else:
        #         next_on = t + inter_barcode_interval  # ?
#
        #     if next_off_arr.size > 0:
        #         next_off = offcode[next_off_arr[0]]
        #     else:
        #         next_off = t + inter_barcode_interval
#
        #     if next_on < next_off:
        #         bits[bit] = 1
#
        #     curr_time += bar_duration

        # barcode = 0

        # least sig left
        # print(bits)
        # print(np.power(np.full_like(a=bits, fill_value=2), np.arange(len(bits), dtype=np.int64)))
        # barcode += bits[bit] * np.pow(2, bit)
        bits = bits * np.power(np.full_like(a=bits, fill_value=2, dtype=np.int64), np.arange(len(bits), dtype=np.int64))
        print(bits)
        barcodes[i] = np.sum(bits)


        # barcodes.append(barcode)

    return barcode_start_times, np.array(barcodes, dtype=np.int64)


def get_barcodes(sync_channel, inter=500000):
    """

    """
    on_times = np.where(np.diff(sync_channel) == 1)[0]
    off_times = np.where(np.diff(sync_channel) == -1)[0]
    start_indices = np.diff(on_times)
    putative_bc_start = np.where(start_indices > inter)[0] + 1
    last = 0
    _on = list()
    _off = list()
    n_barcodes = 0
    for i, elt in enumerate(putative_bc_start):
        _on.append(on_times[last:elt])
        _off.append(off_times[last:elt])
        last = elt
        n_barcodes += 1
    bc_sum = np.zeros(n_barcodes)
    for i in range(n_barcodes):
        s = np.sum(_off[i] - _on[i])
        # s = s - s % 10
        bc_sum[i] = s
    return bc_sum, on_times, putative_bc_start


def find_common_seq(sync_intan, sync_ni):
    """

    """
    delta = len(sync_ni) - len(sync_intan)
    k = -1
    for i in range(delta):  # on cherche l'égalité entre les deux arrays.
        stop = i + len(sync_intan)
        if np.allclose(sync_intan, sync_ni[i:stop]):
            k = i
    return k


def align_sync_channels_and_get_basler_sync(sync_intan, sync_glx):
    """

    """
    start_bc_intan, val_intan = extract_barcodes_from_times(sync_intan)
    start_bc_glx, val_glx = extract_barcodes_from_times(sync_glx)
    idx = find_common_seq(val_intan, val_glx)
    print(idx)
    if idx != -1:
        d = start_bc_glx[idx] - start_bc_intan[0]  # différence entre les deux enregistrements
        return d


if __name__ == "__main__":
    import os
    from pathlib import Path
    import glob
    folder = "C:/data/experiment/MANIGODINE/Manigodine_20220712/MANIGODINE_20220712_SESSION_00"
    ephys_folder = os.path.join(folder, "MANIGODINE_20220712_session00_g0",
                                "MANIGODINE_20220712_session00_g0_t0.nidq.bin")

    p = np.empty(0, dtype=np.int32)
    t = np.empty(0)

    for path in glob.glob(os.path.join(folder, "positions*.bin")):
        p = np.hstack((p, io_bin.read_positions_file(path)))

    triggers = np.load(os.path.join(folder, "dig_in.npy"))
    sync_channel_intan = triggers[0]
    sync_channel_intan = sync_channel_intan.astype(dtype=np.int8)
    nidq = Path(ephys_folder)
    meta_dict = io.read_meta_file(nidq)
    raw = io.memmap_raw(nidq, meta_dict)
    t_start = 0
    t_end = float(meta_dict["fileTimeSecs"])
    sr = io.sample_rate(meta_dict)
    first_sample = int(t_start * sr)
    last_sample = int(t_end * sr)
    d_line_list_dig = np.arange(8, dtype=np.int8)
    d_line_list_analog = [0]
    dw_req = 0
    dig = io.extract_digital(raw, dw_req, d_line_list_dig, meta_dict)
    save_dig = np.vstack((dig[3]+dig[4], dig[2]))
    np.save(os.path.join(folder, "glx_dig_in.npy"), save_dig)
    for i, elt in enumerate(dig):
        print(f"Chan#{i}: {len(np.where(np.diff(elt) == 1)[0])}")
    mn, ma, xa, dw = io.channel_count_ni(meta_dict)
    data = raw[np.arange(raw.shape[0]-1), first_sample:last_sample]
    conv_data = 1e3*io.gain_correct_ni(data, np.arange(raw.shape[0]-1), meta_dict)

    sync_channel_ni = conv_data[1]
    sync_channel_ni /= sync_channel_ni.max()
    sync_channel_ni = np.logical_and(sync_channel_ni > 0.4, sync_channel_ni < 1.5)
    print(np.where(np.diff(sync_channel_ni) == 1)[0])
    print(np.where(np.diff(sync_channel_intan) == 1)[0])
    sync_channel_ni = sync_channel_ni.astype(dtype=np.int8)
    a, b = extract_barcodes_from_times(sync_channel_intan)
    _a, _b = extract_barcodes_from_times(sync_channel_ni)
    r = align_sync_channels_and_get_basler_sync(sync_channel_intan, sync_channel_ni)

