import numpy as np


def _pos_to_tone(x, width, bw, h_bw, n_area):
    """

    """
    if x == -1:
        index = -1
    elif x < h_bw:
        index = 0
    elif x > (width - h_bw):
        index = n_area - 1
    else:
        index = x - h_bw
        index //= bw
        index += 1
    return index


def clean_positions(positions, width=1920, n_freq=33):
    """

    """
    bw = width // (n_freq - 1)
    h_bw = bw // 2
    idx = np.zeros_like(positions, dtype=positions.dtype)
    for i, pos in enumerate(positions):
        idx[i] = _pos_to_tone(pos, width=width, bw=bw, h_bw=h_bw, n_area=n_freq)
    y = np.where(idx == -1)[0]
    for i in y:
        idx[i] = idx[i - 1]
    return idx


def freq_switch(x):
    """

    """
    f_unique = list()
    f_idx = list()
    r = -1
    for i, j in enumerate(x):
        if j != r:
            r = j
            f_unique.append(r)
            f_idx.append(i)
    return np.array(f_unique), np.array(f_idx, dtype=np.int32)


def align(tones_tracking, tones):
    """

    """
    u_t, c_t = np.unique(tones_tracking, return_counts=True)
    _min = u_t[np.argmin(c_t)]
    t_min = np.where(tones_tracking == _min)[0]
    g_min = np.where(tones == _min)[0]
    diff_t = np.diff(t_min)
    diff_g = np.diff(g_min)
    k = list()
    m = 0
    found = False
    begin_g, begin_t, end_g, end_t = 0, 0, 0, 0
    for i, j in enumerate(diff_g):
        if j == diff_t[m]:
            k.append(i)
            m += 1
            if len(k) == len(diff_t):
                begin_g = g_min[k[0]]
                begin_t = t_min[0]
                found = True
                break
        else:
            k = list()
            m = 0
    if found:
        begin_seq = begin_g - begin_t  # indice du début de séquence dans la séquence totale
        end_seq = begin_seq + len(tones_tracking)
        return begin_seq, end_seq
    else:
        return 0


def replace(positions):
    """

    """
    y = np.where(positions == -1)[0]
    diff_y = np.diff(y)
    diff_y = np.vstack((np.arange(1, len(diff_y) + 1), diff_y)).T
    k = 0
    begin = 0
    for i, elt in diff_y:
        if elt != 1:
            positions[y[i - 1]] = positions[y[i - 1] + 1]
            positions[y[i]] = positions[y[i] - 1]
            if k != 0:
                end = y[i - 1]
                filler = np.full(shape=k, fill_value=positions[begin - 1])
                # filler = np.linspace(positions[begin - 1], positions[end], num=k).astype(dtype=positions.dtype)
                positions[begin:end] = filler
                k = 0
        else:
            if k == 0:
                begin = y[i - 1]
            k += 1
    return positions


def sync(positions, n_bins):
    """

    """
    dim1 = len(positions) // n_bins  # samples per bin.
    remainder = len(positions) % n_bins  # éliminer les points en trop.
    positions_binned = np.reshape(positions[:-remainder], (dim1, n_bins)).mean(1)
    return positions_binned


def get_positions(tracking_tones, tones, positions):
    u, c = np.unique(tones, return_counts=True)
    clean_pos = clean_positions(positions)
    f, idx_switch = freq_switch(clean_pos)
    ff = np.zeros_like(f, dtype=np.float64)

    for i, elt in enumerate(f):
        ff[i] = u[elt]

    begin, end = align(tracking_tones, ff)
    positions = replace(positions[idx_switch[begin]:idx_switch[end]])
    return positions


# if __name__ == "__main__":
#     folder = "/home/feral/Desktop/Manigodine/MANIGODINE/Manigodine_20220712/MANIGODINE_20220712_SESSION_00"
#     import os
#     import glob
#     p = np.empty(0, dtype=np.int32)
#     t = np.empty(0)
#     for path in glob.glob(os.path.join(folder, "positions*.bin")):
#         p = np.hstack((p, io.read_positions_file(path)))
#
#     for path in glob.glob(os.path.join(folder, "tones*.bin")):
#         t = np.hstack((t, io.read_tones_file(path)))
#
#     track = np.load(os.path.join(folder, "__t.npy"))
#
#     e_p = get_positions(track, t, p)
#     print(len(e_p))
#     plt.plot(e_p)
#     plt.show()
