from sequences import *
from extraction_utils import extract
from get_data_v2 import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json


ANALOG_TRIGGERS_MAPPING = {"Main": 0, "Playback": 1}
DIGITAL_TRIGGERS_MAPPING = {"Basler": 0, "Sounds": 1}


def read_json(folder):

    key_to_fetch = "Experiment_1"  # on cherche la première expérience. ?

    out = list(glob.glob(os.path.join(folder, "session_*.json")))

    exp_type = ""

    try:
        assert (len(out) == 1), "Glob in folder should be of length 1."
        fname = out[0]
        with open(fname, "r") as f:
            d = json.load(f)

    except AssertionError as error:
        print("Error: ", error)

    try:
        assert (key_to_fetch in d.keys()), f"{key_to_fetch} not in json."
        sub_d = d[key_to_fetch]
        exp_type = sub_d["Type"]

    except AssertionError as error:
        print("Error: ", error)

    # is_v2(d) => utiliser ça.
    version_key = "Version"  # Là on demande si c'est le nouveau format de fichier.
    # todo : la nouvelle fonction doit retourner un json.
    if version_key in d.keys():
        pass

    else:
        pass

    return exp_type


def is_v2(d):
    version_key = "Version"  # Là on demande si c'est le nouveau format de fichier.
    if version_key in d.keys():
        return True

    else:
        return False


def extract_v2(d, triggers, folder):
        seq, length = check_already_extracted(folder)
        #exp_type = read_json(folder)
        exp_type = "Playback"
        seq = extract_according_exp_type(exp_type, triggers=triggers, folder=folder, compatibility=compatibility)

        seq.set_recording_length(length)
        seq.save(folder)
    print("All izz well")


def extract_data(folder, analog_channels=None, digital_channels=None, compatibility=False):

    seq, length = check_already_extracted(folder)

    if seq and length:
        return extract_tt(folder)

    else:
        triggers = extract(folder)
        exp_type = read_json(folder)
        seq = extract_according_exp_type(exp_type, triggers=triggers, folder=folder, compatibility=compatibility)

        seq.set_recording_length(length)
        seq.save(folder)
        return seq


def extract_according_exp_type(type_of, triggers, folder, compatibility=False):
    """

    """
    seq = None

    if type_of == "Playback":

        seq = get_playback(triggers, folder)

        # if compatibility:
        #     seq = get_data_4(triggers, triggers["dig"], folder)  # Pour Manigodine.

    elif type_of == "PureTones":
        seq = get_tonotopy(folder=folder, triggers=triggers)

    return seq


def get_tonotopy(triggers, folder):
    """
    Charge une expérience de tonotopie.
    """
    tt_seq = Tonotopy(n_iter=1)

    tones = np.empty(0)

    for file in glob.glob(os.path.join(folder, "tones_*.bin")):
        tones = np.hstack((tones, np.fromfile(file, dtype=np.double)))

    tt_seq.add(Pair(tones, triggers["tracking"], "PureTones", 0))

    return tt_seq


def get_playback(triggers, folder):
    """
    Charge une expérience Playback.
    """
    l_tracking, l_mock, l_pb, l_warmup = fetch_tones(folder)

    # Cette assertion peut mener à des erreurs.
    assert (len(l_pb) == len(l_tracking) == len(l_mock))

    n_iter = len(l_pb)

    fn = os.path.join(folder, "durations.json")

    # todo : penser à incorporer les données dans le .json de manip.
    if os.path.exists(fn):
        with open(fn, "r") as f:
            d = json.load(f)
        duration_tr = d["tracking"]
        duration_warmup = d["warmup"]
        duration_warmdown = d["warmdown"]

    else:
        duration_tr = 5
        duration_warmup = 10
        duration_warmdown = 10

    c = 0
    sequence = Sequence()
    sequence.add(XPSingleton("warmup", c, 0, duration_warmup, tones=l_warmup[0]))
    c += 1
    for i in range(n_iter):
        sequence.add(XPSingleton("tracking", c, i, duration_tr, tones=l_tracking[i]))
        c += 1
        sequence.add(XPSingleton("mock", c, i, duration_tr, tones=l_mock[i]))
        c += 1
        sequence.add(XPSingleton("playback", c, i, duration_tr, tones=l_pb[i]))
        c += 1

    sequence.add(XPSingleton("warmdown", c, 0, duration_warmdown, tones=l_warmup[1]))

    d_out = divide_triggers_2(triggers, sequence, n_iter=n_iter)
    return d_out


def get_bin_pos(t_0, _t, _size):
    _d = _t - t_0
    _p = int(_d / _size)
    return _p


def iterate_tones_folder(folder, pattern):
    """
    Retourne une liste de np.ndarray avec les tons joués.
    """
    seq_out = list()
    _glob = glob.glob(os.path.join(folder, pattern))
    _glob = list(_glob)
    _glob.sort()
    for file in _glob:
        seq_out.append(np.fromfile(file, dtype=np.double))
    return seq_out


def if_complete_2(analog, sequence, tt, type_of=None):
    if type_of is None:
        xp_list = sequence.get_tracking()
    else:
        xp_list = sequence.get_in_order_for_type(type_of)
    for elt in xp_list:
        t = elt.tones
        triggers, analog = analog[:len(t)], analog[len(t):]
        tt.add(Pair(t, triggers, elt.type, number=elt.n, order=elt.order))
    return tt


def build_pair_from_singleton(analog, tones, singleton):
    p = Pair(tones, analog, singleton.type, number=singleton.n, order=singleton.order)
    return p


def divide_triggers_2(triggers, sequence, n_iter):
    tt_seq = SequenceTT(n_iter=n_iter)
    tt_seq = triggers_tones_inspection_2(tt_seq, triggers, sequence, n_iter)
    return tt_seq


def fetch_tones(folder):
    """

    """
    t_path = os.path.join(folder, "tones")
    tracking_pattern = "tracking_0*.bin"
    mock_pattern = "tracking_mock*.bin"
    pb_pattern = "playback_*.bin"
    warmup_pattern = "warmup_*.bin"

    l_tracking = iterate_tones_folder(t_path, tracking_pattern)

    l_mock = iterate_tones_folder(t_path, mock_pattern)

    l_pb = iterate_tones_folder(t_path, pb_pattern)

    l_warmup = iterate_tones_folder(t_path, warmup_pattern)

    return l_tracking, l_mock, l_pb, l_warmup


def nan_sum(x):
    return np.sum(np.isnan(x[1]))


def has_nan(x):
    return np.isnan(x[1])


# Files part.
def get_recording_length(folder):
    assert (os.path.exists(os.path.join(folder, "recording_length.bin"))), "No length file."
    with open(os.path.join(folder, "recording_length.bin"), "r") as f:
        length = int(f.read())
    return length


def save_recording_length(folder, length):
    with open(os.path.join(folder, "recording_length.bin"), "w") as f:
        f.write('{:03d}\n'.format(length))


def load_files(file_list):
    out = dict()
    file_list.sort()
    for i, elt in enumerate(file_list):
        out[i] = np.load(elt)
    return out


def load_analog_files(file_list):
    out = dict()
    for elt in file_list:
        if re.search("Tracking", elt):
            out["tracking"] = np.load(elt)
        elif re.search("Mock.npy", elt):
            out["mock"] = np.load(elt)
        elif re.search("Playback", elt):
            out["playback"] = np.load(elt)
    return out


def load_digital_files(file_list):
    out = dict()
    for elt in file_list:
        if re.search("Basler", elt):
            out["basler"] = np.load(elt)
        elif re.search("Sounds", elt):
            out["sounds"] = np.load(elt)
    return out


def check_plot_folder_exists(directory):
    path = os.path.join(directory, "plot")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def check_already_extracted(folder):
    tt = False
    length = False

    if os.path.exists(os.path.join(folder, "tt.npz")):
        tt = True

    if os.path.exists(os.path.join(folder, "recording_length.bin")):
        length = True

    return tt, length


def extract_tt(folder):
    tt_file = os.path.join(folder, "tt.npz")
    assert(os.path.exists(tt_file)), "No tt file."
    seq = SequenceTT(folder=folder)  # Aïe ?
    return seq


def check_digital_triggers(folder):
    return check_files(folder, analog=False)


def check_analog_triggers(folder):
    return check_files(folder, analog=True)


def check_files(folder, analog=True):
    """

    """
    if analog:
        fn_pattern = os.path.join(folder, "trig_analog_chan*.npy")
    else:
        fn_pattern = os.path.join(folder, "trig_dig_chan*.npy")
    lf = list(glob.glob(fn_pattern))
    return len(lf), lf


def process_digital_file(folder):
    """
    Prend un dossier en entrée. Cherche les noms de fichiers digitaux et extrait les temps de triggers.
    """
    return process_file(folder, analog=False)


def process_analog_file(folder, compatibility=False):
    """
    Prend un dossier en entrée. Cherche les noms de fichiers analogiques et extrait les temps de triggers.
    """
    return process_file(folder, analog=True, compatibility=compatibility)


def process_file(folder, analog=True, compatibility=False):
    """
    Comment sortir proprement les triggers mock?
    Est-ce que c'est à passer dans une classe
    """
    out = dict()
    if analog:
        f = os.path.join(folder, "analog_in.npy")

        if compatibility:
            fn_pattern = os.path.join(folder, "trig_analog_chan{}.npy")
            func = ut.extract_analog_triggers_compat
        else:
            fn_pattern = os.path.join(folder, "trig_analog_chan_{}.npy")
            func = ut.extract_analog_triggers
    else:
        f = os.path.join(folder, "dig_in.npy")
        fn_pattern = os.path.join(folder, "trig_dig_chan_{}.npy")
        func = ut.extract_digital_triggers
    triggers = np.load(f)
    n_channel = triggers.shape[0]
    for i in range(n_channel):
        events = func(triggers[i])
        if analog:
            if i == ANALOG_TRIGGERS_MAPPING["Main"]:
                if not compatibility:
                    np.save(fn_pattern.format("Tracking"), events[0])
                    np.save(fn_pattern.format("Mock"), events[1])
                    out["tracking"] = events[0]
                    out["mock"] = events[1]
                else:
                    np.save(fn_pattern.format("Tracking"), events)
                    out["tracking"] = events
            elif i == ANALOG_TRIGGERS_MAPPING["Playback"]:
                if not compatibility:
                    events = events[0]
                np.save(fn_pattern.format("Playback"), events)
                out["playback"] = events
        else:

            if i == DIGITAL_TRIGGERS_MAPPING["Basler"]:
                tag = "Basler"
                out["basler"] = events
            elif i == DIGITAL_TRIGGERS_MAPPING["Sounds"]:
                out["sounds"] = events
                tag = "Sounds"
            else:
                tag = i
            np.save(fn_pattern.format(tag), events)
    return out


def triggers_tones_inspection_2(tt_seq, triggers, sequence, n_iter):
    pb_triggers = triggers["playback"]
    tr_triggers = triggers["tracking"]
    mck_triggers = triggers["mock"]
    digital_triggers = triggers["dig"]
    pb_duration = sequence.get_duration_for("playback")

    tr_done = False
    pb_done = False
    mck_done = False

    s_pb = sequence.get_n_tones_for("playback")

    s_mck = sequence.get_n_tones_for("mock")

    s_tr = sum([sequence.get_n_tones_for(elt) for elt in ["tracking", "warmdown", "warmup"]])

    if s_tr == len(tr_triggers):
        tt_seq = if_complete_2(tr_triggers, sequence, tt_seq, type_of=None)
        tr_done = True

    if s_pb == len(pb_triggers):
        tt_seq = if_complete_2(pb_triggers, sequence, tt_seq, type_of="playback")
        pb_done = True

    if s_mck == len(mck_triggers):
        tt_seq = if_complete_2(mck_triggers, sequence, tt_seq, type_of="mock")
        mck_done = True

    where_to_withdraw = list()
    where_to_append = dict()
    if tr_done and pb_done and mck_done:
        return tt_seq, tr_done, pb_done, mck_done

    if not pb_done:
        # assembler les .bin tones
        # regarder quels triggers ne sont pas représentés dans les triggers analogiques et dans les triggers digitaux.

        p = deepcopy(pb_triggers)
        d = deepcopy(digital_triggers)
        d_corr_0 = synchronize_step(d, tr_triggers)
        d = d[np.isnan(d_corr_0[1])]
        a = list()  # triggers analogiques
        b = list()  # triggers digitaux
        for i in range(n_iter):
            start = p[0]
            end = p[0] + sequence.get_duration_for("playback")
            idx_p = np.less_equal(p, end)
            idx_d = np.logical_and(d >= start - 10000, d <= end)

            b.append(d[idx_d])
            a.append(p[idx_p])
            p = p[~idx_p]
        # 1) Assembler
        for i in range(n_iter):
            xp = sequence.get_xp_number("playback", i)
            _tones = xp.tones
            # _tones = np.vstack((np.full_like(_tones, i), _tones))
            # concat_pb_tones.append(_tones)

            if len(_tones) != len(a[i]):
                d_corr = synchronize_step(b[i], a[i], begin=False)
                if has_nan(d_corr[1]):
                    _tones = _tones[~np.isnan(d_corr[1])]
                else:
                    tr = sequence.get_xp_number("tracking", i).tones
                    if len(tr) == len(a[i]):
                        _tones = tr

            tt_seq.add(Pair(_tones, a[i], xp.type, number=xp.n, order=xp.order))

        pb_done = True

    if not tr_done:
        cp_tr_t = deepcopy(tr_triggers)
        # attraper les premiers triggers pour le warmup.
        xp = sequence.get_xp_number("warmup", 0)
        t0 = tr_triggers[0]
        idx = np.equal(xp.tones, 0)

        # 1) on sors le warmup et le warmdown.
        if sum(idx) > 0:
            xp.tones = xp.tones[~idx]
        wp_duration = sequence.get_duration_for("warmup")
        idx = np.less_equal(cp_tr_t, pb_triggers[0] - pb_duration)
        warmup_triggers = cp_tr_t[idx]
        if len(warmup_triggers) != len(xp.tones):
            if len(warmup_triggers) > len(xp.tones):
                d = len(warmup_triggers) - len(xp.tones)
                warmup_triggers = warmup_triggers[d:]
            else:
                # d = len(xp.tones) - len(warmup_triggers)
                xp.tones = xp.tones[:-1]

        tt_seq.add(Pair(xp.tones, warmup_triggers, xp.type, number=xp.n, order=xp.order))
        cp_tr_t = cp_tr_t[~idx]
        idx = np.greater_equal(cp_tr_t, pb_triggers[-1])
        xp = sequence.get_xp_number("warmdown", 0)
        wd_triggers = cp_tr_t[idx]
        tt_seq.add(Pair(xp.tones, wd_triggers, xp.type, number=xp.n, order=xp.order))
        cp_tr_t = cp_tr_t[~idx]

        # 2) on fait l'extraction des blocks tracking.
        if pb_done:
            l_tr_blocks = list()
            for i in range(n_iter):
                tt_after = tt_seq.get_xp_number("playback", i).triggers[0]
                if i > 0:
                    tt_before = tt_seq.get_xp_number("playback", i - 1).triggers[-1]
                    idx = np.logical_and(tt_before <= cp_tr_t, tt_after >= cp_tr_t)
                elif i == 0:
                    idx = np.logical_and(t0 + wp_duration <= cp_tr_t, tt_after >= cp_tr_t)
                l_tr_blocks.append(cp_tr_t[idx])
                cp_tr_t = cp_tr_t[~idx]

            # retirer indice 0 dans les triggers du tracking, si nécessaire.
            for i, elt in enumerate(l_tr_blocks):
                xp = sequence.get_xp_number("tracking", i)
                print(len(xp.tones), len(elt))
                if i == 0 and len(xp.tones) > len(elt):  # cas rare où le ton du warmup se trouve dans le tracking.
                    xp.tones = xp.tones[1:]

                if len(xp.tones) + 1 == len(elt):  # cas commun où le ton du mock se trouve dans le tracking.
                    if i != 0:
                        where_to_withdraw.append(i - 1)
                        mck_xp = sequence.get_xp_number("mock", i - 1)
                        xp.tones = np.hstack((mck_xp.tones[-1], xp.tones))

                if len(elt) + 1 == len(xp.tones):
                    if i != 0:
                        where_to_append[i] = xp.tones[-1]
                        xp.tones = xp.tones[:-1]

                tt_seq.add(Pair(xp.tones, l_tr_blocks[i], xp.type, number=xp.n, order=xp.order))

            tr_done = True

    if not mck_done:
        mck_duration = sequence.get_duration_for("mock")
        l_mck_blocks = list()
        cp_mck_t = deepcopy(mck_triggers)
        if tr_done:
            for i in range(n_iter):
                if len(sequence.get_xp_number("mock", i).tones) == 0:
                    l_mck_blocks.append([])
                    continue
                if i == n_iter - 1:
                    start = tt_seq.get_xp_number("tracking", i).triggers[-1]
                    stop = tt_seq.get_xp_number("warmdown", 0).triggers[0]
                else:
                    start = tt_seq.get_xp_number("tracking", i).triggers[-1]
                    stop = tt_seq.get_xp_number("tracking", i + 1).triggers[0]
                idx = np.logical_and(start <= cp_mck_t, stop >= cp_mck_t)
                l_mck_blocks.append(cp_mck_t[idx])
                cp_mck_t = cp_mck_t[~idx]
        else:
            for i in range(n_iter):
                end = cp_mck_t[0] + mck_duration
                if len(sequence.get_xp_number("mock", i).tones) == 0:
                    l_mck_blocks.append([])
                    continue
                idx = np.less_equal(cp_mck_t, end)
                l_mck_blocks.append(cp_mck_t[idx])
                cp_mck_t = cp_mck_t[~idx]

        for i, elt in enumerate(l_mck_blocks):

            xp = sequence.get_xp_number("mock", i)

            if i in where_to_withdraw:
                tt_seq.add(Pair(xp.tones[:-1], l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))

            elif i in where_to_append.keys():
                tt_seq.add(Pair(xp.tones, l_mck_blocks[i], xp.type, xp.n, xp.order))

            else:
                if len(xp.tones) == len(l_mck_blocks[i]):
                    tt_seq.add(Pair(xp.tones, l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))
                else:
                    tt_seq.add(Pair(xp.tones[:-1], l_mck_blocks[i], xp.type, number=xp.n, order=xp.order))

    return tt_seq


def create_data_folder(folder):
    """
    Créée le dossier data dans le dossier de l'animal.
    """
    data_folder = os.path.join(folder, "data")
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return data_folder


def check_data_folder_exists(folder):
    data_folder = os.path.join(folder, "data")
    if os.path.exists(data_folder):
        return True
    else:
        return False


def load_data_file_if_exists(folder, file):
    """
    Regarde si le fichier existe, le charge si c'est le cas.
    """
    check_data_folder_exists(folder)
    fn = os.path.join(folder, file)
    if os.path.exists(fn):
        return np.load(fn)
    else:
        return False


def create_data_file(data_folder, trial_type, cluster=None, trial_num=None, block=None, sub_block=None):
    fn = f"psth_{trial_type}"
    if cluster is not None:
        fn += f"_cl{cluster}"
    if trial_num is not None:
        fn += f"it{trial_num}"
    if block is not None:
        fn += f"blk{block}"
    if sub_block is not None:
        fn += f"sb{sub_block}"
    fn = os.path.join(data_folder, fn + ".npy")
    return fn


def save_psth_file(x, data_filename):
    # todo : Créer un authorised_type_name de type liste.
    np.save(data_filename, x)
    pass


def load_data_file(data_file):
    return np.load(data_file)