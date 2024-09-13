import numpy as np
import re
import os
from abc import ABC, abstractmethod


def get_pattern_from_type(type_of):
    if type_of == "playback":
        return "pb_"
    elif type_of == "tracking":
        return "tr_"
    elif type_of == "mock":
        return "mk_"
    elif type_of == "warmup":
        return "wp_"
    elif type_of == "warmdown":
        return "wd_"
    else:
        return None


def get_type_from_pattern(pattern):
    if re.match("pb_[0-9]", pattern):
        return "playback"
    elif re.search("tr_[0-9]", pattern):
        return "tracking"
    elif re.search("mk_[0-9]", pattern):
        return "mock"
    elif re.search("wp_[0-9]", pattern):
        return "warmup"
    elif re.search("wd_[0-9]", pattern):
        return "warmdown"
    else:
        return None


def append_zero(i, length):
    assert (length % 10 == 0), "Length must be a power of ten."
    if i > length:
        n = str(i)
    else:
        block_size = int(np.log10(length))
        targets = np.array([10 ** x for x in range(1, block_size)], dtype=int)
        idx = np.less(targets, i)
        n = (block_size - np.sum(idx)) * "0" + str(i)
    return n


class AbstractSequenceTT(ABC):
    """

    """
    def __init__(self, folder, n_iter):
        self.container = dict()
        self.keys = list()
        self.order = np.empty(0, dtype=int)
        self.numbers = np.empty(0, dtype=int)
        self.total_iter = 0
        if n_iter is not None:
            self.total_iter = n_iter
        self.recording_length = 0

        if folder is not None:
            self._load(folder)

    def _load(self, folder):
        d = np.load(os.path.join(folder, "tt.npz"))
        self.recording_length = d["recording_length"][0]
        self.order = d["order"]
        self.total_iter = d["n_iter"][0]
        self.keys = [key.decode() for key in d["keys"]]
        self.numbers = d["numbers"]
        for i, key in enumerate(self.keys):
            tones, triggers = d[key][0], d[key][1]
            self.container[key] = Pair(tones, triggers, type_of=get_type_from_pattern(key), order=self.order[i])

    def get_triggers_all(self, ordered=False):
        pass

    def get_recording_length(self):
        return self.recording_length

    def get_n_iter(self):
        return self.total_iter

    def add(self, pairs):
        pattern = pairs.get_pattern()
        order = pairs.order
        number = pairs.number
        assert (pattern not in self.keys), "Already in DataStructure."
        assert (order not in self.order), "Already in DataStructure."
        self.numbers = np.hstack((self.numbers, number))
        self.order = np.hstack((self.order, order))
        self.keys.append(pattern)
        self.container[pattern] = pairs


class SequenceTT(object):
    """

    """
    def __init__(self, folder=None, n_iter=None):
        self.container = dict()
        self.keys = list()
        self.order = np.empty(0, dtype=int)
        self.numbers = np.empty(0, dtype=int)
        self.total_iter = 0
        self.allowed = ["playback", "tracking", "warmup", "warmdown", "mock", "PLAYBACK", "MAIN", "TRACKING", "MOCK"]
        if n_iter is not None:
            self.total_iter = n_iter
        self.recording_length = 0

        if folder is not None:
            self._load(folder)

    def _load(self, folder):
        d = np.load(os.path.join(folder, "tt.npz"))
        self.recording_length = d["recording_length"][0]
        self.order = d["order"]
        self.total_iter = d["n_iter"][0]
        self.keys = [key.decode() for key in d["keys"]]
        self.numbers = d["numbers"]
        for i, key in enumerate(self.keys):
            tones, triggers = d[key][0], d[key][1]
            self.container[key] = Pair(tones, triggers, type_of=get_type_from_pattern(key), order=self.order[i])

    def get_number_iteration(self):
        return self.total_iter

    def save(self, folder, fn=None):
        """

        """
        if fn is None:
            fn = "tt.npz"
        fn = os.path.join(folder, fn)
        kwargs = dict()
        kwargs["order"] = np.array(self.order)
        kwargs["n_iter"] = np.array([self.total_iter])
        kwargs["recording_length"] = np.array([self.recording_length])
        kwargs["keys"] = self._build_chararray()
        kwargs["numbers"] = self.numbers
        for key in self.container.keys():
            kwargs[key] = self.container[key].get_stacked()
        # kwargs = {key: self.container[key].get_stacked() for key in self.container.keys()}
        np.savez(fn, **kwargs)

    def _build_chararray(self):
        n = np.array(self.keys).shape
        ch = np.chararray(n, itemsize=5)
        for i, elt in enumerate(self.keys):
            ch[i] = elt
        return ch

    def get_container(self):
        return self.container

    def get_xp_type_all(self, type_of, as_tt=True):
        """
        On va chercher toutes les expériences d'un certain type.
        """
        assert (type_of in self.allowed), "Wrong type..."
        pattern = get_pattern_from_type(type_of)
        if as_tt:
            out = SequenceTT()
            for i, k in enumerate(self.keys):
                if re.search(pattern, k):
                    tmp = self.container[k]
                    p = Pair(tmp.tones, tmp.triggers, type_of, number=self.numbers[i], order=self.order[i])
                    out.add(p)
            out.set_n_iter(self.total_iter)
        else:
            out = dict()
            for k in self.container.keys():
                if re.search(pattern, k):
                    out[k] = self.container[k]
        return out

    def merge(self, type_of):
        """
        On lui donne un type d'expériences. Renvoie une paire. Mets touts les triggers et les tones dedans.
        """
        d_out = self.get_xp_type_all(type_of, as_tt=False)
        l_number = list()
        # mettre dans l'ordre
        for k in d_out.keys():
            l_number.append(k)
        l_number.sort()
        tones = list()
        triggers = list()
        for k in l_number:
            p = d_out[k]
            type_of = p.get_type()
            tones.append(p.get_tones())
            triggers.append(p.get_triggers())
        triggers = np.hstack(triggers)
        tones = np.hstack(tones)
        return Pair(tones, triggers, type_of)

    def get_xp_number(self, type_of, n):
        """
        On demande une expérience d'un type donné, à un moment donné.
        """
        assert (type_of in self.allowed), "Wrong type..."
        if type_of not in ["warmup", "warmdown"]:
            assert (n < self.total_iter), "Unavailable."
        pattern = get_pattern_from_type(type_of) + str(n)
        assert (pattern in self.keys), "Not existing"
        return self.container[pattern]

    def get_all_number(self, n):
        """
        On va chercher le triplet Playback, Tracking, Mock.
        """
        assert (n < self.total_iter), "Unavailable."
        pattern = str(n)
        d_out = dict()
        for k in self.container.keys():
            if re.search(pattern, k):
                d_out[k] = self.container[k]

        return d_out

    def get_from_type_and_number(self, type_of, n):
        assert (type_of in self.allowed), "Wrong type..."
        assert (n < self.total_iter), "Unavailable."
        pattern = get_pattern_from_type(type_of) + str(n)
        for k in self.container.keys():
            if re.search(pattern, k):
                return self.container[k]

    def get_all_triggers(self, ordered=False):
        list_triggers = list()
        if ordered:
            keys = list()
        else:
            keys = self.keys
        for elt in keys:
            list_triggers.append(self.container[elt].get_triggers())
        return np.hstack(list_triggers)

    def get_all_triggers_for_type(self, type_of):
        p = self.merge(type_of)  # sort un objet Pair.
        return p.get_triggers()

    def add(self, pairs):
        pattern = pairs.get_pattern()
        order = pairs.order
        number = pairs.number
        assert (pattern not in self.keys), "Already in DataStructure."
        assert (order not in self.order), "Already in DataStructure."
        self.numbers = np.hstack((self.numbers, number))
        self.order = np.hstack((self.order, order))
        self.keys.append(pattern)
        self.container[pattern] = pairs

    def set_recording_length(self, length):
        self.recording_length = length

    def get_recording_length(self):
        return self.recording_length

    def set_n_iter(self, n_iter):
        self.total_iter = n_iter

    def get_n_iter(self):
        return self.total_iter

    def get_borders(self):
        d = dict()
        d_tr = dict()
        d_pb = dict()
        begin, end = self.get_xp_number("warmup", 0).get_begin_and_end_triggers()
        d["warmup"] = [begin, end]
        begin, end = self.get_xp_number("warmdown", 0).get_begin_and_end_triggers()
        d["warmdown"] = [begin, end]
        for i in range(self.total_iter):
            tr = self.get_xp_number("tracking", i)
            pb = self.get_xp_number("playback", i)

            begin, end = tr.get_begin_and_end_triggers()

            if i == 0:
                begin = pb.triggers[0] - 5 * 30000 * 60
            d_tr[i] = [begin, end]

            if i < self.total_iter - 1:
                tr = self.get_xp_number("tracking", i + 1)
            else:
                tr = d["warmdown"][0]
            begin, end = pb.triggers[0], tr.triggers[0]
            d_pb[i] = [begin, end]

        d["tracking"] = d_tr
        d["playback"] = d_pb
        # d[""]
        return d


class TT(object):
    def __init__(self, tones, triggers):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        self.tones = tones
        self.triggers = triggers


class Pair(object):
    def __init__(self, tones, triggers, type_of, number=None, order=None):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        self.tones = tones

        self.triggers = triggers
        self.tt = TT(tones, triggers)

        assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "TRACKING", "MAIN", "PLAYBACK", "MOCK"]), "Wrong type..."
        self.type = type_of

        if order is not None:
            self.order = order
        else:
            self.order = None

        if number is not None:
            self.number = number
            self.pattern = get_pattern_from_type(self.type) + str(self.number)
        else:
            self.number = None
            self.pattern = None

    def get_stacked(self):
        return np.vstack((self.tones, self.triggers))

    def get_tones(self):
        return self.tones

    def get_triggers(self):
        return self.triggers

    def get_pairs(self):
        return self.tt

    def get_pattern(self):
        return self.pattern

    def get_type(self):
        return self.type

    def get_begin_and_end_triggers(self):
        return self.triggers[0], self.triggers[-1]


class XPSingleton(object):
    """
    Bout de session avant processing
    """
    def __init__(self, type_of, order, number, duration, tones, fs=30e3):
        self.t = duration * 60 * fs
        self.order = order
        self.n = number
        assert(type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "TRACKING", "MAIN", "PLAYBACK", "MOCK"]), "Wrong type..."
        self.type = type_of
        if self.type == "warmup":
            self.tag = -1
        elif self.type == "warmdown":
            self.tag = -2
        elif self.type == "tracking":
            self.tag = 0 + self.n
        elif self.type == "mock":
            self.tag = 10 + self.n
        x = append_zero(self.n, 10)
        self.pattern = get_pattern_from_type(self.type) + str(x)
        self.tones = tones


class Sequence(object):
    """
    Regroupe les xp d'une session.
    """
    def __init__(self):
        self.container = dict()
        self.order = list()
        self.patterns = list()
        self.duration = dict()

    def get_n_tones_for(self, type_of):
        l_out = self.get_all_xp_for_type(type_of)
        s = 0
        for elt in l_out:
            s += len(elt.tones)
        return s

    def get_all_xp_for_type(self, type_of):
        assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "TRACKING", "MAIN", "PLAYBACK", "MOCK"]), "Wrong type..."
        pattern = get_pattern_from_type(type_of)
        l_order = list()
        l_out = list()
        for k in self.container.keys():
            if re.search(pattern, k):
                l_order.append(k)
        l_order.sort()
        for k in l_order:
            l_out.append(self.container[k])
        return l_out

    def get_all_number(self, n):
        """
        On va chercher le triplet Playback, Tracking, Mock.
        """
        # assert (n < self.total_iter), "Unavailable."
        pattern = append_zero(n, 10)
        l_type = ["tracking", "playback", "mock"]
        d_out = {key: get_pattern_from_type(key) + pattern for key in l_type}
        for k in self.container.keys():
            for key in d_out.keys():
                if d_out[key] == k:
                    d_out[key] = self.container[k]

        return d_out

    def get_xp_number(self, type_of, n):
        out = self.get_all_xp_for_type(type_of)
        # todo: assert le numéro est OK.
        return out[n]

    def add(self, xp):
        pattern = xp.pattern
        order = xp.order
        assert (pattern not in list(self.container.keys())), "Already in Sequence."
        assert (order not in self.order), "Already in Sequence."
        self.patterns.append(pattern)
        self.order.append(order)
        type_of = xp.type
        if type_of not in list(self.duration.keys()):
            self.duration[type_of] = xp.t
        self.container[pattern] = xp

    def get_duration_for(self, type_of):
        print(type_of)
        #assert (type_of in ["playback", "tracking", "warmup", "warmdown", "mock", "tail"]), "Wrong type..."
        print(self.duration[type_of])
        return self.duration[type_of]

    def get_for_types(self, types):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        assert(type(types) == list or type(types) == str), "Wrong type for types. str or list are required."
        keep = list()
        if type(types) == str:
            pattern_to_search = [get_pattern_from_type(types)]
        else:
            pattern_to_search = [get_pattern_from_type(type_of) for type_of in types]
        for elt in patterns:
            for p in pattern_to_search:
                if re.search(p, elt):
                    keep.append(self.container[elt])
        return keep

    def get_in_order(self, pb=False):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = get_pattern_from_type("playback")
        for elt in patterns:
            if pb:
                if re.search(pattern, elt):
                    keep.append(elt)
            else:
                if not re.search(pattern, elt):
                    keep.append(elt)

        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep

    def get_all_tones_for(self, type_of):
        keep = self.get_in_order_for_type(type_of=type_of)
        out = np.hstack([xp.tones for xp in keep])
        return out

    def get_in_order_for_type(self, type_of):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = get_pattern_from_type(type_of)
        for elt in patterns:
            if re.search(pattern, elt):
                keep.append(elt)
        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep

    def get_tracking(self):
        order = np.array(self.order)
        arg_order = np.argsort(order)
        patterns = [self.patterns[elt] for elt in arg_order]
        keep = list()
        pattern = [get_pattern_from_type("playback"), get_pattern_from_type("mock")]

        for elt in patterns:
            if not re.search(pattern[0], elt) and not re.search(pattern[1], elt):
                keep.append(elt)

        for i, elt in enumerate(keep):
            keep[i] = self.container[elt]
        return keep
