import numpy as np
import os
import glob
import re
from Analyse.GLM.glm import build_dm
import utils as ut


def load_positions_files(folder):
    triggers = np.load(os.path.join(folder, "trig_analog_chan0.npy"))
    if os.path.exists(os.path.join(folder, "recording_length.bin")):
        with open(os.path.join(folder, "recording_length.bin"), "r") as f:
            length = int(f.read())
    # on sélectionne les fichiers dans le glob selon leurs patterns.
    pause_pattern = "positions_Pause_0[0-9]"
    playback_pattern = "positions_Playback_playback_0[0-9]"
    tracking_pattern = "positions_Playback_tracking_0[0-9]"
    warmup_pattern = "positions_Playback_warmup_0[0-9]"

    glob_files = glob.glob(os.path.join(folder, "positions", "positions_*.bin"))
    types_pos_list = [list() for _ in range(4)]
    for file in glob_files:
        if re.search(pause_pattern, file):
            types_pos_list[0].append(file)
        elif re.search(warmup_pattern, file):
            types_pos_list[1].append(file)
        elif re.search(tracking_pattern, file):
            types_pos_list[2].append(file)
        elif re.search(playback_pattern, file):
            types_pos_list[3].append(file)
    out = [["", ""] for _ in glob_files]
    out[0] = [types_pos_list[0][0], "Pause_00"]  # Pause
    out[-1] = [types_pos_list[0][-1], "Pause_01"]
    out[1] = [types_pos_list[1][0], "Warmup_00"]  # Warmup
    out[-2] = [types_pos_list[1][1], "Warmup_01"]
    idx = 2
    for i in range(len(types_pos_list[2])):
        out[idx] = [types_pos_list[2][i], f"Tracking_0{i}"]  # Tracking
        out[idx + 1] = [types_pos_list[3][i], f"Playback_0{i}"]  # Playback
        idx += 2

    positions = list()
    l_triggers = list()
    counter_triggers = 0
    d = dict()
    for elt in out:  # on charge les positions depuis les fichiers enregistrés.
        p = np.fromfile(elt[0], dtype=np.int32)
        positions.append(p)
        counter_triggers += len(p)
        d[elt[1]] = [p, triggers[counter_triggers, len(p)]]
    stack = np.hstack(positions)
    stack = clean_positions(stack)  # nettoyage des positions.
    counter = 0
    for i in range(len(positions)):  # on reconstruit la liste de positions nettoyées
        lx = len(positions[i])
        positions[i] = stack[counter:counter + lx]
        counter += lx

    for elt in out:
        pass
    return d, stack, triggers, length


def clean_positions(positions):
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
                positions[begin:end] = filler
                k = 0
        else:
            if k == 0:
                begin = y[i - 1]
            k += 1
    remainder = np.where(positions == -1)[0]
    if len(remainder) != 0 and k != 0:
        pass

    return positions


class Positions(object):
    def __init__(self, folder, fps=30.):
        self.d, self.positions, self.triggers, self.recording_length = load_positions_files(folder)
        self.fps = fps
        self.fs = 30e3
        self.ratio = int(self.fs // self.fps)
        self.scaled_to_ephys = np.arange(1, self.recording_length + 1) * self.ratio

    def get_positions(self):
        return self.positions

    def get_binned(self, bin_duration):
        assert (bin_duration > 1 / self.fps), f"Bin duration must be greater than {1 / self.fps * 1000} ms"
        bin_size = int(bin_duration * self.fps)

        n_bins, remainder = self.recording_length // bin_size, self.recording_length % bin_size
        triggers_index_list = list()
        triggers = self.triggers[:-remainder]
        _bin_counter = 0
        for i in range(n_bins):
            triggers_index_list.append(np.logical_and(_bin_counter <= triggers, triggers < _bin_counter + bin_size))
            _bin_counter += bin_size

        positions = list()
        for elt in triggers_index_list:
            positions.append(self.positions[elt])
        return positions

    def get_between(self, t0, t1):
        idx = np.logical_and(self.triggers >= t0, self.triggers <= t1)
        triggers = self.triggers[idx]
        triggers -= triggers[0]
        return self.positions[idx], self.triggers[idx]

    def get_binnned_mean(self, bin_duration):
        binned = self.get_binned(bin_duration)
        mean = np.zeros(len(binned))
        for i, elt in enumerate(binned):
            mean[i] = elt.mean()
        return mean

    def get_binned_mean_between(self, t0, t1, bin_duration):
        bin_size = int(bin_duration * self.fps)
        delta = t1 - t0
        delta //= self.ratio
        n_bins, remainder = delta // bin_size, delta % bin_size
        between, triggers = self.get_between(t0, t1)
        binned = np.array_split(between[:-remainder], n_bins)
        mean = np.zeros(len(binned))
        for i, elt in enumerate(binned):
            mean[i] = elt.mean()
        return mean

    def get_binned_around(self, event, time_around):
        time_around = int(time_around * self.fs)
        left, right = event - time_around, event + time_around
        return self.get_between(left, right)

    def get_binned_mean_around(self, event, time_around, bin_duration):
        time_around = int(time_around * self.fs)
        left, right = event - time_around, event + time_around
        return self.get_binned_mean_between(left, right, bin_duration)

    def get_design_matrix(self, t0, t1, bin_duration, len_pad):
        p = self.get_binned_mean_between(t0, t1, bin_duration)
        return build_dm(p, len_pad, norm=True)


