from pathlib import Path
import json
from .Experiment import *
from .Cell import *
import re


def read_spike_sorting_file(path):
    with path.open(mode="r") as f:
        o = json.load(f)
    clusters = o["clusters"]
    spk_times = o["times"]
    return spk_times, clusters


def load_session(folder):
    f = Path(folder)
    spk_file = list(f.glob("ss*.json"))
    spk_times, clusters = None, None
    if len(spk_file) == 1:
        spk_file = spk_file[0]
        spk_times, clusters = read_spike_sorting_file(spk_file)
    session_file = list(f.glob("session*.json"))
    if len(session_file) == 1:
        session_file = session_file[0]
    with session_file.open(mode="r") as f:
        session_json = json.load(f)
    session = Session(session_json, Path(folder), spk_times, clusters)
    return session


class Session(object):
    """

    """
    def __init__(self, session_json, folder, spike_times_file=None, clusters_file=None):
        self.folder = Path(folder)
        self.cells = list()
        self.spike_times_file = None
        self.clusters_file = None
        if spike_times_file:
            self.spike_times_file = self.folder / spike_times_file
        if clusters_file:
            self.clusters_file = self.folder / clusters_file
        self.version = None
        self.versioned = False
        self.exp_num = 0
        self.exp_list = list()
        self.exp_type = list()
        self.name = ""
        self.session_number = 0
        self.date = ""
        self.ephys = False
        self.video = False
        self.video_path = ""
        self.position_file = ""
        self.__extract(session_json)
        if self.spike_times_file is not None and self.clusters_file is not None:
            self._load_spikes()

    def __repr__(self):
        pass

    def get_name(self):
        return self.name

    def _load_spikes(self):
        spike_clusters = np.load(self.clusters_file)
        spike_times = np.load(self.spike_times_file)
        u_spk, c_spk = np.unique(spike_clusters, return_counts=True)
        for i, cluster in enumerate(u_spk):
            _times = spike_times[np.where(spike_clusters == cluster)[0]]
            self.cells.append(Cell(cell_id=cluster, spike_times=_times, count=c_spk[i]))

    def get_session_number(self):
        return self.session_number

    def get_cells(self):
        return self.cells

    def get_cell(self, cluster):
        return self.cells[cluster]

    def get_date(self):
        return self.date

    def get_number_of_experiments(self):
        return self.exp_num

    def get_experiment_types(self):
        return self.exp_type

    def get_experiment(self, protocol):
        try:
            if protocol not in ProtocolType:
                raise RuntimeError
        except RuntimeError:
            print(f"{protocol} is not a legit protocol.")

        try:
            if protocol not in self.exp_type:
                raise RuntimeError
        except RuntimeError:
            print(f"{protocol} not an available protocol in session.")

    def __extract(self, _json):
        exp_list_keys = list()
        exp_list = list()
        exp_dict = dict()
        versioned = False
        pattern = "Experiment_[0-9]"
        for key in _json.keys():
            if key == "Name":
                self.name = _json[key]
            elif key == "Date":
                self.date = _json[key]
            elif key == "Version":
                self.version = _json[key]
                versioned = True
            elif key == "Session Number":
                self.session_number = _json[key]
            elif re.search(pattern, key):
                exp_list_keys.append(key)
            elif key == "EXPERIMENT":
                exp_dict = _json[key]
            elif key == "Ephys recording enabled":
                self.ephys = _json[key]
            elif key == "Ephys folder":
                self.ephys_path = _json[key]
            elif key == "Video recording enabled":
                self.video = _json[key]
            elif key == "Video recording path":
                self.video_path = _json[key]
            elif key == "Binary path":
                self.position_file = _json[key]
        if versioned:
            for key in exp_dict.keys():
                exp_list.append(exp_dict[key])
        else:
            for key in exp_list_keys:
                exp_list.append(_json[key])
        tag = Tag(self.name, self.date, self.session_number)
        for i, exp in enumerate(exp_list):
            self.exp_list.append(exp_factory(i, tag=tag, dict_json=exp, folder=self.folder))
        self.exp_num = len(self.exp_list)
        for elt in self.exp_list:
            protocol_type = elt.get_protocol_type()
            if protocol_type not in self.exp_type:
                self.exp_type.append(protocol_type)
