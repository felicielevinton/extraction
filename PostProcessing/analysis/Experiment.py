"""
Classes qui implémentent les différents types d'expériences
TODO: leur passer des triggers.
"""
from enum import Enum
import abc
from pathlib import Path
from .Mapping import *
from Analyse.ExtractRecordings.load_exp_files import *


class ProtocolType(Enum):
    UNDETERMINED = -1
    PAUSE = 0
    PURE_TONES = 1
    TONOTOPY = 2
    TRACKING = 3
    BEHAVIOUR = 4
    MAPPING_CHANGE = 5
    PLAYBACK = 6
    PERTURBATIONS = 7


class Experiment(abc.ABC):
    """
    Abstraite...
    Représente l'expérience qui a eu lieu. Type / Triggers / Sons...
    Sera créé par Session. lecture du HDF5 de récap des données pour une session.
    Contient les fichiers triggers etc..
    """
    def __init__(self, num_exp, exp_type, tag, dict_json, folder):
        self.protocol = exp_type
        self.num_exp = num_exp
        self.tag = tag
        self.folder = folder
        self.mapping = None
        self.bin_path = None  # ?
        self._extract_json(dict_json=dict_json)

    def get_num_exp(self):
        return self.num_exp

    def get_protocol_type(self):
        return self.protocol

    def get_mapping(self):
        return self.mapping

    def get_tag(self):
        return self.tag

    @abc.abstractmethod
    def _extract_json(self, dict_json):
        pass


class Pause(Experiment):
    def __init__(self, num_exp, tag, dict_json, folder):
        super(Pause, self).__init__(num_exp, ProtocolType.PAUSE, tag, dict_json, folder)

    def _extract_json(self, dict_json):
        pass


class PureTones(Experiment):
    """

    """
    def __init__(self, num_exp, tag, dict_json, folder):
        super(PureTones, self).__init__(num_exp, ProtocolType.PURE_TONES, tag, dict_json, folder)  # WTF?
        self.bin_path_tones = None  # est-il utile de garder la path?
        self.tone_sequence = None
        self._extract_json(dict_json)

    def _extract_json(self, dict_json):
        # todo: extraire mapping, tone sequence.
        self.mapping = create_mapping(dict_json)
        self.bin_path_tones = dict_json["Tones played"]
        self.tone_sequence = read_bin.read_tones_file(self.folder / self.bin_path_tones)


class Tracking(Experiment):
    """

    """
    def __init__(self, num_exp, tag, dict_json, folder):
        super(Tracking, self).__init__(num_exp, ProtocolType.TRACKING, tag, dict_json, folder)
        self.tones_played = None

    def _extract_json(self, dict_json):
        self.mapping = create_mapping(dict_json)
        self.tones_played = read_bin.read_tones_file(self.folder / dict_json["Tones played"])

    def _extract_data_from_json(self, dict_json):
        self.mapping = create_mapping(dict_json)


class MappingChange(Tracking):
    def __init__(self, num_exp, tag, dict_json, folder):
        super(MappingChange, self).__init__(num_exp, tag, dict_json, folder)
        self.protocol = ProtocolType.MAPPING_CHANGE
        self.changed_mappings = None
        self.num_changed_mappings = None
        self.changed_mappings = list()

    def _extract_data_from_json(self, dict_json):
        self.mapping = create_mapping(dict_json)
        self.num_changed_mappings = len(self.changed_mappings)

    def get_changed_mappings(self):
        """
        Retourne tous les mappings de perturbations
        """
        return self.changed_mappings

    def get_changed_mapping_num(self, n):
        if n < self.num_changed_mappings:
            return self.changed_mappings[n]
        else:
            return None


class Playback(Tracking):
    """

    """
    def __init__(self, num_exp, tag, dict_json, folder):
        super(Playback, self).__init__(num_exp, tag, dict_json, folder)
        self.protocol = ProtocolType.PLAYBACK
        self.playback_file = None
        self.bin_path_pb = None

    def _extract_data_from_json(self, dict_json):
        self.mapping = create_mapping(dict_json)
        self.playback_file = dict_json[""]
        self.bin_path_pb = dict_json[""]


class Behaviour(Tracking):
    """

    """
    def __init__(self, num_exp, dict_json, tag, folder):
        super(Behaviour, self).__init__(num_exp, dict_json, tag, folder)
        self.protocol = ProtocolType.BEHAVIOUR
        self.target_frequencies = None
        self.scores = None

    def _extract_data_from_json(self, dict_json):
        self.mapping = create_mapping(dict_json)
        self.scores = np.array(dict_json["Scores"])
        self.target_frequencies = np.array(dict_json["Targets"])

    def get_scores(self):
        return self.scores

    def get_targets(self):
        return self.target_frequencies


def extract_experiment(dict_json):
    """
    On passe le json entier, on récupère les balises "Experiment_"
    on retourne une liste de sous dictionnaires.
    """
    pattern = "Experiment_[0-9]"
    exp_list = list()
    keys = list(dict_json.keys())
    for key in keys:
        if re.match(pattern, key) is not None:
            exp_list.append(dict_json[key])
    return exp_list


def exp_factory(num_exp, dict_json, tag, folder):
    banner = dict_json["Type"]
    if banner == "Pause":
        exp = Pause(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Pure Tones":
        exp = PureTones(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Tracking":
        exp = Tracking(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Mapping Change":
        exp = MappingChange(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Playback":
        exp = Playback(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Behaviour":
        exp = Behaviour(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    elif banner == "Perturbations":
        exp = Tracking(num_exp, tag=tag, dict_json=dict_json, folder=folder)

    else:
        exp = Pause(num_exp, tag=tag, dict_json=dict_json, folder=folder)
    return exp


class Tag(object):
    """
    Objet qui contient 3 infos: le nom / la date / le numéro de session.
    Permet d'éviter l'analyse de sessions différentes ensemble.
    Quand ça devra être évité.
    """

    def __init__(self, name, date, session_number):
        self.name = name
        self.date = date
        self.session_number = session_number

    def get_name(self):
        return self.name

    def get_date(self):
        return self.date

    def get_session_number(self):
        return self.session_number

    def __repr__(self):
        return f"Name: {self.name}; Date: {self.date}; Session: {self.session_number}"

    def __eq__(self, other):
        return self.name == other.name and self.date == other.name and self.session_number == other.session_number

    def __lt__(self, other):
        if self.name != other.name:
            return "Not available"
        if self.date == other.date:
            return self.session_number < other.session_number
        else:
            return self.date < other.date

    def __gt__(self, other):
        if self.name != other.name:
            return "Not available"
        if self.date == other.date:
            return self.session_number > other.session_number
        else:
            return self.date > other.date


