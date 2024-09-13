import re
from enum import Enum


def get_pattern_from_type(type_of):
    """

    :param type_of:
    :return:
    """
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
    elif type_of == "PureTones":
        return "pt_"
    elif type_of == "silence":
        return "si_"
    else:
        return None


def get_type_from_pattern(pattern):
    """

    :param pattern:
    :return:
    """
    """
    Conversion en un type.
    :param pattern:
    :return:
    """
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
    elif re.search("si_[0-9]", pattern):
        return "silence"
    elif re.search("pt_[0-9]", pattern):
        return "PureTones"
    else:
        return None


class ExperimentType(Enum):
    """
    Les différentes expériences.
    """
    UNDEFINED = -1
    PAUSE = 0
    PURE_TONES = 1
    PLAYBACK = 2
    SILENCE = 3
    MAPPING_CHANGE = 4
    TRACKING = 5


def get_str_repr(xp_type):
    """

    :param xp_type:
    :return:
    """
    out = ""

    if xp_type == ExperimentType.PURE_TONES.value:
        out = "Tonotopy"

    elif xp_type == ExperimentType.PLAYBACK.value:
        out = "Playback"

    elif xp_type == ExperimentType.SILENCE.value:
        out = "Silence"

    elif xp_type == ExperimentType.TRACKING.value:
        out = "Tracking"

    elif xp_type == ExperimentType.MAPPING_CHANGE.value:
        out = "MappingChange"

    elif xp_type == ExperimentType.PAUSE.value:
        out = "Pause"

    return out


def get_from_str(xp_type):

    if xp_type == "Tonotopy":
        out = ExperimentType.PURE_TONES

    elif xp_type == "Playback":
        out = ExperimentType.PLAYBACK

    elif xp_type == "Silence":
        out = ExperimentType.SILENCE

    elif xp_type == "Tracking":
        out = ExperimentType.TRACKING

    elif xp_type == "MappingChange":
        out = ExperimentType.MAPPING_CHANGE

    elif xp_type == "Pause":
        out = ExperimentType.PAUSE

    else:
        out = ExperimentType.UNDEFINED

    return out


def get_allowed_keywords(xp_type):

    if xp_type == ExperimentType.PAUSE.value:
        allowed = ["pause"]

    elif xp_type == ExperimentType.PURE_TONES.value:
        allowed = ["PureTones"]

    elif xp_type == ExperimentType.PLAYBACK.value:
        allowed = ["playback", "tracking", "warmup", "warmdown", "mock"]

    elif xp_type == ExperimentType.SILENCE.value:
        allowed = ["tracking", "warmup", "warmdown", "silence"]

    elif xp_type == ExperimentType.TRACKING.value:
        allowed = ["tracking"]

    elif xp_type == ExperimentType.MAPPING_CHANGE.value:
        allowed = ["tracking", "warmup", "warmdown", "mapping_change"]

    else:
        allowed = list()

    return allowed
