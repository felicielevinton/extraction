"""
Classe qui permet de gérer les différents types de triggers qui existent lors de l'acquisition des données.
"""
import numpy as np
from enum import Enum


def open_trigger_nidq_file(nidq_file_path):
    pass


class TriggerType(Enum):
    TONOTOPY = 0
    TRACKING = 1
    PLAYBACK = 2


class Triggers(object):
    def __init__(self, trigger_type, trigger_times, values):
        self.trigger_type = trigger_type
        self.trigger_times = trigger_times
        self.values = values

    def get_type(self):
        return self.trigger_type

    def get_trigger_times(self):
        return self.trigger_times


class TonotopyTriggers(Triggers):
    # todo: passer les valeurs des fréquences jouées
    def __init__(self, trigger_times, values):
        super().__init__(TriggerType.TONOTOPY, trigger_times, values)


class ExperienceChangedTriggers(Triggers):
    def __init__(self, trigger_times, values):
        super().__init__("ExperienceChanged", trigger_times, values)


class TargetTriggers(Triggers):
    def __init__(self, trigger_times, values):
        super().__init__("Target", trigger_times, values)


class ThereminTriggers(Triggers):
    """
    Récupérer les triggers du thérémine classique
    """
    def __init__(self, trigger_times, tone_frequency, values):
        super().__init__("Theremine", trigger_times, values)
        self.tone_frequency = tone_frequency


class FrameTriggers(Triggers):
    """
    Récupérer les triggers des caméras.
    """
    def __init__(self, trigger_times, values):
        super().__init__("Frame", trigger_times, values)
