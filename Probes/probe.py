import numpy as np
import cairo
import matplotlib.pyplot as plt
import os


def generate_old_omnetics():
    """
    top view of old omnetics connector
    :return:
    """
    bottom_line = np.arange(1, 17, dtype=np.uint16)[::-1]
    top_line = np.arange(17, 33, dtype=np.uint16)[::-1]
    return np.vstack((top_line, bottom_line))


def generate_new_omnetics():
    bottom_line = np.array([[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9],
                            [32, 31, 30, 29, 28, 27, 26, 25, 1,  2,  3,  4,  5,  6,  7,  8]])

    return bottom_line


def translate_omnetics_old_to_new(channel, bank):
    """
    why use this function: converts old chan (from Blackrock's .xlsx file) to new nomenclature.
    bank is "A" "B" or "C"
    :param channel:
    :param bank:
    :return:
    """
    # new channel
    bank = bank.upper()
    try:
        bank in ["A", "B", "C"]
    except AssertionError:
        print("Input is not a correct bank.")
    if bank is "A":
        new_channel = 0
    if bank is "B":
        new_channel = 32
    if bank is "C":
        new_channel = 64

    new_omnetics = generate_new_omnetics()
    old_omnetics = generate_old_omnetics()

    chan_position = np.where(old_omnetics == channel)[0]
    return new_omnetics[chan_position]


def imro_file_reader(filename):
    # try:
    #     file_extension = os.path.splitext(filename)[0]
    #     if file_extension == "":
    #         raise Exception
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            pass
    pass


class Probe:
    """
    Le but de cette classe: classe abstraite qui permettra les manipulations basiques sur une électrode
    Correspondance numéro d'électrodes et channel mapping.
    """
    def __init__(self):
        self.shape = None
        self.broken_channels = list()

    def set_shape(self, shape):
        self.shape = shape

    def get_shape(self):
        return self.shape

    def set_broken_channels(self, broken_channels):
        self.broken_channels = broken_channels

    def get_broken_channels(self):
        return self.broken_channels


class UtahArray(Probe):
    """

    """
    pass


class Neuropixels(Probe):

    def __init__(self):
        # super
        pass
