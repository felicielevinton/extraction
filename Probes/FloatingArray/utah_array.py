import numpy as np


def _old_omnetics_connector_32():
    c = np.arange(1, 33)[::-1]
    return np.vstack((c[:16], c[16:]))


def _new_omnetics_connector_32():
    c = np.arange(1, 33)
    br, ur, ul, bl = c[:8], c[8:16][::-1], c[16:24][::-1], c[24:]
    n = np.vstack((np.hstack((ul, ur)), np.hstack((bl, br))))
    return n


def _add_32_to_connector():
    array = np.ones((16, 2)) * 32
    

def _add_bank(bank):
    if bank == "A":
        return _new_omnetics_connector_32()
    elif bank == "B":
        pass