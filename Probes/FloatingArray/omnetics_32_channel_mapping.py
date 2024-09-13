import numpy as np


def _old_omnetics_connector_32():
    c = np.arange(1, 33)[::-1]
    return np.vstack((c[:16], c[16:]))


def _new_omnetics_connector_32():
    c = np.arange(1, 33)
    br, ur, ul, bl = c[:8], c[8:16][::-1], c[16:24][::-1], c[24:]
    return np.vstack((np.hstack((ul, ur)), np.hstack((bl, br))))


def _add_to_connector(connector, to_add):
    return connector + to_add


def _add_32_to_connector(connector):
    return _add_to_connector(connector, np.ones((16, 2)) * 32)


def _add_64_to_connector(connector):
    return _add_to_connector(connector, np.ones((16, 2)) * 64)


def _add_bank(bank):
    if bank == "A":
        return _new_omnetics_connector_32()
    elif bank == "B":
        pass


if __name__ == "__main__":
    bank_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    electrodes = [17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7, 24, 8,
                  25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15, 32, 16]
    n_om = _new_omnetics_connector_32()
    # n_om = np.hstack((n_om[]))
