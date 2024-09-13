import numpy as np
# 384, 768, 961


class Neuropixels:
    def __init__(self):
        self.electrodes = np.arange(1, 961, dtype=np.int16)
        self.line_number = 480
        self.available_electrodes = 960
        self.enabled = None
        self.inactivated = np.copy(self.electrodes)

        # COLORS
        self.black = (0, 0, 0)

        # ODD
        self.far_left = np.arange(1, 961, 4)
        self.middle_left = np.arange(3, 961, 4)

        # EVEN
        self.middle_right = np.arange(2, 961, 4)
        self.far_right = np.arange(4, 961, 4)



