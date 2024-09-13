import numpy as np


class AbstractNeuropixels(object):
    pass


class Neuropixels1Point0(object):
    """

    """

    def __init__(self):
        self.probe_type = 0  # pour Npx2.0 linéaire 21, pour Npx2.0 quatre shanks: 24.
        self.sr_ap = 30e3
        self.sr_lfp = 2.5e3
        self.impedance = 150e3
        self.bank_0_limit = 384
        self.bank_1_limit = 768
        self.bank_2_limit = 960
        self.n_sites_total = 960
        self.bank_size = 384
        self.reference_channels = [0, 1, 192, 576, 960]  # 0 est réf externe, 1 est référence de pointe.
        self.available_gains = [50, 125, 250, 500, 1000, 1500, 2000, 3000]

    def get_bank_0_limit(self):
        return self.bank_0_limit

    def get_bank_1_limit(self):
        return self.bank_1_limit

    def get_bank_2_limit(self):
        return self.bank_2_limit

    def get_reference_channels(self):
        return self.reference_channels

    def get_available_gains(self):
        return self.available_gains

    def get_bank_size(self):
        return self.bank_size

    def get_probe_type(self):
        return self.probe_type

    def get_sampling_rate_lfp(self):
        return self.sr_lfp

    def get_sampling_rate_ap(self):
        return self.sr_ap

    def get_impedance(self):
        return self.impedance


class Neuropixels2point0(object):
    pass


class Neuropixels2point4(object):
    pass

