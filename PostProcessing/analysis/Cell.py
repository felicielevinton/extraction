from enum import Enum
import numpy as np
import Analyse.PostProcessing.tools.utils as nt
import Analyse.GLM.glm as glm
from scipy import signal


class CellRegion(Enum):
    """
    Définit les différentes régions cérébrales
    """
    UNDETERMINED = 0
    AUDITORY = 1
    MOTOR = 2
    HIPPOCAMPUS = 3


class CellType(Enum):
    """
    Définit les différents types cellulaires auxquels je peux être confronté.
    """
    UNDETERMINED = 0
    PYRAMIDAL = 1
    INTERNEURON = 2


class Cell(object):
    """
    Représente les informations d'une unité détectée dans le tri des clous.
    Une cell se créée quand on charge le fichier de récap HDF5. en avril: DOUBT.
    Le fichier de récap sera lu par l'objet Session.
    = Temps de PA
    = numéro de l'unité
    = Position du fichier de waveforms
    """
    def __init__(self, cell_id, spike_times, count,
                 cell_waveforms=None,
                 region=CellRegion.UNDETERMINED,
                 cell_type=CellType.UNDETERMINED):
        # todo: calculer le meilleur canal grâce au ratio signal / bruit
        self.cell_id = cell_id  # numéro de cluster provenant de phy.
        self.spike_times = spike_times
        self.count = count
        self.fs = 30e3
        self.mean_fr = nt.mean_firing_rate(self.spike_times, self.count)
        self.cell_waveforms = cell_waveforms  # passer un chemin d'accès.
        self.region = region                  # Hippocampe, Moteur, Auditif
        self.cell_type = cell_type            # Pyramidale, interneurone etc.
        self.psth = list()

    def __gt__(self, other):
        if other.get_region() == self.region:
            pass
        pass

    def __lt__(self, other):
        pass

    def get_cell_id(self):
        """
        On récupère l'id de la cellule.
        :return:
        """
        return self.cell_id

    def get_spike_times(self):
        return self.spike_times

    def get_spike_times_between(self, t_0, t_1, zero=False):
        x = self.spike_times[np.logical_and(self.spike_times > t_0, self.spike_times < t_1)]
        if zero:
            x -= t_0
        return x

    def get_mean_firing_rate(self):
        return self.mean_fr

    def set_waveforms(self, path):
        self.cell_waveforms = path

    def get_waveforms(self):
        return self.cell_waveforms

    def get_region(self):
        return self.region

    def get_psth(self, triggers, t_pre=0.2, t_post=0.5, bin_size=0.002):
        """

        """
        return nt.psth(self.spike_times, triggers=triggers, t_0=t_pre, t_1=t_post, bin_size=bin_size)

    def get_raster(self, triggers, t_pre=0.2, t_post=0.5):
        """

        """
        return nt.raster(triggers, self.spike_times, t_0=t_pre, t_1=t_post)

    def get_design_matrix(self, t_0, t_1, bins, len_pad):
        """
        Retourne la matrice de design pour la cellule.
        """
        spk = self.get_spike_times_between(t_0=t_0, t_1=t_1, zero=True)
        spk_binned = glm.bin_spikes(spk, bins)
        dm = glm.build_spike_history_dm(spk_binned, len_pad)
        return dm

    def auto_correlation(self, num_lags, bin_size=0.02):
        """

        """
        x = self.spike_times - self.spike_times[0]
        x = x.astype(dtype=np.float64)
        x /= self.fs
        bins = np.arange(x[0], x[-1] + bin_size, bin_size)
        n_bins = bins.size
        binned_spikes = glm.bin_spikes(x, bins)
        xc = signal.correlate(binned_spikes, binned_spikes)
        xc = xc[n_bins - 1 - num_lags:num_lags + n_bins]
        xc[num_lags+1] = 0
        return xc

    def isi(self, bin_size=0.001):
        """

        """
        return nt.isi(self.spike_times, bin_size=bin_size)


class MotorCell(Cell):
    """
    Héritage de Cell, classe spéciale pour les cellules motrices.
    """
    def __init__(self, cell_id, spike_times, cell_waveforms, cell_type=CellType.UNDETERMINED):
        super().__init__(cell_id, spike_times, cell_waveforms, cell_type=cell_type, region=CellRegion.MOTOR)


class AuditoryCell(Cell):
    """
    Héritage de Cell, classe spéciale pour les cellules auditives.
    """
    def __init__(self, cell_id, spike_times, cell_waveforms, cell_type=CellType.UNDETERMINED):
        super().__init__(cell_id, spike_times, cell_waveforms, region=CellRegion.AUDITORY, cell_type=cell_type)


class HippocampusCell(Cell):
    """
    Héritage de Cell, classe spéciale pour les cellules hippocampales.
    """
    def __init__(self, cell_id, spike_times, cell_waveforms, cell_type=CellType.UNDETERMINED):
        super().__init__(cell_id, spike_times, cell_waveforms, region=CellRegion.HIPPOCAMPUS, cell_type=cell_type)
