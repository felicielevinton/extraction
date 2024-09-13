"""
https://docs.python.org/fr/3/reference/datamodel.html
"""
import numpy as np


def mean_firing_rate(spike_times, count=None, fs=30e3):
    t_begin, t_end = spike_times[0], spike_times[-1]
    d = t_end - t_begin  # ?
    d /= fs
    if count:
        count = len(spike_times)
    mean_fr = count / d
    return mean_fr


def optimal_bin_size():
    # k = nombre de PA.
    return np.zeros(100)


def raster(triggers, spike_times):
    """
    Calcul un raster.
    :param triggers:
    :param spike_times:
    :return:
    """
    left_border, right_border = -1000, +5000
    extracted = list()
    for trigger in triggers:
        tmp_sp_times = spike_times - trigger  # la valeur du trigger est déjà soustraite.
        _idx = np.where((spike_times > left_border) & (spike_times < right_border))
        extracted.append(tmp_sp_times[_idx])
    return extracted


def psth(triggers, spike_times, poisson_optimizer=False):
    """
    Calcul du psth.
    :param triggers:
    :param spike_times:
    :param poisson_optimizer:
    :return:
    """
    # calculer le taux de décharge moyen
    # il faut régler la taille des bins combien de temps avant, combien de temps après
    mean_fr = mean_firing_rate(spike_times)
    left_border, right_border = -1000, +5000
    extracted = None
    for trigger in triggers:
        tmp_sp_times = spike_times - trigger  # la valeur du trigger est déjà soustraite.
        _idx = np.where((spike_times > left_border) & (spike_times < right_border))
        if extracted is None:
            extracted = tmp_sp_times[_idx]
        else:
            extracted = np.append(extracted, tmp_sp_times[_idx])
    # choisir méthode de calcul des bins.
    if poisson_optimizer:
        bins = optimal_bin_size()
    else:
        bins = np.histogram_bin_edges(extracted, range=(left_border, right_border))
    hist, _ = np.histogram(extracted, bins, range=(left_border, right_border))
    return hist
