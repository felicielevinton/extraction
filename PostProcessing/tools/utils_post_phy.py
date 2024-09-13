"""
Post phy et post KS3!
"""
import numpy as np
import pandas as pd
import os


def load_cluster_groups(tsv_path):
    """
    Le fichier cluster_groups.tsv indique les clusters considérés comme bons, mua ou bruit.
    :param tsv_path:
    :return:
    """
    return pd.read_csv(tsv_path, sep="\t")


def _get_clusters(pd_dataframe, cluster_type):
    return pd_dataframe[pd_dataframe["group"] == cluster_type]["cluster_id"].values


def get_good_clusters(pd_dataframe):
    return _get_clusters(pd_dataframe, "good")


def get_mua_clusters(pd_dataframe):
    return _get_clusters(pd_dataframe, "mua")


def load_spike_times_file(_dir):
    """
    Charge le fichier "spike_times.npy". A une forme matlabienne.
    :param _dir:
    :return:
    """
    # Transposer, car de type Matlab, extraire indice 0
    return np.load(os.path.join(_dir, "spike_times.npy")).T[0]  # donner un type? uint64


def get_spike_times(_spike_time_array, _id):
    """
    Donne les spike times pour un cluster donné.
    :param _spike_time_array: array de spike times
    :param _id: numéro de cluster.
    :return:
    """
    # à mon avis c'est une erreur.
    return np.where(_spike_time_array == _id)[0]


def load_spike_clusters(_dir):
    """
    Charge le fichier spike_clusters.npy. A une forme matlabienne.
    :param _dir:
    :return:
    """
    # Transposer, car de type Matlab, extraire indice 0
    return np.load(os.path.join(_dir, "spike_clusters.npy")).T[0]  # donner un type? = uint32


# TODO: ajouter les waveforms.
