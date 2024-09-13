"""
Fichiers de création des mappings
"""
import numpy as np
import re


def get_normal_mapping(exp_dict):
    """
    Créée un mapping normal
    # todo changer sauvegarde du mapping dans bb
    """
    return create_mapping(exp_dict)


def get_changed_mappings(exp_dict):
    """
    Créé une liste de mappings changés.
    """
    changed_mapping = list()
    pattern = "PerturbationMapping_[0-9]"
    l_keys = list(exp_dict.keys())
    for key in l_keys:
        if re.match(pattern, key) is not None:
            changed_mapping.append(create_mapping(exp_dict[key]))
    return changed_mapping


def create_mapping(exp_json):
    mid = exp_json["Mid tone"]
    n_freq = exp_json["Num frequencies"]
    n_oct = exp_json["Num octaves"]
    vec_mapping = np.array(exp_json["Mapping"], dtype=np.double)
    return Mapping(n_freq, mid, n_oct, vec_mapping)


class Mapping(object):
    # todo: créer une classe mapping perturbé. avec numéro dans la séquence
    #  et type: gain change ou shifted.
    def __init__(self, n_freq, middle_freq, n_oct, vec_mapping):
        self.n_freq = n_freq
        self.middle_freq = middle_freq
        self.n_oct = n_oct
        self.mapping = vec_mapping

    def get_n_freq(self):
        """
        Retourne le nombre de fréquences
        """
        return self.n_freq

    def get_middle_freq(self):
        """
        Retourne la fréquence du milieu
        """
        return self.middle_freq

    def get_n_oct(self):
        """
        Retourne le nombre d'octaves
        """
        return self.n_oct

    def get_mapping(self):
        """
        Retourne l'array qui contient les fréquences du mapping
        """
        return self.mapping


class ChangedMapping(Mapping):
    def __init__(self, num, n_freq, middle_freq, n_oct, vec_mapping):
        super(Mapping, self).__init__(n_freq, middle_freq, n_oct, vec_mapping)

    def foo(self):
        print(self.n_freq)

