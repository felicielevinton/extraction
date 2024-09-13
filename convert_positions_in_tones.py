import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import re
### dans le cas où les triggers n'ont pas été enregsitrés, on peut les retrouver ici. 


class Mapping(object):
    """
    
    """
    def __init__(self, width, n_freq, mid, octave):
        """
        Construction d'un objet de Mapping.
        :param width: Largeur de l'image en pixels. 
        :param n_freq: Nombre de fréquences.
        :param mid: Fréquence du milieu.
        :param octave: Nombre d'octaves.
        """
        self.bandwidth = width / (n_freq - 1)
        self.half_bandwidth = self.bandwidth // 2
        self.width = width
        self.mid = mid
        self.o = octave
        self.m_numFrequency = n_freq
        self._lut_indices = np.zeros(self.width, dtype=int)
        self.tones = np.zeros(n_freq)
        self._lut_tones = np.zeros(self.width)
        self._build_lut()

    def _build_lut(self):
        """
        Construit la "look-up table" des indices du mapping et également la LUT des fréquences.
        """
        
        def mapping(mid, n, o):
            _t = np.zeros(n)
            m_idx = n // 2
            _t[m_idx] = 0
            s = o / n
            _t[:m_idx] = np.arange((- n // 2) + 1, 0)
            _t[m_idx + 1:] = np.arange(1, n // 2 + 1)
            _t = np.round(mid * np.power(2, _t * s))
            return _t
        
        def func(position):
            if position < self.half_bandwidth:
                index = 0
            elif position > (self.width - self.half_bandwidth):
                index = self.m_numFrequency - 1
            else:
                index = position - self.half_bandwidth
                index //= self.bandwidth
                index += 1
            return int(index)
        
        def func_fill_tones(position, tones):
            return tones[func(position)]
        
        self.tones = mapping(self.mid, self.m_numFrequency, 7.0)
        for i in range(self.width):
            self._lut_indices[i] = func(i)
            self._lut_tones[i] = func_fill_tones(i, self.tones)
            

    def get_start_stop(self, motion):
        """
        Renvoie les indices de départ et d'arrivée pour un mouvement donnée.
        """
        start = self._lut_indices[motion[0]]
        stop = self._lut_indices[motion[1]]
        return start, stop
    
    def convert_position(self, x):
        if not np.isnan(x) and x != -1:
            return self._lut_tones[int(x)]
        else:
            return -1
    
    def convert_to_frequency(self, motion):
        """
        Renvoie les fréquences correspondantes aux positions dans un vecteur.
        :param motion: 
        :return: 
        """
        t = np.zeros(len(motion), dtype=float)
        for i, _p in enumerate(motion):
            if not np.isnan(_p):
                t[i] = self._lut_tones[int(_p)]
            else:
                t[i] = np.nan
        return t

def detect_frequency_switch(vec, mapping):
    """
    Fonction qui a pour objectif de détecter les changements de fréquences.
    :param mapping:
    :param vec: 
    :return: 
    """
    # print(np.unique(vec))
    tone_vec = mapping.convert_to_frequency(vec)
    d = np.diff(tone_vec)
    idx = np.where(d != 0)[0] + 1
    switch = tone_vec[idx]
    return switch






def get_positions_playback(directory, key_to_fetch):
    """"
    pour récupérer les dossiers de positions playback d'une session
    key_to_fetch : 'playback' si on veut les positions en pb ou 'tracking' etc
    """
    matching_files = []
    for root, dirs, files in os.walk(directory+'/positions'):
        for file in files:
            if key_to_fetch in file:
                matching_files.append(os.path.join(root, file))
    return matching_files

def load_bin_file_with_numpy(file_path):
    """
    Function to load the values in the .bin file
    Args:
        file_path (_type_): _description_

    Returns:
        
    """
    return np.fromfile(file_path,  dtype=np.int32)

def get_mock_frequencies(directory):
    """

    Args:
        directory (_type_): the global directory of the folder of the session

    Returns:
        _type_: mock tones for each playback block
    """
    positions_mock = []
    mapping = Mapping(1920, 33, 2000., 7) # pour Altai et Oscypek c'est 
    positions_files = get_positions_playback(directory, 'playback')
    for elt in positions_files:
        #enregistrer les tons dans le bon folder sous le bon nom
        index = elt.find("positions_")
        if index != -1:
            tone_save_name = elt[index + len("positions"):]
        save_name = directory+'/tones/'+tone_save_name
        print(save_name)
        # recupérer les positions
        xy_mock = (load_bin_file_with_numpy(elt)) # dans les fichiers positions il y a un array qui contient 
        # les x et les y (attention à bien charger en int32), les valeurs paires sont les x, les valeurs impaires sont les y
        # Donc ici je ne prends que les valeurs paires
        x_mock = xy_mock[::2]
        tones_mock = detect_frequency_switch(x_mock, mapping)
        np.save(save_name , tones_mock)
    return 'all izz well'


# essayer avec un fichier : 
#get_mock_frequencies('/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240426_SESSION_00')