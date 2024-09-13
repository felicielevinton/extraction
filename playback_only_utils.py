from sequences import *
from extraction_utils import extract
from get_data_v2 import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json
import matplotlib.pyplot as plt
from get_data import *


def find_json_file(directory_path):
    # Use glob to find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    # Check if any JSON files were found
    if json_files:
        return json_files[0]
    else:
        return None
    
def check_block_ended_correctly(data):
    experiment_results = {}
    for block_key, block_data in data.items():
        if block_key.startswith("Block_"):
            experiment_results[block_key] = block_data.get("Experiment ended correctly", None)
    return experiment_results

def iterate_log_for_tones_fn(folder, log, allowed_kw, key_to_fetch):
    """
    Itère dans le .json à la recherche des noms de fichiers sons.
    :param key_to_fetch:
    :param folder:
    :param log:
    :param allowed_kw:
    :return:
    """
    tones_folder = os.path.join(folder, log["Tones folder"])
    tones_fn = {kw: list() for kw in allowed_kw}
    print(tones_folder)
    sub_log = log[key_to_fetch]
    for kw in allowed_kw:
        for key in sub_log.keys():
            if re.match(kw, key):
                tones_fn[kw].append(os.path.join(tones_folder, sub_log[key]["Tones_fn"]))
                if kw == "playback":
                    tones_fn["mock"].append(os.path.join(tones_folder, sub_log[key]["Mock_fn"]))
    return tones_fn

def get_tones(folder, log, allowed_kw, key_to_fetch):
    """
    Charge les fichiers dans un np.ndarray.
    :param folder:
    :param log:
    :param allowed_kw:
    :param key_to_fetch:
    :return:
    """
    print(folder)
    tones_fn = iterate_log_for_tones_fn(folder, log, allowed_kw, key_to_fetch)
    print(tones_fn)
    tones_values = {kw: list() for kw in allowed_kw}
    # C'est à dire : fichier dans le log, mais absent du dossier car vide.
    # Là, je cherche à charger les .bin dans un dictionnaire de listes.
    for key in tones_fn.keys():
        tones_file_for_type = tones_fn[key]
        for file in tones_file_for_type:
            if not os.path.exists(file):
                tones = np.empty(0)
            else:
                tones = np.fromfile(file, dtype=np.double)
            tones_values[key].append(tones)
    return tones_values