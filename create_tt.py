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

#ANALOG_TRIGGERS_MAPPING = {"MAIN": 1, "PLAYBACK": 0, "MOCK": 3, "TARGET": 2}
#ANALOG_TRIGGERS_MAPPING = {"MAIN": 0, "PLAYBACK": 1, "MOCK": 2}
ANALOG_TRIGGERS_MAPPING = {"MAIN": 0, "PLAYBACK": 1}


def find_json_file(directory_path):
    """
    

    Args:
        directory_path (_type_): dossier dans lequel se trouve le json de la session

    Returns:
        _type_: renvoie l'adresse du json
    """
    # Use glob to find all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    # Check if any JSON files were found
    if json_files:
        return json_files[0]
    else:
        return None
    
def get_block_numbers(path):
    """"
    Extraire le nombre de blocs dans une session à partir du json
    arg : path du dossier dans lequel se trouve le json file
    output : int(numero du premier bloc),  int(numéro du dernier bloc)
    """
    files = os.listdir(path)

    # Filter JSON files
    json_files = [file for file in files if file.endswith('.json')]
    # Check if only one JSON file is found
    if len(json_files) == 1:
        json_file_path = os.path.join(path, json_files[0])
        print("Found JSON file:", json_file_path)
        # Load the JSON data from file
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    
        block_keys = [key for key in data if key.startswith("Block_00")]
        block_numbers = [int(block.split('_')[1]) for block in block_keys]
        first_block_number = min(block_numbers)
        max_block_number = max(block_numbers)
        return first_block_number, max_block_number

def check_block_ended_correctly(data):
    """
    Fonction pour savoir quels blocks se sont terminés correctement au cours de l'expérience
    arg : le json
    output: un tableau qui contient le réponse bloc par bloc
    """
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
    tones_fn = iterate_log_for_tones_fn(folder, log, allowed_kw, key_to_fetch)
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

def get_tones_for_tt(path, dowehavemock = False):
   
    # load json file
    json_file_path = find_json_file(path)

    with open(json_file_path, 'r') as f:
            json_file = json.load(f)
    # Vérifier les blocks qui se sont bien terminés
    blocks_ended_correctly = list(check_block_ended_correctly(json_file))

    # Trouver le numéro du premier et du dernier bloc de l'expérience
    first_block, last_block = get_block_numbers(json_file)

    # Calculer le bon nombre de blocs :
    n_blocks = last_block-first_block+1

    #n_blocks =5
    #extraire les canaux analogiques et digitaux avec le bon mapping des canaux digitaux
    #triggers = extract(path)

    #extraire positions et tons block par block
    positions_tr, tones_tr, positions_pb, tones_pb, tones_mck = [], [], [], [], []
    
    # si session playback/tracking classique : 
    
    for block in range(0, n_blocks):
        print(block)
        try:
            if blocks_ended_correctly[block]:
                print('block ended correctly')
                # extraire les positions et les tons
                tones_tr.append(get_tones(path, json_file, ['tracking', 'playback', 'mock'], f"Block_00{block}")['tracking'][0])
                #positions_tr.append(get_positions(path, json_file, ['tracking', 'playback'], f"Block_00{block}")['tracking'][0])
                #tones = extract_tones_path(json_file, block, condition)


            #en playback : convertir les positions en tones pour avoir les MOCK
            # mock = get_tones_from_position (nom donné au pif)
                tones_pb.append(get_tones(path, json_file, ['tracking', 'playback', 'mock'], f"Block_00{block}")['playback'][0])
                positions_pb.append(get_positions(path, json_file, ['tracking', 'playback', 'pbOnly'], f"Block_00{block}")['playback'][0])
                if dowehavemock:
                    tones_mck.append(get_tones(path, json_file, ['tracking', 'playback', 'mock'], f"Block_00{block}")['mock'][0])
                else : 
                    pass
            else:
                break
        except:
            pass
    return tones_tr, tones_pb, tones_mck



def get_triggers_for_tt(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False):
    """
    extraire les triggers pour créer le fichier tt.npz

    Args:
        path (_type_): le dossier dans lequel se trouvent les triggers
        ANALOG_TRIGGERS_MAPPING (_type_): la mapping des triggers analogiques dans ce format #ANALOG_TRIGGERS_MAPPING = {"MAIN": 0, "PLAYBACK": 1, "MOCK": 2}
    Output:
        les triggers
    """
    triggers = extract(path, ANALOG_TRIGGERS_MAPPING)
    analog_channels = triggers['ANALOG']
    digital_channels = triggers['DIGITAL']

    an_tr_trigs = analog_channels['MAIN']
    an_pb_trigs = analog_channels['PLAYBACK']
    if dowehavemock:
        an_mck_trigs = analog_channels['MOCK']
    else : 
        an_mck_trigs = None
    return an_tr_trigs, an_pb_trigs, an_mck_trigs, digital_channels


def associate_tones_and_triggers(tones, triggers, condition):
    """""
    Fonction pour associer les tones an triggers
    est appelée dans create_tt
    
    """
    # Create an empty dictionary to store associations
    tt = {}
    if condition == 'tracking':
        cond = 'tr'
    elif condition == 'playback':
        cond = 'pb'
    elif condition == 'mock':
        cond = 'mck'
    # Iterate over each index and subarray of tones
    for i, block_tones in enumerate(tones): 
        # Get the length of the current subarray
        subarray_length = len(block_tones)
        
        # Take the corresponding triggers for the current subarray
        subarray_triggers = triggers[:subarray_length]
        
        # Associate triggers with tones in a dictionary
        tt[f"{cond}_{i}"] = [block_tones, subarray_triggers]
        
        # Remove the triggers used for this subarray
        triggers = triggers[subarray_length:]
    return tt

def save_recording_length(folder, length):
    with open(os.path.join(folder, "recording_length.bin"), "w") as f:
        f.write('{:03d}\n'.format(length))
    return None
        
def extract_recording_length(triggers):
    return len(triggers['BASLER'])

def create_tt(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False):
    """"
    Fonction ok dans le cas où on a pas de mock triggers 
    
    """
    
    tones_tr, tones_pb, tones_mck = get_tones_for_tt(path, dowehavemock)
    n_iter = len(tones_pb)
    
    an_tr_trigs, an_pb_trigs, an_mck_trigs, digital_channels = get_triggers_for_tt(path, ANALOG_TRIGGERS_MAPPING, dowehavemock)
    
    recording_length = extract_recording_length(digital_channels)
    
    # Create the arrays for 'pb_' and 'tr_' using the provided function
    tt_tr = associate_tones_and_triggers(tones_tr, an_tr_trigs, 'tracking')
    tt_pb = associate_tones_and_triggers(tones_pb, an_pb_trigs, 'playback')
    tt_mck = associate_tones_and_triggers(tones_mck, an_mck_trigs, 'mock')
    
    # Combine 'tt_tr' and 'tt_pb' dictionaries
    tt_tr.update(tt_pb)
    tt_tr.update(tt_mck)
    
    # Create the final object
    tt_object = {
        'n_iter': n_iter,
        'recording_length': recording_length,
        **tt_tr  # Add the combined triggers and tones dictionaries
    }
    np.savez(path+'/tt.npz', tt_object)
    return tt_object




""""      si session tracking pbonly"""


def get_tones_for_tt_pbOnly(path, dowehavemock = False):
   
    # load json file
    json_file_path = find_json_file(path)

    with open(json_file_path, 'r') as f:
            json_file = json.load(f)
    # Vérifier les blocks qui se sont bien terminés
    blocks_ended_correctly = list(check_block_ended_correctly(json_file))
    # Prendre tous les fichiers .bin dans le dossier tones
    try:
        files = os.listdir(path+'/tones')
    except Exception as e:
        print("Erreur lors de la lecture du dossier:", e)
        return

    # Filtrer les fichiers .bin
    bin_files = [file for file in files if file.endswith('.bin')]

    if not bin_files:
        print("Aucun fichier .bin trouvé dans le dossier.")
        return
    
    for bin_file in bin_files:
        file_path = os.path.join(path+'/tones', bin_file)
        try:
            # Charger le fichier .bin en tant que tableau NumPy
            tones_pb = np.fromfile(file_path, dtype=np.double)
        except:
            print('Problemo')
    return tones_pb


def create_tt_playback_only(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False):
    """"
    Fonction ok dans le cas où on a pas de mock triggers 
    
    """
    
    tones_pb = get_tones_for_tt_pbOnly(path, dowehavemock)
    n_iter = len(tones_pb)
    
    an_tr_trigs, an_pb_trigs, an_mck_trigs, digital_channels = get_triggers_for_tt(path, ANALOG_TRIGGERS_MAPPING, dowehavemock)
    recording_length = extract_recording_length(digital_channels)
    if len(tones_pb) == len(an_pb_trigs):
        tt_pb = [tones_pb, an_pb_trigs]
    else:
        min_length = min(len(tones_pb), len(an_pb_trigs))
        tt_pb = [tones_pb[:min_length], an_pb_trigs[:min_length]]
    # Create the final object
    tt_object = {
        'n_iter': n_iter,
        'recording_length': recording_length,
        'pb_0' : tt_pb  # Add the combined triggers and tones dictionaries
    }
    np.savez(path+'/tt.npz', tt_object)
    return tt_object

def create_tt_tracking_only(path, ANALOG_TRIGGERS_MAPPING, dowehavemock=False):
    """"
    Fonction ok dans le cas où on a pas de mock triggers 
    
    """
    
    tones_tr = get_tones_for_tt_pbOnly(path, dowehavemock)
    n_iter = len(tones_tr)
    
    an_pb_trigs, an_tr_trigs, an_mck_trigs, digital_channels = get_triggers_for_tt(path, ANALOG_TRIGGERS_MAPPING, dowehavemock)
    recording_length = extract_recording_length(digital_channels)
    if len(tones_tr) == len(an_tr_trigs):
        tt_pb = [tones_tr, an_tr_trigs]
    else:
        min_length = min(len(tones_tr), len(an_tr_trigs))
        tt_tr = [tones_tr[:min_length], an_tr_trigs[:min_length]]
    # Create the final object
    tt_object = {
        'n_iter': n_iter,
        'recording_length': recording_length,
        'tr_0' : tt_tr  # Add the combined triggers and tones dictionaries
    }
    np.savez(path+'/tt.npz', tt_object)
    return tt_object