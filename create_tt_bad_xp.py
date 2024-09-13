from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
from tonotopy import *
from matplotlib.colors import ListedColormap, Normalize
from format_data import *
from skimage import measure
import matplotlib.colors as colors
from scipy.signal import find_peaks
from extract_data_total import *
import PostProcessing.tools.utils as ut
from PostProcessing.tools.extraction import *
from get_data import *
import re
import numpy as np
import os
import glob
import warnings
from copy import deepcopy
import json
import pickle

def get_triggers(path, analog_line):
    """"
    Récupérer les triggers en tracking
    
     - analog_line : numero de la ligne de triggers analogique. 
      (tracking0, playback1 et mock3 pour les xp de types Playback)
    """
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[analog_line])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times, tracking_only=True)
    return an_times, tones_total


triggers_tr, tones_total_tr = get_triggers(path, analog_line=0)
triggers_pb, tones_total_pb = get_triggers(path, analog_line=1)
triggers_mck, tones_total_mck = get_triggers(path, analog_line=3)
       
condition_tr = np.zeros(len(triggers_tr))
condition_pb = np.ones(len(triggers_pb))
        
trig_times = np.concatenate((triggers_tr, triggers_pb))
tones = np.concatenate((tones_total_tr, tones_total_pb))
condition = np.concatenate((condition_tr, condition_pb))
        

    
sorted_indices = np.argsort(trig_times[:len(tones)])
sorted_indices = sorted_indices[:-1]
sorted_triggers = trig_times[sorted_indices]
sorted_tones = tones[sorted_indices]
sorted_condition = condition[sorted_indices]

import re

def extract_number_from_filename(filename, type):
    """
    Extrait le numéro qui apparaît après le préfixe 'tracking_' dans une chaîne de caractères.
    
    Args:
    - filename (str): Le nom du fichier ou chaîne contenant le préfixe 'tracking_'.
    
    Returns:
    - int: Le numéro extrait, ou None si aucun numéro n'est trouvé.
    """
    # Utilisation d'une expression régulière pour rechercher un numéro après 'type'
    pattern = rf'{type}(\d+)'
    match = re.search(pattern, filename)
    
    if match:
        # Retourne le numéro extrait en tant qu'entier
        return int(match.group(1))
    else:
        # Aucun numéro trouvé
        return None



def get_tracking_tones(folder):
    all_files = glob.glob(os.path.join(folder+'/tones/', "*tracking_*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'tracking_'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'tracking_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs
    # Create an array of the same length as tones filled with the filename

def get_playback_tones(folder):
    # Find all files matching the pattern
    all_files = glob.glob(os.path.join(folder, 'tones', "*playback_*.bin"))

    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)

    # Sort files by the number extracted from their filename
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'playback_'))

    all_tones, all_blocs = [], []
    for file in all_files:
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        
        # Create an array filled with the file number
        blocs = np.full(len(tones), extract_number_from_filename(file, 'playback_'))
        all_blocs.append(blocs)

    return all_tones, all_blocs

def get_tail_tones(folder):
    all_files = glob.glob(os.path.join(folder+'/tones/', "*tail*.bin"))
    all_files.sort(key=lambda x: extract_number_from_filename(x, 'tail_'))
    # Print all matching files
    print("Files matching the pattern:")
    for file in all_files:
        print(file)
    all_tones, all_blocs = [], []
    for file in all_files:
        
        # Load the binary file into a NumPy array
        tones = np.fromfile(file, dtype=np.double)
        
        # Append the tones data to the list
        all_tones.append(tones)
        blocs = np.full(len(tones),extract_number_from_filename(file, 'tail_'))
        all_blocs.append(blocs)
    return all_tones, all_blocs
    # Create an array of the same length as tones filled with the filename
    
def read_json_file(json_file):
    """
    Reads a JSON file and extracts information from specific sections.
    
    Args:
    json_file (str): The path to the JSON file.
    
    Returns:
    list: A list of dictionaries containing the extracted information.
    """
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Liste pour stocker les informations
    extracted_data = []

    # Parcourir les blocs pour extraire les informations
    for block_name, block_data in json_data.items():
        if block_name.startswith("Block_"):
            # Initialiser les variables
            mock_fn = None
            playback_tones_fn = None
            tracking_tones_fn = None
            mapping_change_fn = None

            # Vérifier s'il y a une section 'playback'
            if 'playback'in block_data:
                playback_data = block_data['playback']
                mock_fn = playback_data.get('Mock_fn')
                playback_tones_fn = playback_data.get('Tones_fn')
            
            # Vérifier s'il y a une section 'tracking'
            if 'tracking' in block_data:
                try : 
                    tracking_data = block_data['Tracking']
                except : 
                    tracking_data = block_data['tracking']
                tracking_tones_fn = tracking_data.get('Tones_fn')
        
                
            # Vérifier s'il y a une section 'mapping change'
            if 'MappingChange' in block_data:
                mc_data = block_data['MappingChange']
                mapping_change_fn = mc_data.get('Tones_fn')

            # Si ni 'playback' ni 'tracking' ne sont présents, prendre 'Type' et 'Tones_fn'
            if not playback_tones_fn and not tracking_tones_fn and not mapping_change_fn:
                block_type = block_data.get('Type')
                tones_fn = block_data.get('Tones_fn')
                print(block_type)
                
                if block_type =='TrackingOnly':
                    tones_fn = block_data.get('Tones_fn')
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Tracking Tones_fn": tones_fn
                })
                    
                elif block_type =='PlaybackOnly':
                    extracted_data.append({
                        "Block": block_name,
                        "Type": block_type,
                        "Playback Tones_fn": tones_fn
                })
                    
                else : 
                    extracted_data.append({
                    "Block": block_name,
                    "Type": block_type,
                    "Tail Tones_fn": tones_fn
                })
                
                #extracted_data.append({
                   # "Block": block_name,
                   # "Type": block_type,
                   # "Tail Tones_fn": tones_fn
                #})
            else:
                extracted_data.append({
                    "Block": block_name,
                    "Playback Mock_fn": mock_fn,
                    "Playback Tones_fn": playback_tones_fn,
                    "Tracking Tones_fn": tracking_tones_fn,
                    "Mapping Change Tones_fn": mapping_change_fn
                })
    
    return extracted_data

def comparer(valeur1, valeur2):
    print(valeur1, valeur2)
    if valeur1 != valeur2:
        raise ValueError(f"Erreur : {valeur1} et {valeur2} ne sont pas égales.")

def concatenate_tones_and_labels(extracted_data, folder):
    all_tones = []
    all_labels = []
    all_mock = []

    # Parcourir les blocs dans l'ordre
    for block in extracted_data:
        block_tones = []
        block_labels = []
        block_mock = []

        block_name = block.get('Block')
        #block_type = block.get('Type', '')
        
        tail_tones_fn = block.get('Tail Tones_fn')
        if tail_tones_fn:
            tail_tones_path = folder + '/' + tail_tones_fn
            tail_tones = np.fromfile(tail_tones_path, dtype=np.double)
            block_tones.append(tail_tones)
            
            # Créer les labels correspondants
            block_type = 'Tail'
            tail_labels = [(block_name, block_type)] * len(tail_tones)
            block_labels.append(tail_labels)
            print('tail length = ', len(tail_tones))
            
        # Charger les tracking tones si disponibles
        tracking_tones_fn = block.get('Tracking Tones_fn')
        if tracking_tones_fn:
            tracking_tones_path = folder + '/' + tracking_tones_fn
            tracking_tones = np.fromfile(tracking_tones_path, dtype=np.double)
            block_tones.append(tracking_tones)
            
            # Créer les labels correspondants
            block_type = 'Tracking'
            tracking_labels = [(block_name, block_type)] * len(tracking_tones)
            block_labels.append(tracking_labels)

        # Charger les playback tones si disponibles
        playback_tones_fn = block.get('Playback Tones_fn')
        mock_tones_fn = block.get('Playback Mock_fn')
        if playback_tones_fn:
            playback_tones_path = folder + '/' + playback_tones_fn
            playback_tones = np.fromfile(playback_tones_path, dtype=np.double)
            block_tones.append(playback_tones)
            
            mock_tones_path = folder + '/' + mock_tones_fn
            mock_tones = np.fromfile(mock_tones_path, dtype=np.double)
            block_mock.append(mock_tones)
            
            block_type = 'Playback'
            # Créer les labels correspondants
            playback_labels = [(block_name, block_type)] * len(playback_tones)
            block_labels.append(playback_labels)
            
        # Mapping change
        mc_tones_fn = block.get('Mapping Change Tones_fn')
        if mc_tones_fn:
            mc_tones_path = folder + '/' + mc_tones_fn
            mc_tones = np.fromfile(mc_tones_path, dtype=np.double)
            block_tones.append(mc_tones)
            
            # Créer les labels correspondants
            block_type = 'Mapping Change'
            mc_labels = [(block_name, block_type)] * len(mc_tones)
            block_labels.append(mc_labels)
            

        # Concaténer les tons et labels de ce bloc
        if block_tones:
            concatenated_block_tones = np.concatenate(block_tones)
            concatenated_block_labels = np.concatenate(block_labels)
            all_tones.append(concatenated_block_tones)
            all_labels.append(concatenated_block_labels)
            all_mock.append(block_mock)
        try:
            comparer(len(playback_tones), len(tracking_tones))
        except:
            pass

    # Concaténer tous les blocs ensemble
    if all_tones:
        final_tones = np.concatenate(all_tones)
        final_labels = np.concatenate(all_labels)
        
        return final_tones, final_labels, all_mock
    else:
        return None, None

final_tones, final_labels, all_mock = concatenate_tones_and_labels(extracted_data, path+'headstage_0/tones')