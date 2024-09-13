from load_rhd import *
from quick_extract import *
from get_data import *
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from utils_tonotopy import *
from load_rhd import *
from quick_extract import *
from get_data import *
import PostProcessing.tools.heatmap as hm
from get_data import *
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from tonotopy import *
import shutil


def create_folder(path):
    """
    Checks if a folder exists and if not, creates it
    """
    # Specify the folder path
    folder_path = path

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder doesn't exist, create it
        os.makedirs(folder_path)
        print("Folder created:", folder_path)
    else:
        print("Folder already exists:", folder_path)


def extract_from_rhd(path, sampling_rate):
    """
    Une seule fonction pour extraire depuis le fichier ephys.rhd jusqu'à ?
    input : path du folder où se trouve le fichier rhd
    """
    load_rhd(path+'ephys.rhd', path, digital=True, analog=True, accelerometer=True, filtered=True, export_to_dat=False)
    neural_data = np.load(path + 'filtered_neural_data.npy')
    
    # Divide into two folders for headstage 0 and headstage 1
    #créer les dossiers "headstage_0" et "headstage_1"
    path_0 = path + '/headstage_0'
    path_1 = path + '/headstage_1'
    create_folder(path_0)
    create_folder(path_1)
    
    #headstage 1
    neural_data_0 = neural_data[0:32]
    filter_and_cmr(neural_data_0, sampling_rate, path+'headstage_0/')
    #extract spikes etc
    quick_extract(path_0+'/refiltered_neural_data.npy')
    
    # headstage 2
    neural_data_1 = neural_data[32:64]
    filter_and_cmr(neural_data_1, sampling_rate, path+'headstage_1/')
    quick_extract(path_1+'/refiltered_neural_data.npy')
    
    print("All izz well")
    
def copy_files(path):
    """
    il faut copier les fichiers analog_in, dig_in et acc et tones dans les 
    folders headstage_0 et headstage_1
    """
    
    path_0 = path+'headstage_0/'
    path_1 = path+'headstage_1/'
      #copier le fichier de tones dans les dossiers headstage_0 et headstage_1
    for file_name in os.listdir(path+'tones/'):
        # Chemin complet du fichier source
        source_file = os.path.join(path+'tones/', file_name)
        # Copier le fichier dans le dossier de destination
        shutil.copy(source_file, path_0)
        shutil.copy(source_file, path_1)
        
    shutil.copy(path+'analog_in.npy', path_0)
    shutil.copy(path+'dig_in.npy', path_0)    
    
    shutil.copy(path+'analog_in.npy', path_1)
    shutil.copy(path+'dig_in.npy', path_1)
    print("All izz well")
    
def get_session_type(path):
    """
    Fonction qui renvoie le type de la session parmi TrackingOnly, PlaybackOnly etc
    elle va chercher dans le fichier json le type de session
    """
    # List all files in the folder
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
            
        # Extract the "Type" field
        try : 
            type_value = data['Block_000']['Type']

            if type_value=="Pause":
                type_value = data['Block_001']['Type']
                
            print("Type:", type_value)
        except :
            type_value = [data[key]["Type"] for key in data if key.startswith("Experiment_")][1]
            print("Type:", type_value)       
            
            
    else:
        print("Error: No JSON files found.")
    return type_value



def get_tonotopy_basics(path):
    """
    input : path = adresse du folder où sont stockés les fichiers analog_in.npy et etc
    output : 
    """
    spk = ut.Spikes(path)
    an_triggers = np.load(os.path.join(path, "analog_in.npy"))
    an_times = ut.extract_analog_triggers_compat(an_triggers[0])
    frequencies, tones_total, triggers_spe, tag = get_data(path, trigs=an_times)
    l_spikes = list()
    return spk, an_times, tones_total, frequencies

def psth(spikes, triggers, t_0=0.2, t_1=0.5, bin_size=0.01, bins=None, trigger_unit="seconds", fs=30e3):
    """

    """
    x = raster(spikes, triggers, t_0, t_1, trigger_unit, fs)
    if len(x) == 0:
        return None, None
    x = np.hstack(x)
    if bins is None:
        bins = np.arange(t_0, t_1 + bin_size, bin_size)
    h, b = np.histogram(x, bins)
    h = h.astype(dtype=np.float64)
    h /= (len(triggers) * bin_size)  # donne l'activité
    return h, b


def process_list(lst):
    # Trouver l'index du premier et dernier True
    first_true = next((i for i, x in enumerate(lst) if x), None)
    last_true = next((i for i, x in enumerate(lst[::-1]) if x), None)
    
    last_true = len(lst) - last_true - 1 if last_true is not None else None
    # Transformer le False entouré de True en un True
    for i in range(first_true + 1, last_true):
        if lst[i-1] and lst[i+1] and not lst[i]:
            lst[i] = True
    return lst

def get_spikes_cluster(cluster, spk, t_pre, t_post, bin_width, trigs, tone_sequence):
    """
    input : 
        cluster : numéro de cluster
        t_pre, t_post, bin_width, 
        trigs : an_times
        tone_sequence : tones_total
    output : 
        all_tones_psth : un tableau qui, pour chaque fréquence, contient les psth.  size : [freq x ( nombre de psth par freq)]

    
    """""
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)

    x = spk.get_spike_times(cluster=cluster)
    hist = list()
    #trigs=an_times
    tones, counts = np.unique(tone_sequence, return_counts=True)
    idx = process_list(list(np.greater(counts, 10)))
    tones = tones[idx]
    idx = np.arange(0, len(tones), dtype=int)
    t_0 = t_pre/bin_width
    all_tones_psth=[]
    for tone in tones:
        tone_idx = np.where(tone_sequence == tone)[0]
        trigger_time = trigs[tone_idx]
        h, _ = psth(x, trigger_time, t_0=t_pre, t_1=t_post, bins=psth_bins)
        hist = list(hist)
        hist.append(h)
        if len(hist) > 0:
            hist = np.vstack(hist)
        else:
            hist = np.zeros((len(tones), len(psth_bins)))
        psths = hist
        all_tones_psth.append(psths)
    return all_tones_psth