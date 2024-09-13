import re
import json
import numpy as np
import os
import glob
from scipy import signal
import warnings
from copy import deepcopy

# todo : ajouter les lignes pour l'eau.
ANALOG_TRIGGERS_MAPPING = {"MAIN": 0, "PLAYBACK": 1, "MOCK": 2, "TARGET": 3}
DIGITAL_TRIGGERS_MAPPING = {"BASLER": 0, "MAIN": 8, "SPECIAL": 9, "MOCK": 10, "TARGET": 11,
                            "BARCODE": 12, "EXP": 13, "COND": 14}


def get_digital_mapping(json_file):
    """
    Correction d'un bug causé par l'absence de NIDAQmx.
    """
    if "Version" in json_file:
        if json_file["Version"] == "v2":
            return {"BASLER": 0}
        else:
            return DIGITAL_TRIGGERS_MAPPING

    else:
        return DIGITAL_TRIGGERS_MAPPING


def read_log_file(folder):
    """
    Cherche le type de l'expérience dans le dossier.
    :param folder:
    :return:
    """
    out = list(glob.glob(os.path.join(folder, "session_*.json")))
    # todo : pour le débug -> Retirer sinon.
    # out = ["C:/Users/Flavi/PycharmProjects/Experience/test.json"]

    try:
        assert (len(out) == 1), "Glob in folder should be of length 1."
        file_name = out[0]
        with open(file_name, "r") as f:
            d = json.load(f)

    except AssertionError as error:
        print("Error: ", error)

    return d



def get_n_of_experiment(log):
    """
    Itère sur le .json à la rechercher des bannières "Experiment_". Retourne le nombre d'expériences.
    :param log: Le .json
    :return: Un entier : le nombre d'expériences.
    """
    counter = 0
    beacons = list()
    for key in log.keys():
        if re.match("Block_", key):
            beacons.append(key)
            counter += 1
    return counter


def get_n_iter(log):
    n_iter = 0

    return n_iter


def associate_tones_and_triggers(tones, triggers):
   
    # Create an empty list to store associations
    associations = []

    # Iterate over each subarray of tones
    for block_tones in tones:
        # Get the length of the current subarray
        subarray_length = len(block_tones)
        
        # Take the corresponding triggers for the current subarray
        subarray_triggers = triggers[:subarray_length]
        
        # Associate triggers with tones in a dictionary
        subarray_associations = dict(zip(block_tones, subarray_triggers))
        
        # Add associations for the current subarray to the list
        associations.append(subarray_associations)
        
        # Remove the triggers used for this subarray
        triggers = triggers[subarray_length:]
    return associations




def associate_tones_and_triggers_pbOnly(tones, triggers):
    """
    Fonction dans le cas où on a un playbackonly type of session

    Args:
        tones (_type_): _description_
        triggers (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    # Create an empty list to store associations
    associations = []

    # Iterate over each subarray of tones
        # Get the length of the current subarray
    subarray_length = len(tones)
        
        # Take the corresponding triggers for the current subarray
    subarray_triggers = triggers[:subarray_length]
        
        # Associate triggers with tones in a dictionary
    subarray_associations = dict(zip(tones, subarray_triggers))
        
        # Add associations for the current subarray to the list
    associations.append(subarray_associations)
        
        # Remove the triggers used for this subarray
    triggers = triggers[subarray_length:]
    return associations

def extract_positions_path(json_file, block, condition):
    """"
    input :
    json_data : adress du json
    block : numéro du block d'interet
    condition : "tracking" ou "playback" (attention sans majuscule)

    output :
        path des positons du block dans la condition donnée
    """
    try:
        positions_fn = json_file.get(f"Block_00{block}", {}).get(condition, {}).get("Positions_fn")
        return positions_fn
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None

def extract_tones_path(json_file, block, condition):
    """"
    input :
    json_data : le json loadé avec open(json_file_path, 'r') as f puis json.load(f)
    block : numéro du block d'interet
    condition : "tracking" ou "playback" (attention sans majuscule)

    output :
        path des tons du block dans la condition donnée
    """
    try:
        positions_fn = json_file.get(f"Block_00{block}", {}).get(condition, {}).get("Tones_fn")
        return positions_fn
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        return None


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
    print(tones_folder)
    tones_fn = {kw: list() for kw in allowed_kw}
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


def iterate_log_for_positions_fn(folder, log, allowed_kw, key_to_fetch):
    """
    Itère dans le .json à la recherche des noms de fichiers positions.
    :param folder:
    :param log:
    :param allowed_kw:
    :param key_to_fetch:
    :return: Un dictionnaire clés : Types d'expérience. Valeurs : liste de fréquences.
    """
    positions_folder = os.path.join(folder, log["Binary path"])
    positions_fn = {kw: list() for kw in allowed_kw}
    sub_log = log[key_to_fetch]
    for kw in allowed_kw:
        for key in sub_log.keys():
            if re.match(kw, key):
                positions_fn[kw].append(os.path.join(positions_folder, sub_log[key]["Positions_fn"]))
    return positions_fn


def get_positions(folder, log, allowed_kw, key_to_fetch):
    """
    Charge les fichiers dans un np.ndarray.
    :param folder:
    :param log:
    :param allowed_kw:
    :param key_to_fetch:
    :return: Un dictionnaire clés : Types d'expérience. Valeurs : liste de positions.
    """
    if "mock" in allowed_kw:  # Retirer le KW "mock" de cette liste.
        new_allowed_kw = list()
        for kw in allowed_kw:
            if kw != "mock":
                new_allowed_kw.append(kw)
        allowed_kw = new_allowed_kw
    positions_fn = iterate_log_for_positions_fn(folder, log, allowed_kw, key_to_fetch)
    positions_values = {kw: list() for kw in allowed_kw}
    # Là, je cherche à charger les .bin dans un dictionnaire de listes.
    for key in positions_fn.keys():
        positions_file_for_type = positions_fn[key]
        for file in positions_file_for_type:
            if not os.path.exists(file):
                positions = np.empty(0)
            else:
                positions = np.fromfile(file, dtype=np.int16)
            positions_values[key].append(positions)
    return positions_values


def get_exp_type(log, key_to_fetch):
    """

    :param log:
    :param key_to_fetch:
    :return:
    """
    exp_type = ""

    try:
        assert (key_to_fetch in log.keys()), f"{key_to_fetch} not in json."
        sub_d = log[key_to_fetch]
        exp_type = sub_d["Type"]

    except AssertionError as error:
        print("Error: ", error)

    return exp_type


def check_if_block_complete(log):
    key_to_fetch = "Experiment ended correctly"
    if key_to_fetch in list(log.keys()):
        return log[key_to_fetch]
    else:
        return False


def extract(folder, ANALOG_TRIGGERS_MAPPING):
    """
    Extraction des canaux analogiques et digitaux.
    :param folder: Dossier de l'expérience.
    :return: Dictionnaire avec les clés : LENGTH, DIGITAL, ANALOG
    """
    output = dict()
    dig_file = os.path.join(folder, "dig_in.npy")
    analog_file = os.path.join(folder, "analog_in.npy")

    session_file = read_log_file(folder)


    assert (os.path.exists(dig_file)), "No digital triggers file in directory."
    d_trigs = np.load(dig_file)
    length = d_trigs.shape[1]
    save_recording_length(folder, length)
    output["LENGTH"] = length

    # DIGITAL
    output["DIGITAL"] = extract_digital_lines(folder, d_trigs, get_digital_mapping(json_file=session_file))

    # ANALOG
    a_trigs = np.load(analog_file)
    output["ANALOG"] = extract_analog_lines(folder, a_trigs, ANALOG_TRIGGERS_MAPPING)

    return output


def extract_digital_lines(folder, digital_channels, mapping):
    """
    Extraction des lignes digitales.
    :param folder: Dossier pour la sauvegarde.
    :param digital_channels: Un np.ndarray.
    :param mapping: Cartographie des canaux digitaux.
    :return: Dictionnaire.
    """
    digital_triggers = dict()
    fn_pattern = os.path.join(folder, "trig_dig_chan_{}.npy")
    print(fn_pattern)
    for key, line_number in mapping.items():
        if key not in ["EXP", "COND"]:
            print(digital_channels[line_number])
            events = detect_digital_triggers(digital_channels[line_number])
        else:
            continue
        np.save(fn_pattern.format(key), events)
        digital_triggers[key] = events

    if "COND" in list(mapping.keys()) and "EXP" in list(mapping.keys()):
        digital_triggers["XP_PAUSE"] = get_pause(digital_channels[mapping["COND"]], digital_channels[mapping["EXP"]])
        digital_triggers["XP_NORMAL"] = get_normal(digital_channels[mapping["COND"]], digital_channels[mapping["EXP"]])
        digital_triggers["XP_SPECIAL"] = get_special(digital_channels[mapping["COND"]], digital_channels[mapping["EXP"]])
        digital_triggers["XP_WARMUP"] = get_warmup(digital_channels[mapping["COND"]], digital_channels[mapping["EXP"]])
    return digital_triggers


def extract_analog_lines(folder, analog_channels, mapping):
    """
    Extraction des lignes analogiques.
    :param folder: Dossier pour la sauvegarde.
    :param analog_channels: Un np.ndarray.
    :param mapping: Cartographie des canaux analogiques.
    :return: Dictionnaire.
    """
    analog_triggers = dict()
    fn_pattern = os.path.join(folder, "trig_analog_chan_{}.npy")
    for key, line_number in mapping.items():
        events = detect_analog_triggers(analog_channels[line_number], min_time_between=0.002)
        np.save(fn_pattern.format(key), events)
        analog_triggers[key] = events
    return analog_triggers


def get_audio_mapping(log):
    """
    Pas utile pour le moment.
    :param log:
    :return:
    """
    mapping = log["Mapping"]
    mid_tone = log["Mid tone"]
    n_freqs = log["Num frequencies"]
    n_oct = log["Num octaves"]
    pass


def append_zero(i, length):
    """

    :param i:
    :param length:
    :return:
    """
    assert (length % 10 == 0), "Length must be a power of ten."
    if i > length:
        n = str(i)
    else:
        block_size = int(np.log10(length))
        targets = np.array([10 ** x for x in range(1, block_size)], dtype=int)
        idx = np.less(targets, i)
        n = (block_size - np.sum(idx)) * "0" + str(i)
    return n


def get_recording_length(folder):
    """
    On cherche la durée de l'enregistrement.
    """
    assert (os.path.exists(os.path.join(folder, "recording_length.bin"))), "No length file."
    with open(os.path.join(folder, "recording_length.bin"), "r") as f:
        length = int(f.read())
    return length


def save_recording_length(folder, length):
    """
    Sauver la durée d'enregistrement.
    """
    with open(os.path.join(folder, "recording_length.bin"), "w") as f:
        f.write('{:03d}\n'.format(length))


def check_digital_triggers(folder):
    """
    Demande les fichiers digitaux.
    """
    return check_files(folder, analog=False)


def check_analog_triggers(folder):
    """
    Demande les fichiers analogiques.
    """
    return check_files(folder, analog=True)


def check_files(folder, analog=True):
    """
    On lui dit quel modèle chercher. Et sort une liste de fichiers.
    """
    if analog:
        fn_pattern = os.path.join(folder, "trig_analog_chan*.npy")
    else:
        fn_pattern = os.path.join(folder, "trig_dig_chan*.npy")
    lf = list(glob.glob(fn_pattern))
    return len(lf), lf


def detect_digital_triggers(digital_channel, min_time_between=0.001, fs=30e3):
    """
    Détection des triggers digitaux.
    :param digital_channel:
    :param min_time_between:
    :param fs:
    :return:
    """
    distance = int(min_time_between * fs)
    out = signal.find_peaks(digital_channel, height=1, distance=distance, plateau_size=[10, 1000])[1]
    return out["left_edges"]


def detect_analog_triggers(analog_channel, min_time_between=0.005, fs=30e3):
    """
    Détections des triggers analogiques.
    :param analog_channel:
    :param min_time_between:
    :param fs:
    :return:
    """
    distance = int(min_time_between * fs)
    analog_channel = np.where(analog_channel <= 2, 0, 1)
    fp_out = signal.find_peaks(analog_channel, height=1, distance=distance, plateau_size=[10, 1000])
    return fp_out[1]["left_edges"]


def findpeaks_both_edges(logic):
    """
    Simple fonction utilitaire pour faire une liste à partir d'un array 2D.
    Prends le rising edge et le falling edge. Pour les triggers de conditions et d'expérience.
    :param logic: Un np.ndarray de booléens
    :return: Liste de np.ndarray.
    """
    fp = signal.find_peaks(logic, height=1, plateau_size=[10, 1000])[1]
    out = np.vstack((fp["left_edges"], fp["right_edges"]))
    result = [out[:, i] for i in range(out.shape[1])]
    return result


def get_pause(cond, exp):
    """
    Les pauses sont signalées par l'état HAUT des DEUX canaux EXP et COND.
    :param cond:
    :param exp:
    :return: Liste de np.ndarray.
    """
    logic = np.logical_and(cond, exp)
    return findpeaks_both_edges(logic)


def get_normal(cond, exp):
    """
    Les trackings dits "normaux" sont signalés par l'état BAS des DEUX canaux EXP et COND.
    :param cond:
    :param exp:
    :return: Liste de np.ndarray.
    """
    logic = np.logical_and(np.logical_not(cond), np.logical_not(exp))
    return findpeaks_both_edges(logic)


def get_special(cond, exp):
    """
    Les spéciaux sont représentés par l'état BAS de EXP et l'état HAUT de COND.
    :param cond:
    :param exp:
    :return: Liste de np.ndarray.
    """
    logic = np.logical_and(cond, np.logical_not(exp))
    return findpeaks_both_edges(logic)


def get_warmup(cond, exp):
    """
    Les warmups sont représentés par l'état HAUT de EXP et l'état BAS de COND.
    :param cond:
    :param exp:
    :return: Liste de np.ndarray.
    """
    logic = np.logical_and(exp, np.logical_not(cond))
    return findpeaks_both_edges(logic)


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
        type_value = data['Block_000']['Type']

        if type_value=="Pause":
            type_value = data['Block_001']['Type']
            
        print("Type:", type_value)

    else:
        print("Error: More than one JSON file found or no JSON files found.")
    return type_value
