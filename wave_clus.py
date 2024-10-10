import numpy as np
from scipy.io import savemat
import scipy.io as sio
import os
import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit
import spikeinterface.widgets as sw
from convert_positions_in_tones import *
from utils_extraction import *
import spikeinterface
import zarr as zr
from  pathlib import Path
import tqdm
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from probeinterface import Probe, ProbeGroup
import matplotlib
import pickle

# chemin = 'Z:/eTheremin/ALTAI/'
# sessions = ['ALTAI_20240718_SESSION_00','ALTAI_20240722_SESSION_02','ALTAI_20240723_SESSION_00','ALTAI_20240725_SESSION_00',
#             'ALTAI_20240726_SESSION_01','ALTAI_20240806_SESSION_00','ALTAI_20240807_SESSION_00','ALTAI_20240809_SESSION_00',
#             'ALTAI_20240814_SESSION_00','ALTAI_20240822','ALTAI_20240823_SESSION_01','ALTAI_20240824_SESSION_00',
#             'ALTAI_20240826_SESSION_01','ALTAI_20240826_SESSION_02','ALTAI_20240826_SESSION_03','ALTAI_20240827_SESSION_00',
#             'ALTAI_20240827_SESSION_01','ALTAI_20240902_SESSION_00','ALTAI_20240902_SESSION_01','ALTAI_20240902_SESSION_01',
#             'ALTAI_20240912_SESSION_01','ALTAI_20240914_SESSION_00','ALTAI_20240917','ALTAI_20240918_SESSION_00']
#sessions =  [nom for nom in os.listdir(chemin) if os.path.isdir(os.path.join(chemin, nom))]

chemin = 'Z:/eTheremin/MUROLS/MUROLS_20230218/'
sessions = ['MUROLS_20230218_SESSION_01']
# for session in sessions:
#     path = chemin + session
#     print(session)
#     save_path = 'Y:/eTheremin/clara/' + session +'/'#+ '/filtered/std.min =5 bis/'

#     matplotlib.use('Agg')
#     sr=30e3
#     fs = sr
#     path_f = path+'/headstage_0/'
#     if os.path.exists(path_f + 'good_clusters.npy'):
#         num_channel = np.load(path_f + 'good_clusters.npy', allow_pickle = True)
#     else : 
#         num_channel = np.arange(32)
#     all_files_exist = all(os.path.exists(save_path + 'C' + str(k) + '.mat') for k in num_channel)
#     if all_files_exist:
#             print(f"Tous les fichiers 'C' pour la session {session} existent déjà. Passer à la session suivante.")
#             continue  # Passer à la session suivante si tous les fichiers 'C' existent déjà

#     if os.path.exists(path_f +'neural_data.npy'):
#         if not os.path.exists(path_f +'filtered_neural_data.npy'):
#             neural_data = np.load(path_f +'neural_data.npy')
#             sig = neural_data
#             n_cpus = os.cpu_count()
#             full_raw_rec = se.NumpyRecording(traces_list=np.transpose(sig), sampling_frequency=sr)
#             # Convertir le type de données avant d'appliquer le filtre
#             full_raw_rec = full_raw_rec.astype('float32')  # Vous pouvez aussi utiliser 'int16'
#             raw_rec = full_raw_rec #ici si tu prends tous les canaux
#             #raw_rec = full_raw_rec.remove_channels(["CH0", "CH4", "CH7", "CH26", "CH19","CH12", "CH22", "CH30"]) #,"CH12", "CH13","CH14", "CH15", "CH16", "CH17", "CH18", "CH19", "CH21", "CH22", "CH23","CH31"  ]) # ici c'est si tu veux retirer certains canaux pour faire la CMR donc tu veux retirer les canaux morts
#             recording_cmr = si.common_reference(raw_rec, reference='global', operator='median') # ici on fait la CMR (retirer les artefacts communs à tous les charnels
#             recording_f = si.bandpass_filter(recording_cmr, freq_min=300, freq_max=3000) # filtre passe-bande ensuite
#             filtered_neural_signal = recording_f.get_traces().astype(np.float32)
#              # pour récupérer le signal neural filtré
#             np.save(path_f + 'filtered_neural_data.npy', filtered_neural_signal)
#         data_t = np.load(path_f + 'filtered_neural_data.npy', allow_pickle = True)
#         data = data_t.transpose() # pour les données filrées
#         print(data.shape)
#         print(num_channel)
#         for k in num_channel:
#             print(k)
#             data_C = data[k,:]
#             print(data_C.shape)  # Pour vérifier la forme du tableau
#             print(data_C)        # Pour vérifier son contenu
#             data_dict = {'data': data_C,'sr':30000}
#             savemat(save_path + 'C'+ str(k) +'.mat',data_dict)
#             print('ok')


chemin = 'Z:/eTheremin/MUROLS/MUROLS_20230218/'
sessions = ['MUROLS_20230218_SESSION_01']
print('1')
chunk_size = 10**6  # Taille du chunk (exemple: 1 million d'échantillons)
sr = 30e3  # Fréquence d'échantillonnage
fs = sr

matplotlib.use('Agg')

# Boucle pour chaque session
for session in sessions:
    path = chemin + session
    print(f"Traitement de la session : {session}")
    save_path = 'Y:/eTheremin/clara/' + session + '/'

    path_f = path + '/headstage_0/'

    # Vérification des clusters
    if os.path.exists(path_f + 'good_clusters.npy'):
        num_channel = np.load(path_f + 'good_clusters.npy', allow_pickle=True)
    else:
        num_channel = np.arange(32)

    # Vérifier si tous les fichiers existent déjà
    all_files_exist = all(os.path.exists(save_path + 'C' + str(k) + '.mat') for k in num_channel)
    if all_files_exist:
        print(f"Tous les fichiers 'C' pour la session {session} existent déjà. Passer à la session suivante.")
        continue  # Passer à la session suivante si tous les fichiers 'C' existent

    # Vérification et chargement des données neuronales
    if os.path.exists(path_f + 'neural_data.npy'):

        # Charger les données neuronales
        neural_data = np.load(path_f + 'neural_data.npy')

        sig = neural_data
        full_raw_rec = se.NumpyRecording(traces_list=np.transpose(sig), sampling_frequency=sr)

        # Convertir le type de données avant d'appliquer le filtre
        full_raw_rec = full_raw_rec.astype('float32')

        # Appliquer la référence commune et le filtre passe-bande sur l'enregistrement entier
        recording_cmr = si.common_reference(full_raw_rec, reference='global', operator='median')
        recording_f = si.bandpass_filter(recording_cmr, freq_min=300, freq_max=3000)  # Appliquer le filtre sur l'enregistrement

        # Ajuster la forme de filtered_neural_signal pour correspondre à celle de chunk_filtered (échantillons, canaux)
        filtered_neural_signal = np.empty((neural_data.shape[1], neural_data.shape[0]), dtype=np.float32)  # (échantillons, canaux)

        # Traitement par chunks
        for start in range(0, neural_data.shape[1], chunk_size):
            end = min(start + chunk_size, neural_data.shape[1])  # Fin du chunk
            print(f"Traitement du chunk de {start} à {end}")

            # Récupérer le chunk filtré de l'enregistrement
            chunk_filtered = recording_f.get_traces(start_frame=start, end_frame=end).astype(np.float32)

            # Insérer le chunk filtré directement dans filtered_neural_signal (sans transposition)
            filtered_neural_signal[start:end, :] = chunk_filtered


        # Sauvegarder les données filtrées
        np.save(path_f + 'filtered_neural_data_2.npy', filtered_neural_signal)

        # Charger les données filtrées
        data_t = np.load(path_f + 'filtered_neural_data_2.npy', allow_pickle=True)
        data = data_t.transpose()  # Transposition des données pour aligner les canaux

        print(f"Forme des données filtrées : {data.shape}")
        print(f"Canaux : {num_channel}")

        # Sauvegarde des données pour chaque canal
        for k in num_channel:
            print(f"Traitement du canal : {k}")
            data_C = data[k, :]
            print(f"Forme des données pour le canal {k} : {data_C.shape}")

            # Sauvegarder dans un fichier .mat
            data_dict = {'data': data_C, 'sr': sr}
            savemat(save_path + 'C' + str(k) + '.mat', data_dict)
            print(f"Fichier C{k}.mat sauvegardé avec succès.")

    else:
        print(f"Les données neuronales pour la session {session} n'existent pas dans {path_f}")
