import numpy as np
import scipy.io

def convert_mat_to_npy(mat_file, npy_file):
    # Charger le fichier .mat#create_data_features(path, bin_width, sr)
    mat_data = scipy.io.loadmat(mat_file)
    
    # Si tu veux sauvegarder tout le contenu du fichier .mat dans un fichier .npy
    #np.save(npy_file, mat_data)
    
    np.save(npy_file, mat_data['cluster_class'])

#Convertir .mat en .npy

def create_spikes_clusters(path,channel, mat_file, npy_file):
    spk_clus_f = []
    spk_times_f = []
    #print('Z:/eTheremin/OSCYPEK/OSCYPEK/OSCYPEK_20240710_SESSION_00/Nouveaudossier/times_C_{channel}.npy')


    convert_mat_to_npy(mat_file, npy_file)
    #load le fichier .npy
    ss = np.load(npy_file, allow_pickle=True)

    # diviser le fichier en temps de spikes et clusters associés
    spk_clus = ss[:,0]
    spk_clus = [x + channel*100 for x in spk_clus]
    spk_clus = [int(elt) for elt in spk_clus]
    spk_times = ss[:, 1]
    spk_clus_f = spk_clus_f + spk_clus
    spk_times_f = np.concatenate((spk_times_f, spk_times))

    #save
    np.save(path + '/ss_C' + str(channel) + '_spike_clusters.npy',spk_clus_f)
    np.save(path + '/ss_C' + str(channel) + '_spike_times.npy',spk_times_f)