import numpy as np
import scipy.io

def convert_mat_to_npy(mat_file, npy_file):
    # Charger le fichier .mat#create_data_features(path, bin_width, sr)
    mat_data = scipy.io.loadmat(mat_file)
    
    # Si tu veux sauvegarder tout le contenu du fichier .mat dans un fichier .npy
    #np.save(npy_file, mat_data)
    
    np.save(npy_file, mat_data['cluster_class'])

#Convertir .mat en .npy


def create_spikes_clusters(save_path, num_channel):
    spk_clus_f = []
    spk_times_f = []
    # Parcourir chaque canal
    for channel in num_channel:
        mat_file = save_path + 'times_C' + str(channel) + '.mat'
        npy_file = save_path + 'times_C' + str(channel) + '.npy'
        convert_mat_to_npy(mat_file, npy_file)  # Convertit le fichier .mat en .npy
        # Charger le fichier .npy
        ss = np.load(npy_file, allow_pickle=True)

        # Diviser le fichier en temps de spikes et clusters associés
        spk_clus = ss[:, 0]
        spk_clus = [x + channel * 100 for x in spk_clus]  # Ajoute le décalage du canal
        spk_clus = [int(elt) for elt in spk_clus]
        spk_times = ss[:, 1]
        
        # Ajouter les valeurs au tableau final
        spk_clus_f.extend(spk_clus)
        spk_times_f.extend(spk_times)

    # Combiner spk_times_f et spk_clus_f dans une liste de tuples
    combined = list(zip(spk_times_f, spk_clus_f))

    # Trier en fonction de spk_times_f (le premier élément de chaque tuple)
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Séparer les listes triées
    spk_times_f_sorted, spk_clus_f_sorted = zip(*combined_sorted)

    # Convertir en listes (si nécessaire)
    spk_times_f_sorted = list(spk_times_f_sorted)
    spk_clus_f_sorted = list(spk_clus_f_sorted)

    # Sauvegarder les résultats triés
    np.save(save_path + '/ss_spike_clusters.npy', spk_clus_f_sorted)
    np.save(save_path + '/ss_spike_times.npy', spk_times_f_sorted)
