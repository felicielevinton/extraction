import matplotlib.pyplot as plt
import numpy as np
from format_data import get_playback_indexes
import math
from format_data import get_sem

colors = ['yellow', 'orange', 'red', 'purple', 'cyan', 'green', 'blue', 'black' ]


def histogram_delta(features):
    delta = []
    pb_index = get_playback_indexes(features)
    for idx in pb_index:
        delta.append(math.log2(abs(features[idx]['Played_frequency']/features[idx]['Mock_frequency'])))
    return delta
        
    
    

def delta_frequency_psth(octaves, data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Pour voir, pour chaque neurone, la différence de psth en fonction de la différence entre la 
    Played_frequency et la mock_frequency
    
    input: 
     - octave: nombre d'octaves au dela duquel on considère que le delta_f est grand
      -data, features, good_clusters
    output : 
     - une liste contenant par cluster le psth moyen quand delta(bin)<diff et une liste contenant par cluster le psth moyen quand delta(bin)>diff
      - une liste contenant les écarts entre les fréquences
    """
    equal, delta =[], []  
    for cluster in good_clusters:
        big_mean_cluster, small_mean_cluster = [], []
        delta_f = []
        for bin in range(len(features)):
            diff = abs(math.log2(features[bin]['Played_frequency']/features[bin]['Mock_frequency']))
            delta_f.append(diff)
            #print(diff)
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and diff<=octaves :
                small_mean_cluster.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and diff>octaves :
                big_mean_cluster.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        equal.append(np.nanmean(small_mean_cluster, axis=0))
        delta.append(np.nanmean(big_mean_cluster, axis=0))
    print("BIG", len(big_mean_cluster))
    print("SMALL", len(small_mean_cluster))
    return equal, delta, delta_f 

def get_delta_f(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Pour voir, pour chaque neurone, la différence de psth en fonction de la différence entre la 
    Played_frequency et la mock_frequency
    
    input: 
      -data, features, good_clusters
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
      - une liste contenant les écarts entre les fréquences jouées et mock en playback [chgt de freq]
    """
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        delta_f = []
        for bin in range(len(features)):
            #print(diff)
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 :
                psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                #diff = abs(math.log2(features[bin]['Played_frequency']/features[bin]['Mock_frequency'])) #si on veut une valeur absolue dans la distribution des deltaf
                diff = math.log2(features[bin]['Played_frequency']/features[bin]['Mock_frequency'])
                delta_f.append(diff)
        psth.append(psth_clus)
    return psth, delta_f


def plot_distribution(deltaf, n_bins):
    """""
    Histogramme avec la distribution des deltas F
    """
    plt.hist(deltaf, bins=n_bins, alpha=0.7, color='b', edgecolor='black')

    # Ajouter des étiquettes et un titre
    plt.xlabel('octave (mock-played)')
    plt.ylabel('Fréquence')
    plt.title('Distribution (mock-played frequencies)')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Afficher l'histogramme
    plt.show()

def deltaf_for_cluster(psth, deltaf, octaves_threshold, gc):
    """""
    Fonction qui permet de voir cluster par cluster 
    le psth en fonction de la différence entre mock et played
    groupé selon des intervals.
    on compte en combien de d'octaves on veut découper le deltaf. deltaf est en octave (log2)
    
    output : renvoie un tableau [cluster x interval x bins]
    """
    
    psth_interval = []
    
    intervals = np.arange(np.min(deltaf)+1, np.max(deltaf) + octaves_threshold, octaves_threshold)
    print(intervals)
    indices = np.digitize(deltaf, intervals, right=True).astype(int)
    
    all_clus = []
    for i, clus in enumerate(gc):
        psth_clus = psth[i]
        mean_psth=[]
        for octave in range(len(intervals)):  
        # Get the indices where the condition is met
            selected_indices = np.where(indices == octave)[0]
            selected_arrays = [psth_clus[index] for index in selected_indices]
            mean_psth.append(np.nanmean(selected_arrays, axis=0))
        all_clus.append(mean_psth)
        #for octave in range(0,5):
           # plt.plot(mean_psth[octave], label = f"{octave} octave")
            #plt.legend()
        #plt.title(f'Psth en fonction de (mock-played) pour cluster {clus}')
        #plt.show()

    return(all_clus, intervals)

def plot_psth_function_of_deltaf(psth, deltaf, octave_threshold, good_clusters, psth_bins):
    """""""""
    Fonction qui permet de plot pour une session, le psth moyen en fonction de delta F
    input : psth d'une session, delta f d'une session, good_clusters, psth bins
    
    output: 2 figures par session qui représentent 1) histogramme de la répartition des deltaf 
            2)le psth moyen par octave de deltaf
    """""
    average, intervals = deltaf_for_cluster(psth, deltaf, octave_threshold, good_clusters)
    for interval in range(len(average[0])):
        psth_interval = [ligne[interval] for ligne in average]
        sem_interval = get_sem(np.array(psth_interval))
        
        average_interval = np.nanmean(psth_interval, axis=0)

        

        plt.plot(psth_bins[:-1], average_interval, label = f'{intervals[interval]} octave(s)', c = colors[interval])
        plt.fill_between(psth_bins[:-1], np.array(average_interval) - np.array(sem_interval), np.array(average_interval) + np.array(sem_interval), alpha=0.2, color = colors[interval])
        plt.fill_between(psth_bins[:-1], np.array(average_interval) - np.array(sem_interval), np.array(average_interval) + np.array(sem_interval), alpha=0.2, color = colors[interval])
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend()
        plt.title('psth moyen en fonction de deltaf pour une session')
    plt.show()