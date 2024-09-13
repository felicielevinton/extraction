from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
import json
n_blocs = 10

"""""
Contient les fonctions nécessaires pour formater les données

 - moyenne de psth pour chaque cluster
 - plot 
"""

def get_all_response(data, features, t_pre, t_post, bin_width, good_clusters):
    """""
    Fonction qui renvoie pour chaque neurone toutes les réponses
    
    input : data, features, t_pre, t_post, bin_width, good_clusters
    output : [neurones x nbr_de_chgt_freq(t_post-t_pre)] tableau qui contient pour chaque neurone, tous les psth liés aux changements de fréquences (non moyennés)
    
    """
    tracking, playback=[], []
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        for bin in range(len(features)):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0:
                mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1:
                mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        tracking.append(mean_psth_tr)
        playback.append(mean_psth_pb)
    return tracking, playback


def get_mean_neurone(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par neurone
    
    input: fichier data.npy d'une session, features.npy, t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        for bin in range(len(features)):
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0:
                    mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1:
                    mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback


def get_mean_neurone_spaced_frequency(data, features, t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par neurone
    Attention ici je ne prends que les changements de fréquence qui sont 
    séparés de plus de 200ms (pour vérifier que les oscillations sont bien
    dûes aux changements de fréquence précédents le stim d'intéret)
    --> si tu veux l'utiliser : change l'appel à la fonction dans get_mean_psth
    input: fichier data.npy d'une session, features.npy, t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        previousbin=0
        for bin in range(len(features)):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0 and bin-previousbin>0.2/bin_width:
                mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                previousbin=bin
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and bin-previousbin>0.2/bin_width:
                mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                previousbin=bin
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback


def get_mean_neurone_for_block(data, features, block, t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par neurone
    pour un block en particulier (block = block)
    """
    tracking, playback=[], []    
    for cluster in good_clusters:
        mean_psth_tr, mean_psth_pb = [], []
        for bin in range(len(features)):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0 and features[bin]['Block']==block:
                mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and features[bin]['Block']==block:
                mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback


def get_mean_neurone_indexes(data, features, indexes,  t_pre, t_post, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen par neurones uniquement pour certains indices de changement de fréquence
    
    input: fichier data.npy d'une session, features.npy,
        indexes : indices des changements de fréquences auxquels on s'intéresse
        t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy
        output : 2 listes [neurones, bins] pour tracking et playback
    
    """
    psth = []    
    for cluster in good_clusters:
        mean_psth = []
        for bin in indexes:
                mean_psth.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(np.nanmean(mean_psth, axis=0))
    return psth



def get_mean_psth(path, t_pre, t_post, bin_width, good_cluster = True):
    """"
    Fonction qui renvoie le psth moyen par neurone en tracking et en playback
    
    Renvoie 2 tableaux de format neurones x bin
    input : path (vers le folder où sont tous les folders), une liste contenant les folders
    output: listes [bins] pour tracking et playback contenant le psth moyen
    """
    tracking, playback = [], []
    data = np.load(path+'/data.npy')
    features = np.load(path+'/features.npy', allow_pickle=True)
    if good_cluster : 
        #good_clusters = np.load(path+'/good_clusters_playback.npy', allow_pickle=True)
        good_clusters = np.arange(32)
        tracking_session, playback_session = get_mean_neurone(data, features, t_pre, t_post, bin_width, good_clusters) #inclus tous les changements de frequence
            #tracking_session, playback_session = get_mean_neurone_spaced_frequency(data, features, t_pre, t_post, bin_width, good_clusters) # si tu veux ne prendre que les changements de freq espacés de 200ms
    else : 
            #je prends les mauvais clusters
        good_clusters = np.load(path+'/good_clusters_playback.npy', allow_pickle=True)
        all_clusters = list(range(len(good_cluster)))
        bad_clusters = [chiffre for chiffre in all_clusters if chiffre not in good_clusters]
        tracking_session, playback_session = get_mean_neurone(data, features, t_pre, t_post, bin_width, bad_clusters)#inclus tous les changements de frequence
            #tracking_session, playback_session = get_mean_neurone_spaced_frequency(data, features, t_pre, t_post, bin_width, bad_clusters)# si tu veux ne prendre que les changements de freq espacés de 200ms
    return tracking_session, playback_session

def get_sem(neurones):
    """""
    Fonction qui renvoie la sem pour un tableau de format (neurones x bin)
    
    input : un tableau [neurones, bins]
    output: liste [bins] contenant la SEM
    """
    sem = []
    for bin in range(len(neurones[0])):
        sem.append(np.nanstd(np.array(neurones)[:,bin])/np.sqrt(len(neurones)))
    return sem

def plot_mean_psth(tracking, playback, psth_bins, title, path, remove_baseline = False):
    """""
    plot tracking vs playback moyenné sur tous les clusters
    
    
    input : tracking, playback =  get_mean_psth(folders, t_pre, t_post, bin_width)
    output : plot of tracking vs playback with SEM
    """
    # Calcul des SEM accross neurones
    tr_sem = get_sem(tracking)
    pb_sem = get_sem(playback)
    # Calcul de la moyenne des psth moyens de chaque neurone
    if remove_baseline:
        mean_tracking = np.nanmean(tracking, axis=0)-np.nanmean(np.nanmean(tracking, axis=0), axis=0)
        mean_playback = np.nanmean(playback, axis=0)-np.nanmean(np.nanmean(playback, axis=0), axis=0)
    else : 
        mean_tracking = np.nanmean(tracking, axis=0)
        mean_playback = np.nanmean(playback, axis=0)
    
    plt.plot(psth_bins[:-1], mean_tracking, c='red', label='Closed loop')
    plt.plot(psth_bins[:-1], mean_playback, c='black', label='Open loop')
    plt.fill_between(psth_bins[:-1], mean_tracking - tr_sem, mean_tracking + tr_sem, color='red', alpha=0.2)
    plt.fill_between(psth_bins[:-1], mean_playback - pb_sem, mean_playback + pb_sem, color='black', alpha=0.2)
    plt.legend()
    plt.title(str(title))
    plt.xlabel('Time (s)')
    plt.ylabel('psth')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig(path+'/tracking_vs_playback.png')
    plt.show()
    print('all izz well')
    
    
def get_mean_psth_in_bandwidth(data, features, bandwidth, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, renvoie la moyenne des psth pour toutes les fréquences comprises dans la badnwidth du cluster
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback), bandwidth
    output : 
     - une liste contenant le psth moyen par cluster [cluster x [t_pre, t_post] ] in la bandwidth
      et une autre out la bandwidth
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    else : 
        c=1
        
    
    in_psth, out_psth=[] , []
    for idx, cluster in enumerate(good_clusters):
        psth_clus, out_clus = [], []
        low_f, high_f = bandwidth[idx][0],  bandwidth[idx][1]
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c:
                    if low_f<=features[bin]['Played_frequency']<=high_f:
                        psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                    else:
                        out_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        if len(psth_clus)==0:
            psth_clus = [[np.nan]*(len(psth_bins)-1)]*2
        if len(out_clus)==0:
            out_clus = [[np.nan]*(len(psth_bins)-1)]*2
        in_psth.append(np.nanmean(psth_clus, axis=0))
        out_psth.append(np.nanmean(out_clus, axis=0))
       
    return in_psth, out_psth    
    




    
def get_playback_indexes(features):
    """""
    renvoie les indices des bin en playback
    
    input : data.npy, features.npy
    output : liste contenant les indices des bins de playback
    """
    playback_indexes = [index for index, value in enumerate(features) if value.get('Condition') == 1]
    return playback_indexes

def get_tracking_indexes(features):
    """""
    renvoie les indices des bin en tracking
    
    input : data.npy, features.npy
    output : liste contenant les indices des bins de tracking
    """
    tracking_indexes = [index for index, value in enumerate(features) if value.get('Condition') == 0]
    return tracking_indexes

def get_frequency_changes(features):
    frequency_changes_indexes = [index for index, value in enumerate(features) if value.get('Frequency_changes') > 0]
    return frequency_changes_indexes  

def get_mock_changes(features):
    mock_changes_indexes = [index for index in range(1, len(features)-1) if (features[index].get('Frequency_changes')!= features[index-1].get('Frequency_changes') or features[index].get('Frequency_changes')!= features[index+1].get('Frequency_changes')) ]
    return mock_changes_indexes 


 
 ####### using Bandwidth ######
 
def get_mean_neurone_in_bandwidth(data, features, t_pre, t_post, bandwidth, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) dans la bandwidth par neurone
    
    input: fichier data.npy d'une session, features.npy, t_post, t_pre,bandwidth (étalement fréquentiel),  bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for i, cluster in enumerate(good_clusters):
        bd = bandwidth[i]
        if np.isnan(bd).any():
            pass
        else:
            mean_psth_tr, mean_psth_pb = [], []
            for bin in range(len(features)):
                if features[bin]['Frequency_changes']>0 and bd[0]<=features[bin]['Played_frequency']<=bd[1] and features[bin]['Condition']==0:
                    mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                if features[bin]['Frequency_changes']>0 and bd[0]<=features[bin]['Played_frequency']<=bd[1] and features[bin]['Condition']==1:
                    mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            tracking.append(np.nanmean(mean_psth_tr, axis=0))
            playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback



def get_mean_neurone_out_bandwidth(data, features, t_pre, t_post, bandwidth, bin_width, good_clusters):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) OUT of the bandwidth par neurone
    
    input: fichier data.npy d'une session, features.npy, t_post, t_pre,bandwidth (étalement fréquentiel),  bin_width, fichier ggod_playback_clusters.npy
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    tracking, playback=[], []    
    for i, cluster in enumerate(good_clusters):
        bd = bandwidth[i]
        if np.isnan(bd).any():
            pass
        else:
            mean_psth_tr, mean_psth_pb = [], []
            for bin in range(len(features)):
                if features[bin]['Frequency_changes']>0 and (bd[0]>features[bin]['Played_frequency'] or features[bin]['Played_frequency']>bd[1]) and features[bin]['Condition']==0:
                    mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                if features[bin]['Frequency_changes']>0 and (bd[0]>features[bin]['Played_frequency'] or features[bin]['Played_frequency']>bd[1]) and features[bin]['Condition']==1:
                    mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            tracking.append(np.nanmean(mean_psth_tr, axis=0))
            playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback




def get_mean_per_f_neurone(data, features, unique_tones, t_pre, t_post, bin_width, good_clusters, condition, path, folder):
    """
    Fonction qui renvoie le psth moyen (tracking et playback) par fréquence pour chaque neurone
    
    input: fichier data.npy d'une session, features.npy, t_post, t_pre, bin_width, fichier ggod_playback_clusters.npy, direction du swipe(parmi 'up' or 'down')
    output : 2 listes [neurones, bins] pour tracking et playabck
    
    """
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    if condition=="tracking":
        c = 0
    else : 
        c=1
    
    heatmaps_up, heatmaps_down = [], []
    # for one frequency 
    for cluster in good_clusters:
        hm_up, hm_down = [], []
        for frequency in unique_tones:
            f_up, f_down=[], []    
            for bin in range(len(features)):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==condition and features[bin]['Played_frequency']==frequency :
                    if features[bin-1]['Played_frequency']<features[bin]['Played_frequency']:
                        f_up.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==condition and features[bin]['Played_frequency']==frequency:
                    if features[bin-1]['Played_frequency']>features[bin]['Played_frequency']:
                        f_down.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            hm_up.append(np.nanmean(f_up, axis=0))
            hm_down.append(np.nanmean(f_down, axis=0))
        heatmaps_up.append(hm_up)
        heatmaps_down.append(hm_down)
        
    np.save(path+folder+'/heatmaps_down.npy',heatmaps_down )
    np.save(path+folder+'/heatmaps_up.npy',heatmaps_up )
    return heatmaps_up, heatmaps_down





def get_mean_neurone_for_block_in_bd(data, features, block, t_pre, t_post, bin_width, good_clusters, bandwidth):
    """
    Fonction qui renvoie le psth moyen dans la bandwidth de chaque neurone (tracking et playback) par neurone
    pour un block en particulier (block = block)
    """
    tracking, playback=[], []    
    for i, cluster in enumerate(good_clusters):
        mean_psth_tr, mean_psth_pb = [], []
        bd = bandwidth[i]
        for bin in range(len(features)):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==0 and features[bin]['Block']==block and bd[0]<=features[bin]['Played_frequency']<=bd[1]:
                mean_psth_tr.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==1 and features[bin]['Block']==block and bd[0]<=features[bin]['Played_frequency']<=bd[1]:
                mean_psth_pb.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        tracking.append(np.nanmean(mean_psth_tr, axis=0))
        playback.append(np.nanmean(mean_psth_pb, axis=0))
    return tracking, playback

def est_premier(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True