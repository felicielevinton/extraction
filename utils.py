import numpy as np
from scipy.signal import find_peaks


def est_premier(nombre):
    if nombre <= 1:
        return False
    elif nombre <= 3:
        return True
    elif nombre % 2 == 0 or nombre % 3 == 0:
        return False
    i = 5
    while i * i <= nombre:
        if nombre % i == 0 or nombre % (i + 2) == 0:
            return False
        i += 6
    return True


def get_plot_geometry(good_clusters):
    n_clus = len(good_clusters)
    if est_premier(n_clus):
        n_clus=n_clus-1

    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
        
        #print(num_columns)
    num_rows = -(-n_clus // num_columns)
    return num_rows, num_columns



def process_cluster_order(cluster_order):
    # Convertir le tableau en entier si nécessaire (facultatif, en fonction de ton besoin)
    # Convertir en tableau NumPy si ce n'est pas déjà fait
    cluster_order = np.array(cluster_order)
    
    # Vérifier si c'est bien un tableau 2D
    if cluster_order.ndim != 2:
        raise ValueError(f"cluster_order doit être un tableau 2D, mais a la forme {cluster_order.shape}")
    
    # Suppression des sous-listes dupliquées
    unique_order = np.unique(cluster_order, axis=0)

    # Tri d'abord par le deuxième élément, puis par le premier
    sorted_indices = np.lexsort((unique_order[:, 1], unique_order[:, 0]))
    sorted_unique_order = unique_order[sorted_indices]

    return sorted_unique_order



def get_better_plot_geometry(cluster_order,good_cluster):
    # Calculate number of rows and columns for subplots
    #cluster_order c'est une liste de sous-listes [channel,cluster] de len 3000000 qqchose
    order = process_cluster_order(cluster_order) #numéros des channels 
    num_plots = len(order)
    num_cols = int(np.max(order[:, 1])) + 1
    num_rows = len(good_cluster) + 1
    return num_plots, num_rows, num_cols

def get_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition):
    """
    Pour voir, pour chaque neurone, les psth
    
    input: 
      -data, features, good_clustersn condition ("tracking" or "playback)
    output : 
     - une liste contenant le psth moyen par cluster pour chaque changement de fréquence en playback [neurones x chgt de freq x [t_pre, t_post] ]
    """
    if condition=="tracking":
        c = 0
    elif condition == "playback" : 
        c=1
    elif condition== "tail":
        c = -1
    elif condition =="mapping change":
        c = 2
    if condition=="tonotopy":
        c = 0
    
    
    psth=[] 
    for cluster in good_clusters:
        psth_clus = []
        for bin in range(len(features)):
            #print(diff)
            if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
                if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                    psth_clus.append(data[cluster][bin-int(t_pre/bin_width):bin+int(t_post/bin_width)])
        psth.append(psth_clus)
    return psth

def get_played_frequency(features, t_pre, t_post, bin_width, condition):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    if condition=="tracking":
        c = 0
    elif condition=="playback":
        c=1
    elif condition=="tail":
        c = -1
    elif condition == "mappingchange":
        c = 2
    frequency = []
    for bin in range(len(features)):
        if bin-int(t_pre/bin_width)>0 and bin+int(t_post/bin_width)<len(features):
            if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
                frequency.append(features[bin]['Played_frequency'])
    return frequency
        


def get_mock_frequency(features):
    """"
    Fonction pour récupérer la fréquence jouée pour chaque psth défini dans get_psth
    """
    c=1
    frequency = []
    for bin in range(len(features)):
        if features[bin]['Frequency_changes']>0 and features[bin]['Condition']==c :
            frequency.append(features[bin]['Mock_frequency'])
    return frequency
        






def get_sustained_activity(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    return (np.nanmean(psth[0: int(t_pre/bin_width)-2]))




def get_sustained_activity_nan(psth, t_pre, t_post, bin_width):
    """""
    Fonction qui renvoie l'activité moyenne d'un seul psth
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    --> dans la cas où on aurait des nan gênants
    
    
    """
    if psth is not np.nan:
    
        return (np.nanmean(psth[0: int(t_pre/bin_width)-2]))
    else:
        return np.nan




def mean_maxima(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau cont les indices sont compris
    entre t0 et t1
    """
    # Find peaks in the array
    pics, _ = find_peaks(arr[t0:t1], distance=thresh)

    # Check if there are at least two peaks
    if len(pics) >= 2:
        # Get the indices of the two maximum values
        max_indices = np.argsort(arr[pics])[-2:]

        # Calculate the mean of the two maximum values
        mean = np.mean(arr[pics][max_indices])

        # Get the actual maximum values
        max_values = arr[pics][max_indices]
    else:
        mean = np.nan
        max_values = np.nan

    return mean, pics, max_values


def mean_maxima_nan(arr, thresh, t0, t1):
    """
    Renvoie la moyenne des deux points max d'un tableau cont les indices sont compris
    entre t0 et t1
    
    --> cas où on aurait des nan gênants
    """
    # Find peaks in the array
    if arr is not np.nan:
        pics, _ = find_peaks(arr, distance=thresh)

        # Check if there are at least two peaks
        if len(pics) >= 2:
            # Get the indices of the two maximum values
            max_indices = np.argsort(arr[pics])[-2:]

            # Calculate the mean of the two maximum values
            mean = np.mean(arr[pics][max_indices])

            # Get the actual maximum values
            max_values = arr[pics][max_indices]
        else:
            mean = np.nan
            max_values = np.nan
    else:
        mean = np.nan
        max_values = np.nan
        pics=np.nan
        

    return mean, pics, max_values


def get_total_evoked_response(psth, t_pre, t_post, bin_width, thresh):
    """"
    Function qui renvoie la total evoked reponse pour un tableau contenant des psth
    input : un tableau psth contenant des psth
    output : un tableau contenant la total evoked response pour chaque psth
    
    """
    total_evoked_response = []
    for elt in psth:
        total_evoked_response.append(mean_maxima(elt, thresh)[0])
    return total_evoked_response


def get_indexes(tableau, a):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est égale à a

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i] == a:
            indices_a.append(i)

    return indices_a

def get_indexes_in(tableau, a, b):
    """
    pour trouver les indices des elements dans tableau dont 
    la valeur est comprise entre a et b

    Args:
        tableau (_type_): _description_
        a (_type_): _description_

    Returns:
        les indices de a dans le tableau 
    """
    indices_a = []

    for i in range(len(tableau)):
        if tableau[i]>=a and tableau[i]<=b:
            indices_a.append(i)

    return indices_a





def get_sustained_activity_OLD(psth, t_pre, t_post, bin_width):
    """""
    PAS UTILE POUR L'INSTANT !!!
    Fonction qui renvoie l'activité moyenne d'un tableau de PSTH
    input : un tableau contenant des PSTH
    output : sustained activity pour chaque PSTH
    
    
    """
    sustained = []
    for elt in psth:
        sustained.append(np.nanmean(elt[0: int(t_pre/bin_width)-2]))
    return sustained 


def indices_valeurs_egales(tableau, valeur_cible):
    """
    

    Args:
        tableau (_type_): un tableau
        valeur_cible (_type_): la valeur qu'on recherche dans le tableau

    Returns:
        indices: les indices des éléments dans le tableau dont la valeur est égale à la valeur cible
    """
    indices = []
    for i in range(len(tableau)):
        if tableau[i] == valeur_cible:
            indices.append(i)
    return indices


def indices_valeurs_comprises(tableau, valeur_min, valeur_max):
    
    """"
       Args:
        tableau (_type_): un tableau
        valeur_min, valeur_max (_type_): valeurs qui définissent l'intervalle dans lequel on cherche des valeurs dans le tableau

        Returns:
            indices: les indices des éléments dans le tableau dont la valeur est comprise dans l'intervalle.
    """
    indices = []
    for i in range(len(tableau)):
        if valeur_min<=tableau[i]<valeur_max:
            indices.append(i)
    return indices



def moyenne_psth_par_frequence(psth, mock_freq, unique_tones, min_presentations):
    
     
    """"
    Fonction qui, pour toutes les fréquences de la bandwidth d'un neurone, fait la moyenne par mock frequency.
       Args:
        psth: un tableau qui contient des psth (get_psth)
        mock_freq : tableau qui contient les mock frequencies associées aux psth
        unique_tones : tableau contenantles tons joués
        min_presentations (nombre): nombre minimun de présentations d'une fréquence pour qu'on la retienne
        
        
        Returns:
            psth_per_frequency : les psth moyens par mock_frequency
            n_presnetation : le nombre de présentations des f à une mock frequency donnée
    """
    # Créer une liste pour regrouper les valeurs de psth par fréquence
    psth_per_frequency, n_presentations=[], []
    for f in unique_tones:#je prends une position cible
        tab=[]
        for i in range(len(mock_freq)): # je regarde sur chaque position si je suis a la position cible
            if mock_freq[i]==f: # si je suis à cette position alors j'ajoute
                tab.append(psth[i])
        n = len(tab)
        if n >min_presentations: #au moins 10 occurrences de la condition
            psth_per_frequency.append(np.nanmean(tab, axis=0))
            n_presentations.append(n)
        else : 
            psth_per_frequency.append(np.nan)
            n_presentations.append(n)
    return psth_per_frequency, n_presentations