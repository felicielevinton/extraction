from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
import json
fs = 30e3
#n_blocs = 3

""""
Contient les fonctions pour formater les données depuis les tt.npz
"""
def get_folders(path):
    folders = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
    return folders

#path = "/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/FRINAULT/FRINAULT_20230218/FRINAULT_20230218_SESSION_00"
#path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/FRINAULT/FRINAULT_20230220/FRINAULT_20230220_SESSION_00'
def create_data(path, bin_width, n_blocs):
    """"
    Mettre en forme les données en :
    data.npy qui contient les spikes binnés pour chaque cluster
    features.npy qui contient les infos sur chaque bin
    
    input : path contenant le tt.npz et tout
            bin_width
    """
    tt = np.load(path+"/tt.npz", allow_pickle=True)
    tt = tt['arr_0'].item()
    #data = pd.read_hdf(path+'/data.h5')
    
    file = path+'/recording_length.bin'
    with open(file, 'rb') as file:
        recording_length = file.read()
    recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
    recording_length = int(''.join(filter(str.isdigit, recording_length)))

    print(recording_length)
    #extraire recording_length OK ca marche

    #avec les clusters et la classe de Flavien
    spike = Spikes(path, recording_length=int(recording_length))

    id = tt.files


    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(int(spike.n_clusters)):
        t_spk.append(spike.get_spike_times(cluster)) #spikes times
        c_spk.append(np.full_like(t_spk[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/fs
    c_spk = c_spk/fs


    ## faire les bins : 
    min_value = t_spk.min()  # Get the minimum value of 'spike_time'
    max_value = t_spk.max()  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define custom bin edges

    ## histogramme par cluster
    unique_clusters = np.unique(c_spk)

    histograms_per_cluster = {}

    for cluster in unique_clusters:
        spike_times_cluster = [time for time, clus in zip(t_spk, c_spk) if clus == cluster]
        # Now spike_times_cluster contains spike times for the current cluster
        
        # Perform histogram for the current cluster
        hist, bin_edges = np.histogram(spike_times_cluster, bins=bins)
        histograms_per_cluster[cluster] = (hist, bin_edges)

    print(histograms_per_cluster)
    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+'/data.npy', data)


    #### TRIGGERS

    t_stim, f_stim, b_stim, type_stim= [], [], [], [] # stimulus times, frequencies, bloc, type_stim(tracking/playback/mock)
    mock_stim, f_mock_stim, b_mock_stim,mock_type_stim = [], [], [], []
    #for bloc in range(n_blocs):
    for bloc in range(1, n_blocs): #ca commence à 1 pour BUrrata mais à zéro pour les autres !!
        print('traitement du bloc', bloc)
        
        # Tracking
        #extraire les triggers et les freq
        t_stim.append(tt['tr_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['tr_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['tr_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['tr_'+str(bloc)][1]), 0))
        
        # Playback
        t_stim.append(tt['pb_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['pb_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['pb_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['pb_'+str(bloc)][1]), 1))
        
        # Mock on verra plus tard 
        mock_stim.append(tt['mk_'+str(bloc)][1]) #time triggers for bloc 
        f_mock_stim.append(tt['mk_'+str(bloc)][0]) #frequencies for bloc
        b_mock_stim.append(np.full_like(tt['mk_'+str(bloc)][1], bloc))
        mock_type_stim.append(np.full(len(tt['mk_'+str(bloc)][1]), 'mock'))
        
    t_stim = np.hstack(t_stim)
    f_stim = np.hstack(f_stim)
    b_stim = np.hstack(b_stim)
    type_stim = np.hstack(type_stim)
    
    unique_tones = sorted(np.unique(f_stim))

    mock_stim = np.hstack(mock_stim)
    f_mock_stim=np.hstack(f_mock_stim)

    t_stim = t_stim/fs
    mock_stim=mock_stim/fs


    # true stims
    interpolated_freq = np.interp(bins, t_stim, f_stim)
    interpolated_blocks = np.interp(bins, t_stim, b_stim)
    interpolated_type_stim = np.interp(bins, t_stim, type_stim)
    bin_stim, _ = np.histogram(t_stim, bins=bins)
    
    #mock frequencies
    interpolated_mock_freq = np.interp(bins, mock_stim, f_mock_stim)

    #corriger l'interpolation qui sinon fait des moyennes de fréquences
    #for bin in range(len(interpolated_freq)):
        #closest_index = np.argmin(np.abs(unique_tones - interpolated_freq[bin]))
        #interpolated_freq[bin]= unique_tones[closest_index] 
        
    #for bin in range(len(interpolated_mock_freq)):
        #closest_index = np.argmin(np.abs(unique_tones - interpolated_mock_freq[bin]))
        #interpolated_mock_freq[bin]= unique_tones[closest_index] 
    
    
    # Create a dictionary to store information for each time bin
    features = {}
    for i, bin in enumerate(bins[:-1]):
        features[bin] = {
            'Played_frequency': interpolated_freq[i],
            'Block': interpolated_blocks[i],
            'Condition': interpolated_type_stim[i],
            'Frequency_changes': bin_stim[i],
            'Mock_frequency':interpolated_mock_freq[i]
        }
        
        
        
    features = list(features.values())
    np.save(path+'/features.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
        
    print('all izz well')


def create_data_v2(path, bin_width, n_blocs):
    """"
    Mettre en forme les données en :
    data.npy qui contient les spikes binnés pour chaque cluster
    features.npy qui contient les infos sur chaque bin
    AVEC MOCK FREQUENCIES
    
    input : path contenant le tt.npz et tout
            bin_width
    """
    tt = np.load(path+"/tt.npz", allow_pickle=True)
    tt = tt['arr_0'].item()

    #data = pd.read_hdf(path+'/data.h5')
    
    file = path+'/recording_length.bin'
    with open(file, 'rb') as file:
        recording_length = file.read()
    recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
    recording_length = int(''.join(filter(str.isdigit, recording_length)))

    print(recording_length)
    #extraire recording_length OK ca marche

    #avec les clusters et la classe de Flavien
    spike = Spikes(path, recording_length=int(recording_length))

    


    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(int(spike.n_clusters)):
        t_spk.append(spike.get_spike_times(cluster)) #spikes times
        c_spk.append(np.full_like(t_spk[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/fs
    c_spk = c_spk/fs


    ## faire les bins : 
    min_value = t_spk.min()  # Get the minimum value of 'spike_time'
    max_value = t_spk.max()  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define custom bin edges

    ## histogramme par cluster
    unique_clusters = np.unique(c_spk)

    histograms_per_cluster = {}

    for cluster in unique_clusters:
        spike_times_cluster = [time for time, clus in zip(t_spk, c_spk) if clus == cluster]
        # Now spike_times_cluster contains spike times for the current cluster
        
        # Perform histogram for the current cluster
        hist, bin_edges = np.histogram(spike_times_cluster, bins=bins)
        histograms_per_cluster[cluster] = (hist, bin_edges)

    print(histograms_per_cluster)
    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+'/data.npy', data)


    #### TRIGGERS

    t_stim, f_stim, b_stim, type_stim= [], [], [], [] # stimulus times, frequencies, bloc, type_stim(tracking/playback/mock)
    mock_stim, f_mock_stim, b_mock_stim,mock_type_stim = [], [], [], []
    for bloc in range(1, n_blocs+1):
        print('traitement du bloc', bloc)
        
        # Tracking
        #extraire les triggers et les freq
        t_stim.append(tt['tr_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['tr_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['tr_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['tr_'+str(bloc)][1]), 0))
        
        # Playback
        t_stim.append(tt['pb_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['pb_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['pb_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['pb_'+str(bloc)][1]), 1))
        
        # Mock on verra plus tard 
        mock_stim.append(tt['mk_'+str(bloc)][1]) #time triggers for bloc 
        f_mock_stim.append(tt['mk_'+str(bloc)][0]) #frequencies for bloc
        b_mock_stim.append(np.full_like(tt['mk_'+str(bloc)][1], bloc))
        mock_type_stim.append(np.full(len(tt['mk_'+str(bloc)][1]), 'mock'))
        
    t_stim = np.hstack(t_stim)
    f_stim = np.hstack(f_stim)
    b_stim = np.hstack(b_stim)
    type_stim = np.hstack(type_stim)
    
    unique_tones = sorted(np.unique(f_stim))

    mock_stim = np.hstack(mock_stim)
    f_mock_stim=np.hstack(f_mock_stim)

    t_stim = t_stim/fs
    mock_stim=mock_stim/fs
    
    print(f"Shape of t_stim: {t_stim.shape}")
    print(f"Shape of f_stim: {f_stim.shape}")
    print(f"Shape of bins: {bins.shape}")

    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)

    previous_frequency = None
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        print(f"stimuli_in_bin indices: {np.where(stimuli_in_bin)}")
        print(f"f_stim values in bin {i}: {f_stim[stimuli_in_bin]}")
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            previous_frequency = interpolated_freq[i]  # Update previous frequency
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                
    interpolated_blocks = np.interp(bins, t_stim, b_stim)
    interpolated_type_stim = np.interp(bins, t_stim, type_stim)

    #2. Mock stims
    mock_stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_mock_frequencies = np.zeros(len(bins) - 1)

    # Initialize previous frequency
    previous_frequency = None

    # Iterate over each bin
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (mock_stim >= bin_start) & (mock_stim < bin_end)
        stimuli_in_bin  = stimuli_in_bin[:-1] # ici c'est au piiiiiiiif
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            mock_stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_mock_frequencies[i] = f_mock_stim[stimuli_in_bin][0]
            previous_frequency = interpolated_mock_frequencies[i]  # Update previous frequency
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_mock_frequencies[i] = previous_frequency

    
    # Create a dictionary to store information for each time bin
    features = {}
    for i, bin in enumerate(bins[:-1]):
        features[bin] = {
            'Played_frequency': interpolated_freq[i],
            'Block': interpolated_blocks[i],
            'Condition': interpolated_type_stim[i],
            'Frequency_changes': stimulus_presence[i],
            'Mock_frequency':interpolated_mock_frequencies[i]
        }
        
        
        
    features = list(features.values())
    np.save(path+'/features.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
        
    print('all izz well')
    
def create_data_v2_no_mock(path, bin_width, n_blocs):
    """"
    Mettre en forme les données en :
    data.npy qui contient les spikes binnés pour chaque cluster
    features.npy qui contient les infos sur chaque bin
    SANS MOCK
    
    input : path contenant le tt.npz et tout
            bin_width
    """
    tt = np.load(path+"/tt.npz", allow_pickle=True)
    tt = tt['arr_0'].item()

    #data = pd.read_hdf(path+'/data.h5')
    
    file = path+'/recording_length.bin'
    with open(file, 'rb') as file:
        recording_length = file.read()
    recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
    recording_length = int(''.join(filter(str.isdigit, recording_length)))

    print(recording_length)
    #extraire recording_length OK ca marche

    #avec les clusters et la classe de Flavien
    spike = Spikes(path, recording_length=int(recording_length))

    


    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(int(spike.n_clusters)):
        t_spk.append(spike.get_spike_times(cluster)) #spikes times
        c_spk.append(np.full_like(t_spk[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/fs
    c_spk = c_spk/fs


    ## faire les bins : 
    min_value = t_spk.min()  # Get the minimum value of 'spike_time'
    max_value = t_spk.max()  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define custom bin edges

    ## histogramme par cluster
    unique_clusters = np.unique(c_spk)

    histograms_per_cluster = {}

    for cluster in unique_clusters:
        spike_times_cluster = [time for time, clus in zip(t_spk, c_spk) if clus == cluster]
        # Now spike_times_cluster contains spike times for the current cluster
        
        # Perform histogram for the current cluster
        hist, bin_edges = np.histogram(spike_times_cluster, bins=bins)
        histograms_per_cluster[cluster] = (hist, bin_edges)

    print(histograms_per_cluster)
    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+'/data.npy', data)


    #### TRIGGERS

    t_stim, f_stim, b_stim, type_stim= [], [], [], [] # stimulus times, frequencies, bloc, type_stim(tracking/playback/mock)
    mock_stim, f_mock_stim, b_mock_stim,mock_type_stim = [], [], [], []
    for bloc in range(n_blocs):
        print('traitement du bloc', bloc)
        
        # Tracking
        #extraire les triggers et les freq
        t_stim.append(tt['tr_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['tr_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['tr_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['tr_'+str(bloc)][1]), 0))
        
        # Playback
        t_stim.append(tt['pb_'+str(bloc)][1]) #time triggers for bloc 
        f_stim.append(tt['pb_'+str(bloc)][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['pb_'+str(bloc)][1], bloc))
        type_stim.append(np.full(len(tt['pb_'+str(bloc)][1]), 1))
        
        # Mock on verra plus tard 
        #mock_stim.append(tt['mk_'+str(bloc)][1]) #time triggers for bloc 
        #f_mock_stim.append(tt['mk_'+str(bloc)][0]) #frequencies for bloc
       # b_mock_stim.append(np.full_like(tt['mk_'+str(bloc)][1], bloc))
       # mock_type_stim.append(np.full(len(tt['mk_'+str(bloc)][1]), 'mock'))
        
    t_stim = np.hstack(t_stim)
    f_stim = np.hstack(f_stim)
    b_stim = np.hstack(b_stim)
    type_stim = np.hstack(type_stim)
    
    unique_tones = sorted(np.unique(f_stim))

    #mock_stim = np.hstack(mock_stim)
    #f_mock_stim=np.hstack(f_mock_stim)

    t_stim = t_stim/fs
    #mock_stim=mock_stim/fs
    

    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)

    previous_frequency = None
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            previous_frequency = interpolated_freq[i]  # Update previous frequency
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                
    interpolated_blocks = np.interp(bins, t_stim, b_stim)
    interpolated_type_stim = np.interp(bins, t_stim, type_stim)


    
    # Create a dictionary to store information for each time bin
    features = {}
    for i, bin in enumerate(bins[:-1]):
        features[bin] = {
            'Played_frequency': interpolated_freq[i],
            'Block': interpolated_blocks[i],
            'Condition': interpolated_type_stim[i],
            'Frequency_changes': stimulus_presence[i]
        }
        
        
        
    features = list(features.values())
    np.save(path+'/features.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
        
    print('all izz well')
    
def create_data_condition(path, bin_width, n_blocs, condition):
    """"
    Pour les cas tracking only ou playback only (pas de bloc, juste un long tracking ou un long playback)
    Mettre en forme les données en :
    data.npy qui contient les spikes binnés pour chaque cluster
    features.npy qui contient les infos sur chaque bin
    
    input : path contenant le tt.npz et tout
            bin_width
            
            
            A FAIRE !!!!!!
    """
    tt = np.load(path+"/tt.npz", allow_pickle=True)
    tt = tt['arr_0'].item()

    #data = pd.read_hdf(path+'/data.h5')
    
    file = path+'/recording_length.bin'
    with open(file, 'rb') as file:
        recording_length = file.read()
    recording_length = recording_length.decode('utf-8')

    # Extract only the numbers using a simple filter
    recording_length = int(''.join(filter(str.isdigit, recording_length)))

    print(recording_length)
    #extraire recording_length OK ca marche

    #avec les clusters et la classe de Flavien
    spike = Spikes(path, recording_length=int(recording_length))

    


    ##NEURO
    t_spk, c_spk = [], [] #spike times, cluster
    #for cluster in range(spike.get_n_clusters()):
    for cluster in range(int(spike.n_clusters)):
        t_spk.append(spike.get_spike_times(cluster)) #spikes times
        c_spk.append(np.full_like(t_spk[cluster], cluster))
    t_spk = np.hstack(t_spk)
    c_spk = np.hstack(c_spk)

    # mettre en secondes 
    t_spk = t_spk/fs
    c_spk = c_spk/fs


    ## faire les bins : 
    min_value = t_spk.min()  # Get the minimum value of 'spike_time'
    max_value = t_spk.max()  # Get the maximum value of 'spike_time'

    bins = np.arange(min_value, max_value + bin_width, bin_width)  # Define custom bin edges

    ## histogramme par cluster
    unique_clusters = np.unique(c_spk)

    histograms_per_cluster = {}

    for cluster in unique_clusters:
        spike_times_cluster = [time for time, clus in zip(t_spk, c_spk) if clus == cluster]
        # Now spike_times_cluster contains spike times for the current cluster
        
        # Perform histogram for the current cluster
        hist, bin_edges = np.histogram(spike_times_cluster, bins=bins)
        histograms_per_cluster[cluster] = (hist, bin_edges)

    print(histograms_per_cluster)
    data = [histograms_per_cluster[key][0] for key in histograms_per_cluster]
    np.save(path+'/data.npy', data)


    #### TRIGGERS

    t_stim, f_stim, b_stim, type_stim= [], [], [], [] # stimulus times, frequencies, bloc, type_stim(tracking/playback/mock)
    mock_stim, f_mock_stim, b_mock_stim,mock_type_stim = [], [], [], []
  
    if condition=='tracking':
            # Tracking
            #extraire les triggers et les freq
        t_stim.append(tt['tr_0'][1]) #time triggers for bloc 
        f_stim.append(tt['tr_0'][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['tr_0'][1], 0))
        type_stim.append(np.full(len(tt['tr_0'][1]), 0))
    if condition=='playback':
        # Playback
        t_stim.append(tt['pb_0'][1]) #time triggers for bloc 
        f_stim.append(tt['pb_0'][0]) #frequencies for bloc
        b_stim.append(np.full_like(tt['pb_0'][1], 0))
        type_stim.append(np.full(len(tt['pb_0'][1]), 1))
        
        # Mock on verra plus tard 
        #mock_stim.append(tt['mk_'+str(bloc)][1]) #time triggers for bloc 
        #f_mock_stim.append(tt['mk_'+str(bloc)][0]) #frequencies for bloc
       # b_mock_stim.append(np.full_like(tt['mk_'+str(bloc)][1], bloc))
       # mock_type_stim.append(np.full(len(tt['mk_'+str(bloc)][1]), 'mock'))
        
    t_stim = np.hstack(t_stim)
    f_stim = np.hstack(f_stim)
    b_stim = np.hstack(b_stim)
    type_stim = np.hstack(type_stim)
    
    unique_tones = sorted(np.unique(f_stim))

    #mock_stim = np.hstack(mock_stim)
    #f_mock_stim=np.hstack(f_mock_stim)

    t_stim = t_stim/fs
    #mock_stim=mock_stim/fs
    

    #need to interpolate between two stims to get the frequency in between
    # 1. True stims
    stimulus_presence = np.zeros(len(bins) - 1, dtype=bool)
    interpolated_freq = np.zeros(len(bins) - 1)

    previous_frequency = None
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Check if any stimuli fall within the current bin
        stimuli_in_bin = (t_stim >= bin_start) & (t_stim < bin_end)
        
        if np.any(stimuli_in_bin):
            # If stimuli are present, set stimulus_presence to True for this bin
            stimulus_presence[i] = True

            # Calculate the frequency associated with the bin (assuming frequency remains constant within the bin)
            # You can simply take the frequency of the first stimulus within the bin
            interpolated_freq[i] = f_stim[stimuli_in_bin][0]
            previous_frequency = interpolated_freq[i]  # Update previous frequency
        else:
            # If no stimulus in the bin, set bin_frequencies to the previous frequency
            if previous_frequency is not None:
                interpolated_freq[i] = previous_frequency
                
    interpolated_blocks = np.interp(bins, t_stim, b_stim)
    interpolated_type_stim = np.interp(bins, t_stim, type_stim)


    
    # Create a dictionary to store information for each time bin
    features = {}
    for i, bin in enumerate(bins[:-1]):
        features[bin] = {
            'Played_frequency': interpolated_freq[i],
            'Block': interpolated_blocks[i],
            'Condition': interpolated_type_stim[i],
            'Frequency_changes': stimulus_presence[i]
        }
        
        
        
    features = list(features.values())
    np.save(path+'/features.npy', features)
    
    np.save(path+'/unique_tones.npy', unique_tones)

    #with open(path+'/features.json', 'w') as json_file:
        #json.dump(features, json_file)   
        
        
    print('all izz well')