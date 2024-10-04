
import matplotlib.pyplot as plt
import pandas as pd
from PostProcessing.tools.utils import *
import csv
from format_data import *
import pandas as pd
import os
import scipy.io
import math
from utils import *
import argparse
import PostProcessing.tools.heatmap as hm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import ListedColormap
from skimage import measure
import matplotlib.colors as colors
from utils import *

def smooth_2d(heatmap, n):
    """"
    input : a heatmap, n (size of the kernel)
    output : value of a heatmap that is smoothed
    
    """
    hm = np.copy(heatmap)
    hm = cv.GaussianBlur(hm, (n, n), 0)
    return hm


def get_tonotopy(data, features, t_pre, t_post, bin_width, good_clusters, unique_tones, max_freq, min_freq, condition, save_name):
    """""
    
    Fonction qui pour une session renvoie
    les heatmaps (psth x freq) pour la tonotopie mais ne les plot pas
    une heatmap par neurone
    uniquement les good_clusters
    attention : les heatmaps sont brutes (pas de traitements, ni smoothed... etc)
    
    input : data, features, t_pre, t_post (pour le psth), bins, good_clusters et condition ("tracking" ou "playback)
            unique_tones : ce sont les tons uniques qui ont été joués pendant la session (33 en tout)
            max_freq, min_freq : indices min et max des fréquences extrêmes à partir desquelles on ne prend pas les psth pour les heatmap
            (car pas assez de présentations donc ca déconne) min_freq = 5, max_freq = 7
            condition : 'tracking' ou 'playback
    ouput : 1 tableau contenant 1 heatmap par good_cluster 
            heatmap non smoothée
    """
    
    #je prends les psth de chaque neurones et la fréquence associée à chaque psth
    psth = get_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition)
    tones = get_played_frequency(features, t_pre, t_post, bin_width, condition)

    
    
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    n_clus = len(good_clusters)
    

    tones = np.array(tones)
    unique_tones_test = np.unique(tones)
    
    heatmaps = []

    for clus in range(n_clus):  
        clus_psth = np.array(psth[clus])
        average_psth_list = []
        
        for tone in unique_tones:
            mask = (tones == tone)
            if len(clus_psth[mask])>0: #au moins 20 présentations d'une fréquence
                average_psth = np.mean(clus_psth[mask], axis=0)
                average_psth_list.append(average_psth)
            else:
                average_psth_list.append(np.zeros_like(psth_bins[:-1]))
    
        average_psths_array = np.array(average_psth_list)
        
        t_0 = int(t_pre/bin_width)
        # faire la moyenne sur toute la heatmap
        #mu = np.nanmean(average_psths_array[:][0:t_0], axis=0)
        #mu = np.nanmean(mu, axis=0)
        
        #je retire la moyenne de la heatmap avant le stim
        #trouver le bin du stim
        
        
        #heatmap = average_psths_array[min_freq:-max_freq]-mu
        #heatmap = average_psths_array-mu
        heatmap = average_psths_array
        #centrer en 0
        n = 9 #n = nombre de noyaux de flou gaussien, doit être un nombre impair positif
        heatmap = smooth_2d(heatmap, n) 
        heatmaps.append(heatmap)
        np.save('spike_sorting_' + save_name, np.array(heatmaps))
    
    return heatmaps

def detect_peak(hm, cluster):
    """"
    input: hm heatmaps , cluster numéro du cluster
    output : 
    """
    hm = hm[cluster]
    n = 3
    kernel = np.outer(signal.windows.gaussian(n, n), signal.windows.gaussian(n, n))
    hm_mask = np.empty_like(hm)
    for i in range(hm.shape[0]):
        if hm[i].std() == 0:
            hm_mask[i] = hm[i]
        else:
            hm_mask[i] = (hm[i] - hm[i].mean()) / hm[i].std()
    hm_mask = signal.convolve(hm_mask, kernel, "same")
    hm_mask -= hm_mask.mean()
    hm_mask /= hm_mask.std()
    hm_clean = np.copy(hm_mask)
    idx = np.logical_and(hm_mask > -3, hm_mask < 3)
        # hm_mask = np.where(hm_mask >= 3, 0, hm_mask)  # todo: détection des creux aussi.
    hm_mask[idx] = 0
    fp = findpeaks(method='topology', scale=True, denoise=10, togray=True, imsize=hm.shape[::-1], verbose=0)
    res = fp.fit(hm_mask)
    peak_position = res["groups0"][0][0]  # tuple qui indique la position du pic.
    return hm_clean, peak_position


def get_contour(hm, threshold):
    """"
    Fonction qui détermine les contours d'une heatmap
    
    input : une heatmap d'un cluster (hm), threshold : sensibilité pour la détection de contour
    output: renvoie les coordonnées du/des contour(s)
    """
    # Find contours in the heatmap
    contours = measure.find_contours(hm, level=threshold)

    return contours


def get_plot_coords(channel_number,n_rows, n_cols):
    """
    Fonction qui calcule la position en 2D d'un canal sur une Microprobe.
    Retourne la ligne et la colonne.
    """
    row = channel_number // n_cols  # Division entière pour obtenir la ligne
    col = channel_number % n_cols   # Reste de la division pour obtenir la colonne

    # Vérification que le canal est bien dans les limites du tableau
    if row >= n_rows:
        raise ValueError("Le numéro du canal dépasse le nombre de lignes disponibles dans le tableau.")

    return row, col






def plot_heatmap_bandwidth(heatmaps, threshold, good_cluster, cluster_order, count_dict, k, unique_tones, min_freq, max_freq, bin_width, psth_bins, t_pre, path, folder, condition):
    """
    Best function pour déterminer la bandwidth et plotter la heatmap et les contours de la bandwidth
    input : heatmaps(contenant plusieurs clusters), le threshold pour la detection du pic, good_clusters
        unique_tones (les fréquences jouées), min_freq, max_freq : les indices des fréquences qu'on exclut (pas assez de présentations)
        condition : 'tracking' ou 'playback'
    output : save plot des heatmap avec la bandwidth entourée .png
            save tableau des heatmaps telles que plottée (avec les psth) .npy
            save tableau contenant les bandwidth de chaque cluster .npy
    """
    
    # pour les plots:
    num_plots, num_rows, num_columns = get_better_plot_geometry(cluster_order, good_cluster)
    print((num_plots, num_rows, num_columns))

    # Create a figure with subplots
    fig_width = 4 * num_columns  # Ajustez selon vos besoins
    fig_height = 4 * num_rows  # Ajustez selon vos besoins

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(fig_width, fig_height))
    fig.suptitle(f'Heatmaps clusters {condition}', y=1.02)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.subplots_adjust() 
    
    bandwidth = []
    plotted_heatmap = []
    peaks = []
    
    sorted_k = k[np.lexsort((k[:, 1], k[:, 0]))]
    unique_values, indices = np.unique(sorted_k[:, 0], return_inverse=True)
    sorted_k[:, 0] = indices  # Les valeurs des channels ramenées à leur indice de ligne

    for cluster in range(num_plots):
        row, col = int(sorted_k[cluster][0]), int(sorted_k[cluster][1])
        cluster_key = tuple(k[cluster])

        # Ajouter la condition pour ne plotter que si count_dict[cluster_key] > 10000
        if count_dict[cluster_key] > 10000:
            print(f"Plotting heatmap for cluster {cluster} with count {count_dict[cluster_key]}")
            heatmap_cluster = np.array(heatmaps[cluster])
            hm, peak = detect_peak(heatmaps, cluster)
            abs_max = np.max(abs(heatmap_cluster[3:-3])) * 0.4
            contours = get_contour(hm, threshold)

            t_0 = int(t_pre / bin_width)
            prestim_hm = heatmap_cluster[:, :t_0]
            mean_freq = np.mean(prestim_hm, axis=1)

            for i in range(heatmap_cluster.shape[0]):
                heatmap_cluster[i] -= mean_freq[i]
            
            smoothed = smooth_2d(heatmap_cluster, 5)

            vmin = -3 * np.std(smoothed)
            vmax = 3 * np.std(smoothed)

            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            img = axes[row, col].pcolormesh(smoothed, cmap='seismic', norm=norm)
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_title(f'Cluster {k[cluster]} nbr {count_dict[cluster_key]}')
            axes[row, col].axvline(x=t_0, color='black', linestyle='--')

            max_length = 0    
            x_c, y_c, minf, maxf = np.nan, np.nan, 0, 0

            for contour in contours:
                if ((contour[:, 1] > t_0 - 5).all() and (contour[:, 1] < t_0 + 10).all()):
                    if len(contour[:, 0]) > max_length:
                        max_length = len(contour[:, 0])
                        x_c = contour[:, 1]
                        y_c = contour[:, 0]
                        maxf = np.max(contour[:, 0])
                        minf = np.min(contour[:, 0])
                        if maxf < len(unique_tones) - 1:
                            maxf += 1

            if max_length == 0 or maxf == 0:
                bandwidth.append([np.nan, np.nan])
                peaks.append(np.nan)
            else:
                bandwidth.append([unique_tones[int(minf)], unique_tones[int(maxf)]])
                peaks.append(unique_tones[peak[0]])

            plotted_heatmap.append(smoothed)
        else:
            print(f"Skipping cluster {cluster} with count {count_dict[cluster_key]} (too small)")

    plt.tight_layout()  
    plt.savefig(path + folder + f'/heatmap_{condition}.png')  # save the figure of the heatmap
    np.save(path + folder + f'/heatmap_bandwidth.npy', bandwidth)  # save the values of the bandwidth
    np.save(path + folder + f'/heatmap_plot_{condition}.npy', plotted_heatmap)  # save the values of the heatmap as it is plotted 
    np.save(path + folder + f'/best_frequency_{condition}.npy', peaks)

    return 'all izz well'
