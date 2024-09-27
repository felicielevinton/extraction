
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


def get_plot_coords(channel_number):
    """
    Fonction qui calcule la position en 2D d'un canal sur une Microprobe.
    Retourne la ligne et la colonne.
    """
    if channel_number in list(range(8)):
        row = 3
        col = channel_number % 8

    elif channel_number in list(range(8, 16)):
        row = 1
        col = 7 - channel_number % 8

    elif channel_number in list(range(16, 24)):
        row = 0
        col = 7 - channel_number % 8

    else:
        row = 2
        col = channel_number % 8

    return row, col





def plot_heatmap_bandwidth(heatmaps,threshold, gc,unique_tones, min_freq, max_freq, bin_width, psth_bins, t_pre,path, folder, condition):
    """""
    Best function pour déterminer la bandwidth et plotter la heatmap et les contours de la bandwidth
    input : heatmaps(contenant plusieurs clusters), le threshold pour la detection du pic, good_clusters
        unique_tones (les fréquences jouées), min_freq, max_freq : les indices des fréquences qu'on exclut (pas assez de présentations)
        condition : 'tracking' ou 'playback'
    output : save plot des heatmap avec la bandwidth entourée .png
            save tableau des heatmaps telles que plottée (avec les psth) .npy
            save tableau contenant les bandwidth de chaque cluster .npy
            
    """
    
    # pour les plots:

    #num_rows, num_columns = get_plot_geometry(gc)
    
    num_plots, num_rows, num_columns = get_better_plot_geometry(gc)


    # Create a figure with subplots
    #fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig, axes = plt.subplots(1, len(gc), figsize=(16, 8))
    fig.suptitle(f'Heatmaps clusters {condition}', y=1.02)
    plt.subplots_adjust() 
    
     # Flatten the axis array if it's more than 1D
    #if num_rows > 1 and num_columns > 1:
        #axes = axes.flatten()
    #
    bandwidth = []
    plotted_heatmap = []
    peaks = []
    for cluster in range(num_plots):
        if cluster < num_plots:
            row, col = get_plot_coords(cluster)
            #print(cluster)
            heatmap_cluster = np.array(heatmaps[cluster])
            hm, peak = detect_peak(heatmaps, cluster)
            #heatmap_min = np.min(heatmap_cluster)
            #heatmap_max = np.max(abs(heatmap_cluster))
            #abs_max = max(abs(heatmap_min), abs(heatmap_max))
            #abs_max = np.max(abs(heatmap_cluster[min_freq:-max_freq]))
            abs_max = np.max(abs(heatmap_cluster[3:-3]))*0.4
            contours = get_contour(hm, threshold)
            #j'essaye en prenant la absolute value de hm
            #contours = get_contour(np.abs(hm), threshold)
            
        # Je retire la moyenne pre-stim ligne par ligne (fréquence par fréquence)
            t_0 = int(t_pre/bin_width)
            prestim_hm = heatmap_cluster[:, :t_0]
            mean_freq = np.mean(prestim_hm, axis=1)

            for i in range(heatmap_cluster.shape[0]):  # Parcours des lignes de A
                heatmap_cluster[i] -= mean_freq[i]
            
            
            smoothed = smooth_2d(heatmap_cluster, 5)
            
            #je mets des zeros aux frequences trop hautes et trop basses où je n'ai pas
            #assez de présentations
            lowf = np.zeros((min_freq+1, len(smoothed[0])))
            highf = np.zeros((max_freq+1, len(smoothed[0])))
            
            milieu = np.concatenate((lowf, smoothed[min_freq:-max_freq]))

            # Concaténation à l'arrière
            milieu = np.concatenate((milieu, highf))


            # vmin = np.min(milieu)  # Valeur minimale dans ta matrice
            # vmax = np.max(milieu)  # Valeur maximale dans ta matrice

            vmin = -3 * np.std(milieu)  # Valeur minimale dans ta matrice
            vmax = 3 * np.std(milieu)  # Valeur maximale dans ta matrice

            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # Normalisation centrée sur 0
            #img = axes[row, col].pcolormesh(milieu, cmap=create_centered_colormap(abs_max), vmin=-abs_max, vmax=abs_max)
            #img = axes[row, col].pcolormesh(milieu, cmap='seismic', norm=norm)
            img = axes[cluster].pcolormesh(milieu, cmap='seismic', norm=norm)

            #axes[row, col].set_yticks(np.arange(len(unique_tones)), unique_tones)
            axes[cluster].set_xlabel('Time')
            #axes[row, col].set_ylabel('Frequency [Hz]')
            axes[cluster].set_title(f'Cluster {gc[cluster]}')
            axes[cluster].axvline(x=t_0, color='black', linestyle='--') # to print a vertical line at the stim onset time
        

            #Je ne prends la réponse qu'entre 40 et 60ms
            #max = 0
            #min = len(unique_tones[min_freq:-max_freq])-2
            max_length =  0    
            x_c, y_c, minf, maxf = np.nan, np.nan, 0,0 # au cas où on trouve pas de contour
            for contour in contours:
                if ((contour[:, 1] > t_0-5).all() and (contour[:, 1] < t_0+10).all()):
                    if len(contour[:, 0])>max_length:
                        max_length = len(contour[:, 0])
                        x_c = contour[:, 1]
                        y_c = contour[:, 0]
                        maxf = np.max(contour[:, 0])
                        minf = np.min(contour[:, 0])
                        test = contour[:, 0]
                        if maxf<len(unique_tones)-1:
                            maxf+=1
            #axes[row, col].plot(x_c, y_c, linewidth=2, color='green')
            print(x_c, y_c)
            #print(plotted_freq[int(min)], plotted_freq[int(max)])
            # je mets np.nan dans bandwidth si je ne trouve pas de contour
            if max_length==0 or maxf==0:
                bandwidth.append([np.nan, np.nan])
                peaks.append(np.nan)
            else : 
            
            #je prends +1 dans le maxf
                bandwidth.append([unique_tones[int(minf)], unique_tones[int(maxf)]])
                peaks.append(unique_tones[peak[0]])
            plotted_heatmap.append(milieu)
            #cbar_ax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
            #fig.colorbar(img, cax=cbar_ax)
        # Hide any unused subplots
    # for ax in axes[num_plots:]:
    #     ax.axis('off')
    plt.tight_layout()  
    plt.savefig(path+folder+f'/heatmap_{condition}.png') # save the figure of the heatmap
    np.save(path+folder+f'/heatmap_bandwidth.npy', bandwidth) # save the values of the bandwidth
    np.save(path+folder+f'/heatmap_plot_{condition}.npy',plotted_heatmap ) # save the values of the heatmap as it is plotted 
    np.save(path+folder+f'/best_frequency_{condition}.npy', peaks)
    return ('all izz well')