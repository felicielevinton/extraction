from kneed import DataGenerator, KneeLocator
from quick_extract import *
from get_data import *
from load_rhd import *
import matplotlib.pyplot as plt
from ExtractRecordings.manual.simple_sort import*
import pandas as pd
from PostProcessing.tools.utils import *
import csv
from format_data import *
import pandas as pd
from create_data import *
import os
import scipy.io
from delta_frequency import *
import math
from utils import *
import argparse
import PostProcessing.tools.heatmap as hm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.colors import ListedColormap
from skimage import measure
import matplotlib.colors as colors
    

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
        heatmaps.append(heatmap)
        np.save(save_name, np.array(heatmaps))
    
    return heatmaps





def old_plot_tonotopy(data, features, t_pre, t_post, bin_width, good_clusters, unique_tones, max_freq, min_freq, condition, save_name):
    """""
    OLD VERSION
    Fonction qui pour une session renvoie
    les heatmaps (psth x freq) pour la tonotopie
    une heatmap par neurone
    uniquement les good_clusters
    
    input : data, features, t_pre, t_post (pour le psth), bins, good_clusters et condition ("tracking" ou "playback)
            unique_tones : ce sont les tons uniques qui ont été joués pendant la session (33 en tout)
            max_freq, min_freq : indices min et max des fréquences extrêmes à partir desquelles on ne prend pas les psth pour les heatmap
            (car pas assez de présentations donc ca déconne) min_freq = 5, max_freq = 7
            condition : 'tracking' ou 'playback'
    ouput : 1 heatmap par good_cluster
    """
    #je prends les psth de chaque neurones et la fréquence associée à chaque psth
    psth = get_psth(data, features, t_pre, t_post, bin_width, good_clusters, condition)
    tones = get_played_frequency(features, condition)

    
    
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    
    n_clus = len(good_clusters)

    print(n_clus)
    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
    
    print(num_columns)
    
    num_rows = -(-len(good_clusters) // num_columns) 

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 25))
    fig.suptitle(f'Heatmaps clusters in {condition}', y=1.02)

    tones = np.array(tones)
    unique_tones_test = np.unique(tones)
    
    all_clus = []

    for clus, ax in enumerate(axes.flatten()):  # Use flatten to iterate over the 1D array
        try:
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
            mu = np.nanmean(average_psths_array[:][0:t_0], axis=0)
            mu = np.nanmean(mu, axis=0)
            
            #je retire la moyenne de la heatmap avant le stim
            #trouver le bin du stim
            
            
            heatmap = average_psths_array[min_freq:-max_freq]-mu
            #heatmap = average_psths_array-mu
            
            all_clus.append(heatmap)

            #blue_white_red = LinearSegmentedColormap.from_list('blue_white_red', [(0, 'blue'), (0.5, 'white'), (1, 'red')])
            smooth_heatmap = smooth_2d(heatmap, 3)
            sns.heatmap(smooth_heatmap, cmap=create_colormap(), yticklabels=unique_tones[min_freq:-max_freq], ax=ax,
                        cbar_kws={'label': 'PSTH Mean'})
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(f'Cluster {good_clusters[clus]}')
            ax.invert_yaxis()
        except:
            #all_clus.append(np.full_like(heatmap, np.nan))
            pass

    # Adjust layout
    plt.tight_layout()
    plt.savefig(save_name)
    # Show the figure
    plt.show()
    
    return all_clus

def create_colormap():
    r = np.hstack((np.ones(50), np.linspace(1, 0, 50)))
    g = np.hstack((np.linspace(0, 1, 50), np.linspace(1, 0, 50)))
    b = np.hstack((np.linspace(0, 1, 50), np.ones(50)))
    rgb = np.vstack((b, g, r)).transpose()
    cmap = ListedColormap(rgb, "yves")
    return cmap

def get_contour(hm, threshold):
    """"
    Fonction qui détermine les contours d'une heatmap
    
    input : une heatmap d'un cluster (hm), threshold : sensibilité pour la détection de contour
    output: renvoie les coordonnées du/des contour(s)
    """
    # Find contours in the heatmap
    contours = measure.find_contours(hm, level=threshold)

    return contours


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

def old_get_bandwidth(heatmap,threshold, gc,unique_tones, min_freq, max_freq, path, folder, condition):
    """""
    OLD VERSION
    Best function pour déterminer la bandwidth
    input : une heatmap d'un cluster, le threshold pour la detection du pic, good_clusters
            condition : "tracking" ou 'playback'
    output : Save un tableau bandwidth.npy qui comprend pour
        chaque cluster [freq min, freq max] de la bandwidth en fréquence
    """
    # pour les plots:
    n_clus = len(gc)
    
    #
    bandwidth = []
    for cluster in range(n_clus):
        #print(cluster)
        hm, peak = detect_peak(heatmap, cluster)
        #plt.pcolormesh(hm, cmap=create_colormap())
        #plt.yticks(np.arange(len(unique_tones[min_freq:-max_freq])), unique_tones[min_freq:-max_freq])
        contours = get_contour(hm, threshold)
        #Je ne prends la réponse qu'entre 40 et 60ms
        max = 0
        min = len(unique_tones[min_freq:-max_freq])-1
        for contour in contours:
            if ((contour[:, 1] > 40).all() and (contour[:, 1] < 60).all()):
                #plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')
                mc = np.max(contour[:, 0])
                minc = np.min(contour[:, 0])
                if mc>max:
                    max = mc
                if minc<min:
                    min = minc
        plotted_freq = unique_tones[min_freq:-max_freq]
        #print(plotted_freq[int(min)], plotted_freq[int(max)])
        #plt.close()
        bandwidth.append([plotted_freq[int(min)], plotted_freq[int(max)]])
    np.save(path+folder+f'/bandwidth_{condition}.npy', bandwidth)
    return ('all izz well')





def old_plot_bandwidth(heatmap,threshold, gc,unique_tones, min_freq, max_freq, path, folder, condition):
    """""
    OLD VERSION
    Best function pour déterminer la bandwidth
    input : une heatmap d'un cluster, le threshold pour la detection du pic, good_clusters
    output : plot des heatmap avec la bandwidth entourée 
    """
    
    # pour les plots:
    n_clus = len(gc)
    if est_premier(n_clus):
        n_clus=n_clus-1

    print(n_clus)
    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
    
    #print(num_columns)
    
    num_rows = -(-n_clus // num_columns) 

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(23, 30))
    fig.suptitle('Heatmaps clusters', y=1.02)
    plt.subplots_adjust() 
    
    
    #
    bandwidth = []
    for cluster, ax in enumerate(axes.flatten()):
        #print(cluster)
        heatmap_cluster = np.array(heatmap[cluster])
        hm, peak = detect_peak(heatmap, cluster)
        #heatmap_min = np.min(heatmap_cluster)
        #heatmap_max = np.max(abs(heatmap_cluster))
        #abs_max = max(abs(heatmap_min), abs(heatmap_max))
        abs_max = np.max(abs(heatmap_cluster))
    # Normalize the colormap for the current cluster
    
    
        img = ax.pcolormesh(heatmap_cluster, cmap=create_centered_colormap(abs_max), vmin=-abs_max, vmax=abs_max)
        ax.set_yticks(np.arange(len(unique_tones[min_freq:-max_freq])), unique_tones[min_freq:-max_freq])
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Cluster {gc[cluster]}')
    
        contours = get_contour(hm, threshold)
        #Je ne prends la réponse qu'entre 40 et 60ms
        max = 0
        min = len(unique_tones[min_freq:-max_freq])-2
        for contour in contours:
            #if ((contour[:, 1] > 40).all() and (contour[:, 1] < 60).all()):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='green')
            mc = np.max(contour[:, 0])
            minc = np.min(contour[:, 0])
            if mc>max:
                max = mc
            if minc<min:
                min = minc

        plotted_freq = unique_tones[min_freq:-max_freq]
        #print(plotted_freq[int(min)], plotted_freq[int(max)])
        
        bandwidth.append([plotted_freq[int(min)], plotted_freq[int(max)]])
        #cbar_ax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        #fig.colorbar(img, cax=cbar_ax)
    plt.tight_layout()  
    plt.savefig(path+folder+f'/heatmap_bandwidth_{condition}.png')
    return ('all izz well')


def create_centered_colormap(x):
    """
    Pour créer une colormap centrée en zero (zero = blanc)
    """
    blue_white_red = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    
    mid = 0.5
    # Adjust the colorscale based on the maximum and minimum values
    #blue_white_red = [(c[0] * (x + mid), c[1]) if c[0] != mid else (mid, c[1]) for c in colorscale]

    return LinearSegmentedColormap.from_list('custom_colormap', blue_white_red)


def plot_heatmap_bandwidth_avant_derniere(heatmaps,threshold, gc,unique_tones, min_freq, max_freq, path, folder, bin_width, psth_bins, t_pre, condition):
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
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(24, 18))
    fig.suptitle('Heatmaps clusters', y=1.02)
    plt.subplots_adjust() 
    
     # Flatten the axis array if it's more than 1D
    if num_rows > 1 and num_columns > 1:
        axes = axes.flatten()
    #
    bandwidth = []
    plotted_heatmap = []
    peaks = []
    for cluster, ax in enumerate(axes):
        if cluster < num_plots:
            #print(cluster)
            heatmap_cluster = np.array(heatmaps[cluster])
            hm, peak = detect_peak(heatmaps, cluster)
            #heatmap_min = np.min(heatmap_cluster)
            #heatmap_max = np.max(abs(heatmap_cluster))
            #abs_max = max(abs(heatmap_min), abs(heatmap_max))
            #abs_max = np.max(abs(heatmap_cluster[min_freq:-max_freq]))
            abs_max = np.max(abs(heatmap_cluster[3:-3]))
            contours = get_contour(hm, threshold)
            
        # Je retire la moyenne pre-stim ligne par ligne (fréquence par fréquence)
            t_0 = int(t_pre/bin_width)
            prestim_hm = heatmap_cluster[:, :t_0]
            mean_freq = np.mean(prestim_hm, axis=1)

            for i in range(heatmap_cluster.shape[0]):  # Parcours des lignes de A
                heatmap_cluster[i] -= mean_freq[i]
            
            
            smoothed = smooth_2d(heatmap_cluster, 3)
            
            #je mets des zeros aux frequences trop hautes et trop basses où je n'ai pas
            #assez de présentations
            lowf = np.zeros((min_freq+1, len(psth_bins[:-1])))
            highf = np.zeros((max_freq+1, len(psth_bins[:-1])))
            
            milieu = np.concatenate((lowf, smoothed[min_freq:-max_freq]))

            # Concaténation à l'arrière
            milieu = np.concatenate((milieu, highf))
            
            
            img = ax.pcolormesh(milieu, cmap=create_centered_colormap(), vmin=-abs_max, vmax=abs_max)
            ax.set_yticks(np.arange(len(unique_tones)), unique_tones)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(f'Cluster {gc[cluster]}')
            ax.axvline(x=t_0, color='black', linestyle='--') # to print a vertical line at the stim onset time
        

            #Je ne prends la réponse qu'entre 40 et 60ms
            #max = 0
            #min = len(unique_tones[min_freq:-max_freq])-2
            max_length =  0    
            x_c, y_c, minf, maxf = np.nan, np.nan, 0,0 # au cas où on trouve pas de contour
            for contour in contours:
                if ((contour[:, 1] > 40).all() and (contour[:, 1] < 60).all()):
                    if len(contour[:, 0])>max_length:
                        max_length = len(contour[:, 0])
                        x_c = contour[:, 1]
                        y_c = contour[:, 0]
                        maxf = np.max(contour[:, 0])
                        minf = np.min(contour[:, 0])
                        test = contour[:, 0]
                        if maxf<len(unique_tones)-1:
                            maxf+=1
            ax.plot(x_c, y_c, linewidth=2, color='green')
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
    for ax in axes[num_plots:]:
        ax.axis('off')
    plt.tight_layout()  
    plt.savefig(path+folder+f'/heatmap_bandwidth_{condition}.png') # save the figure of the heatmap
    np.save(path+folder+f'/heatmap_bandwidth_{condition}.npy', bandwidth) # save the values of the bandwidth
    np.save(path+folder+f'/heatmap_plot_{condition}.npy',plotted_heatmap ) # save the values of the heatmap as it is plotted 
    np.save(path+folder+f'/best_frequency_{condition}.npy', peaks)
    return ('all izz well')


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
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('Heatmaps clusters', y=1.02)
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
            
            
            smoothed = smooth_2d(heatmap_cluster, 3)
            
            #je mets des zeros aux frequences trop hautes et trop basses où je n'ai pas
            #assez de présentations
            lowf = np.zeros((min_freq+1, len(smoothed[0])))
            highf = np.zeros((max_freq+1, len(smoothed[0])))
            
            milieu = np.concatenate((lowf, smoothed[min_freq:-max_freq]))

            # Concaténation à l'arrière
            milieu = np.concatenate((milieu, highf))
            
            img = axes[row, col].pcolormesh(milieu, cmap=create_centered_colormap(abs_max), vmin=-abs_max, vmax=abs_max)
            #axes[row, col].set_yticks(np.arange(len(unique_tones)), unique_tones)
            axes[row, col].set_xlabel('Time')
            #axes[row, col].set_ylabel('Frequency [Hz]')
            axes[row, col].set_title(f'Cluster {gc[cluster]}')
            axes[row, col].axvline(x=t_0, color='black', linestyle='--') # to print a vertical line at the stim onset time
        

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
            axes[row, col].plot(x_c, y_c, linewidth=2, color='green')
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
    for ax in axes[num_plots:]:
        ax.axis('off')
    plt.tight_layout()  
    plt.savefig(path+folder+f'/heatmap_{condition}.png') # save the figure of the heatmap
    np.save(path+folder+f'/heatmap_bandwidth.npy', bandwidth) # save the values of the bandwidth
    np.save(path+folder+f'/heatmap_plot_{condition}.npy',plotted_heatmap ) # save the values of the heatmap as it is plotted 
    np.save(path+folder+f'/best_frequency_{condition}.npy', peaks)
    return ('all izz well')








def plot_inout(heatmaps, bandwidth, gc,unique_tones, min_freq, max_freq, psth_bins, path, folder, condition):
    """""
    Best function pour plotter, par cluster, les psth in et out la bandwidth
    input : une heatmap d'un cluster, le threshold pour la detection du pic, good_clusters
    output : plot des heatmap avec la bandwidth entourée 
    """
    # pour les plots:
    n_clus = len(gc)
    if est_premier(n_clus):
        n_clus=n_clus-1

    print(n_clus)
    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
    
    #print(num_columns)
    
    num_rows = -(-n_clus // num_columns) 

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(24, 48))
    fig.suptitle('Heatmaps clusters', y=1.02)
    plt.subplots_adjust() 
    
    
    
    for cluster, ax in enumerate(axes.flatten()):
        hm = heatmaps[cluster]
        lowf, highf = find_indexes(unique_tones, bandwidth[cluster][0],bandwidth[cluster][1] )
        lowf, highf = lowf[0], highf[0]
        in_bd = [hm[i] for i in range(lowf, highf)]
        #out_bd = None
        #out_bd_low = [hm[i] for i in range(0, lowf) ] 
        #out_bd_up = ([hm[i] for i in range(highf+1, len(hm))] )
        
        #ici j'enlève HF et BF
        out_bd_low = [hm[i] for i in range(min_freq, lowf) ] 
        out_bd_up = ([hm[i] for i in range(highf+1, len(hm)-max_freq)] )
        out_bd_low.extend(out_bd_up)

        if len(in_bd)==0:
            pass
        else : 
            ax.plot(psth_bins[:-1], np.nanmean(in_bd, axis=0), color='red')
        ax.set_title(f'Cluster {gc[cluster]}')
        if len(out_bd_low)==0:
            pass
        else : 
            ax.plot(psth_bins[:-1], np.nanmean(out_bd_low, axis=0), color='blue')
    return in_bd, out_bd_low



def find_indexes(tableau, a, b):
    """
    Fonction utile pour inout

    Args:
        tableau (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        les indices de a et b dans le tableau 
    """
    indices_a = []
    indices_b = []

    for i in range(len(tableau)):
        if tableau[i] == a:
            indices_a.append(i)
        elif tableau[i] == b:
            indices_b.append(i)

    return indices_a, indices_b

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


    
def plot_heatmap(heatmaps, psth_bins, gc, unique_tones, min_freq, max_freq, title):
    """"
    fonction uniquement pour plotter des heatmaps # pas sure de son utilité
    input : heatmaps au format .npy, psth_bins, gc = good_clusters, unique_tones et un titre pour le plot
    output : un plot
    """
    n_clus = len(gc)
    if est_premier(n_clus):
        n_clus=n_clus-1

    print(n_clus)
    num_columns = 4 
    if n_clus % 5 == 0:
        num_columns = 5
    elif n_clus % 3 == 0:
        num_columns = 3
    elif n_clus % 4 != 0:
        num_columns = 2
    
    # indices de l'onset dans psth_bins
    onset = np.argmin(np.abs(psth_bins))
    
    num_rows = -(-n_clus // num_columns) 

    # Create a figure with subplots
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(23, 30))
    fig.suptitle('Heatmaps clusters', y=1.02)
    plt.subplots_adjust() 
    
    for cluster, ax in enumerate(axes.flatten()):
        #print(cluster)
        heatmap_cluster = np.array(heatmaps[cluster])
        abs_max = np.max(abs(heatmap_cluster))
    
        img = ax.pcolormesh(heatmap_cluster, cmap=create_centered_colormap(abs_max), vmin=-abs_max, vmax=abs_max)
        ax.set_yticks(np.arange(len(unique_tones[min_freq:-max_freq])), unique_tones[min_freq:-max_freq])
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(f'Cluster {gc[cluster]}')
            #Je ne prends la réponse qu'entre 40 et 60ms
        max = 0
        min = len(unique_tones[min_freq:-max_freq])-2
        ax.axvline(x=onset, color='black', linestyle='--', linewidth=1.5)
        plotted_freq = unique_tones[min_freq:-max_freq]
    plt.tight_layout()  
    return ('all izz well')


def get_heatmaps_swipe_up(data, features, gc, condition, unique_tones, t_pre, t_post, bin_width, path, folder ):
    
    if condition=='tracking':
        c = 0
    else:
        c = 1
        
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    heatmaps_up = []
    for cluster in range(len(gc)):
        print(cluster)
        hm=[]
        for frequency in  unique_tones:
            response = []
            for bin_idx, feature in enumerate(features):
                if feature['Frequency_changes'] > 0 and feature['Condition'] == c and feature['Played_frequency'] == frequency:
                    if features[bin_idx-1]['Played_frequency']<features[bin_idx]['Played_frequency']:
                        response.append(data[cluster][bin_idx-int(t_pre/bin_width):bin_idx+int(t_post/bin_width)])
            if len(response)==0:
                response = [[np.nan]*len(psth_bins[:-1])]*2
            hm.append(np.nanmean(response, axis=0))
        heatmaps_up.append(hm)
    np.save(path+folder+f'/heatmaps_up_{condition}.npy',heatmaps_up )
    return heatmaps_up

def get_heatmaps_swipe_down(data, features, gc, condition, unique_tones, t_pre, t_post, bin_width, path, folder ):
    
    if condition=='tracking':
        c = 0
    else:
        c = 1
        
    psth_bins = np.arange(-t_pre, t_post + bin_width, bin_width)
    heatmaps_down = []
    for cluster in range(len(gc)):
        print(cluster)
        hm=[]
        for frequency in  unique_tones:
            response = []
            for bin_idx, feature in enumerate(features):
                if feature['Frequency_changes'] > 0 and feature['Condition'] == c and feature['Played_frequency'] == frequency:
                    if features[bin_idx-1]['Played_frequency']>features[bin_idx]['Played_frequency']:
                        response.append(data[cluster][bin_idx-int(t_pre/bin_width):bin_idx+int(t_post/bin_width)])
            if len(response)==0:
                response = [[np.nan]*len(psth_bins[:-1])]*2
            hm.append(np.nanmean(response, axis=0))
        heatmaps_down.append(hm)
    np.save(path+folder+f'/heatmaps_down_{condition}.npy',heatmaps_down )
    return heatmaps_down


def get_mock_tonotopy(data, features, t_pre, t_post, bin_width, good_clusters, unique_tones, max_freq, min_freq, save_name):
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
    psth = get_psth(data, features, t_pre, t_post, bin_width, good_clusters, 'playback')
    tones = get_mock_frequency(features)

    
    
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
        heatmaps.append(heatmap)
        np.save(save_name, np.array(heatmaps))
    
    return heatmaps


def get_baseline_per_tones(psth_tracking, psth_playback, f_tracking, f_playback, gc, unique_tones, t_pre, t_post, bin_width):
#num_rows, num_columns = get_plot_geometry(gc)
#fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(24, 48))   
    all_tr, all_pb=[], [] 
    for cluster in range(len(gc)):    
        baseline, baseline_tr = [], []
        for f in unique_tones:
            indexes = get_indexes(f_playback, f)
            f_psth = [psth_playback[cluster][i] for i in indexes]
            f_mean_psth = np.nanmean(f_psth, axis=0)
            f_baseline = get_sustained_activity(f_mean_psth, t_pre, t_post, bin_width)
            baseline.append(f_baseline)
            
            #tracking
            indexes_tr = get_indexes(f_tracking, f)
            f_psth_tr = [psth_tracking[cluster][i] for i in indexes_tr]
            f_mean_psth_tr = np.nanmean(f_psth_tr, axis=0)
            f_baseline_tr = get_sustained_activity(f_mean_psth_tr, t_pre, t_post, bin_width)
            baseline_tr.append(f_baseline_tr)
        all_tr.append(baseline_tr)
        all_pb.append(baseline)
    return all_tr, all_pb