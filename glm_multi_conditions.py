import argparse
import os.path
import numpy as np
import PostProcessing.tools.heatmap as hm
import pandas as pd
import get_data as gd
from .GLM import glm
import PostProcessing.tools.utils as ut


def parse_args():
    parser = argparse.ArgumentParser(prog="GLM")
    parser.add_argument("--folders", type=str, nargs="+", help="Path to folder having data.")
    parser.add_argument("--save", type=str, help="Folder to save data.")
    parser.add_argument("--compatibility", type=bool, help="Old data.")
    opt = parser.parse_args()
    return opt


def glm_main(folders, save_folder):
    # les deux noms de fichiers à trouver -> alpha et lambdas et scores.
    params_files_glm = ["alphas.npy", "lambdas.npy", "good_clusters.npy"]
    plot_path = gd.check_plot_folder_exists(save_folder)
    for sess_num, folder in enumerate(folders):
        sequence = gd.extract_data(folder)
        for file in params_files_glm:
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                print(f"{file} is not in folder {os.path.split(folder)[-1]}. It is required, thus: ABORT.")
                break

        gc = np.load("good_clusters.npy")  # todo: idée, ajouter les bons clusters dans l'objet spikes.
        lambdas = np.load("lmabdas.npy")
        alphas = np.load("alphas.npy")
        train_set = sequence.get_xp_number("warmup", 0)
        spk = ut.Spikes(folder, recording_length=sequence.get_recording_length())

        # construire la DM du stim,

        # Itérer sur les neurones.

        # Entraîner

        # Extraire les tracking et les playback.


    return