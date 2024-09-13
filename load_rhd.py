import os
import numpy as np
import argparse
from ExtractRecordings.load_intan_rhd_format.intanutil import load_intan_rhd_format as load
#from sorting.spike_sorter import filtering


def parse_args():
    parser = argparse.ArgumentParser(prog="LoadRHD")
    parser.add_argument("--path", type=str, help="Chemin d'accès vers le fichier RHD")
    parser.add_argument("--save_path", type=str, help="Nom de Sauvegarde.", default=None)
    parser.add_argument("--float", type=bool, help="Convertir en float.", default=False)
    parser.add_argument("--save", type=bool, help="Sauvegarder les données extraites", default=True)
    parser.add_argument("--filtered", type=bool, help="Data has been filtered.", default=True)
    parser.add_argument("--analog", type=bool, help="Obtenir les enregistrements analogiques", default=True)
    parser.add_argument("--digital", type=bool, help="Obtenir les enregistrements digitaux", default=True)
    parser.add_argument("--accelerometer", type=bool, help="Obtenir les données de l'accéléromètre", default=True)
    parser.add_argument("--dat", type=bool, help="Vers un fichier .dat", default=False)
    return parser.parse_args()


def load_rhd(path, save_path, digital=True, analog=True, accelerometer=True, filtered=True, export_to_dat=False):

    # ajouter filtrage et common average
    a = load.read_data(path, data_in_float=False)
    #save_path = os.path.split(path)[0]
    if filtered:
        fn = os.path.join(save_path, "filtered_neural_data.{}")
    else:

        fn = os.path.join(save_path, "neural_data.{}")
    if not export_to_dat:
        fn = fn.format("npy")
        np.save(fn, a["amplifier_data"])
    else:
        fn = fn.format("dat")
        a["amplifier_data"].tofile(fn, sep="", format="%U")
    if digital:
        np.save(os.path.join(save_path, "dig_in.npy"), a["board_dig_in_data"])
    if analog:
        np.save(os.path.join(save_path, "analog_in.npy"), a["board_adc_data"])
    if accelerometer:
        np.save(os.path.join(save_path, "accelerometer.npy"), a["aux_input_data"])
    return fn


if __name__ == "__main__":
    options = parse_args()
    load_rhd(options.path, options.digital, options.analog, options.accelerometer, options.filtered, options.dat)
