import numpy as np
from tqdm import tqdm
import os
from quick_extract import quick_extract
from ExtractRecordings.manual import simple_sort as ss
from load_rhd import load_rhd
import argparse
from sorting.spike_sorter import filtering
from ExtractRecordings.load_intan_rhd_format.intanutil import load_intan_rhd_format as load


def parse_args():
    parser = argparse.ArgumentParser(prog="Extract")
    parser.add_argument("--path", type=str, help="Chemin d'accès vers le fichier .rhd")
    parser.add_argument("--mode", type=str, help="Méthode d'extraction.", default="relative")
    parser.add_argument("--dat", type=bool, help="Vers un fichier .dat", default=False)
    parser.add_argument("--threshold", type=float, default=-3.7)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_args()
    root_dir = os.path.split(options.path)[0]

    # 1. Convertir le RHD vers un point ".dat"
    fn = load_rhd(path=options.path, export_to_dat=True)

    # 2) étape de filtrage et CAR.

    # 3) Extraction des évènements.
    quick_extract(fn)
