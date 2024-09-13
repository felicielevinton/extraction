import PostProcessing.tools.accelerometer as au
import PostProcessing.tools.positions as po
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(prog="Accelerometer")
    parser.add_argument("--folder", type=str, help="Path to folder having data.")
    _opt = parser.parse_args()
    return _opt


if __name__ == "__main__":
    folder = "C:/Users/Flavi/data/EXPERIMENT/MANIGODINE/MANIGODINE_20221107/MANIGODINE_20221107_SESSION_00"
    data = au.Accelerometer(folder)
    out, positions = po.load_positions_files(folder)

