import Analyse.PostProcessing.analysis as nt
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    folder = "C:/data/experiment/MANIGODINE/MANIGODINE_20220712/MANIGODINE_20220712_SESSION_00"
    session = nt.load_session(folder)
    cell = session.cells[4]
    isi, bins = cell.isi(bin_size=0.001)
    plt.plot(cell.auto_correlation(num_lags=30))
    plt.show()
    plt.semilogy(isi)
    plt.show()

