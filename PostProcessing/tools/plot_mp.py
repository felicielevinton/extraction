import matplotlib.pyplot as plt
import os
import numpy as np


def plot_mp_fma_32_sd(psth, tag, folder, cmap="bwr", l_ex=None, r_ex=None):
    # order = np.arange(32)
    if r_ex > 0:
        r_ex *= -1
    fig, axes = plt.subplots(4, 8)
    plt.title(f"Heatmap {tag}")

    for i in range(8):
        axes[3, i % 8].pcolormesh(psth[i][l_ex:r_ex], cmap=cmap)
        axes[3, i % 8].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

    for i in range(8, 16):
        axes[1, 7 - i % 8].pcolormesh(psth[i][l_ex:r_ex], cmap=cmap)
        axes[1, 7 - i % 8].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

    for i in np.arange(16, 24):
        axes[0, 7 - i % 8].pcolormesh(psth[i][l_ex:r_ex], cmap=cmap)
        axes[0, 7 - i % 8].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

    for i in range(24, 32):
        axes[2, i % 8].pcolormesh(psth[i][l_ex:r_ex], cmap=cmap)
        axes[2, i % 8].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

    for axe in axes:
        for ax in axe:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

    plt.savefig(os.path.join(folder, f"heatmap_{tag}.png"), dpi=240)
    plt.close()
