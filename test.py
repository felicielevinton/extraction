import matplotlib.pyplot as plt
import argparse
import get_data as gd
import numpy as np
import os
from findpeaks import findpeaks
from findpeaks import stats
import PostProcessing.tools.utils as ut
import PostProcessing.tools.heatmap as hm
import PostProcessing.tools.accelerometer as acu
from scipy import signal
from contourpy import contour_generator
from contourpy.util.mpl_renderer import MplRenderer as Renderer
import cv2


def peak_and_contour_finding(heatmap):
    # trouver min vs max
    n = 5
    gb = cv2.GaussianBlur(heatmap, (n, n), 0)
    gb -= gb.mean()
    gb /= gb.std()
    max_val, min_val = gb.max(), np.abs(gb.min())
    is_valley = False
    if min_val > max_val:
        gb = np.square(gb)
        gb = np.where(gb >= 4, gb, 0)
        is_valley = True
    else:
        gb = np.where(gb >= 2, gb, 0)
    fp = findpeaks(method='topology', scale=True, denoise="lee_enhanced", togray=True, imsize=gb.shape[::-1], verbose=0)
    res = fp.fit(gb)
    persistence = res["persistence"]
    # keys in pd.DataFrame are: x, y, birth_level, death_level, score, peak, valley.
    best_score_arg = persistence["score"].argmax()
    series = persistence.iloc[best_score_arg]
    # assert (series["peak"] is True)
    x, y = series["x"], series["y"]
    xx = np.arange(gb.shape[1])
    yy = np.arange(gb.shape[0])
    xx, yy = np.meshgrid(xx, yy)
    contour = contour_generator(xx, yy, gb)
    lines = contour.lines(2)  # on veut supérieur à deux
    line = None
    for elt in lines:
        elt = np.transpose(elt)
        e = np.sort(elt)
        x_axis, y_axis = e[0], e[1]
        if x_axis[0] <= x <= x_axis[-1] and y_axis[0] <= y <= y_axis[-1]:
            line = elt
            break
    if line is None:
        pass
    plt.pcolormesh(gb)
    if line is not None:
        plt.plot(line[0], line[1], c="r")
    plt.plot(x, y, marker="+", c="k")
    plt.show()
    return x, y, line, is_valley


if __name__ == "__main__":
    folder = "C:/Users/Flavi/data/EXPERIMENT/MUROLS/MUROLS_20230224/MUROLS_20230224_SESSION_00"
    sequence = gd.extract_data(folder)
    fn = os.path.join(folder, "heatmap_playback.npz")
    if os.path.exists(fn):
        hm_total = hm.load_heatmap(fn)
        for cluster in hm_total.get_clusters():
            c = hm_total.get_hm_1_cluster(cluster)
            x, y, line, is_valley = ut.peak_and_contour_finding(c)

    # # https://erdogant.github.io/findpeaks/pages/html/index.html
    # cluster_list = [0, 14]
    # threshold = 2
    # res_list = list()
    # for cluster in cluster_list:
    #     c = hm_total.get_hm_1_cluster(cluster)
    #     gb = cv2.GaussianBlur(c, (5, 5), 0)
    #     gb -= gb.mean()
    #     gb /= gb.std()
    #     fp = findpeaks(method='topology', scale=True, denoise="lee_enhanced", togray=True, imsize=gb.shape[::-1],
    #                    verbose=0)
    #     res = fp.fit(gb)
    #     res_list.append(res)
    #     if cluster == 14:
    #         pass
    #     foo = np.where(gb >= 2, gb, 0)
    #     x = np.arange(foo.shape[1])
    #     y = np.arange(foo.shape[0])
    #     xv, yv = np.meshgrid(x, y)
    #     contour = contour_generator(xv, yv, foo)
    #     lines = contour.lines(2)[0]
    #     plt.plot(lines[:, 0], lines[:, 1], c="r")
    #     plt.pcolormesh(gb)
    #     plt.show()
#
    # Comment savoir si la vallée est plus profonde que le pic?




