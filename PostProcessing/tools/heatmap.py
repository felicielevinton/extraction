import numpy as np
from copy import deepcopy
from .utils import psth, qx, mean_smoothing, peak_and_contour_finding, find_temporal_span, find_spectral_span
import matplotlib.pyplot as plt
import os
from findpeaks import findpeaks
from scipy import signal
from zetapy import getZeta
import cv2 as cv
from .extraction import *
import matplotlib
from scipy.stats import norm, mode
from skimage import feature


def plot_psth_multisession(session, folder, vector_heatmap_playback, vector_heatmap_tracking):
    # for session in range(len(vector_heatmap_playback)):
    fig, axes = plt.subplots(4, 8)
    plt.title(f"PSTH session#{session}")
    plot(axes, vector_heatmap_tracking)
    plot(axes, vector_heatmap_playback)
    for axe in axes:
        for ax in axe:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
    if folder is not None:
        plt.savefig(os.path.join(folder, f"psth_session_{session}.png"), dpi=240)
        plt.close()
    else:
        plt.show()


def plot_tc_multisession(session, folder, vector_heatmap_playback, vector_heatmap_tracking):
    fig, axes = plt.subplots(4, 8)
    plt.title(f"TC session#{session}")
    for i in vector_heatmap_playback.keys():
        tmp = vector_heatmap_tracking[i]
        tones_tr, t = tmp[0], tmp[1]
        tmp = vector_heatmap_playback[i]
        tones_pb, p = tmp[0], tmp[1]

        row, col = get_plot_coords(i)

        axes[row, col].plot(np.log2(tones_tr / 2000), t, linewidth=0.5)
        axes[row, col].plot(np.log2(tones_pb / 2000), p, linewidth=0.5)
        axes[row, col].axhline(qx(t, -3.0), c="purple", linewidth=0.4)
        axes[row, col].axhline(qx(p, -3.0), c="red", linewidth=0.4)
        axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)
        axes[row, col].set_xticks(list())
        axes[row, col].set_yticks(list())

    for axe in axes:
        for ax in axe:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
    if folder is not None:
        plt.savefig(os.path.join(folder, f"tc_session_{session}.png"), dpi=240)
        plt.close()
    else:
        plt.show()


def plot(axes, y, x=None):
    for i in y.keys():

        row, col = get_plot_coords(i)

        if x is not None:
            axes[row, col].plot(x, y[i], linewidth=0.5)
        else:
            axes[row, col].plot(y[i], linewidth=0.5)
        axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)
        axes[row, col].set_xticks(list())
        axes[row, col].set_yticks(list())


def colormesh():
    pass


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


def substract(hm1, hm2):
    """
    Substraction de deux Heatmap.
    """
    hm_cp = deepcopy(hm1)
    return hm_cp - hm2


def concatenate(*args):
    """
    Concatenation de deux Heatmaps.
    """
    hm_cp = deepcopy(args[0])
    args = args[1:]
    for _hm in args:
        hm_cp = hm_cp.concatenate(_hm)
    return hm_cp


def tc_common(hm_tracking, hm_playback, session, folder):
    """
    Comparer les TC de deux Heatmaps.
    """
    positions = hm_playback.get_best_time()
    vector_playback = hm_playback.get_tuning_curves()
    vector_tracking = hm_tracking.get_tuning_curves(positions)
    plot_tc_multisession(session, folder, vector_playback, vector_tracking)


def tc_common_2(hm_tracking, hm_playback, session, folder):
    """
    Comparer les TC de deux Heatmaps.
    """
    hm_tmp = Heatmap()
    for i in range(32):
        hm_tmp.psths[i] = np.dstack((hm_tracking.get_hm_1_cluster(i), hm_playback.get_hm_1_cluster(i))).mean(2)
    hm_tmp.bins = hm_tracking.bins
    hm_tmp.clusters = hm_tracking.get_clusters()
    positions = hm_tmp.get_best_time()
    vector_playback = hm_playback.get_tuning_curves(positions)
    vector_tracking = hm_tracking.get_tuning_curves(positions)
    plot_tc_multisession(session, folder, vector_playback, vector_tracking)


def tc_common_3(hm_tracking, hm_playback, session, folder):
    """
    Comparer les TC de deux Heatmaps.
    """
    psths = dict()
    for i in range(32):
        hm = np.vstack((hm_tracking.get_hm_1_cluster(i), hm_playback.get_hm_1_cluster(i)))
        m, sigma = hm.mean(1), hm.std(1)
        psths[i] = [m, sigma]
    # hm_tmp.bins = hm_tracking.bins
    # hm_tmp.clusters = hm_tracking.get_clusters()
    # positions = hm_tmp.get_best_time()
    vector_playback = hm_playback.get_tuning_curves(scaling=psths)
    vector_tracking = hm_tracking.get_tuning_curves(scaling=psths)
    plot_tc_multisession(session, folder, vector_playback, vector_tracking)


def psth_common(hm_tracking, hm_playback, session, folder):
    positions = hm_playback.get_best_tone()
    vector_playback = hm_playback.get_bf_psth()
    vector_tracking = hm_tracking.get_bf_psth(positions)
    plot_psth_multisession(session, folder, vector_playback, vector_tracking)


def psth_common_2(hm_tracking, hm_playback, session, folder, smooth=False):
    hm_tmp = Heatmap()
    for i in range(32):
        hm_tmp.psths[i] = np.dstack((hm_tracking.get_hm_1_cluster(i), hm_playback.get_hm_1_cluster(i))).mean(2)
    hm_tmp.clusters = hm_tracking.get_clusters()
    hm_tmp.tones = hm_tracking.get_tones()
    positions = hm_tmp.get_best_tone()
    vector_playback = hm_playback.get_bf_psth()
    vector_tracking = hm_tracking.get_bf_psth(positions)
    if smooth:
        for key in vector_playback.keys():
            x = vector_playback[key]
            x -= x.mean()
            # x /= x.std()
            x = mean_smoothing(x, size=10, pad_size=50)
            vector_playback[key] = x
            x = vector_tracking[key]
            x -= x.mean()
            # x /= x.std()
            x = mean_smoothing(x, size=10, pad_size=50)
            vector_tracking[key] = mean_smoothing(x, size=10, pad_size=50)
    plot_psth_multisession(session, folder, vector_playback, vector_tracking)


class Heatmap(object):
    """
    Objet qui facilite la manipulation des Heatmaps.
    """
    def __init__(self, tones=None, clusters=None, psths=None, bins=None):
        if tones is None and clusters is None and psths is None:
            self.empty = True
        else:
            self.empty = False

        if tones is None:
            self.tones = np.empty(0, dtype=np.double)
            self.idx = np.empty(0, dtype=int)
        else:
            self.tones = tones
            self.idx = np.arange(len(tones), dtype=int)

        if clusters is None:
            self.clusters = np.empty(0, dtype=int)
        else:
            self.clusters = clusters

        if psths is None:
            self.psths = dict()
        else:
            self.psths = psths

        if bins is None:
            self.bins = np.empty(0, dtype=np.double)
        else:
            self.bins = bins

    def is_empty(self):
        return self.empty

    def get_tones(self):
        return self.tones

    def get_clusters(self):
        return self.clusters

    def get_bins(self):
        return self.bins

    def get_hm_1_cluster(self, cluster):
        return self.psths[cluster]

    def get_heatmap(self):
        return self.psths

    def get_psth_at(self, tone, cluster):
        assert(tone in self.tones), f"{tone}Hz is not an available frequency."
        assert(cluster in self.clusters), f"{cluster} is not an available cluster."
        idx = np.where(self.tones == tone)[0][0]
        return self.psths[cluster][idx]

    def plot(self, n_clus, tag, folder=None, cmap="bwr", l_ex=None, r_ex=None, ext="png"):
        if r_ex is not None:
            if r_ex > 0:
                r_ex *= -1
        fig, axes = plt.subplots(4, 8)
        plt.title(f"Heatmap {tag}")
        heatmaps = []
        #for i in range(32):
        for i in range(n_clus):
            row, col = get_plot_coords(i)
            heatmaps.append(self.psths[i][l_ex:r_ex])
            axes[row, col].pcolormesh(self.psths[i][l_ex:r_ex], cmap=cmap)
            axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)
            axes[row, col].set_xticks(list())
            axes[row, col].set_yticks(list())

        for axe in axes:
            for ax in axe:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        if folder is not None:
            plt.savefig(os.path.join(folder, f"heatmap_{tag}.{ext}"), dpi=240)
            plt.close()
        else:
            plt.show()
        return heatmaps   

    def get_bf_psth_for(self, cluster, position=None):

        if type(position) == list():
            assert (len(position) == 2), "Can't interpret."
            relevant_tones = position

        else:
            if type(position) != list and position is not None:
                bf = position

            else:
                bf = self.get_best_frequency(cluster)
            relevant_tones = [2 ** -0.5 * bf, 2 ** 0.5 * bf]
        ix = np.logical_and(self.tones >= relevant_tones[0],
                            self.tones <= relevant_tones[1])
        return self.psths[cluster][ix].mean(0)

    def get_bf_psth(self, best_tone_response=None):
        best_psth = dict()
        for i in self.clusters:
            if best_tone_response is not None:
                best_psth[i] = self.get_bf_psth_for(i, best_tone_response[i])
            else:
                best_psth[i] = self.get_bf_psth_for(i)
        return best_psth

    def get_position_for(self, cluster):
        _, bf_positions = self.detect_peak(cluster)
        return bf_positions

    def get_positions(self):
        positions = dict()
        for cluster in self.clusters:
            positions[cluster] = self.get_position_for(cluster)
        return positions

    def get_activity_at_peak(self, cluster):
        f, t = self.get_position_for(cluster)
        return self.psths[cluster][f, t]

    def get_best_time_for(self, cluster):
        _, bf_positions = self.detect_peak(cluster)
        return self.bins[bf_positions[1]], bf_positions[1]

    def get_best_time(self):
        positions = dict()
        for cluster in self.clusters:
            positions[cluster] = self.get_best_time_for(cluster)
        return positions

    def get_best_frequency_for(self, cluster):
        _, peak_coords = self.detect_peak(cluster)
        return self.tones[peak_coords[0]]

    def get_spectral_span_for(self, cluster):
        _, peak_coords = self.detect_peak(cluster)
        return self.tones[peak_coords[0]]

    def get_best_tone(self):
        positions = dict()
        for cluster in self.clusters:
            positions[cluster] = self.get_best_frequency_for(cluster)
        return positions

    def get_tuning_curve_for(self, cluster, position=None, m=None, std=None):
        """

        """
        conv_hm = self.get_hm_1_cluster(cluster)
        smooth, bf_positions = self.detect_peak(cluster)

        if position is None:
            best_tone = bf_positions[1]
        else:
            best_tone = position

        for i in range(conv_hm.shape[0]):
            if m is not None and std is not None:
                conv_hm[i] = (conv_hm[i] - m[i]) / std[i]
            else:
                conv_hm[i] = (conv_hm[i] - conv_hm[i].mean()) / conv_hm[i].std()
        kernel = np.outer(signal.windows.gaussian(3, 3), signal.windows.gaussian(3, 3))
        conv_hm = signal.convolve(conv_hm, kernel, "same")
        # conv_hm -= conv_hm.mean()
        # conv_hm /= conv_hm.std()
        tones = self.get_tones()
        tc = conv_hm[:, best_tone]
        return tones, tc

    def get_tuning_curves(self, positions=None, scaling=None):
        tc_dict = dict()
        for i in self.clusters:
            if scaling is not None:
                tones, tc = self.get_tuning_curve_for(i, m=scaling[i][0], std=scaling[i][1])
            elif positions is not None:
                tones, tc = self.get_tuning_curve_for(i, positions[i][1])
            else:
                tones, tc = self.get_tuning_curve_for(i)
            tc_dict[i] = [tones, tc]
        return tc_dict

    def identify_best_frequency(self):
        n_bins = len(self.bins)
        l_bf = list()
        for cluster in self.clusters:
            # Where and when?
            argmax = np.argmax(self.psths[cluster])
            f, t = argmax // n_bins, argmax % n_bins
            l_bf.append([f, t])
        return l_bf

    def plot_mean_psth(self, folder, tag, sess_num):
        fig, axes = plt.subplots(4, 8)
        for i in range(32):
            row, col = get_plot_coords(i)
            axes[row, col].plot(self.psths[i].mean(0), linewidth=0.5, c="purple")
            axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

        for axe in axes:
            for ax in axe:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        plt.title(f"Heatmap {tag}")
        if folder is not None:
            plt.savefig(os.path.join(folder, f"psth_{tag}_{sess_num}.png"), dpi=240)
            plt.close()
        else:
            plt.show()

    def smooth(self):
        kernel = signal.windows.gaussian(M=4, std=3)
        smooth = dict()
        for cluster in self.clusters:
            hm = np.copy(self.psths[cluster])
            for idx in self.idx:
                hm[idx] = signal.fftconvolve(hm[idx], kernel, "same")
            smooth[cluster] = hm
        return smooth

    def smooth_2d(self):
        n = 5
        # kernel = np.outer(signal.windows.gaussian(n, n), signal.windows.gaussian(n, n))
        smooth = dict()
        for cluster in self.clusters:
            hm = np.copy(self.psths[cluster])
            hm = cv.GaussianBlur(hm, (n, n), 0)
            smooth[cluster] = hm
        return smooth

    def plot_smooth_2d(self, tag, folder=None, cmap="bwr", l_ex=None, r_ex=None, num=None, ext="png"):
        if r_ex is not None:
            if r_ex > 0:
                r_ex *= -1
        fig, axes = plt.subplots(4, 8)
        smooth = self.smooth_2d()
        # cm = [[ones(1,50);linspace(1,0,50)],
        # [linspace(0,1,50);linspace(1,0,50)],
        # [linspace(0,1,50);ones(1,50)]];
        r = np.hstack((np.ones(50), np.linspace(1, 0, 50)))
        g = np.hstack((np.linspace(0, 1, 50), np.linspace(1, 0, 50)))
        b = np.hstack((np.linspace(0, 1, 50), np.ones(50)))
        rgb = np.vstack((r, g, b)).transpose()
        print(np.hstack((np.ones(50), np.linspace(1, 0, 50))).shape)
        cmap = matplotlib.colors.ListedColormap(rgb, "yves")
        for i in range(32):
            row, col = get_plot_coords(i)
            axes[row, col].pcolormesh(self.bins, np.log2(self.tones), smooth[i][l_ex:r_ex], cmap="bwr")
            if i == 0:
                tones = np.array([self.tones[p] for p in range(0, len(self.tones), 3)])
                axes[row, col].set_xlabel("Time[s]", fontsize=5)
                axes[row, col].set_ylabel("Frequency[Hz]", fontsize=5)
                axes[row, col].set_yticks(np.log2(tones))
                axes[row, col].set_xticks([0, 0.200, 0.400])
                axes[row, col].set_xticklabels([str(x) for x in [0, 0.200, 0.400]], color="k", size=10)
                axes[row, col].set_yticklabels([str(round(x)) for x in tones], color="k", size=10)
                axes[row, col].tick_params(axis="both",  # changes apply to the x-axis
                                           which="both",  # both major and minor ticks are affected
                                           bottom=True,
                                           left=True,  # ticks along the bottom edge are off
                                           top=False,  # ticks along the top edge are off
                                           labelbottom=True)  # labels along the bottom edge are off

                plt.setp(axes[row, col].get_xticklabels(), visible=True, fontsize=5)
                plt.setp(axes[row, col].get_yticklabels(), visible=True, fontsize=5)

            else:
                axes[row, col].tick_params(axis="both",          # changes apply to the x-axis
                                           which="both",      # both major and minor ticks are affected
                                           bottom=False,
                                           left=False,        # ticks along the bottom edge are off
                                           top=False,         # ticks along the top edge are off
                                           labelbottom=False)  # labels along the bottom edge are off
                plt.setp(axes[row, col].get_xticklabels(), visible=False, fontsize=5)
                plt.setp(axes[row, col].get_yticklabels(), visible=False, fontsize=5)
                #         plt.setp(ax.get_yticklabels(), visible=False)
            # axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)
        # plt.xticks([], [])
        # plt.yticks(list(), list())
        for axe in axes:
            for ax in axe:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        # plt.title(f"Heatmap {tag}")
        if folder is not None:
            plt.savefig(os.path.join(folder, f"sm2d_heatmap_{tag}_{num}.{ext}"), dpi=240)
            plt.close()
        else:
            plt.show()

    def plot_smooth(self, tag, folder=None, cmap="bwr", l_ex=None, r_ex=None, num=None, ext="png"):
        """

        """
        if r_ex is not None:
            if r_ex > 0:
                r_ex *= -1
        fig, axes = plt.subplots(4, 8)

        smooth = self.smooth()

        for i in range(32):
            row, col = get_plot_coords(i)
            axes[row, col].pcolormesh(smooth[i][l_ex:r_ex], cmap=cmap)
            axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)

        for axe in axes:
            for ax in axe:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        plt.title(f"Heatmap {tag}")
        if folder is not None:
            plt.savefig(os.path.join(folder, f"smooth_heatmap_{tag}_{num}.{ext}"), dpi=240)
            plt.close()
        else:
            plt.show()

    def plot_bf(self, tag, folder=None, cmap="bwr", ext="png"):
        """

        """
        fig, axes = plt.subplots(4, 8)
        plt.title(f"Heatmap bf {tag}")
        for i in range(32):
            row, col = get_plot_coords(i)
            smooth, bf = self.detect_peak(i)
            axes[row, col].pcolormesh(smooth, cmap=cmap)
            axes[row, col].axvline(0, linewidth=1, c="y")
            axes[row, col].axvline(bf[1], linewidth=1, c="g")
            axes[row, col].axhline(bf[0], linewidth=1, c="g")
            axes[row, col].set_title(f"Chan #{i}", y=0.95, fontsize="xx-small", linespacing=0.1)
            axes[row, col].set_xticks(list())
            axes[row, col].set_yticks(list())
        for axe in axes:
            for ax in axe:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
        if folder is not None:
            plt.savefig(os.path.join(folder, f"bf_heatmap_{tag}.{ext}"), dpi=240)
            plt.close()
        else:
            plt.show()

    def plot_left_vs_right(self, trigs, spikes, tone_sequence, session, folder, tag, smooth=False, t_pre=0.1, t_post=1):
        lr_clusters = dict()
        clusters = list()
        tones, counts = np.unique(tone_sequence, return_counts=True)

        for cluster in range(spikes.get_n_clusters()):
            clusters.append(cluster)
            x = spikes.get_spike_times(cluster=cluster)
            hist_right = list()
            hist_left = list()
            for tone in self.tones:
                tone_idx = np.where(tone_sequence == tone)[0]
                tone_idx_right = list()
                tone_idx_left = list()
                for elt in tone_idx:
                    if tone_sequence[elt] > tone_sequence[elt-1] and elt != 0:
                        tone_idx_left.append(elt)
                    elif tone_sequence[elt] < tone_sequence[elt-1] and elt != 0:
                        tone_idx_right.append(elt)
                    elif elt == 0:
                        if tone_sequence[elt] > tone_sequence[elt+1]:
                            tone_idx_right.append(elt)
                        else:
                            tone_idx_left.append(elt)
                tone_idx_right = np.array(tone_idx_right, dtype=int)
                tone_idx_left = np.array(tone_idx_left, dtype=int)
                # TODO: éliminer les fréquences trop peu présentées de la réprésentation graphique.
                if len(tone_idx_left) > 0:  # va vers la gauche
                    h_left, _ = psth(x, trigs[tone_idx_left], bins=self.bins)
                else:
                    h_left = np.zeros(len(self.bins) - 1)
                if len(tone_idx_right) > 0:  # va vers la droite
                    h_right, _ = psth(x, trigs[tone_idx_right], bins=self.bins)
                else:
                    h_right = np.zeros(len(self.bins) - 1)
                hist_left.append(h_left)
                hist_right.append(h_right)
            hist_left = np.vstack(hist_left)
            hist_right = np.vstack(hist_right)
            if smooth:
                lr_clusters[cluster] = [cv.GaussianBlur(hist_left, (5, 5), 0), cv.GaussianBlur(hist_right, (5, 5), 0)]
            else:
                lr_clusters[cluster] = [hist_left, hist_right]

        plot_sub_figures(lr_clusters, session, folder, tag=tag)

    def plot_rl(self, sequence, spikes, session, folder, smooth=True):
        """
        Va plotter pour tracking et playback la différence: furet va à droite, furet va à gauche.
        Ce qui est à gauche: quand le furet va à gauche
        Ce qui est à droite, quand le furet va à droite.
        """

        lr_clusters_pb = dict()
        lr_clusters_tr = dict()
        n_iter = sequence.get_number_iteration()
        n_cluster = spikes.get_n_clusters()
        tones = list()

        coming_from_left_tones = list()  # f1 < f0: le furet va à droite.
        coming_from_right_tones = list()  # f1 > f0: le furet va à gauche.

        cfl_triggers_pb = list()
        cfr_triggers_pb = list()
        cfl_triggers_tr = list()
        cfr_triggers_tr = list()

        for i in range(n_iter):
            tones.append(sequence.get_xp_number("playback", i).tones)
        tones = np.hstack(tones)
        tones, c = np.unique(tones, return_counts=True)
        idx = np.greater(c, 50)
        tones = tones[idx]

        for i in range(n_iter):
            xp_0 = sequence.get_xp_number("playback", i)
            xp_1 = sequence.get_xp_number("tracking", i)
            t = xp_0.tones[1:]
            delayed_tones = xp_0.tones
            tr_0 = xp_0.triggers[1:]
            if i == 0:
                tr_1 = xp_1.triggers
            else:
                tr_1 = xp_1.triggers[1:]
            for j, elt in enumerate(t):
                if elt > delayed_tones[j]:
                    coming_from_right_tones.append(elt)
                    cfr_triggers_tr.append(tr_1[j])
                    cfr_triggers_pb.append(tr_0[j])
                else:
                    coming_from_left_tones.append(elt)
                    cfl_triggers_tr.append(tr_1[j])
                    cfl_triggers_pb.append(tr_0[j])

        d_lr = {"cfr": np.hstack(coming_from_right_tones), "cfl": np.hstack(coming_from_left_tones)}
        d_triggers_pb = {"cfr": np.hstack(cfr_triggers_pb), "cfl": np.hstack(cfl_triggers_pb)}
        d_triggers_tr = {"cfr": np.hstack(cfr_triggers_tr), "cfl": np.hstack(cfl_triggers_tr)}

        for cluster in range(n_cluster):
            x = spikes.get_spike_times(cluster=cluster)
            out_pb = lr_helper(d_lr, x, tones, d_triggers_pb, self.bins)
            out_tr = lr_helper(d_lr, x, tones, d_triggers_tr, self.bins)

            if smooth:
                lr_clusters_pb[cluster] = [cv.GaussianBlur(elt, (5, 5), 0) for elt in out_pb]
                lr_clusters_tr[cluster] = [cv.GaussianBlur(elt, (5, 5), 0) for elt in out_tr]

            else:
                lr_clusters_pb[cluster] = out_pb
                lr_clusters_tr[cluster] = out_tr
        plot_sub_figures(lr_clusters_pb, session, folder, tag="Playback")
        plot_sub_figures(lr_clusters_tr, session, folder, tag="Tracking")

    def compute_heatmap(self, trigs, spikes, tone_sequence, t_pre=0.1, t_post=1, bin_size=0.01):
        assert(self.empty is True), "Heatmap already done."
        if len(self.bins) == 0:
            self.bins = np.arange(-t_pre, t_post + bin_size, bin_size)
        tones, counts = np.unique(tone_sequence, return_counts=True)
        #idx = process_list(list(np.greater(counts, 30)))
        idx = process_list(list(np.greater(counts, 10)))
        # print(list(np.greater(counts, 30)), idx)
        self.tones = tones[idx]
        self.idx = np.arange(0, len(self.tones), dtype=int)
        clusters = list()
        for cluster in range(spikes.get_n_clusters()):
            clusters.append(cluster)
            x = spikes.get_spike_times(cluster=cluster)
            hist = list()
            for tone in self.tones:
                tone_idx = np.where(tone_sequence == tone)[0]
                trigger_time = trigs[tone_idx]
                h, _ = psth(x, trigger_time, t_0=t_pre, t_1=t_post, bins=self.bins)
                hist.append(h)
            if len(hist) > 0:
                hist = np.vstack(hist)
            else:
                hist = np.zeros((len(self.tones), len(self.bins)))

            self.psths[cluster] = hist
        self.clusters = np.array(clusters, dtype=int)
        self.empty = False

    def compute_heatmap_with_stats(self, trigs, spikes, folder, clusters=None):
        assert(self.empty is True), "Heatmap already done."
        # self.tones = np.unique(tone_sequence)
        # self.idx = np.arange(0, len(self.tones), dtype=int)
        # clusters = list()
        good_clusters = list()

        if clusters is not None:
            iterator = clusters
        else:
            iterator = list(range(spikes.get_n_clusters()))
        for cluster in iterator:
            # clusters.append(cluster)
            x = spikes.get_spike_times(cluster=cluster)
            a, b = getZeta(x * (1 / 30000), trigs * (1 / 30000))
            if a < 0.001:
                good_clusters.append(cluster)
        good_clusters = np.array(good_clusters)
        np.save(os.path.join(folder, "good_clusters_playback.npy"), good_clusters)
        self.empty = False

    def concatenate(self, other):
        self._check_bins(other)
        empty = self._check_empty(other)
        if empty is not None:
            if empty:
                return Heatmap(tones=self.tones, clusters=self.clusters, psths=self.psths, bins=self.bins)
            else:
                return Heatmap(tones=other.tones, clusters=other.clusters, psths=other.psths, bins=other.bins)
        else:
            clusters = self._check_cluster(other)
            idx, tones, other_is_shorter, idx_ex = self._check_tones(other)
            psths = dict()
            if other_is_shorter is None:
                tones = self.tones
                for key in clusters:
                    psths[key] = np.hstack((self.psths[key], other.psths[key]))
            else:
                for key in clusters:
                    if other_is_shorter:
                        psths[key] = np.hstack((self.psths[key][idx], other.psths[key][idx_ex][0]))
                    else:
                        psths[key] = np.hstack((self.psths[key][idx_ex][0], other.psths[key][idx]))
            return Heatmap(tones=tones, clusters=clusters, psths=psths, bins=self.bins)

    def __sub__(self, other):
        self._check_bins(other)
        empty = self._check_empty(other)
        if empty is not None:
            if empty:
                return Heatmap(tones=self.tones, clusters=self.clusters, psths=self.psths, bins=self.bins)
            else:
                return Heatmap(tones=other.tones, clusters=other.clusters, psths=other.psths, bins=other.bins)
        else:
            clusters = self._check_cluster(other)
            idx, tones, other_is_shorter, idx_ex = self._check_tones(other)
            psths = dict()
            if other_is_shorter is None:
                tones = self.tones
                psths = dict()
                for key in clusters:
                    psths[key] = self.psths[key] - other.psths[key]
            else:
                self.tones = tones
                for key in clusters:
                    if other_is_shorter:
                        psths[key] = self.psths[key][idx] - other.psths[key][idx_ex][0]
                    else:
                        psths[key] = self.psths[key][idx_ex][0] - other.psths[key][idx]
            return Heatmap(tones=tones, clusters=clusters, psths=psths, bins=self.bins)

    def _check_cluster(self, other):
        assert (len(self.clusters) != 0 and len(other.clusters) != 0), "Clusters are not registered."

        if not np.array_equal(self.clusters, other.clusters):
            kept_clusters = list()
            if len(self.clusters) > len(other.clusters):
                base_array, shorter_array = self.tones, other.tones
            else:
                base_array, shorter_array = other.tones, self.tones

            for elt in base_array:
                if elt in shorter_array:
                    kept_clusters.append(elt)

            clusters = np.array(kept_clusters, dtype=np.double)

        else:
            clusters = self.clusters
        return clusters

    def _check_tones(self, other):
        assert (len(self.tones) != 0 and len(other.tones) != 0), "Tones are not registered."
        if not np.array_equal(self.tones, other.tones):
            print("INFO: frequencies are partly different in the two heatmaps, removing the unique.")
            kept_tones = list()
            idx = list()
            idx_ex = list()
            if len(self.tones) > len(other.tones):
                other_is_shorter = True
                base_array, shorter_array = self.tones, other.tones
            else:
                other_is_shorter = False
                base_array, shorter_array = other.tones, self.tones
            for elt in base_array:
                if elt in shorter_array:
                    kept_tones.append(elt)
                    idx.append(np.where(base_array == elt)[0][0])
            idx = np.array(idx, dtype=int)
            tones = np.array(kept_tones, dtype=np.double)
            for elt in shorter_array:
                if elt not in base_array:
                    idx_ex.append(np.where(shorter_array == elt)[0][0])
            if len(idx_ex) == 0:
                idx_ex = None
            else:
                idx_ex = np.array(idx_ex, dtype=int)
            return idx, tones, other_is_shorter, idx_ex
        else:
            return None, None, None, None

    def _check_empty(self, other):
        assert(not self.empty and not other.empty), "Both heatmap are empty"
        if self.empty:
            return False
        elif other.empty:
            return True
        return None

    def _check_bins(self, other):
        assert(np.array_equal(self.bins, other.bins)), "Bins different. Abort"

    def detect_peak_and_contours(self, cluster, contour_std=2):
        """
        Retourne la position du peak, l'étalement temporel de la réponse et l'étalement spectral de cette dernière.
        """
        hm = self.psths[cluster]
        x, y, line, is_valley = peak_and_contour_finding(hm, contour_std=contour_std)
        line = np.transpose(line)
        if line.shape != ():
            temporal_span = find_temporal_span(line)
            temporal_span = [self.bins[temporal_span[0]], self.bins[temporal_span[1]]]
            spectral_span = find_spectral_span(line)
            spectral_span = [self.tones[spectral_span[0]], self.tones[spectral_span[1]]]
            return [x, y], temporal_span, spectral_span
        else:
            return [x, y], None

    def detect_peak(self, cluster):
        hm = self.psths[cluster]
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

    def get_best_frequency(self, cluster):
        _, peak_coord = self.detect_peak(cluster)
        return self.tones[peak_coord[0]]

    def save(self, folder, typeof):
        fn = os.path.join(folder, f"heatmap_{typeof}.npz")
        kwargs = {str(key): self.psths[key] for key in self.psths.keys()}
        kwargs["tones"] = self.tones
        kwargs["bins"] = self.bins
        kwargs["clusters"] = self.clusters
        kwargs["idx"] = self.idx
        np.savez(fn, **kwargs)


def load_heatmap(fn):
    """
    On passe un nom de fichier. Charge un objet Heatmap
    """
    hm = np.load(fn)
    tones = hm["tones"]
    clusters = hm["clusters"]
    bins = hm["bins"]
    psths = dict()
    for cluster in clusters:
        psths[cluster] = hm[str(cluster)]
    return Heatmap(tones=tones, clusters=clusters, psths=psths, bins=bins)


def lr_helper(directed_tones, x, tones, triggers, bins):
    assert (len(list(directed_tones.keys())) == 2)
    assert (len(list(triggers.keys())) == 2)
    hist_l = list()
    hist_r = list()

    for t in tones:
        h_l = extract(t, directed_tones["cfl"], triggers["cfl"], x, bins)
        h_r = extract(t, directed_tones["cfr"], triggers["cfr"], x, bins)
        hist_l.append(h_l)
        hist_r.append(h_r)

    hist_l = np.vstack(hist_l)
    hist_r = np.vstack(hist_r)
    return [hist_r, hist_l]


def plot_sub_figures(lr_clusters, session, folder, tag):
    r = np.zeros((4, 8), dtype=int)
    p = np.zeros((4, 8), dtype=int)
    for row in range(4):
        if row == 0:
            r[row] = np.arange(16, 24)[::-1]
            p[row] = np.ones_like(r[row]) * 7 - r[row] % 8
        elif row == 1:
            r[row] = np.arange(8, 16)[::-1]
            temp = np.ones_like(r[row]) * 7 - r[row] % 8
            p[row] = temp[::-1]
        elif row == 2:
            r[row] = np.arange(24, 32)
            p[row] = r[row] % 8
        else:
            r[row] = np.arange(8)
            p[row] = r[row]
    fig = plt.figure(constrained_layout=True, figsize=(64, 16))
    plt.title(f"Heatmap LR {session}")
    subfigs = fig.subfigures(4, 8)  # , wspace=0.07)
    for row in range(4):
        for col in range(8):
            sf = subfigs[row, col].subplots(1, 2)
            id_cell = r[row, col]
            if (row + col) % 2 == 0:
                subfigs[row, col].set_facecolor("0.75")
            subfigs[row, col].suptitle(f"Channel {id_cell}")
            sf[0].pcolormesh(lr_clusters[id_cell][0])
            sf[1].pcolormesh(lr_clusters[id_cell][1])
            sf[0].set_xticks(list())
            sf[0].set_yticks(list())
            sf[1].set_xticks(list())
            sf[1].set_yticks(list())
    if folder is not None:
        plt.savefig(os.path.join(folder, f"LR_heatmap_{tag}_{session}.png"), dpi=240)
        plt.close()


def extract(t, directed_tones, trigs, x, bins):
    idx = np.equal(t, directed_tones)
    _t = directed_tones[idx]
    _tr = trigs[idx]
    h, _ = psth(x, _tr, bins=bins)
    return h


def heatmap_channel_factory(x, cluster, tone_sequence, triggers, type_of=None, bins=None, t_pre=0.1, t_post=1, bin_size=0.01):
    if bins is None:
        bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    tones, counts = np.unique(tone_sequence, return_counts=True)
    tones = tones[np.greater(counts, 15)]
    hist = list()
    for tone in tones:
        tone_idx = np.equal(tone_sequence, tone)
        h, _ = psth(x, triggers[tone_idx], t_0=t_pre, t_1=t_post, bins=bins)
        hist.append(h)
    hist = np.vstack(hist)
    return HeatmapChannel(cluster, tones, hist, type_of, bins, t_pre, t_post, bin_size)


def process_list(lst):
    # Trouver l'index du premier et dernier True
    first_true = next((i for i, x in enumerate(lst) if x), None)
    last_true = next((i for i, x in enumerate(lst[::-1]) if x), None)
    
    last_true = len(lst) - last_true - 1 if last_true is not None else None
    # Transformer le False entouré de True en un True
    for i in range(first_true + 1, last_true):
        if lst[i-1] and lst[i+1] and not lst[i]:
            lst[i] = True
    return lst


class HeatmapChannel(object):
    """
    Renommer STRF?
    """
    def __init__(self, cluster, tones, heatmap, type_of, bins, t_pre=0.1, t_post=1, bin_size=0.01):
        self.cluster = cluster
        self.tones = tones
        self.heatmap = heatmap
        self.type_of = type_of
        self.t_pre = t_pre
        self.t_post = t_post
        self.bin_size = bin_size
        self.bins = bins
        self.best_frequency = 0
        self.peak_time = 0

    def smooth(self, m=4, std=3):
        kernel = signal.windows.gaussian(M=m, std=std)
        hm = np.copy(self.heatmap)
        for i, elt in hm:
            hm[i] = signal.fftconvolve(elt, kernel, "same")
        return hm

    def detect_peak(self):
        hm = self.heatmap
        hm_mask = np.empty_like(hm)
        for i in range(hm.shape[0]):
            if hm[i].std() == 0:
                hm_mask[i] = hm[i]
            else:
                hm_mask[i] = (hm[i] - hm[i].mean()) / hm[i].std()
        kernel = np.outer(signal.windows.gaussian(3, 3), signal.windows.gaussian(3, 3))
        hm_mask = signal.convolve(hm_mask, kernel, "same")
        hm_mask -= hm_mask.mean()
        hm_mask /= hm_mask.std()
        hm_clean = np.copy(hm_mask)
        hm_mask = np.where(hm_mask >= 3, hm_mask, 0)  # todo: permettre le négatif.
        fp = findpeaks(method='topology', scale=True, denoise=10, togray=True, imsize=hm.shape[::-1], verbose=0)
        res = fp.fit(hm_mask)
        peak_position = res["groups0"][0][0]  # tuple qui indique la position du pic
        return hm_clean, peak_position
