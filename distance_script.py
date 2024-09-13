import os.path
from distance import *
import argparse
from get_all_good_sessions import get_good_sessions
import pandas as pd
import get_data as gd


def parse_args():
    parser = argparse.ArgumentParser(prog="distance")
    parser.add_argument("--name", type=str, help="Ferret Name.")
    parser.add_argument("--compatibility", type=bool, help="Old data.")
    opt = parser.parse_args()
    return opt


def distance(folders, save_folder, absolute=False, error_bars=False):
    """

    """
    # PARAMS
    bin_size = 0.0025

    plot_path = gd.check_plot_folder_exists(save_folder)
    if absolute:
        n_delta = np.arange(33, dtype=int)
    else:
        n_delta = np.arange(-33, 33, dtype=int)
    d_cluster = dict()
    df_session = pd.DataFrame(index=n_delta, columns=np.arange(len(folders), dtype=int))
    df_counts = pd.DataFrame(index=n_delta, columns=np.arange(len(folders), dtype=int))

    for sess_num, folder in enumerate(folders):
        if not os.path.exists(folder):
            print(f"{folder} does not exist. Skipping.")
            continue
        fn = os.path.split(folder)[-1]

        sequence = gd.extract_data(folder)
        wp = sequence.merge("warmup")
        good_clusters = np.load(os.path.join(folder, "good_clusters_playback.npy"))
        keep = None
        recording_length = sequence.get_recording_length()
        spk = ut.Spikes(folder, recording_length=recording_length)

        # C'est là où on assigne la fréquence qui aurait dû être jouée
        pb_tr, pb_tones_all, pb_triggers_all, tr_tones_all, tr_triggers_all, mck_tones_all = virtual_tones(sequence)

        fn = os.path.join(folder, "heatmap_playback.npz")
        if os.path.exists(fn):
            hm_total = hm.load_heatmap(fn)
        else:
            hm_total, _ = build_heatmap(tr_tones_all, pb_tones_all, tr_triggers_all, pb_triggers_all, spk)
        df = pd.DataFrame(index=n_delta, columns=good_clusters)
        df_tmp_counts = pd.DataFrame(index=n_delta, columns=good_clusters)
        for cluster in good_clusters:

            if keep is not None and cluster not in keep:
                continue

            out = compute_activity_with_distance(spk, cluster, hm_total, pb_tones_all,
                                                 pb_triggers_all, pb_tr, bin_size, first_trigger=wp.triggers[0],
                                                 absolute=absolute)

            if out is None:
                break

            else:
                a, delta_list, cpd = out[0], out[1], out[2]

            ser = pd.Series(a, index=np.array(delta_list))

            df[cluster] = ser

            df_tmp_counts[cluster] = cpd

            if cluster not in d_cluster.keys():
                d_cluster[cluster] = pd.DataFrame(index=n_delta)

            d_cluster[cluster][sess_num] = ser

        df_session[sess_num] = df.mean(1, skipna=True)

        df_counts[sess_num] = df_tmp_counts.sum(1, skipna=True)  # somme par delta

    plot_distance(df_session, bin_size, df_counts=df_counts, folder=plot_path, absolute=absolute, error_bars=error_bars)


if __name__ == "__main__":
    options = parse_args()

    good_dirs, save_dir = get_good_sessions(options.name)

    good_dirs.sort()

    eb = False

    distance(good_dirs, save_dir, False, error_bars=eb)
    distance(good_dirs, save_dir, True, error_bars=eb)

    # eb = False
    # distance(options.folders, options.save, False, error_bars=eb)
    # distance(options.folders, options.save, True, error_bars=eb)

    # plt.plot(ax, gen_data(a, *out.x), c="r", linewidth=0.5)

    # lsq_y = list()
    # lsq_x = list()
    # for elt in act_tot:
    #     a[:len(elt)] += elt
    #     counts[:len(elt)] += 1
    #     lsq_x.append(np.arange(len(elt)))
    #     lsq_y.append(elt)
    # x0 = np.array([1.0, 1.0, 0.0])
    # idx = np.equal(clean_delta, delta)
    # a = np.array(count[idx])

    # Partie: Régression linéaire
    # def fun(r, t, y):
    #     return r[0] + r[1] * np.exp(r[2] * t) - y

    # def gen_data(t, h, b, c, noise=0., n_outliers=0, seed=None):
    #     rng = np.random.default_rng(seed)
    #     y = h + b * np.exp(t * c)
    #     error = noise * rng.standard_normal(t.size)
    #     outliers = rng.integers(0, t.size, n_outliers)
    #     error[outliers] *= 10
    #     return y + error

    # lsq_x = np.hstack(lsq_x)
    # lsq_y = np.hstack(lsq_y)
    # out = least_squares(fun=fun, x0=x0, loss="soft_l1", f_scale=0.1, args=(lsq_x, lsq_y))
    # regression = linear_model.LinearRegression()
    # regression.fit(ax, a)
    # plt.plot(lsq_x, lsq_y, "o", c="purple")
    # plt.show()

    # max_length = 0
    # cluster_max = 0
    # for key in foo.keys():
    #     length = len(foo[key][0])
    #     if max_length < length:
    #         max_length = length
    #         cluster_max = key
    # array_mean_activity = np.zeros(max_length)
    # ref_deltas = foo[cluster_max][1]
    # counts = np.zeros(max_length)
    # for key in foo.keys():
    #     act, tmp_deltas = foo[key][0], foo[key][1]
    #     array_mean_activity[:len(act)] += act
    #     counts[:len(act)] += 1
    # array_mean_activity /= counts
    # act_tot.append(array_mean_activity[1:-1])  # ut.z_score(array_mean_activity)
    # max_length = 0
    # for elt in act_tot:
    #     length = len(elt)
    #     if max_length < length:
    #         max_length = length
