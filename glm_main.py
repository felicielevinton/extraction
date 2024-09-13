import pyglmnet
import numpy as np
import time
import GLM.glm as glm
import GLM.glm_mode as glm_mode
import matplotlib.pyplot as plt
from tqdm import tqdm
import PostProcessing.tools.utils as ut
import PostProcessing.tools.accelerometer as ax
import PostProcessing.tools.heatmap as hm
import PostProcessing.tools.positions as pos
import os
import argparse
import glob
import ExtractRecordings.load_exp_files.read_bin as io


def extract_cell_spikes(spikes, n_cluster, t_0, t_1, bins, zero=True):
    cell_area = spikes.get_spike_times_between(cluster=n_cluster, t_0=t_0, t_1=t_1, zero=zero)
    return glm.bin_spikes(cell_area, bins)


def get_r2(spk_binned, prediction):
    mean_squared_error = np.mean((spk_binned - prediction) ** 2)
    squared_error = np.mean((spk_binned - np.mean(spk_binned)) ** 2)
    return 1 - mean_squared_error / squared_error


def parse_args():
    parser = argparse.ArgumentParser(prog="GLM")
    parser.add_argument("--folder", type=str, help="neural recordings in .npy format and double fp precision.")
    parser.add_argument("--cells", nargs="+", type=int, help="Cells choice for GLM fit")
    parser.add_argument("--design_matrix_type", default="onset2D", type=str, help="")
    parser.add_argument("--positions", type=bool, default=False, help="Add positions design matrix")
    parser.add_argument("--spike_history", type=bool, default=False, help="Add spike history in design matrix")
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--accelerometer", type=bool, default=False)
    parser.add_argument("--exclude", type=bool, default=False)
    opt = parser.parse_args()
    assert opt.design_matrix_type in ("positive", "onset2D"), "Not a valid design matrix type"
    assert opt.mode in ("cross_validation", "permutation", "test"), "Not a valid mode"
    return opt


if __name__ == "__main__":
    options = parse_args()
    dm_type = options.design_matrix_type
    mode = options.mode
    folder = options.folder
    spike_history = options.spike_history
    accelerometer = options.accelerometer
    positions = options.positions
    exclude = options.exclude
    spk = ut.Spikes(folder)
    n_cells = options.cells
    alpha, _lambda = glm.load_alpha_lambda_values(folder)
    len_pad = 25
    len_pad_sh = 20
    len_pad_ax = 20
    len_pad_pos = 20
    n_permutation = 500
    params = glm_mode.get_params_dict(dm_type, mode,
                                      spike_history, accelerometer,
                                      positions, n_cells, folder,
                                      alpha=alpha, _lambda=_lambda,
                                      len_pad=len_pad,
                                      len_pad_sh=len_pad_sh,
                                      len_pad_ax=len_pad_ax,
                                      len_pad_pos=len_pad_pos)

    triggers = io.read_dig_in(options.folder)
    tones_tracking = np.empty(0)
    tonotopy_tones = np.empty(0)
    for elt in glob.glob(os.path.join(options.folder, "tones_*.bin")):
        tonotopy_tones = np.hstack((tonotopy_tones, io.read_tones_file(elt)))
    for elt in glob.glob(os.path.join(options.folder, "tracking_*.bin")):
        tones_tracking = np.hstack((tones_tracking, io.read_tones_file(elt)))
    start = int(len(tonotopy_tones) / 2)
    if start != 0:
        triggers = triggers[start:-start]

    data = glm.Data(tones_tracking, triggers, len_pad=len_pad)
    # ATTENTION: pas comme ça...
    if dm_type == "positive":
        dm_train = data.get_positive_lag_dm(dataset="train")
        dm_test = data.get_positive_lag_dm(dataset="test")
    else:
        dm_train = data.get_onset_dm(dataset="train")
        dm_test = data.get_onset_dm(dataset="test")

    if mode == "permutation":
        glm_mode.permutation(data, spk, params, n_permutation)

    elif mode == "cross_validation":
        glm_mode.cross_validation(data, spk, params)

    elif mode == "test":
        glm_mode.test(data, spk, params)

    tones_tracking = tones_tracking[1:-1]
    u, n_tones = glm.get_tones_and_num(tones_tracking)
    freq_played = u
    b_s, dt = glm.get_maximum_bin_size(triggers, fs=30e3)
    print(f"bin duration = {dt * 1000:.2f} ms")
    if exclude:
        # Quand on veut exclure des fréquences trop peu présentées.
        y = ~np.logical_or(tones_tracking == u[0], tones_tracking == u[-1])
        idx = np.where(y != 0)[0]
        array_for_splitting = np.where(np.diff(idx) != 1)[0] + 1
        tones_tracking_pruned = np.split(tones_tracking[idx], array_for_splitting)
        triggers_pruned = np.split(triggers[idx], array_for_splitting)
        triggers_pruned.sort(key=len, reverse=True)
        tones_tracking_pruned.sort(key=len, reverse=True)
        triggers_train, tones_train = triggers_pruned[0], tones_tracking_pruned[0]
        triggers_test, tones_test = triggers_pruned[1], tones_tracking_pruned[1]

    else:
        percent = 0.7
        up_to = int(percent * len(tones_tracking))
        triggers_train, tones_train = triggers[:up_to], tones_tracking[:up_to]
        triggers_test, tones_test = triggers[up_to:], tones_tracking[up_to:]

    # calcul de mu et sigma
    n_bins, bins, samples, triggers_zeroed = glm.max_onset_on_0(b_s, triggers, tones_tracking)
    mu, sigma = glm.get_mean_std(triggers_zeroed, tones_tracking, samples, b_s, n_bins)

    n_bins_train, bins_train, samples_train, triggers_zeroed_train = glm.max_onset_on_0(b_s,
                                                                                        triggers_train, tones_train)

    n_bins_test, bins_test, samples_test, triggers_zeroed_test = glm.max_onset_on_0(b_s, triggers_test, tones_test)

    if dm_type == "positive":
        dm_train, binned_stim_train = glm.build_stim_dm_2d_onset(triggers_zeroed_train, tones_train, len_pad, b_s,
                                                                 samples_train, n_bins_train, freq_played=freq_played,
                                                                 norm=[mu, sigma], shifted=True, delta=1)

        dm_test, binned_stim_test = glm.build_stim_dm_2d_onset(triggers_zeroed_test, tones_test, len_pad, b_s,
                                                               samples_test, n_bins_test,
                                                               freq_played=freq_played, norm=[mu, sigma],
                                                               shifted=True, delta=1)

    # Onset
    else:
        dm_train, binned_stim_train = glm.build_stim_dm_2d_onset(triggers_zeroed_train, tones_train, len_pad, b_s,
                                                                 samples_train, n_bins_train, freq_played=freq_played,
                                                                 norm=[mu, sigma], shifted=False)

        dm_test, binned_stim_test = glm.build_stim_dm_2d_onset(triggers_zeroed_test, tones_test, len_pad, b_s,
                                                               samples_test, n_bins_test,
                                                               freq_played=freq_played, norm=[mu, sigma], shifted=False)

    plt.figure(figsize=[12, 8])
    plt.imshow(dm_train[1000:1400], aspect='auto', interpolation='nearest')
    plt.savefig(os.path.join(options.folder, f"test_dm.png"), dpi=240)
    plt.close()
    u_binned_test = freq_played
    # idée: entraîner le glm sur warmup. Puis prendre warmout pour calculer un R2. Puis calculer
    # le R2 pour chaque tracking. Comparer cas où il ya l'accéléromètre et les positions.
    # Puis le cas avec juste les onsets
    # 1) performance de prédiction GLM stims + positions + ax +/- sh vs stim seulement.
    # 2) performance de prédiction avec entraînement sur tracking et prédictions sur playback.
    # 3) performance de prédictions avec entraînement sur tracking et prédictions sur tracking interleaved.

    if mode == "permutation":
        _time = time.gmtime()

        dir_name = f"results/perm_{_time.tm_year}{_time.tm_mon}{_time.tm_mday}"
        if not os.path.exists(os.path.join(options.folder, dir_name)):
            os.mkdir(os.path.join(options.folder, dir_name))

        perms = list()
        perms_test = list()
        # _p = np.random.randint(100, 2600, n_permutation)
        _p = np.random.choice(np.arange(100, 2600), size=n_permutation, replace=False)
        # s'assurer que c'est différent.
        for i in tqdm(range(n_permutation)):
            # rng = np.random.default_rng()
            tt = np.copy(tones_tracking)
            tt = np.roll(tt, _p[i])
            dm_perm, _ = glm.build_stim_dm_2d_onset(triggers_zeroed_train, tt[:up_to], len_pad, b_s,
                                                    samples_train, n_bins_train, freq_played=freq_played,
                                                    norm=[mu, sigma], shifted=False)
            # rng.shuffle(tt)
            plt.figure(figsize=[12, 8])
            plt.imshow(dm_perm, aspect='auto', interpolation='nearest')
            plt.title(f"Perm#{i} onset")
            plt.savefig(os.path.join(options.folder, dir_name, f"perm{i}.png"), dpi=240)
            plt.close()
            perms.append(tt[:up_to])
            perms_test.append(tt[up_to:])

        perms = np.vstack(perms)

        for n_cell in n_cells:
            perm_score = np.zeros(n_permutation)
            file_name = f"Cell#{n_cell}"
            cell_train = spk.get_spike_times_between_(cluster=n_cell, t_0=triggers_train[0],
                                                      t_1=samples_train[-1] + triggers_train[0], zero=True)
            cell_train_binned = glm.bin_spikes(cell_train, bins_train)
            cell_test = spk.get_spike_times_between_(cluster=n_cell, t_0=triggers_test[0],
                                                     t_1=samples_test[-1] + triggers_test[0], zero=True)
            cell_test_binned = glm.bin_spikes(cell_test, bins_test)
            if spike_history:
                dm_sh = glm.build_spike_history_dm(cell_train_binned, len_pad_sh)
                dm_train = np.concatenate((dm_train, dm_sh), axis=1)
                dm_sh_test = glm.build_spike_history_dm(cell_test_binned, len_pad_sh)
                dm_test = np.concatenate((dm_test, dm_sh_test), axis=1)
            _glm_train = pyglmnet.GLM(distr="poisson", alpha=alpha, reg_lambda=_lambda,
                                      score_metric="pseudo_R2")
            _glm_train.fit(X=dm_train, y=cell_train_binned)
            score = _glm_train.score(X=dm_test, y=cell_test_binned)
            del _glm_train
            score = np.array([score])
            np.save(os.path.join(options.folder, dir_name, file_name + "_score_gt.npy"), score)

            for i in tqdm(range(n_permutation)):
                perm = perms[i]
                dm_perm, _ = glm.build_stim_dm_2d_onset(triggers_zeroed_train, perm, len_pad, b_s,
                                                        samples_train, n_bins_train, freq_played=freq_played,
                                                        norm=[mu, sigma], shifted=False)

                if spike_history:
                    dm_perm = np.concatenate((dm_perm, dm_sh), axis=1)

                _glm = pyglmnet.GLM(distr="poisson", alpha=alpha, reg_lambda=_lambda,
                                    score_metric="pseudo_R2")
                _glm.fit(X=dm_perm, y=cell_train_binned)
                psc = _glm.score(X=dm_test, y=cell_test_binned)
                perm_score[i] = psc
                del _glm
                np.save(os.path.join(options.folder, dir_name, file_name + "_perm_test.npy"), perm_score)

    elif mode == "cross_validation":
        alphas = np.linspace(0.45, 0.60, 10, endpoint=True)
        # alphas = np.array([0.5])
        lambdas = np.logspace(np.log(0.01), np.log(0.0001), num=10, base=np.e)
        np.save(os.path.join(options.folder, "results", "alphas.npy"), alphas)
        np.save(os.path.join(options.folder, "results", "lambda.npy"), lambdas)
        for n_cell in n_cells:
            file_name = f"Cell#{n_cell}"

            cell_train = spk.get_spike_times_between_(cluster=n_cell, t_0=triggers_train[0],
                                                      t_1=samples_train[-1] + triggers_train[0], zero=True)
            cell_train_binned = glm.bin_spikes(cell_train, bins_train)

            if options.spike_history:
                dm_sh = glm.build_spike_history_dm(cell_train_binned, len_pad_sh)
                dm_train = np.concatenate((dm_train, dm_sh), axis=1)

            results = np.zeros((len(alphas), len(lambdas)))
            for i, alpha in enumerate(alphas):
                print(f"Cellule #{n_cell}, Activité moyenne: {cell_train_binned.mean()}. alpha = {alpha}")
                glms = pyglmnet.GLMCV(distr="poisson", verbose=True, alpha=alpha, score_metric="pseudo_R2", cv=3,
                                      reg_lambda=lambdas, max_iter=10000)  # , solver="cdfast")
                glms.fit(X=dm_train, y=cell_train_binned)
                results[i] = glms.scores_
                np.save(os.path.join(options.folder, "results", file_name + f"_results_2.npy"), results)

    elif mode == "test":
        for cell in n_cells:
            file_name = f"Cell#{cell}"
            # print(file_name)
            cell_train = spk.get_spike_times_between_(cluster=cell, t_0=triggers_train[0],
                                                      t_1=samples_train[-1] + triggers_train[0], zero=True)
            cell_train_binned = glm.bin_spikes(cell_train, bins_train)
            cell_test = spk.get_spike_times_between_(cluster=cell, t_0=triggers_test[0],
                                                     t_1=samples_test[-1] + triggers_test[0], zero=True)
            cell_test_binned = glm.bin_spikes(cell_test, bins_test)
            if options.spike_history:
                dm_sh = glm.build_spike_history_dm(cell_train_binned, len_pad_sh)
                dm_train = np.concatenate((dm_train, dm_sh), axis=1)
                dm_sh_test = glm.build_spike_history_dm(cell_test_binned, len_pad_sh)
                dm_test = np.concatenate((dm_test, dm_sh_test), axis=1)
            _glm = pyglmnet.GLM(distr="poisson", verbose=True, alpha=alpha, score_metric="pseudo_R2",
                                reg_lambda=_lambda, max_iter=10000)  # , solver="cdfast")
            _glm.fit(X=dm_train, y=cell_train_binned)
            rate_pred_onset = _glm.predict(X=dm_test)
            pseudo_r2 = _glm.score(X=dm_test, y=cell_test_binned)
            centers_test_bin = np.arange(n_bins_test) * (dt / 60)
            plt.stem(centers_test_bin[1100:1200], cell_test_binned[1100:1200])
            plt.plot(centers_test_bin[1100:1200], rate_pred_onset[1100:1200], linewidth=0.5, c="purple")
            plt.title(f"prediction for Cell#{cell}")
            plt.savefig(os.path.join(options.folder, f"pred_{file_name}.png"), dpi=240)
            plt.close()
            n_bin_spec = len_pad
            spec = np.empty((len(u_binned_test[2:]), n_bin_spec))
            spec_pred = np.empty((len(u_binned_test[2:]), n_bin_spec))
            for i, elt in enumerate(u_binned_test[2:]):
                x = np.zeros(n_bin_spec)
                x_pred = np.zeros(n_bin_spec)
                bin_tone = np.where(binned_stim_test == elt)[0]
                for foo in bin_tone[:-1]:
                    x += cell_test_binned[foo:foo + n_bin_spec]
                    x_pred += rate_pred_onset[foo:foo + n_bin_spec]
                x /= len(bin_tone)
                x_pred /= (len(bin_tone) - 1)
                spec[i] = x
                spec_pred[i] = x_pred
            plt.figure(figsize=[12, 4])
            fig, axes = plt.subplots(1, 2)
            axes[0].pcolormesh(spec)
            axes[0].set_title("Ground Truth")
            axes[1].pcolormesh(spec_pred)
            axes[1].set_title(f"Prediction r2 = {pseudo_r2:.4f}")
            plt.savefig(os.path.join(options.folder, f"pred_spec_{file_name}.png"), dpi=240)
            plt.close()

        # rate_pred_onset = glm.get_rate_prediction(cst_onset, dm_test, stim_filters_1_vec)
        # r2 = best_glm.score(dm_test, cell_test_binned)
        # plt.figure(figsize=[12, 4])
        # plt.semilogx(glms.reg_lambda, glms.scores_, c="r")
        # plt.savefig(os.path.join(options.folder, f"lambdas_{file_name}.png"), dpi=240)
        # plt.close()
        # idx_best = np.argmax(glms.scores_)
        # best_glm = glms.glm_
        # cst_onset = glms.beta0_
        # stim_filters_1_vec = glms.beta_
        # if options.spike_history:
        #     dm_test = np.concatenate((dm_test, dm_sh_test), axis=1)
        #     file_name += f"_sh"
        # stim_filters_1 = np.reshape(stim_filters_1_vec[:len_pad * n_tones], (n_tones, len_pad))
        # rate_pred_onset = best_glm.predict(dm_test)
        # rate_pred_onset = glm.get_rate_prediction(cst_onset, dm_test, stim_filters_1_vec)
        # r2 = best_glm.score(dm_test, cell_test_binned)
        # positions = np.load(os.path.join(options.folder, "clean_pos.npy"))
        # triggers_pos = np.load(os.path.join(options.folder, "test_2_trigger_frame.npy"))
        # mu_pos = positions.mean()
        # std_pos = positions.std()
        # idx_train = np.logical_and(triggers_pos == triggers_train[0], triggers_pos < triggers_train[-1])
        # triggers_pos_train = triggers_pos[idx_train]
        # positions_train = positions[idx_train]
        # idx_test = np.logical_and(triggers_pos >= triggers_test[0], triggers_pos < triggers_test[-1])
        # triggers_pos_test = triggers_pos[idx_test]
        # positions_test = positions[idx_test]
        # p_train = np.zeros_like(samples_train)
        # triggers_pos_train -= triggers_pos_train[0]
        # next_idx = 0
        # for i in range(len(bins_train) - 1):
        #     tmp = positions_train[np.logical_and(bins_train[i] <= triggers_pos_train,
        #                                          bins_train[i + 1] >= triggers_pos_train)]
        #     if tmp.size != 0:
        #         next_idx = tmp[-1]
        #         p_train[i] = np.mean(tmp)
        #     if tmp.size == 0:
        #         p_train[i] = next_idx
        # p_train = p_train.astype(dtype=np.float64)
        # p_train -= mu_pos
        # p_train /= std_pos
        # p_test = np.zeros_like(samples_test)
        # triggers_pos_test -= triggers_pos_test[0]
        # next_idx = positions_test[0]
        # for i in range(len(bins_test) - 1):
        #     tmp = positions_test[np.logical_and(bins_test[i] <= triggers_pos_test,
        #                                         bins_test[i + 1] >= triggers_pos_test)]
        #     if tmp.size != 0:
        #         next_idx = tmp[-1]
        #         p_test[i] = np.mean(tmp)
        #     if tmp.size == 0:
        #         p_test[i] = next_idx
        # p_test = p_test.astype(dtype=np.float64)
        # p_test -= mu_pos
        # p_test /= std_pos
        # thread_pool = list()
        # for i, n_cell in enumerate(n_cells):
        #     cell_train = spk.get_spike_times_between(cluster=n_cell, t_0=triggers_train[0],
        #                                              t_1=samples_train[-1] + triggers_train[0], zero=True)
        #     cell_train_binned = glm.bin_spikes(cell_train, bins_train)
        #     print(f"Cellule #{n_cell}, Activité moyenne: {cell_train_binned.mean()}")
        #     # Les threads sont créés ici.
        #     glm_thread = GLMThread(f"Cluster {i}", cell_train_binned, dm_train, len_pad_sh, options.spike_history)
        #     thread_pool.append(glm_thread)
        #     # glm_thread.run()
        #
        # for thread in thread_pool:
        #     thread.run()
