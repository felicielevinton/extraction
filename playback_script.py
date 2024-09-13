from playback import *
import argparse
import PostProcessing.tools.heatmap as hm
from get_all_good_sessions import get_good_sessions
import pandas as pd
import get_data as gd


def parse_args():
    parser = argparse.ArgumentParser(prog="Playback")
    parser.add_argument("--name", type=str, help="Ferret Name.")
    # parser.add_argument("--folders", type=str, nargs="+", help="Path to folder having data.")
    # parser.add_argument("--save", type=str, help="Folder to save data.")
    parser.add_argument("--compatibility", type=bool, help="Old data.")
    opt = parser.parse_args()
    return opt


def playback(folders, save_folder, compatibility=False):
    t_pre = 0.05
    t_post = 0.300
    bin_size = 0.005
    sr = 30e3
    plot_path = gd.check_plot_folder_exists(save_folder)

    for sess_num, folder in enumerate(folders):
        plot_path_session = gd.check_plot_folder_exists(folder)
        sequence = gd.extract_data(folder)

        recording_length = sequence.get_recording_length()
        spk = ut.Spikes(folder, recording_length=recording_length)

        pb = sequence.merge("playback")

        tr = sequence.merge("tracking")

        wp = sequence.merge("warmup")

        wd = sequence.merge("warmdown")

        # hm_warm

        hm_corr = hm.Heatmap()
        hm_corr.compute_heatmap(tone_sequence=np.hstack((wp.tones, tr.tones, wd.tones)),
                                trigs=np.hstack((wp.triggers, tr.triggers, wd.triggers)),
                                spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
        # hm_corr.plot_smooth("correlated", plot_path, num=sess_num, ext="png")
        hm_corr.plot_smooth_2d("correlated", plot_path_session, num=sess_num, ext="png")

        # hm_warmup = hm.Heatmap()
        # hm_warmup.compute_heatmap(tone_sequence=wp.tones, trigs=wp.triggers,
        #                           spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
        # hm_warmup.plot_smooth("warmup", plot_path, num=sess_num, ext="png")

        hm_tracking = hm.Heatmap()
        hm_tracking.compute_heatmap(tone_sequence=tr.tones, trigs=tr.triggers,
                                    spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)
        hm_tracking.save(folder=folder, typeof="tracking")
        hm_tracking.plot_smooth_2d("tracking", plot_path_session, num=sess_num, ext="png")

        hm_playback = hm.Heatmap()
        hm_playback.compute_heatmap(tone_sequence=pb.tones, trigs=pb.triggers,
                                    spikes=spk, t_pre=t_pre, t_post=t_post, bin_size=bin_size)

        hm_delta = hm.substract(hm_tracking, hm_playback)
        hm_delta.save(folder=folder, typeof="delta")
        hm_delta.plot_smooth_2d("delta", plot_path_session, num=sess_num, ext="png")

        hm_playback.save(folder=folder, typeof="playback")
        hm_playback.plot_smooth_2d("playback", plot_path_session, num=sess_num, ext="png")

        # hm_playback.plot_rl(sequence, spk, 0, plot_path, smooth=True)

        hm.psth_common_2(hm_tracking, hm_playback, sess_num, plot_path_session, smooth=True)
        hm.tc_common_3(hm_tracking, hm_playback, sess_num, plot_path_session)


if __name__ == "__main__":
    options = parse_args()
    good_dirs, save_dir = get_good_sessions(options.name)
    good_dirs.sort()
    playback(good_dirs, save_dir)
