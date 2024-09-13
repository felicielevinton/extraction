import spikeinterface.full as si
#import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spiketoolkit
import spikeinterface.widgets as sw
import numpy as np
from convert_positions_in_tones import *
from utils_extraction import *
import numpy as np
import spikeinterface
import zarr as zr
import os
from  pathlib import Path
import tqdm
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from probeinterface import Probe, ProbeGroup
import matplotlib
import pickle
from spikeinterface import extractors
from spikeinterface.sorters import run_sorter

matplotlib.use('Agg')
sr=30e3
# adresse pour OSCYPEK : /mnt/working2/felicie/data2/eTheremin/OSCYPEK/OSCYPEK

root = '/mnt/working2/felicie/data2/eTheremin/ALTAI/ALTAI_20240910_SESSION_00/'
path = root+'headstage_0/' 
#path=root
#neural_data = np.load(root +'/neural_data.npy')
#pour Burrata : s
neural_data = np.load(path +'/neural_data.npy')
sig = neural_data


def get_fma_probe():
    """ Implements the FMA probe using Probeinterface.
    """
    ### The following distances are in mm:
    inter_hole_spacing = 0.4  # along one row
    inter_row_spacing = np.sqrt(0.4**2-0.2**2)  # between row/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240419_SESSION_01/headstage_1/psth_figure_spikeinterface.png

    # We need to remove the ground from these positions:
    mask = np.zeros((16, 2), dtype=bool)
    mask[[0, 15], [0, 1]] = True
    positions = positions[np.logical_not(mask.reshape(-1))]

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions,
                       shapes='circle', shape_params={'radius': 100})
    polygon = [(0, 0), (0, 16000), (800, 16000), (800, 0)]
    probe.set_planar_contour(polygon)

    ## This mapping is ordered from 32 to 1, so we use[::-1] to invert it and make it 1 to 32
    mapping = np.arange(32).reshape(16, 2)[::-1].reshape(-1)
    mapping = mapping[np.logical_not(mask.reshape(-1))]

    probe.set_device_channel_indices(mapping)

    return probe


n_cpus = os.cpu_count()

full_raw_rec = se.NumpyRecording(traces_list=np.transpose(sig), sampling_frequency=sr)
#raw_rec = full_raw_rec.set_probe(probe)
#raw_rec = raw_rec.remove_channels(["CH2"])
recording_cmr = si.common_reference(full_raw_rec, reference='global', operator='median')
recording_f = si.bandpass_filter(full_raw_rec, freq_min=300, freq_max=9000)

recording_ss = run_sorter(sorter_name='waveclus', recording=recording_f)