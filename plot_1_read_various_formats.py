"""
Read various format into SpikeInterface
=======================================

SpikeInterface can read various formats of "recording" (traces) and "sorting" (spike train) data.

Internally, to read different formats, SpikeInterface either uses:
  * a wrapper to `neo <https://github.com/NeuralEnsemble/python-neo>`_ rawio classes
  * or a direct implementation

Note that:

  * file formats contain a "recording", a "sorting",  or "both"
  * file formats can be file-based (NWB, ...)  or folder based (SpikeGLX, OpenEphys, ...)

In this example we demonstrate how to read different file formats into SI
"""

import matplotlib.pyplot as plt

import spikeinterface.core as si
import spikeinterface.extractors as se

##############################################################################
# Let's download some datasets in different formats from the
# `ephy_testing_data <https://gin.g-node.org/NeuralEnsemble/ephy_testing_data>`_ repo:
#
#   * MEArec: a simulator format which is hdf5-based. It contains both a "recording" and a "sorting" in the same file.
#   * Spike2: file from spike2 devices. It contains "recording" information only.


file_path = '/mnt/working2/felicie/Python_theremin/Analyse/Analyse/Experiment/BURRATA/BURRATA_20240410_SESSION_04/ephys.rhd'
recording = se.IntanRecordingExtractor(file_path, stream_name='RHD2000 amplifier channel')


##############################################################################
# Now that we have downloaded the files, let's load them into SI.
#
# The :py:func:`~spikeinterface.extractors.read_spike2` function returns one object,
# a :py:class:`~spikeinterface.core.BaseRecording`.
#
# Note that internally this file contains 2 data streams ('0' and '1'), so we need to specify which one we
# want to retrieve ('0' in our case).
# the stream information can be retrieved by using the :py:func:`~spikeinterface.extractors.get_neo_streams` function.



##############################################################################
# The :py:func:`~spikeinterface.extractors.read_mearec` function returns two objects,
# a :py:class:`~spikeinterface.core.BaseRecording` and a :py:class:`~spikeinterface.core.BaseSorting`:

recording, sorting = se.read_mearec(mearec_folder_path)
print(recording)
print(type(recording))
print()
print(sorting)
print(type(sorting))

##############################################################################
#  The :py:func:`~spikeinterface.extractors.read_mearec` function is equivalent to:

recording = se.MEArecRecordingExtractor(mearec_folder_path)
sorting = se.MEArecSortingExtractor(mearec_folder_path)

##############################################################################
# SI objects (:py:class:`~spikeinterface.core.BaseRecording` and :py:class:`~spikeinterface.core.BaseSorting`)
# can be plotted quickly with the :py:mod:`spikeinterface.widgets` submodule:

import spikeinterface.widgets as sw

w_ts = sw.plot_traces(recording, time_range=(0, 5))
w_rs = sw.plot_rasters(sorting, time_range=(0, 5))

plt.show()
