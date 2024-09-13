import numpy as np
from scipy import signal
from kneed import KneeLocator
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import os
from tqdm import tqdm


gain = 0.195
offset = 32768


class Cluster(object):
    """
    Objet qui représente un cluster nouvellement détecté.
    """
    def __init__(self, cluster_number, channel_number, times, heights, pcs, template):
        self.cluster = cluster_number
        self.channel = channel_number
        self.times = times
        self.heights = heights
        self.template = template
        pass

    def get_templates(self, fd):
        pass


class Clustering(object):
    """
    Gestion de la sortie du Spike Sorting
    """
    def __init__(self, fd):
        self.descriptor = fd
        self.n_cluster = 0
        self.cluster_list = list()
        self.clusters = dict()
        self.files = {
            "tfsi": "template_feature_spikes_ids.npy",
            "tfi": "template_feature_ind.npy",
            "t": "templates.npy",
            "tf": "template_features.npy",
            "sp_tmp": "spike_templates.npy",
        }
        pass

    def create_new_cluster(self, channel_num, times, heights, pcs):
        new_cluster = Cluster(self.n_cluster, channel_number=channel_num, times=times, heights=heights, pcs=pcs)
        self.cluster_list.append(self.n_cluster)
        self.n_cluster += 1
        pass

    def _save_templates_file(self, folder):
        """
        Quatre fichiers de templates à créer (un 5ème est similar_templates.npy),
        / templates.npy => [n_cluster, n_samples, n_channel], juste les templates moyennés.
        / template_feature_ind.npy =>
        / template_features.npy => sortir la PCA des templates.
        / template_feature_spikes_ids.npy =>
        """
        pass

    def _save_spikes(self):
        """
        Trois fichiers:
        / spike_clusters  ??
        / spike_templates ??
        / spike_times
        """
        pass

    def _save_pcs(self):
        """
        pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single matrix giving the PC values for each spike.
        The channels that those features came from are specified in pc_features_ind.npy. E.g. the value
        at pc_features[123, 1, 5] is the projection of the 123rd spike onto the 1st PC on the channel
        given by pc_feature_ind[5].
        pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32 matrix specifying which pcFeatures
        are included in the pc_features matrix.
        """


class Template(object):
    def __init__(self, tag, template, pcs):
        self.tag = tag
        self.template = template
        self.pcs = pcs

    def get_template(self):
        return self.template

    def get_tag(self):
        return self.tag

    def pcs(self):
        return self.pcs


class TemplatesContainer(object):
    """
    Gestion des templates trouvés lors de la phase de template matching.
    """
    def __init__(self, channel_number):
        self.channel_number = channel_number
        self.tags = list()
        self.templates_container = dict()

    def add_template(self, tag, template, pcs):
        assert (tag not in self.tags), "Template already registered."
        assert (len(template) == 64), "Template is not 64 samples long."
        self.templates_container[tag] = Template(tag=tag, template=template, pcs=pcs)
        self.tags.append(tag)

    def get_available_templates(self):
        if -1 in self.tags:
            # Il y a un cluster de bruit.
            return len(self.tags) - 1
        else:
            return len(self.tags)

    def get_template(self, tag):
        """
        Attention vérifier le retour de cette méthode. Si None,
        le cluster numéro Tag n'existe pas dans le container. Ne lève pas d'exception.
        """
        if tag in list(self.templates_container.keys()):
            return self.templates_container[tag].get_template()
        else:
            return None


def load_spike_data(path):
    ext = os.path.splitext(path)[-1]
    if ext == ".dat":
        x = np.memmap(path, dtype=np.uint16, mode="r", offset=0, order="C")
        return x.reshape((32, len(x) // 32))
    else:
        return np.lib.format.open_memmap(path, mode="r")


def from_chan_and_spk_times_get_templates(fd, ):
    """
    Passer un descripteur de fichier en arguments, des spikes times
    """
    pass


def convert(x):
    if x.dtype == np.uint16:
        x = np.multiply(gain, x.astype(np.int32) - offset)
    return x


def filtering(x):
    """
    x is a channel.
    Filtre forward backward avec filtre digital passe bande Butterworth d'ordre 3 entre 300Hz et 3kHz.
    """
    # sos = signal.butter(N=3, Wn=[300, 3000], fs=30e3, btype="bandpass", output="sos")
    sos = signal.ellip(N=2, rp=1, rs=100, Wn=[300, 3000], fs=30e3, btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x, padtype="odd")


def filtering_script(x, n_channel, folder):
    """
    Prend un numpy array en entrée.
    Sauvegarde ce ndarray après filtrage.
    """
    filename = os.path.join(folder, "ephys.dat")
    fp = np.memmap(filename, dtype=np.uint16, mode="w+", shape=x.shape)
    for i in tqdm(range(n_channel)):
        y = x[i]
        fp[i] = filtering(y)

    return fp


def common_average_referencing(fd, n_channel=32):
    mean = 0
    pass


def threshold_quiroga_mode(x):
    """
    Moins sensible que le RMS aux cellules aux forts taux de décharge.
    """
    return 5 * np.median(np.absolute(x) / 0.6745)


def find_peaks(chan, threshold, distance=0.0005, fs=30e3):
    """

    """
    threshold = np.abs(threshold)
    distance *= fs
    opp_chan = chan * -1
    _spk_times, _height = signal.find_peaks(opp_chan, threshold, distance=distance)
    return _spk_times, _height["peak_heights"] * -1


def extract_templates(x, spk_times, n_samples=64):
    """
    Extraction des formes d'ondes des évènements qui dépassent le seuil.
    """
    # function that allows to extract the waveforms of the spikes (for each peak)
    # in a window that contains sampling points
    # arguments:number of points around the spike point (sampling), number of peaks (n)
    # matrix containing the time of each peak (spk_times),data (sig)
    wf = list()
    before, after = 20, n_samples - 20  # 20 avant, 44 après. Permet d'avoir la phase de repolarisation.
    length = len(x)
    for elt in spk_times:  # à changer pour si jamais (d'ou le n-1)
        try:
            d_b = elt - before
            d_e = elt + after
            assert (d_b > 0 and d_e < length)
            wf.append(x[elt - before:elt + after])
        except AssertionError:
            continue

    wf = np.vstack(wf)
    return wf


def median_absolute_deviation(x):
    m = np.median(x)
    mad = np.absolute(x - m)
    mad = np.median(mad)
    return mad


def mad_norm(x):
    mad = median_absolute_deviation(x)
    return (x - np.median(x)) / mad


def template_matching(x, threshold, n_templates=2500, n_components=16):
    """
    C'est la fonction qui va se charger de récupérer les templates pour la réalisation d'un algorithme de template
    Matching.
    Principales étapes:
    1/ Extraction d'évènements. Avec un grand paramètre de distance. Pour permttre une bonne extraction de templates.
    2/
    retourne les templates trouvés.
    """
    # 1) extraire avec grande distance entre évènements détectés.

    spk_times, spk_heights = find_peaks(x, threshold=threshold, distance=0.015)
    # 2) sélectionner n_templates au hasard.
    if len(spk_times) < n_templates:
        pass
    else:
        idx = np.random.choice(np.arange(len(spk_times)), size=n_templates, replace=False)
        spk_times = spk_times[idx]
        spk_heights = spk_heights[idx]
    wf = extract_templates(x, spk_times)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(wf)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=25)
    labels = clusterer.fit_predict(pcs)
    labels_tag, counts = np.unique(labels, return_counts=True)
    t = TemplatesContainer(0)
    for label in labels_tag:
        idx = np.equal(labels, label)
        cluster = wf[idx]
        cluster -= cluster.mean()
        cluster /= cluster.std()
        t.add_template(tag=label, template=cluster.mean(0), pcs=pcs[idx])
    import matplotlib.pyplot as plt

    ex0 = t.get_template(0)
    ex1 = t.get_template(1)

    center_0 = l2_norm(ex0)
    center_1 = l2_norm(ex1)
    plt.plot(ex0)
    plt.plot(ex1)
    plt.show()
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(ex0)
    ax[0, 1].plot(ex1)
    foo = list()
    for i, elt in enumerate(wf):
        cc = np.array([correlation_coefficient(wf[i], ex0), correlation_coefficient(wf[i], ex1)])
        foo.append(cc)
    foo = np.vstack(foo)

    foo_l2 = list()
    for i, elt in enumerate(wf):
        cc = np.array()
    # faire ça sur les PCA??
    belongs = np.argmax(foo, axis=1)
    print("Hre")
    for i, elt in enumerate(wf):
        conv = signal.fftconvolve(wf[i], ex0[::-1], mode="same")
        if labels[i] == 0:
            c = "r"
        else:
            c = "k"
        ax[1, 0].plot(conv, c=c, linewidth=0.5)

    for i, elt in enumerate(wf):
        conv = signal.fftconvolve(wf[i], ex1[::-1], mode="same")
        if labels[i] == 1:
            c = "r"
        else:
            c = "k"
        ax[1, 1].plot(conv, c=c, linewidth=0.5)
    plt.show()
    return t


def correlation_coefficient(X, Y):
    assert(len(X) == len(Y))
    zX = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, keepdims=True)
    zY = (Y - np.mean(Y, axis=0, keepdims=True)) / np.std(Y, axis=0, keepdims=True)
    corr = np.sum(zX * zY) / len(X)
    return corr


def l2_norm(x):
    return np.sqrt(x ** 2)


def clustering():
    return


def contamination_ratio(list_sample, fs, refractory, censored, n_samples):
    """
    return ratio of contamination in 1 cluster base on its refractory period contamination, don't work very well for
    unit with freq < 1 Hz and N spike < 4000 (censored = 0.5e-3 and refractory 1e-3)

    it's supposed that contamination can be in refractory period of an other contamination spike

    - list_sample : list or np.array with sample of each spike of 1 cluster
    - sampling_rate : int
    - refractory_time : time considered as refractory time, in second.
    - censored_time : min time between spikes when there is the spike detection.
    - n_of_samples : number of sample in the recording.
    """

    assert (refractory > censored), "censored period is higher than refractory period"

    isi = np.diff(list_sample)
    isi_time = isi / fs

    # count event between censored time and refractory time
    ref_violation_count = np.nonzero((isi_time < refractory) * (isi_time > censored))[0].size

    ratio_contamination = (ref_violation_count * n_samples / fs) / (len(list_sample) ** 2 * (refractory - censored))

    return ratio_contamination
