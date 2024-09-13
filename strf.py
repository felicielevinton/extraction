import numpy as np
import mne
from mne.decoding import ReceptiveField
from pathlib import Path
import librosa
import sounddevice as sd


def find_spikes(spikes, t_0, t_1, trig=None, trigger_unit="s", fs=30e3):
    """
    Prend en arguments des temps exprimés en nb de samples

    """
    if trigger_unit == "s":
        t_0 = int(t_0 * fs)
        t_1 = int(t_1 * fs)
    if trig is not None:
        t_0 = trig - t_0
        t_1 = trig + t_1
    else:
        trig = t_0
    x = np.where(np.logical_and(spikes > t_0, spikes < t_1))[0]
    # LOGICAL_AND.non_zero() plus rapide peut-être?
    x = spikes[x]
    x -= trig
    x = x.astype(np.double)
    x /= fs
    return x


def _get_trigger_double(trigger):
    start, stop = trigger
    return start, stop


def _get_trigger_duration_mode(trigger, length):
    start, stop = trigger, trigger + length
    return start, stop


def _make_length_triggers(triggers, length):
    l_triggers = list()
    for trigger in triggers:
        l_triggers.append((trigger, trigger + length))
    return l_triggers


def activity(spikes, triggers, duration_stimuli, trigger_type="duration", t_bin=0.005, fs=30e3):
    """

    @param spikes
    @param triggers
    @param trigger_type
    @param duration_stimuli
    @param t_bin
    @param fs
    """
    if trigger_type == "duration":
        length = int(duration_stimuli * fs)
        triggers = _make_length_triggers(triggers, length)

    n_bin = int(duration_stimuli / t_bin)
    hist = np.empty((0, n_bin))
    for trigger in triggers:
        start, stop = trigger
        x = find_spikes(spikes, t_0=start, t_1=stop)
        h, b = np.histogram(x, n_bin)
        hist = np.vstack((hist, h))
    return hist


def load_torcs(path, sr=96e3):
    p = Path(path)
    wav_list = list(p.glob("*.wav"))
    sound_dict = {wav.stem: librosa.load(wav, sr=sr, mono=False)[0][0] for wav in wav_list}
    return sound_dict


def load_sequence_file():
    # sera un fichier .bin
    pass


# on prend la séquence de torcs joués
def down_sampling_audio(sequence, sound_dict):
    sound_dict = {key: mne.filter.resample(sound_dict[key]) for key in sound_dict.keys()}
    x = [sound_dict[elt] for elt in sequence]
    return x


def compute_strf(stimuli, neural_activity):
    # https://mne.tools/dev/generated/mne.decoding.ReceptiveField.html#mne.decoding.ReceptiveField
    # dans l'exemple: X est l'audio = Torcs sous échantilloné
    # y = activité
    t_min = 0.0  # défini a priori
    t_max = 0.0  # défini a priori
    sfreq = 0.0  # sampling rate / n_decim
    freqs = 0.0  # nombre de fréquences à traiter...
    x_train, x_test = stimuli[:-1], stimuli[-1]
    y_train, y_test = neural_activity[:-1], neural_activity[-1]
    x, y = [np.rollaxis(ii, -1, 0) for ii in (stimuli, neural_activity)]
    alphas = np.logspace(-3, 3, 7)
    scores = np.zeros_like(alphas)  # scores?
    models = list()  # models?
    for ii, alpha in enumerate(alphas):
        rf = ReceptiveField(t_min, t_max, sfreq, freqs, estimator=alpha)
        rf.fit(x, y)
        scores[ii] = rf.score(x, y)
        models.append(rf)

    times = rf.delays_ / float(rf.sfreq)
    ix_best_alpha = np.argmax(scores)
    best_model = models[ix_best_alpha]
    coefs = best_model.coef_[0]
    best_prediction = best_model.predict(x_test)[:, 0]
    # après avoir calculé les strf, comment choisir le meilleur modèle sur des critères objectifs
    # on compare la meilleure prédiction de l'activité à partir d'un son.
    # mne semble avoir des fonctions pour ça


def select_best_model():
    pass
