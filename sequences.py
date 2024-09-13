from experiment_type import *
from extraction_utils import *
from abc import ABC, abstractmethod

# Comment sauvegarder les positions et les triggers positions dans un même .npz ?


class BuildSession(object):
    """
    Pendant le processus de création de la session. À la première extraction.
    """
    def __init__(self, folder):
        self.folder = folder
        self.fn = os.path.join(self.folder, "{}_{}_session.npz")  # pour la sauvegarde.
        self.vector_xp = list()

    def build(self):
        """
        Lit le .json. Extrait les triggers.
        :return:
        """
        # Quand il y aura plusieurs expériences par Session. Il faudra boucler sur les "Experiment_"
        log = read_log_file(self.folder)
        n_exp, beacons = get_n_of_experiment(log)
        output = extract(self.folder)
        # ATTENTION : faire attention aux triggers cond et exp.

        for i, beacon in enumerate(beacons):
            xp_type_str = get_exp_type(log, beacon)  # On connaît le type de l'expérience.
            xp_type = get_from_str(xp_type_str)

            # Calculer n_iter.
            # Trouver n_iter avec les chiffres accolés à une balise.
            allowed = get_allowed_keywords(xp_type)

            # Extraire les fichiers fréquences.
            if xp_type != ExperimentType.PAUSE:
                tones = get_tones(self.folder, log, allowed, beacon)

            # Extraire les fichiers positions.
            positions_fn = get_positions(self.folder, log, allowed, beacon)

            if xp_type == ExperimentType.PAUSE:
                self._build_pause(extracted=output, positions=positions_fn, num_exp=i)

            elif xp_type == ExperimentType.TRACKING:
                pass

            elif xp_type == ExperimentType.PLAYBACK:
                pass

    def save(self):
        pass


    def _build_pause(extracted, positions, num_exp):
        """
        Construite une Session Pause.
        :param extracted:
        :param positions:
        :param num_exp:
        :return:
        """
        digital_triggers = extracted["DIGITAL"]
        pause_triggers = digital_triggers["XP_PAUSE"][num_exp]
        start, stop = pause_triggers[0], pause_triggers[1]
        camera_triggers = np.logical_and(digital_triggers["BASLER"] >= start, digital_triggers["BASLER"] <= stop)
        pause_positions = positions["pause"][num_exp]
        assert (len(pause_positions) == len(camera_triggers))
        # todo : Ne pas oublier d'enlever les triggers extraits.
        # ajouter dans le vecteur


    def _build_tracking(extracted, positions, num_exp):
        pass

    def _build_playback(extracted, positions, num_exp, n_iter):
        pass


    def _build_silence(extracted, positions, num_exp, n_iter):
        pass


    def _build_mapping_change(extracted, positions, num_exp, n_iter):
        pass


class LoadSession(object):
    """
    Charger le .npz et crée le bon objet de Session.
    """
    def __init__(self, folder):
        self.folder = folder
        self.fn = os.path.join(self.folder, "tt.npz")
        assert(os.path.exists(self.fn)), f"Cannot Find a sequence file in {self.folder}"

    def load(self):
        """
        Méthode qui va retourner le bon objet.
        :return: Héritage d'AbstractSession.
        """
        d = np.load(self.fn, allow_pickle=True)

        xp_type = d["Type"]

        if xp_type == ExperimentType.PURE_TONES.value:
            seq = Tonotopy(d)

        elif xp_type == ExperimentType.PLAYBACK.value:
            seq = Playback(d)

        elif xp_type == ExperimentType.SILENCE.value:
            seq = Silence(d)

        elif xp_type == ExperimentType.TRACKING.value:
            seq = Tracking(d)

        elif xp_type == ExperimentType.MAPPING_CHANGE.value:
            seq = MappingChange(d)

        else:
            seq = None

        return seq


class Trial(object):
    """

    """
    def __init__(self, tones, triggers, positions, positions_triggers, type_of, number=None, order=None):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        assert (len(positions) == len(positions_triggers))
        self.tones = tones
        self.triggers = triggers
        self.positions = positions
        self.positions_triggers = positions_triggers
        self.quad = Quad(tones, triggers, positions, positions_triggers)

        assert (type_of in ["pause", "playback", "tracking", "warmup", "warmdown", "mock", "PureTones", "silence"]), "Wrong type..."
        self.type = type_of

        if order is not None:
            self.order = order
        else:
            self.order = None

        if number is not None:
            self.number = number
            self.pattern = get_pattern_from_type(self.type) + str(self.number)
        else:
            self.number = None
            self.pattern = None

    def get_stacked(self):
        return np.vstack((self.tones, self.triggers))

    def get_tones(self):
        return self.tones

    def get_triggers(self):
        return self.triggers

    def get_pairs(self):
        return self.quad

    def get_pattern(self):
        return self.pattern

    def get_type(self):
        return self.type

    def get_positions(self):
        return self.positions

    def get_positions_triggers(self):
        return self.positions_triggers

    def get_begin_and_end_triggers(self):
        return self.triggers[0], self.triggers[-1]


class Quad(object):
    def __init__(self, tones, triggers, positions, positions_triggers):
        assert (len(tones) == len(triggers)), "Tones and Triggers have different length."
        self.tones = tones
        self.triggers = triggers
        self.positions = positions
        self.positions_triggers = positions_triggers
