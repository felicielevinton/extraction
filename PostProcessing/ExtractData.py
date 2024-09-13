class ExtractData:
    """
    Classe ExtractData: se charge après le spike sorting. Va créer les fichiers de triggers,
    ceux de données neuronales. Va chercher les waveforms pour chaque unité = utilisation de np.memmap
    https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
    Va changer selon la méthode employée pour l'acquisition de données.
    """
    def __int__(self, name):
        self.name = name
        pass

    def get_last(self):
        """
        Va récupérer la dernière session.
        :return:
        """
        pass

    def get_triggers(self):
        pass

    def waveforms_files(self):
        pass

