class Ferret(object):
    """
    Cette classe va créer un container d'objets sessions. On pourra essayer d'obtenir
    des données sur plusieurs sessions comme ça => Analyse d'une évolution dans le temps e.g.
    """

    def __init__(self, name, directories):
        self.name = name
        self.directories = directories
        self.n_sessions = len(directories)

    def __repr__(self):
        out = f"Name: {self.name}\nn_sess: {self.n_sessions}"
        return out

    def __str__(self):
        pass

    def load(self):
        pass


