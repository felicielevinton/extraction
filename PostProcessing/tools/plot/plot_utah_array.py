import numpy as np
import matplotlib.pyplot as plt


def _draw_array():
    pass


class Electrode:
    """
    Base class for electrode on Utah Array.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


class NotConnected(Electrode):
    """
    Hérite de la classe électrode. Désigne une position de l'array qui n'est pas connectée.
    """
    def __init__(self, x, y):
        Electrode.__init__(self, x, y)
        self.color = "black"
        pass


class Broken(Electrode):
    """
    Electrode cassée, à déterminer selon l'impédance.
    """
    def __init__(self, x, y, pin):
        Electrode.__init__(self, x, y)
        self.pin = pin
        self.color = "silver"


class Array:
    """

    """
    def __init__(self, electrodes_per_array, array_number=1):
        pass

    def foo(self):
        pass
