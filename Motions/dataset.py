import pandas as pd
from enum import Enum
import tensorflow as tf
from pathlib import Path
import numpy as np


class MotionType(Enum):
    Steady = 0
    Right = 1
    Left = 2


def int_to_motion_type(label):
    motion_type = MotionType.Steady
    if label == 0:
        motion_type = MotionType.Steady
    elif label == 1:
        motion_type = MotionType.Right
    elif label == 2:
        motion_type = MotionType.Left
    return motion_type


def motion_type_to_array(motion_type):
    """
    Sera utile selon la sortie du réseau.
    """
    try:
        assert motion_type in MotionType
        raise AssertionError
    except AssertionError:
        print(f"{motion_type} is not a valid motion type.")
    x = np.zeros(3, dtype=int)
    if motion_type == MotionType.Steady:
        x[0] = 1
    elif motion_type == MotionType.Right:
        x[1] = 1
    elif motion_type == MotionType.Left:
        x[2] = 1
    return x


def normalize(move, image_width):
    """
    Pas une normalisation, mais je ne sais pas quel nom lui donner...
    On passe une vecteur de positions en int, on ressort un vecteur de positions en float,
    centrées sur 0. Les valeurs négatives (resp. positives) étant à gauche (resp. droite).
    """
    hw = image_width / 2
    move = (move - hw) / hw
    return move


def series_to_array(series, dtype):
    x = list()
    for serie in series:
        x.append(serie)
    return np.array(x)


class DatasetFactory(object):
    def __init__(self, n_points, path=None, load=None):
        self.columns = ["motions", "labels", "net_out"]
        if path:
            self.path = path
            if load:
                self.df = pd.read_excel(self.path)
        else:
            self.path = None
            self.df = pd.DataFrame(data=None, columns=self.columns)

    def append(self, move, label, image_width):
        move = normalize(move, image_width)
        motion_type = int_to_motion_type(label)
        motion_type = motion_type_to_array(motion_type)
        d = {self.columns[0]: [move], self.columns[1]: label, self.columns[3]: [motion_type]}
        tmp = pd.DataFrame(data=d, columns=self.columns, index=[self.df.shape[0]])
        self.df = pd.concat((self.df, tmp))

    def set_path(self, path):
        self.path = path

    def get_path(self):
        return self.path

    def print_num_examples(self):
        # self.df.columns
        pass

    def save(self, path=None):
        if not path:
            if not self.path:
                print("Error")
            else:
                self.df.to_excel(self.path)
        else:
            self.set_path(path)
        pass

    def to_tf_record(self):
        labels = self.df[self.columns[2]]  # "net_out"
        tensor_labels = tf.convert_to_tensor(labels)

        motions = self.df[self.columns[0]]  # "motions"
        tensor_motions = tf.convert_to_tensor(motions)

        # tf.io.serialize_tensor()

