import json
import os
import glob


class LogSessionReader(object):
    def __init__(self):
        self.kwords_session = ["Name", "Date", "Session ended correctly",
                               "Experiment_",
                               "Positions saving enabled", "Binary path"
                               "Ephys recording enabled", "Ephys folder",
                               "Video recording enabled", "Video recording path"]

    def foo(self):
        pass


# ['Experience has started', 'Experiment ended correctly',
# 'Mapping', 'Mid tone', 'Num frequencies', 'Num octaves', 'PerturbationMapping_',
# 'Type']
class ExpLogReader(object):
    def __init__(self):
        self.kwords_exp = []


# gestion de l'héritage ?
class PauseReader(object):
    def __init__(self):
        pass


def load_json(path):
    with open(path, "r") as f:
        json_out = json.load(f)
    return json_out


if __name__ == "__main__":
    data = load_json(path="../../../playground/exp_files/session_FOO_20220227.json")
    # print(data.keys())
    print(data["Experiment_1"].keys())
