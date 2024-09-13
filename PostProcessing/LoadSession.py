import json
import numpy as np


class LoadSession(object):
    def __init__(self, path):
        # Tester si Json ou pas.
        self.path = path
        self.session = None
        self.load()
        pass

    def load(self):
        with open(self.path, "r") as f:
            self.session = json.load(f)


