import os
from pathlib import Path


BASE = "C:/Users/Flavi/data/EXPERIMENT/"


def get_good_sessions(ferret):
    good_sessions = list()
    ferret_dir = Path(BASE)
    ferret_dir = Path(os.path.join(BASE, ferret))
    days = list(ferret_dir.glob(ferret + "*"))
    to_find = ["tt.npz", "spike_clusters.npy", "spike_times.npy"]
    for day in days:
        day_dir = Path(day)
        sessions = list(day_dir.glob(ferret + "*"))
        for session in sessions:
            go = 0
            session_dir = Path(session)
            files = list(session_dir.glob("*"))
            files = [os.path.split(file)[-1] for file in files]
            for elt in to_find:
                if elt in files:
                    go += 1
            if go == len(to_find):
                good_sessions.append(session)
    return good_sessions, ferret_dir.resolve().as_posix()

