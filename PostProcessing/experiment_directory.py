import json
import os
import glob
from pathlib import Path


DATA_PATH = "data"


def add_zero(num, max_length):
    str_num = str(num)
    try:
        num_zero = max_length - len(str_num)
        if num_zero < 0:
            raise IOError
    except IOError:
        print("error")
        return str_num
    return num_zero * "0" + str_num


def create_day_name(name, day):
    return name + "_" + day


def add_session(name_day, session_number):
    return name_day + "_" + "SESSION" + add_zero(session_number, 2)


def load(name, date, session):
    path = load_animal(name)
    path = path.joinpath(date)


def load_animal(name):
    path = Path.home().joinpath(DATA_PATH)
    if path.joinpath(name).is_dir():
        return path.joinpath(name)


def days(path):
    path.glob("*")
