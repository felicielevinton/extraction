import cairo

# TODO: create an image surface from a numpy array


class Square(object):
    def __init__(self):
        self.coordinates = None
        self.value = None

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value


class NpxSketch(object):
    def __init__(self):
        self.electrodes_number = 960
        self.line_number = 480
        self.column_number = 4
        pass