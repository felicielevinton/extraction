import cairo
import re
import numpy as np

# TODO créer un carré et le remplir selon un code couleur.

SQUARE_SIZE = 32


def colormap_save(txt_file, name_for_saving):
    with open(txt_file, "r") as f:
        lines = f.readlines()

    cmp = list()

    for line in lines:
        cmp.append([int(s) for s in line.split() if s.isdigit()])
    cmp_np = np.array(cmp)
    cmp_np /= 255
    np.save(name_for_saving, cmp_np)


def convert_values_for_colorspace(data):
    """
    Multiply by 128 and add 128
    :param data: array
    :return:
    """
    copy_data = data - data.min()
    copy_data /= copy_data.max()

    # data_256


def colormap_hot():
    pass


def contours(electrodes):
    """
    Retourner une surface à la place.
    :param electrodes: tuple for electrodes disposition
    :return: surface
    """
    width = electrodes[0] * SQUARE_SIZE
    height = electrodes[1] * SQUARE_SIZE
    step_x = 1 / electrodes[0]
    step_y = 1 / electrodes[1]
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                 width,
                                 height)
    return surface


def draw_square(width, height, n_square_width, n_square_height):
    # on passe la taille totale du contexte, la taille d'un carré
    # le nombre de carrés
    pass


WIDTH, HEIGHT = 256, 256
square_size = 32
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
ctx.scale(WIDTH, HEIGHT)
pat = cairo.SolidPattern(1, 1, 1, 1.0)
# pat.add_color_stop_rgba(1, 0.7, 0, 0, 0.5)
# pat.add_color_stop_rgba(0, 0.9, 0.7, 0.2, 1)
ctx.rectangle(0, 0, 1, 1)  # Rectangle(x0, y0, x1, y1)
ctx.set_source(pat)
ctx.fill()
# square = cairo.Rectangle(0, 0, 1, 1)
# ctx.translate(0.1, 0.1)
step = 0.125
x_coord = np.arange(0, 1 + step, step)
y_coord = np.arange(0, 1 + step, step)
ctx.move_to(0, 0)
ctx.rectangle(0, 0, step, step)
ctx.set_source_rgb(0.2, 0.3, 0.5)
ctx.fill()

# for x in x_coord:
#     for y in y_coord:
#         ctx.rectangle(x, y, step, step)
# ctx.rectangle(0, 0, 0.125, 0.125)
# ctx.rectangle(0.125, 0, 0.125, 0.125)
# ctx.arc(0.2, 0.1, 0.1, -math.pi / 2, 0)
# ctx.line_to(0.5, 0.1)
# ctx.curve_to(0.5, 0.2, 0.5, 0.4, 0.2, 0.8)
ctx.close_path()
ctx.set_source_rgb(0, 0, 0)  # Solid color
ctx.set_line_width(0.005)
ctx.stroke()
surface.write_to_png("png/example.png")