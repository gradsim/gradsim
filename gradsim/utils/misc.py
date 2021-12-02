import colorsys
import random
import string


def random_bright_color():
    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
    r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
    return r, g, b


def random_string(stringLength=8):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))
