"""
LetterTemplates.py - Trazo canonico (de una sola pasada) de cada letra, en unidades de x-height.
Se usan para (a) generar datos sinteticos de entrenamiento y (b) dibujar los iconos de referencia.
Cada letra es UN trazo continuo (sin levantar el lapiz): i/j/t/f/x/k/ñ se adaptan a ese esquema.
"""
import numpy as np

ALPHABET = list("abcdefghijklmnñopqrstuvwxyz")  # 27 letras


def arc(cx, cy, rx, ry, a0, a1):
    n = max(4, int(abs(a1 - a0) / 6))
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)


def line(*pts):
    return np.array(pts, dtype=float)


def join(*parts):
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p[1:] if np.allclose(out[-1][-1], p[0]) else p)
    return np.concatenate(out)


def _n_shape():
    return join(line((0, 0), (0, .7)), arc(.4, .7, .4, .3, 180, 0), line((.8, .7), (.8, 0)))


TEMPLATES = {
    "a": join(arc(.4, .5, .4, .5, 50, 360), line((.8, .5), (.8, 0)), line((.8, 0), (1.0, .15))),
    "b": join(line((0, 2), (0, 0)), line((0, 0), (0, .35)), arc(.4, .5, .4, .5, 195, -150)),
    "c": arc(.4, .5, .4, .5, 45, 315),
    "d": join(arc(.4, .5, .4, .5, 50, 360), line((.8, .5), (.8, 2))),
    "e": join(line((0, .5), (.8, .5)), arc(.4, .5, .4, .5, 0, 320)),
    "f": join(arc(.55, 1.7, .25, .3, 30, 180), line((.3, 1.7), (.3, 0))),
    "g": join(arc(.4, .5, .4, .5, 50, 360), line((.8, .5), (.8, -.5)), arc(.4, -.5, .4, .5, 0, -130)),
    "h": join(line((0, 2), (0, 0)), line((0, 0), (0, .7)),
              arc(.4, .7, .4, .3, 180, 0), line((.8, .7), (.8, 0))),
    "i": join(line((.2, 1), (.2, 0)), line((.2, 0), (.45, .25))),
    "j": join(line((.4, 1), (.4, -.6)), arc(0, -.6, .4, .4, 0, -150)),
    "k": join(line((0, 2), (0, 0)), line((0, 0), (0, .45)), line((0, .45), (.7, 1)),
              line((.7, 1), (.1, .5)), line((.1, .5), (.75, 0))),
    "l": line((.2, 2), (.2, 0)),
    "m": join(line((0, 0), (0, .7)), arc(.3, .7, .3, .3, 180, 0),
              line((.6, .7), (.6, 0)), line((.6, 0), (.6, .7)), arc(.9, .7, .3, .3, 180, 0),
              line((1.2, .7), (1.2, 0))),
    "n": _n_shape(),
    "ñ": join(_n_shape(), line((.8, 0), (.8, 1.15)), line((.8, 1.15), (.62, 1.5), (.4, 1.25), (.2, 1.55))),
    "o": arc(.4, .5, .4, .5, 80, 440),
    "p": join(line((0, -1), (0, 1)), line((0, 1), (0, .65)), arc(.4, .5, .4, .5, 165, -150)),
    "q": join(arc(.4, .5, .4, .5, 50, 360), line((.8, .5), (.8, -1))),
    "r": join(line((0, 0), (0, .65)), arc(.35, .65, .35, .35, 180, 50)),
    "s": join(arc(.4, .75, .35, .25, 25, 270), arc(.4, .25, .35, .25, 90, -160)),
    "t": join(line((.3, 1.7), (.3, .25)), arc(.55, .25, .25, .25, 180, 300)),
    "u": join(line((0, 1), (0, .4)), arc(.4, .4, .4, .4, 180, 360), line((.8, .4), (.8, 1)),
              line((.8, 1), (.8, 0))),
    "v": line((0, 1), (.4, 0), (.8, 1)),
    "w": line((0, 1), (.3, 0), (.6, .8), (.9, 0), (1.2, 1)),
    "x": line((0, 1), (.8, 0), (.8, 1), (0, 0)),
    "y": line((0, 1), (.4, 0), (.8, 1), (.1, -1)),
    "z": line((0, 1), (.8, 1), (0, 0), (.8, 0)),
}
assert sorted(TEMPLATES) == sorted(ALPHABET) and len(ALPHABET) == 27
