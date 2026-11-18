"""
Gestures.py - Garabatos y figuras que NO son letras, para calibrar y probar el rechazo
(el sistema debe decir "no se reconocio ningun gesto" en vez de inventar una letra).
Unidades de x-height, como las muestras reales.
"""
import math

import numpy as np


def _smooth(x, w):
    pad = w // 2
    xp = np.pad(x, ((pad, w - 1 - pad), (0, 0)), mode="edge")
    k = np.ones(w) / w
    return np.stack([np.convolve(xp[:, d], k, mode="valid") for d in range(x.shape[1])], axis=1)


def random_curve(rng):
    n = int(rng.integers(120, 400))
    st = _smooth(rng.normal(0, 1, (n, 2)), int(rng.integers(8, 30)))
    st = st / (np.linalg.norm(st, axis=1, keepdims=True) + 1e-9) * rng.uniform(0.02, 0.05)
    return [np.cumsum(st, axis=0)]


def zigzag(rng):
    k = int(rng.integers(4, 10))
    xs = np.arange(k) * rng.uniform(0.2, 0.5)
    ys = np.where(np.arange(k) % 2 == 0, 0.0, rng.uniform(0.8, 2.0))
    return [np.column_stack([xs, ys])]


def spiral(rng):
    t = np.linspace(0, rng.uniform(2, 3.5) * 2 * math.pi, 160)
    r = 0.1 + 0.55 * t / (2 * math.pi)
    return [np.column_stack([r * np.cos(t), r * np.sin(t)])]


def long_line(rng):
    a, L = rng.uniform(0, 2 * math.pi), rng.uniform(1.5, 4.0)
    return [np.array([[0, 0], [L * math.cos(a), L * math.sin(a)]])]


def scribble(rng):
    pts = [np.zeros(2)]
    for _ in range(int(rng.integers(8, 20))):
        pts.append(pts[-1] + rng.normal(0, 0.5, 2))
    return [np.array(pts)]


def dots(rng):
    return [rng.uniform(0, 3, 2) + np.array([[0, 0], [0.02, 0.01]]) for _ in range(int(rng.integers(3, 7)))]


def polygon(rng):
    k = int(rng.integers(3, 6))
    a = np.sort(rng.uniform(0, 2 * math.pi, k))
    r = rng.uniform(0.6, 1.4)
    p = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return [np.vstack([p, p[:1]])]


FAMILIES = {"curva aleatoria": random_curve, "zigzag": zigzag, "espiral": spiral,
            "linea larga": long_line, "garabato": scribble, "puntos sueltos": dots,
            "poligono": polygon}


def make_gestures(n_per_family: int, rng):
    return [(name, f(rng)) for name, f in FAMILIES.items() for _ in range(n_per_family)]
