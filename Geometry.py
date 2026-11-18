"""
Geometry.py - Rasgos geometricos globales de la letra, su verosimilitud por clase y un
refinador por pares. Solo NumPy.

El Chain Code (HMM) describe la FORMA local del trazo pero casi no "ve" el tamano global:
'l' (palo largo) vs 'i' (palo corto + punto), 'a' vs 'd' (solo cambia la altura del palo) o
'x' vs 'y' (mismo primer trazo, cambia el largo de la pierna) producen cadenas muy parecidas.

Rasgos (en unidades de x-height):
    altura, ancho, longitud total, n.o de trazos, tamano del ultimo trazo (un punto ~ 0),
    longitud del primer trazo, razon ultimo/primero (x vs y: la pierna de la y es mas larga),
    vtope, dx: donde cae el ultimo trazo respecto al PRIMERO (en unidades de la longitud del primero).
                El punto de la i cae arriba (+0.2); un toque suelto al final de la l cae abajo (-1.0).
                (fix de la l: sin esto una l con toque suelto es indistinguible de una i)

Cada clase = gaussiana diagonal:   log P(O|lambda_c) + GEO_WEIGHT * log N(f | mu_c, sigma_c)
"""
from typing import Dict, Sequence

import numpy as np

GEO_WEIGHT = 1.0
MIN_STD = 0.05
# Los rasgos de POSICION (vtope, dx) son mas variables de lo que dicen 150 muestras de una sola mano:
# quien escribe pequeno deja el punto de la i relativamente mas lejos. Tolerancia minima mayor.
POS_STD_MIN = 0.15
POS_DIMS = (7, 8)
FEATURE_NAMES = ["altura", "ancho", "longitud", "n_trazos", "tamano_ultimo_trazo", "long_primer_trazo",
                 "razon_ultimo/primero", "ultimo_vs_tope", "ultimo_dx"]


def _plen(s):
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0


def features(segments: Sequence[np.ndarray]) -> np.ndarray:
    """segments: lista de trazos (u, v) en unidades de x-height (v hacia arriba)."""
    segs = [np.asarray(s, dtype=float)[:, :2] for s in segments if len(s) >= 1]
    P = np.vstack(segs)
    h, w = float(np.ptp(P[:, 1])), float(np.ptp(P[:, 0]))
    L = sum(_plen(s) for s in segs)
    last = segs[-1]
    dot = float(np.ptp(last[:, 0]) + np.ptp(last[:, 1]))
    L1, Ll = _plen(segs[0]), _plen(last)
    ratio = Ll / L1 if (len(segs) > 1 and L1 > 0.05) else 1.0
    vtop = dx = 0.0
    if len(segs) > 1 and L1 > 0.05:
        first = segs[0]
        cl, cf = last.mean(axis=0), (first.max(axis=0) + first.min(axis=0)) / 2
        vtop = float(np.clip((cl[1] - first[:, 1].max()) / L1, -2.0, 2.0))
        dx = float(np.clip((cl[0] - cf[0]) / L1, -2.0, 2.0))
    return np.array([h, w, L, float(min(len(segs), 4)), dot, L1, min(ratio, 4.0), vtop, dx])


class GeometryModel:
    def __init__(self, mean: Dict[str, np.ndarray], std: Dict[str, np.ndarray]):
        self.names = list(mean.keys())
        self.mean = np.array([mean[n] for n in self.names])
        self.std = np.array([std[n] for n in self.names])
        if len(self.std):
            self._floor()

    @staticmethod
    def fit(feats_by_letter: Dict[str, np.ndarray]) -> "GeometryModel":
        mean = {c: F.mean(axis=0) for c, F in feats_by_letter.items()}
        std = {c: F.std(axis=0) + MIN_STD for c, F in feats_by_letter.items()}
        return GeometryModel(mean, std)

    def _floor(self):
        for d in POS_DIMS:
            self.std[:, d] = np.maximum(self.std[:, d], POS_STD_MIN)
        return self

    def loglik(self, f: np.ndarray, order: Sequence[str]) -> np.ndarray:
        idx = [self.names.index(n) for n in order]
        z = (f[None, :] - self.mean[idx]) / self.std[idx]
        return (-0.5 * z ** 2 - np.log(self.std[idx])).sum(axis=1)

    def to_dict(self) -> dict:
        return {"names": self.names, "mean": self.mean.tolist(), "std": self.std.tolist(),
                "weight": GEO_WEIGHT, "features": FEATURE_NAMES}

    @staticmethod
    def from_dict(d: dict) -> "GeometryModel":
        m = GeometryModel({}, {})
        m.names = list(d["names"])
        m.mean, m.std = np.array(d["mean"]), np.array(d["std"])
        return m._floor()


# ------------------------------------------------------------------ refinador por pares
PAIRS = [("x", "y"), ("d", "a"), ("p", "b"), ("r", "n"), ("h", "n"), ("l", "i")]


class PairRefiner:
    """Cuando las dos mejores candidatas son un par que el HMM confunde, decide una regresion
    logistica (gradiente descendente en NumPy) sobre los rasgos geometricos."""

    def __init__(self, params=None):
        self.params = params or {}     # "a|b" -> {mu, sd, w, b};  P(a) = sigmoid(w.z + b)

    @staticmethod
    def fit(feats_by_letter, pairs=PAIRS, iters=3000, lr=0.1):
        params = {}
        for a, b in pairs:
            if a not in feats_by_letter or b not in feats_by_letter:
                continue
            X = np.vstack([feats_by_letter[a], feats_by_letter[b]])
            y = np.r_[np.ones(len(feats_by_letter[a])), np.zeros(len(feats_by_letter[b]))]
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Z = (X - mu) / sd
            w, bias = np.zeros(Z.shape[1]), 0.0
            for _ in range(iters):
                p = 1 / (1 + np.exp(-(Z @ w + bias)))
                g = p - y
                w -= lr * (Z.T @ g / len(y) + 0.01 * w)
                bias -= lr * g.mean()
            params[f"{a}|{b}"] = {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(bias)}
        return PairRefiner(params)

    def prob_first(self, f, a, b):
        if f"{a}|{b}" in self.params:
            k, sign = f"{a}|{b}", 1
        elif f"{b}|{a}" in self.params:
            k, sign = f"{b}|{a}", -1
        else:
            return None
        P = self.params[k]
        z = ((np.asarray(f) - P["mu"]) / P["sd"]) @ np.asarray(P["w"]) + P["b"]
        p = 1 / (1 + np.exp(-z))
        return float(p if sign == 1 else 1 - p)

    def refine(self, ranking, f):
        """ranking [(letra, p)] -> el par top-2 se redecide; la masa del par se conserva."""
        (a, pa), (b, pb) = ranking[0], ranking[1]
        q = self.prob_first(f, a, b)
        if q is None:
            return ranking
        tot = pa + pb
        new = sorted([(a, tot * q), (b, tot * (1 - q))], key=lambda t: -t[1])
        return new + list(ranking[2:])

    def to_dict(self):
        return self.params

    @staticmethod
    def from_dict(d):
        return PairRefiner(d)
