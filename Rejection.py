"""
Rejection.py - "No se reconocio ningun gesto".  Solo NumPy.

Dos criterios independientes (basta uno para rechazar):

 1) UMBRAL DE PROBABILIDAD (sugerido por el profesor, 20 %):
        max_c P(c | O) < PROB_THRESHOLD        -> detecta AMBIGUEDAD (ninguna letra destaca).

 2) AJUSTE ABSOLUTO (solo con la letra terminada). El posterior es RELATIVO (suma 100 % entre
    las 27 letras): un garabato puede terminar con 95 % en alguna letra, solo dice "cual se
    parece mas", no "se parece".
        z  = (log P(O|lambda_c*)/T - mu_c) / sigma_c  <  Z_MIN     (la forma local no encaja)
        d2 = distancia de Mahalanobis de los rasgos a la clase c*  >  GEO_D2_MAX
    mu_c, sigma_c salen de las muestras de entrenamiento de la letra c.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

PROB_THRESHOLD = 0.20
Z_MIN = -3.0
GEO_D2_MAX = 50.0       # 9 rasgos (con los 2 de posicion): el limite crece con el numero de rasgos
Z_LIVE = -4.0           # ajuste absoluto DURANTE el trazo (mas tolerante: el trazo esta incompleto)
LEN_FACTOR = 1.35       # un trazo con mas simbolos que 1.35 x (la secuencia mas larga de esa letra) no es esa letra


@dataclass
class Verdict:
    recognized: bool
    reason: str = ""          # "", "ambiguo", "ajuste", "geometria", "vacio"
    z: float = 0.0
    d2: float = 0.0

    @property
    def message(self) -> str:
        return {
            "vacio": "Dibuja un trazo para empezar.",
            "ambiguo": "Ninguna letra supera el {:.0f} %.".format(PROB_THRESHOLD * 100),
            "largo": "El trazo es demasiado largo para ser una sola letra.",
            "ajuste": "El trazo no se parece a ninguna letra (la forma no encaja con ningún modelo).",
            "geometria": "El tamaño/proporción del trazo no corresponde a ninguna letra.",
        }.get(self.reason, "")


class Rejector:
    def __init__(self, mu: Dict[str, float], sd: Dict[str, float], prob_threshold: float = PROB_THRESHOLD,
                 z_min: float = Z_MIN, geo_d2_max: float = GEO_D2_MAX, len_max: Optional[Dict[str, int]] = None,
                 z_live: float = Z_LIVE, len_factor: float = LEN_FACTOR):
        self.mu, self.sd = mu, sd
        self.prob_threshold, self.z_min, self.geo_d2_max = prob_threshold, z_min, geo_d2_max
        self.len_max = len_max or {}
        self.z_live, self.len_factor = z_live, len_factor

    @staticmethod
    def fit(clf, sequences_by_letter: Dict[str, list], max_per_letter: int = 60) -> "Rejector":
        mu, sd, len_max = {}, {}, {}
        for ch, seqs in sequences_by_letter.items():
            len_max[ch] = int(max(len(s) for s in seqs))
            i = clf.names.index(ch)
            v = []
            for s in seqs[:max_per_letter]:
                clf.reset()
                clf.sync(list(s))
                v.append(clf.scores(final=True, with_extra=False)[i] / len(s))
            mu[ch], sd[ch] = float(np.mean(v)), float(np.std(v) + 1e-6)
        clf.reset()
        return Rejector(mu, sd, len_max=len_max)

    def prob_only(self, top_prob: float, n_symbols: int) -> Verdict:
        if n_symbols < 1:
            return Verdict(False, "vacio")
        if top_prob < self.prob_threshold:
            return Verdict(False, "ambiguo")
        return Verdict(True)

    def live_z(self, clf, best: str, n_symbols: int) -> float:
        """z del ajuste del modelo ganador con lo dibujado hasta ahora (sin exigir estado final)."""
        i = clf.names.index(best)
        ll = clf.scores(final=False)[i] / max(n_symbols, 1)
        return float((ll - self.mu[best]) / self.sd[best])

    def assess_live(self, clf, best: str, top_prob: float, n_symbols: int) -> Verdict:
        """Rechazo MIENTRAS se dibuja: umbral de probabilidad + largo + ajuste absoluto en vivo.
        Sin esto un garabato largo termina con 100 % en alguna letra (el posterior es relativo)."""
        v = self.prob_only(top_prob, n_symbols)
        if not v.recognized:
            return v
        if self.len_max and n_symbols > self.len_factor * self.len_max.get(best, 10 ** 9):
            return Verdict(False, "largo")
        z = self.live_z(clf, best, n_symbols)
        if z < self.z_live:
            return Verdict(False, "ajuste", z)
        return Verdict(True, "", z)

    def assess_final(self, clf, best: str, top_prob: float, n_symbols: int, geo=None,
                     feats: Optional[np.ndarray] = None) -> Verdict:
        v = self.prob_only(top_prob, n_symbols)
        if not v.recognized:
            return v
        i = clf.names.index(best)
        ll = clf.scores(final=True, with_extra=False)[i] / max(n_symbols, 1)
        z = float((ll - self.mu[best]) / self.sd[best])
        d2 = 0.0
        if geo is not None and feats is not None:
            k = geo.names.index(best)
            d2 = float((((feats - geo.mean[k]) / geo.std[k]) ** 2).sum())
        if self.len_max and n_symbols > self.len_factor * self.len_max.get(best, 10 ** 9):
            return Verdict(False, "largo", z, d2)
        if z < self.z_min:
            return Verdict(False, "ajuste", z, d2)
        if d2 > self.geo_d2_max:
            return Verdict(False, "geometria", z, d2)
        return Verdict(True, "", z, d2)

    def to_dict(self) -> dict:
        return {"mu": self.mu, "sd": self.sd, "prob_threshold": self.prob_threshold,
                "z_min": self.z_min, "geo_d2_max": self.geo_d2_max, "len_max": self.len_max,
                "z_live": self.z_live, "len_factor": self.len_factor}

    @staticmethod
    def from_dict(d: dict) -> "Rejector":
        return Rejector(d["mu"], d["sd"], d.get("prob_threshold", PROB_THRESHOLD),
                        d.get("z_min", Z_MIN), d.get("geo_d2_max", GEO_D2_MAX), d.get("len_max"),
                        d.get("z_live", Z_LIVE), d.get("len_factor", LEN_FACTOR))
