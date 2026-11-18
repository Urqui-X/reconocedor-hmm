"""
Session.py - Logica de reconocimiento en tiempo real (sin interfaz grafica).
FeatureExtractor + HMMEngine + Geometry + Rejection:
    puntos -> simbolos -> posterior -> (refinador de pares) -> (umbral / ajuste) -> proyeccion escalada.
La GUI solo dibuja lo que devuelve este modulo (se puede probar sin ventanas).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from FeatureExtractor import XH_RAD, ChainCodeExtractor2D, ChainCodeExtractorSphere
from Geometry import GEO_WEIGHT, GeometryModel, PairRefiner, features
from HMMEngine import HMMClassifier
from Rejection import Rejector, Verdict


@dataclass
class Result:
    ranking: List[Tuple[str, float]]
    best: str
    n_symbols: int
    projection: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    recognized: bool = True          # False -> "no se reconocio ningun gesto"
    verdict: Verdict = field(default_factory=lambda: Verdict(True))
    scale: float = 1.0               # escala con la que se dibuja la proyeccion
    display_ranking: Optional[List[Tuple[str, float]]] = None   # lo que muestra el panel (ver abajo)


class _SessionBase:
    def __init__(self, clf: HMMClassifier, K: int, geo: Optional[GeometryModel] = None,
                 rejector: Optional[Rejector] = None, refiner: Optional[PairRefiner] = None):
        self.clf, self.geo, self.rejector, self.refiner = clf, geo, rejector, refiner
        self.K = K

    def _symbols(self, valid) -> List[int]:
        raise NotImplementedError

    def _uv(self, valid) -> List[np.ndarray]:
        raise NotImplementedError

    def _project(self, valid, future, scale) -> np.ndarray:
        raise NotImplementedError

    def _empty_projection(self) -> np.ndarray:
        raise NotImplementedError

    def process(self, seg, final: bool = False) -> Result:
        return self.process_segments([seg], final=final)

    def process_segments(self, segments, final: bool = False) -> Result:
        valid = [np.asarray(s, dtype=float) for s in segments if len(s) >= 2]
        syms = self._symbols(valid)
        self.clf.sync(syms)
        feats = None
        if final and self.geo is not None and valid:
            feats = features(self._uv(valid))
            self.clf.set_extra(GEO_WEIGHT * self.geo.loglik(feats, self.clf.names))
        else:
            self.clf.set_extra(None)
        ranking = self.clf.ranking(final)
        if final and feats is not None and self.refiner is not None and len(ranking) > 1:
            ranking = self.refiner.refine(ranking, feats)
        best, top = ranking[0]
        if self.rejector is None:
            verdict = Verdict(True)
        elif final:
            verdict = self.rejector.assess_final(self.clf, best, top, len(syms), self.geo, feats)
        else:
            verdict = self.rejector.assess_live(self.clf, best, top, len(syms))
        proj, scale = self._empty_projection(), 1.0
        if verdict.recognized and not final and len(syms) and valid:   # finalizada: sin autocompletado
            _, future, scale = self.clf.predict_projection()
            proj = self._project(valid, self._until_lift(future), scale)
        # Si el trazo se rechaza por NO PARECERSE a ninguna letra (ajuste / largo / geometria), el
        # posterior "100 % para la letra X" es engañoso (solo dice cual se parece mas): el panel
        # muestra 0 % en todas. Si es solo AMBIGUO (ninguna supera el 20 %), se muestran los valores reales.
        display = ranking
        if not verdict.recognized and verdict.reason in ("ajuste", "largo", "geometria"):
            display = [(c, 0.0) for c in self.clf.names]
        return Result(ranking, best, len(syms), proj, verdict.recognized, verdict, scale, display)

    def _until_lift(self, future):
        """La proyeccion cubre solo el resto del trazo actual: en un 'levantar el lapiz' no se
        sabe donde empieza el siguiente trazo, y no se dibuja una linea inventada."""
        out = []
        for s in future:
            if s >= self.K:
                break
            out.append(s)
        return out

    def reset(self):
        self.clf.reset()
        self.clf.set_extra(None)


class Session2D(_SessionBase):
    """Puntos en 'unidades de letra' (u derecha, v arriba; 1.0 = altura de la x)."""

    def __init__(self, clf, K, geo=None, rejector=None, refiner=None):
        super().__init__(clf, K, geo, rejector, refiner)
        self.ext = ChainCodeExtractor2D(K)

    def _symbols(self, valid):
        return self.ext.symbols_segments(valid)

    def _uv(self, valid):
        return valid

    def _empty_projection(self):
        return np.zeros((0, 2))

    def _project(self, valid, future, scale):
        return self.ext.symbols_to_points(valid[-1][-1], future, scale)


class Session3D(_SessionBase):
    """Puntos (x,y,z) unitarios en el sistema de la VISTA (x derecha, y arriba, z hacia el usuario)."""

    def __init__(self, clf, K, geo=None, rejector=None, refiner=None):
        super().__init__(clf, K, geo, rejector, refiner)
        self.ext = ChainCodeExtractorSphere(K)

    def _symbols(self, valid):
        return self.ext.symbols_segments(valid)

    def _uv(self, valid):
        return [s[:, :2] / XH_RAD for s in valid]

    def _empty_projection(self):
        return np.zeros((0, 3))

    def _project(self, valid, future, scale):
        return self.ext.symbols_to_points(valid[-1][-1], future, scale)
