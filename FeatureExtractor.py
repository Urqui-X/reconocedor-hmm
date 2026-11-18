"""
FeatureExtractor.py - Procesamiento geometrico, cuantizacion angular y conversion esferica.

Unidades del "lienzo de letra": 1.0 = altura de la letra 'x' (x-height).
    baseline v=0, x-height v=1, ascendente v=2, descendente v=-1.

2D : trazo (u,v) -> remuestreo a paso fijo -> vectores (du,dv) -> Chain Code de K sectores.
3D : trazo (x,y,z) sobre esfera unitaria -> (theta, phi) -> (d_theta, d_phi)
     -> componentes tangenciales  norte = -d_theta,  este = d_phi * sin(theta)
     -> Chain Code de K sectores.
"""
import math
from typing import List, Sequence

import numpy as np

K_DEFAULT = 16         # sectores de direccion del Chain Code
N_SYMBOLS = K_DEFAULT + 1   # + 1 simbolo "levantar el lapiz" (indice K)
STEP_UNITS = 0.12      # paso de remuestreo (en x-heights)
XH_RAD = 0.30          # x-height sobre la esfera, en radianes
TWO_PI = 2 * math.pi


# ----------------------------------------------------------------- utilidades
def quantize_angles(psi: np.ndarray, K: int) -> np.ndarray:
    """Sector k = round(psi / (2pi/K)) mod K  (centros en k*2pi/K)."""
    return (np.floor(((np.asarray(psi) % TWO_PI) + math.pi / K) / (TWO_PI / K)).astype(int)) % K


def sector_angle(k: int, K: int) -> float:
    return k * TWO_PI / K


def resample_polyline(pts: np.ndarray, step: float, sphere: bool = False) -> np.ndarray:
    """Remuestrea a longitud de arco constante, empezando en pts[0] (el prefijo es estable)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    if sphere:
        dots = np.clip(np.einsum("ij,ij->i", pts[:-1], pts[1:]), -1.0, 1.0)
        seg = np.arccos(dots)
    else:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < step:
        return pts[:1].copy()
    targets = np.arange(0.0, total + 1e-9, step)
    out = np.stack([np.interp(targets, cum, pts[:, d]) for d in range(pts.shape[1])], axis=1)
    if sphere:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


# ------------------------------------------------------------------------ 2D
class ChainCodeExtractor2D:
    def __init__(self, K: int = K_DEFAULT, step: float = STEP_UNITS):
        self.K, self.step = K, step

    def symbols(self, uv: np.ndarray) -> List[int]:
        rs = resample_polyline(uv, self.step)
        if len(rs) < 2:
            return []
        d = np.diff(rs, axis=0)
        psi = np.arctan2(d[:, 1], d[:, 0])
        return quantize_angles(psi, self.K).tolist()

    @property
    def lift(self) -> int:
        """Indice del simbolo 'levantar el lapiz' (despues de las K direcciones)."""
        return self.K

    def symbols_segments(self, segments: Sequence[np.ndarray]) -> List[int]:
        """Varios trazos independientes. No inventa una linea al levantar el lapiz: entre dos
        trazos inserta el simbolo LIFT (asi 'i' = palo + LIFT + punto se distingue de 'l')."""
        out: List[int] = []
        for i, segment in enumerate(segments):
            if i > 0:
                out.append(self.lift)
            out.extend(self.symbols(np.asarray(segment, dtype=float)))
        return out

    def symbols_to_points(self, start: Sequence[float], symbols: Sequence[int], scale: float = 1.0) -> np.ndarray:
        pos = np.asarray(start, dtype=float)
        out = [pos.copy()]
        for k in symbols:
            a = sector_angle(k, self.K)
            pos = pos + scale * self.step * np.array([math.cos(a), math.sin(a)])
            out.append(pos.copy())
        return np.array(out)


# ------------------------------------------------------------------------ 3D
def to_spherical(p: np.ndarray):
    """(x,y,z) unitario -> (theta, phi). theta: angulo polar desde +y, phi = atan2(x, z)."""
    p = np.asarray(p, dtype=float)
    theta = np.arccos(np.clip(p[..., 1], -1.0, 1.0))
    phi = np.arctan2(p[..., 0], p[..., 2])
    return theta, phi


def tangent_basis(p: np.ndarray):
    """Base (este, norte) del plano tangente en p."""
    theta, phi = to_spherical(p)
    if math.sin(theta) < 1e-6:
        east = np.array([1.0, 0.0, 0.0])
        north = np.array([0.0, 0.0, -1.0 if p[1] > 0 else 1.0])
        return east, north
    east = np.array([math.cos(phi), 0.0, -math.sin(phi)])
    north = np.array([-math.cos(theta) * math.sin(phi), math.sin(theta),
                      -math.cos(theta) * math.cos(phi)])
    return east, north


class ChainCodeExtractorSphere:
    def __init__(self, K: int = K_DEFAULT, step: float = STEP_UNITS * XH_RAD):
        self.K, self.step = K, step   # step en radianes

    def symbols(self, pts3: np.ndarray) -> List[int]:
        rs = resample_polyline(pts3, self.step, sphere=True)
        if len(rs) < 2:
            return []
        theta, phi = to_spherical(rs)
        d_theta = np.diff(theta)
        d_phi = np.diff(phi)
        d_phi = (d_phi + math.pi) % TWO_PI - math.pi            # envolver a [-pi, pi)
        theta_mid = 0.5 * (theta[:-1] + theta[1:])
        north = -d_theta                                          # theta crece hacia el sur
        east = d_phi * np.sin(theta_mid)
        psi = np.arctan2(north, east)
        return quantize_angles(psi, self.K).tolist()

    @property
    def lift(self) -> int:
        return self.K

    def symbols_segments(self, segments: Sequence[np.ndarray]) -> List[int]:
        out: List[int] = []
        for i, segment in enumerate(segments):
            if i > 0:
                out.append(self.lift)
            out.extend(self.symbols(np.asarray(segment, dtype=float)))
        return out

    def symbols_to_points(self, start: np.ndarray, symbols: Sequence[int], scale: float = 1.0) -> np.ndarray:
        p = np.asarray(start, dtype=float)
        p = p / np.linalg.norm(p)
        step = scale * self.step
        out = [p.copy()]
        for k in symbols:
            a = sector_angle(k, self.K)
            east, north = tangent_basis(p)
            t = math.cos(a) * east + math.sin(a) * north
            p = p * math.cos(step) + t * math.sin(step)
            p /= np.linalg.norm(p)
            out.append(p.copy())
        return np.array(out)
