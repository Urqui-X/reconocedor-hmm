"""
real_dataset.py - Carga el dataset de trazos reales (glifo-dataset-*.json).

Formato del JSON:  { letra: [ muestra, ... ] }
    muestra = [ trazo, ... ]            (una letra puede tener varios trazos)
    trazo   = [ [x, y, t_ms], ... ]     (pixeles de pantalla, y hacia abajo)

Se convierte al "lienzo de letra" (u, v) usado por el resto del proyecto:
    u = x / xh,  v = -y / xh      (1.0 = altura de la x)
xh se mide del propio dataset: altura mediana de las letras de una sola altura-x.
"""
import json
import os
from typing import Dict, List

import numpy as np

XH_LETTERS = "acemnorsuvwz"
DEFAULT_PATH = os.path.join("data", "glifo-dataset.json")


def load_raw(path: str) -> Dict[str, list]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def measure_xheight(raw: Dict[str, list]) -> float:
    hs = []
    for ch in XH_LETTERS:
        for sample in raw.get(ch, []):
            P = np.array([p for st in sample for p in st], dtype=float)
            if len(P):
                hs.append(np.ptp(P[:, 1]))
    return float(np.median(hs))


def to_uv_segments(sample, xh: float) -> List[np.ndarray]:
    segs = []
    for st in sample:
        a = np.asarray(st, dtype=float)[:, :2]
        if len(a) == 0:
            continue
        if len(a) == 1:                       # un clic sin mover = PUNTO (i, j, ñ...): se conserva
            a = np.vstack([a, a + [0.0, 0.5]])  # 2 puntos casi iguales, igual que hace la GUI
        segs.append(np.stack([a[:, 0] / xh, -a[:, 1] / xh], axis=1))
    return segs


def load_uv(path: str = DEFAULT_PATH):
    """(dict letra -> lista de muestras; cada muestra = lista de segmentos (u,v)), xh_px)."""
    raw = load_raw(path)
    xh = measure_xheight(raw)
    out = {ch: [s for s in (to_uv_segments(m, xh) for m in muestras) if s]
           for ch, muestras in raw.items()}
    return out, xh
