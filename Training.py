"""
Training.py - Secuencias de entrenamiento desde el dataset REAL (glifo-dataset.json),
augmentacion, entrenamiento por letra (Baum-Welch) y evaluacion de la tuberia completa.
Para 3D, cada letra se proyecta sobre la esfera igual que lo hace la GUI.
"""
import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from FeatureExtractor import K_DEFAULT, XH_RAD, ChainCodeExtractor2D, ChainCodeExtractorSphere
from Geometry import GEO_WEIGHT, GeometryModel, PairRefiner, features
from HMMEngine import HMM, HMMClassifier
from Rejection import Rejector

STATES_PER_SYMBOL = 3.0      # N estados = longitud media de la secuencia / 3  (acotado a 5..22)
TEMPERATURE = 2.5


# ------------------------------------------------------------------ augmentacion
def augment_segments(segs: Sequence[np.ndarray], rng: np.random.Generator, strength: float = 0.5):
    """Variabilidad (escala, proporcion, rotacion, inclinacion) aplicada a TODOS los trazos por igual."""
    c = np.vstack(segs).mean(axis=0)
    s = math.exp(rng.normal(0, 0.10 * strength))
    ax, ay = rng.uniform(0.9, 1.12, 2) ** strength
    th = math.radians(rng.normal(0, 5 * strength))
    sh = rng.normal(0, 0.06 * strength)
    M = np.array([[ax * s, sh], [0, ay * s]]) @ np.array([[math.cos(th), -math.sin(th)],
                                                         [math.sin(th), math.cos(th)]])
    return [(np.asarray(sg) - c) @ M.T + c + rng.normal(0, 0.004, np.shape(sg)) for sg in segs]


def segments_to_sphere(segs, rng):
    """Coloca la letra (todos sus trazos) en un punto aleatorio del disco visible de la esfera."""
    c0 = np.vstack(segs).mean(axis=0)
    for _ in range(200):
        r, a = 0.62 * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        centre = np.array([r * math.cos(a), r * math.sin(a)])
        out, ok = [], True
        for sg in segs:
            xy = (np.asarray(sg) - c0) * XH_RAD + centre
            r2 = (xy ** 2).sum(axis=1)
            if r2.max() >= 0.96:
                ok = False
                break
            out.append(np.column_stack([xy, np.sqrt(1 - r2)]))
        if ok:
            return out
    return None


def encode(segs, mode: str, rng=None, K: int = K_DEFAULT):
    """Letra (u,v) -> (simbolos, rasgos). En 3D se proyecta a la esfera primero."""
    if mode == "2d":
        return ChainCodeExtractor2D(K).symbols_segments(segs), features(segs)
    view = segments_to_sphere(segs, rng)
    if view is None:
        return [], None
    return (ChainCodeExtractorSphere(K).symbols_segments(view),
            features([np.asarray(s)[:, :2] / XH_RAD for s in view]))


def build_set(samples, mode, rng, n_aug, strength=0.5, K=K_DEFAULT):
    seqs, feats = [], []
    for segs in samples:
        for v in [segs] + [augment_segments(segs, rng, strength) for _ in range(n_aug)]:
            s, f = encode(v, mode, rng, K)
            if len(s) >= 3:
                seqs.append(np.array(s, dtype=int))
                feats.append(f)
    return seqs, np.array(feats)


def train_letter(seqs, K=K_DEFAULT, per_state=STATES_PER_SYMBOL, n_iter=30):
    N = int(np.clip(round(float(np.mean([len(s) for s in seqs])) / per_state), 5, 22))
    m = HMM.left_to_right(N, K + 1, n_dir=K)
    m.init_emissions_from_data(seqs)
    m.fit(seqs, n_iter=n_iter)
    return m


@dataclass
class Bundle:
    models: dict
    geo: GeometryModel
    rejector: Rejector
    refiner: PairRefiner


def fit_bundle(train: Dict[str, list], mode: str, rng, n_aug: int, strength=0.5, K=K_DEFAULT, verbose=True):
    models, feats, seqs_by = {}, {}, {}
    for letter, samples in train.items():
        seqs, F = build_set(samples, mode, rng, n_aug, strength, K)
        models[letter] = train_letter(seqs, K)
        feats[letter], seqs_by[letter] = F, seqs
        if verbose:
            print(f"[{mode}] {letter}: N={models[letter].N:2d} estados, {len(seqs)} secuencias", flush=True)
    rej = Rejector.fit(HMMClassifier(models, TEMPERATURE), seqs_by)
    return Bundle(models, GeometryModel.fit(feats), rej, PairRefiner.fit(feats))


# ------------------------------------------------------------------ evaluacion
def make_session(b: Bundle, mode: str, temperature=TEMPERATURE, K=K_DEFAULT, use_refiner=True, use_rejector=True):
    from Session import Session2D, Session3D
    clf = HMMClassifier(b.models, temperature)
    cls = Session2D if mode == "2d" else Session3D
    return cls(clf, K, b.geo, b.rejector if use_rejector else None, b.refiner if use_refiner else None)


def to_view(segs, mode, rng):
    return segs if mode == "2d" else segments_to_sphere(segs, rng)


def split_dataset(data, frac, seed):
    rng = np.random.default_rng(seed)
    tr, te = {}, {}
    for ch, ms in data.items():
        idx = rng.permutation(len(ms))
        n = int(round(frac * len(ms)))
        tr[ch] = [ms[i] for i in idx[:n]]
        te[ch] = [ms[i] for i in idx[n:]]
    return tr, te


def evaluate(b: Bundle, test, mode, gestures=None, seed=321, use_refiner=True, use_rejector=True):
    """Tuberia completa sobre letras NO vistas (y garabatos). Devuelve metricas y matriz de confusion."""
    rng = np.random.default_rng(seed)
    sess = make_session(b, mode, use_refiner=use_refiner, use_rejector=use_rejector)
    names = sess.clf.names
    conf = np.zeros((len(names), len(names)), dtype=int)
    n = ok = rej = 0
    for ch, samples in test.items():
        for segs in samples:
            view = to_view(segs, mode, rng)
            if view is None:
                continue
            sess.reset()
            r = sess.process_segments(view, final=True)
            n += 1
            conf[names.index(ch), names.index(r.best)] += 1
            if not r.recognized:
                rej += 1
            ok += (r.recognized and r.best == ch)
    out = {"n": n, "accuracy_argmax": float(np.trace(conf) / max(n, 1)),
           "accuracy_all": ok / n, "rejected_letters": rej / n,
           "accuracy_when_recognized": ok / max(n - rej, 1), "wrong_accepted": (n - rej - ok) / n}
    if gestures:
        per = {}
        for name, segs in gestures:
            view = to_view(segs, mode, rng)
            if view is None:
                continue
            sess.reset()
            per.setdefault(name, []).append(not sess.process_segments(view, final=True).recognized)
        out["gestures_rejected"] = {k: float(np.mean(v)) for k, v in per.items()}
        out["gestures_rejected_all"] = float(np.mean([x for v in per.values() for x in v]))
    return out, conf, names
