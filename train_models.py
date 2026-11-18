"""
train_models.py - Entrenamiento OFFLINE (Baum-Welch, NumPy) de los 27 modelos HMM, 2D y 3D,
a partir del dataset REAL de trazos (data/glifo-dataset.json).

Uso:
    python train_models.py                       # ambos modos
    python train_models.py --mode 2d --dataset ruta.json
    python train_models.py --no-holdout          # entrena directo con todo (sin evaluar)

Por modo: (1) reserva 20 % de cada letra y evalua la tuberia completa con muestras NO vistas;
(2) reentrena con el 100 % y guarda models/hmm_<modo>.json + models/report_<modo>.json.
"""
import argparse
import json
import os
import time

import numpy as np

from FeatureExtractor import K_DEFAULT
from Gestures import make_gestures
from Training import TEMPERATURE, evaluate, fit_bundle, split_dataset
from real_dataset import DEFAULT_PATH, load_uv

AUG = {"2d": (0, 0.5), "3d": (3, 0.5)}      # (copias aumentadas por muestra, fuerza)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2d", "3d", "both"], default="both")
    ap.add_argument("--dataset", default=DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--no-holdout", action="store_true")
    a = ap.parse_args()
    data, xh = load_uv(a.dataset)
    print(f"Dataset: {sum(len(v) for v in data.values())} muestras, {len(data)} letras, x-height = {xh:.0f} px", flush=True)
    os.makedirs("models", exist_ok=True)
    for mode in (["2d", "3d"] if a.mode == "both" else [a.mode]):
        t0 = time.time()
        n_aug, strength = AUG[mode]
        report = {"mode": mode, "dataset": os.path.basename(a.dataset), "xh_px": xh}
        if not a.no_holdout:
            tr, te = split_dataset(data, 1 - a.holdout, a.seed)
            b = fit_bundle(tr, mode, np.random.default_rng(a.seed), n_aug, strength, verbose=False)
            g = make_gestures(80, np.random.default_rng(4))
            res, conf, names = evaluate(b, te, mode, g)
            res0, _, _ = evaluate(b, te, mode, None, use_refiner=False, use_rejector=False)
            report.update({"holdout": res, "holdout_sin_refinador_ni_rechazo": res0,
                           "confusion": conf.tolist(), "names": names})
            print(f"[{mode}] holdout: solo HMM+geometria {res0['accuracy_argmax']:.1%} | con refinador {res['accuracy_argmax']:.1%} | "
                  f"tuberia: bien {res['accuracy_all']:.1%}, rechazadas {res['rejected_letters']:.1%}, "
                  f"error aceptado {res['wrong_accepted']:.1%} | garabatos rechazados {res['gestures_rejected_all']:.1%}", flush=True)
        b = fit_bundle(data, mode, np.random.default_rng(a.seed), n_aug, strength)
        out = {"mode": mode, "K": K_DEFAULT, "n_symbols": K_DEFAULT + 1, "alphabet": list(data),
               "xh_px": xh, "temperature": TEMPERATURE, "geometry": b.geo.to_dict(),
               "rejector": b.rejector.to_dict(), "refiner": b.refiner.to_dict(),
               "models": {k: m.to_dict() for k, m in b.models.items()}}
        with open(f"models/hmm_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(out, f)
        with open(f"models/report_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(report, f)
        print(f"[{mode}] guardado models/hmm_{mode}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
