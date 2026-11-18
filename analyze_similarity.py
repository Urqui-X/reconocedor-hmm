"""
analyze_similarity.py - Analisis de SIMILITUD geometrica entre letras y justificacion de los
resultados del lienzo. (Solo NumPy + matplotlib para las figuras; NO es parte de la app.)

Genera:
    analisis/confusion_2d.png        confusion 5-fold (sin y con refinador por pares)
    analisis/distancia_modelos.png   distancia entre los HMM de cada par de letras
    analisis/pares_confundibles.png  trazos reales superpuestos de los pares mas confundidos
    analisis/rasgos_pares.png        el rasgo geometrico que separa cada par
    analisis/rechazo.png             por que el umbral del 20 % solo no basta
    analisis/resultados.json         todas las cifras (las usa ANALISIS_SIMILITUD.md)
    models/similarity.json           "letras parecidas" + motivo (lo muestra la GUI)
    models/icons.json                muestra real representativa de cada letra (iconos de la GUI)

Uso:  python analyze_similarity.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from FeatureExtractor import K_DEFAULT, ChainCodeExtractor2D, resample_polyline
from Geometry import FEATURE_NAMES, features
from Gestures import make_gestures
from HMMEngine import HMMClassifier
from Training import evaluate, fit_bundle, make_session
from real_dataset import DEFAULT_PATH, load_uv

OUT = "analisis"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def draw_sample(ax, segs, color, alpha=1.0, lw=1.4, dx=0.0):
    allp = np.vstack(segs)
    u0, v0 = allp[:, 0].min(), allp[:, 1].min()
    for s in segs:
        a = np.asarray(s) - [u0, v0]
        ax.plot(a[:, 0] + dx, a[:, 1], color=color, alpha=alpha, lw=lw, solid_capstyle="round")
        ax.plot(a[0, 0] + dx, a[0, 1], "o", color="#16a34a", ms=3, alpha=alpha)


def top_feature(Fa, Fb):
    """Rasgo con mayor separacion estandarizada entre dos letras: (indice, d')."""
    d = np.abs(Fa.mean(0) - Fb.mean(0)) / np.sqrt(0.5 * (Fa.var(0) + Fb.var(0)) + 1e-9)
    i = int(np.argmax(d))
    return i, float(d[i])


def reason(Fa, Fb):
    i, d = top_feature(Fa, Fb)
    name = FEATURE_NAMES[i]
    ma, mb = Fa.mean(0)[i], Fb.mean(0)[i]
    if d < 1.0:
        return "forma y tamaño casi idénticos: solo cambia la posición de los giros"
    return f"se separan sobre todo por «{name}» ({ma:.1f} vs {mb:.1f})"


def agglomerative_order(D):
    """Orden de hojas de un agrupamiento jerarquico de enlace promedio (NumPy puro)."""
    clusters = {i: [i] for i in range(len(D))}
    while len(clusters) > 1:
        keys = list(clusters)
        best, pair = 1e18, None
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                d = np.mean([D[i, j] for i in clusters[keys[x]] for j in clusters[keys[y]]])
                if d < best:
                    best, pair = d, (keys[x], keys[y])
        a, b = pair
        clusters[a] = clusters[a] + clusters.pop(b)
    return next(iter(clusters.values()))


def cross_validate(data, k=5, seed=0):
    names = list(data)
    folds = [{c: [] for c in names} for _ in range(k)]
    rng = np.random.default_rng(seed)
    for c, ms in data.items():
        for j, i in enumerate(rng.permutation(len(ms))):
            folds[j % k][c].append(ms[i])
    conf0 = np.zeros((len(names), len(names)), dtype=int)
    conf1 = conf0.copy()
    first = None
    for f in range(k):
        train = {c: [s for g in range(k) if g != f for s in folds[g][c]] for c in names}
        b = fit_bundle(train, "2d", np.random.default_rng(f), 0, verbose=False)
        r0, c0, _ = evaluate(b, folds[f], "2d", None, use_refiner=False, use_rejector=False)
        r1, c1, _ = evaluate(b, folds[f], "2d", None, use_refiner=True, use_rejector=False)
        conf0 += c0
        conf1 += c1
        print(f"  fold {f+1}/{k}: HMM+geometria {r0['accuracy_argmax']:.1%} | con refinador {r1['accuracy_argmax']:.1%}", flush=True)
        if first is None:
            first = (b, folds[f])
    return names, conf0, conf1, first


def model_distance(clf, data, ext, max_per=60):
    names = clf.names
    M = np.zeros((len(names), len(names)))
    for i, c in enumerate(names):
        rows = []
        for segs in data[c][:max_per]:
            syms = ext.symbols_segments(segs)
            clf.reset(); clf.sync(syms); clf.set_extra(None)
            rows.append(clf.scores(final=True, with_extra=False) / len(syms))
        M[i] = np.mean(rows, axis=0)
    D = 0.5 * (np.diag(M)[:, None] - M + np.diag(M)[None, :] - M.T)
    np.fill_diagonal(D, 0.0)
    return np.maximum(D, 0.0)


def make_icons(data):
    icons = {}
    for ch, ms in data.items():
        ns = np.array([min(len(s), 4) for s in ms])
        modal = np.bincount(ns).argmax()
        F = np.array([features(s) for s in ms])
        med, sd = np.median(F, axis=0), F.std(axis=0) + 1e-3
        cand = [i for i in range(len(ms)) if ns[i] == modal]
        best = min(cand, key=lambda i: float((((F[i] - med) / sd) ** 2).sum()))
        icons[ch] = [np.round(resample_polyline(np.asarray(s), 0.07), 3).tolist() for s in ms[best]]
    return icons


def main():
    data, xh = load_uv(DEFAULT_PATH)
    names = list(data)
    iu = {c: i for i, c in enumerate(names)}
    ext = ChainCodeExtractor2D(K_DEFAULT)
    F = {c: np.array([features(s) for s in ms]) for c, ms in data.items()}
    R = {"n_muestras": int(sum(len(v) for v in data.values())), "xh_px": xh}

    print("1) Validación cruzada 5-fold (2D)…", flush=True)
    names, conf0, conf1, (b0, test0) = cross_validate(data)
    R["cv_accuracy_hmm_geo"] = float(np.trace(conf0) / conf0.sum())
    R["cv_accuracy_con_refinador"] = float(np.trace(conf1) / conf1.sum())
    R["cv_accuracy_por_letra_hmm_geo"] = {c: float(conf0[i, i] / conf0[i].sum()) for i, c in enumerate(names)}
    R["cv_accuracy_por_letra_con_refinador"] = {c: float(conf1[i, i] / conf1[i].sum()) for i, c in enumerate(names)}
    off = sorted([(names[i], names[j], int(conf0[i, j])) for i in range(len(names)) for j in range(len(names))
                  if i != j and conf0[i, j] > 0], key=lambda t: -t[2])
    off1 = sorted([(names[i], names[j], int(conf1[i, j])) for i in range(len(names)) for j in range(len(names))
                   if i != j and conf1[i, j] > 0], key=lambda t: -t[2])
    R["cv_confusiones_hmm_geo"], R["cv_confusiones_con_refinador"] = off[:15], off1[:15]
    print(f"   HMM+geometria {R['cv_accuracy_hmm_geo']:.1%}  ->  con refinador {R['cv_accuracy_con_refinador']:.1%}", flush=True)

    print("2) Umbral y rechazo…", flush=True)
    gest = make_gestures(80, np.random.default_rng(4))
    sess = make_session(b0, "2d", use_rejector=False)
    top_l, top_g = [], []
    for ch, ms in test0.items():
        for s in ms:
            sess.reset(); top_l.append(sess.process_segments(s, final=True).ranking[0][1])
    for _, segs in gest:
        sess.reset(); top_g.append(sess.process_segments(segs, final=True).ranking[0][1])
    top_l, top_g = np.array(top_l), np.array(top_g)
    R["umbral_prob"] = {str(t): {"letras_rechazadas": float((top_l < t).mean()),
                                 "garabatos_rechazados": float((top_g < t).mean())} for t in (0.2, 0.3, 0.5)}
    R["pipeline"], _, _ = evaluate(b0, test0, "2d", gest)
    print("  ", json.dumps(R["umbral_prob"]), flush=True)

    print("3) Distancia entre modelos…", flush=True)
    clf, _ = HMMClassifier.load("models/hmm_2d.json", 1.0)
    D = model_distance(clf, data, ext)
    sim = {}
    for a in names:
        lst = []
        for j in np.argsort(D[iu[a]]):
            b = names[j]
            if b == a:
                continue
            lst.append({"letter": b, "distance": round(float(D[iu[a], j]), 3),
                        "confusion_pct": round(100 * float(conf0[iu[a], j] / conf0[iu[a]].sum()), 1),
                        "reason": reason(F[a], F[b])})
            if len(lst) == 3:
                break
        sim[a] = lst
    json.dump(sim, open("models/similarity.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(make_icons(data), open("models/icons.json", "w"))
    R["mas_parecidas"] = {a: [(s["letter"], s["distance"]) for s in v] for a, v in sim.items()}
    R["rasgos_medios"] = {c: dict(zip(FEATURE_NAMES, np.round(F[c].mean(0), 2).tolist())) for c in names}

    print("4) Figuras…", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 7.2))
    for ax, conf, ttl in ((axs[0], conf0, "HMM + rasgos"), (axs[1], conf1, "+ refinador por pares")):
        Pn = conf / conf.sum(axis=1, keepdims=True)
        im = ax.imshow(Pn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.set_xlabel("predicha"); ax.set_ylabel("real")
        ax.set_title(f"{ttl}: {np.trace(conf)/conf.sum():.1%} (5-fold, datos no vistos)")
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j and conf[i, j] >= 3:
                    ax.text(j, i, int(conf[i, j]), ha="center", va="center", color="#b91c1c", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_2d.png", dpi=130); plt.close(fig)

    order = agglomerative_order(D)
    lab = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(D[np.ix_(order, order)], cmap="viridis_r")
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
    ax.set_title("Distancia entre HMM (oscuro = más parecidas; letras agrupadas)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="distancia (nats/símbolo)")
    fig.tight_layout(); fig.savefig(f"{OUT}/distancia_modelos.png", dpi=130); plt.close(fig)

    pairs, seen = [], set()
    for a, b, n in off:
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b, n + int(conf0[iu[b], iu[a]])))
        if len(pairs) == 6:
            break
    R["pares_figura"] = pairs
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (a, b, n) in zip(axs.ravel(), pairs):
        for c, col, dx in ((a, "#2563eb", 0.0), (b, "#dc2626", 1.7)):
            for k in rng.choice(len(data[c]), 8, replace=False):
                draw_sample(ax, data[c][k], col, 0.35, 1.2, dx)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"«{a}» (azul) vs «{b}» (rojo) · {n} confusiones\n{reason(F[a], F[b])}", fontsize=8.5)
    fig.suptitle("Trazos reales superpuestos de los pares más confundidos por el HMM (punto verde = inicio)")
    fig.tight_layout(); fig.savefig(f"{OUT}/pares_confundibles.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (a, b, n) in zip(axs.ravel(), pairs):
        i, d = top_feature(F[a], F[b])
        lo, hi = min(F[a][:, i].min(), F[b][:, i].min()), max(F[a][:, i].max(), F[b][:, i].max())
        bins = np.linspace(lo, hi, 25)
        ax.hist(F[a][:, i], bins, alpha=.6, color="#2563eb", label=a)
        ax.hist(F[b][:, i], bins, alpha=.6, color="#dc2626", label=b)
        ax.set_title(f"«{a}» vs «{b}»: {FEATURE_NAMES[i]} (d' = {d:.1f})", fontsize=9)
        ax.legend()
    fig.suptitle("Rasgo que mejor separa cada par (d' alto = separable; d' < 1 = prácticamente igual)")
    fig.tight_layout(); fig.savefig(f"{OUT}/rasgos_pares.png", dpi=130); plt.close(fig)
    R["rasgo_discriminante"] = {f"{a}|{b}": {"rasgo": FEATURE_NAMES[top_feature(F[a], F[b])[0]],
                                             "d_prima": round(top_feature(F[a], F[b])[1], 2)} for a, b, _ in pairs}

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 21)
    axs[0].hist(top_l, bins, alpha=.7, label="letras reales", color="#2563eb")
    axs[0].hist(top_g, bins, alpha=.7, label="garabatos (no son letras)", color="#dc2626")
    axs[0].axvline(0.2, color="k", ls="--"); axs[0].text(0.21, axs[0].get_ylim()[1] * .9, "umbral 20 %")
    axs[0].set_xlabel("probabilidad de la letra ganadora"); axs[0].set_title("El posterior solo es RELATIVO"); axs[0].legend()
    gr = R["pipeline"]["gestures_rejected"]
    axs[1].barh(list(gr), [100 * v for v in gr.values()], color="#16a34a")
    axs[1].set_xlim(0, 100); axs[1].set_xlabel("% rechazados (umbral + ajuste absoluto)")
    axs[1].set_title(f"Garabatos rechazados {100*R['pipeline']['gestures_rejected_all']:.0f} % · letras rechazadas {100*R['pipeline']['rejected_letters']:.1f} %")
    fig.tight_layout(); fig.savefig(f"{OUT}/rechazo.png", dpi=130); plt.close(fig)

    json.dump(R, open(f"{OUT}/resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("Listo.", flush=True)


if __name__ == "__main__":
    main()
