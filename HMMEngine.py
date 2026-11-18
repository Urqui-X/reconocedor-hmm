"""
HMMEngine.py  -  Cadenas Ocultas de Markov implementadas desde cero (solo NumPy).

Modelo  lambda = (A, B, pi)
    A  (N x N): a_ij = P(q_{t+1}=j | q_t=i)
    B  (N x K): b_j(k) = P(O_t=k | q_t=j)
    pi (N)    : pi_i = P(q_1=i)

Algoritmos:
    * Forward escalado         alpha_t(i)           -> verosimilitud P(O|lambda)
    * Backward escalado        beta_t(i)
    * Viterbi (log)            delta_t(i), psi_t(i) -> mejor secuencia de estados
    * Baum-Welch multi-secuencia (entrenamiento offline)
    * Proyeccion de trayectoria futura (simbolos restantes del modelo)
    * HMMClassifier: forward incremental en lote para los 27 modelos y
      posterior P(c | O_1:t) con prior uniforme.
"""
import json
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

_TINY = 1e-300
SCALE_MIN, SCALE_MAX = 0.6, 1.8   # limites del reescalado de la proyeccion
DWELL_MAX = 8                    # tope de simbolos por estado en la proyeccion (evita trazos fantasma)
PROJ_MAX = 90                     # tope de simbolos proyectados (~11 x-heights)


class HMM:
    def __init__(self, A, B, pi, name: str = "", final_dwell: float = 4.0, n_dir: int = None):
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.pi = np.asarray(pi, dtype=np.float64)
        self.N, self.K = self.B.shape
        # K simbolos = n_dir direcciones (circulares) + simbolos extra (p.ej. "levantar el lapiz")
        self.n_dir = int(n_dir) if n_dir is not None else self.K
        self.name = name
        self.final_dwell = float(final_dwell)  # duracion esperada en el ultimo estado

    # ------------------------------------------------------------------ init
    @staticmethod
    def left_to_right(N: int, K: int, name: str = "", n_dir: int = None) -> "HMM":
        """Topologia Bakis (izquierda-derecha con salto de a lo sumo 1 estado)."""
        A = np.zeros((N, N))
        for i in range(N - 2):
            A[i, i], A[i, i + 1], A[i, i + 2] = 0.70, 0.25, 0.05
        A[N - 2, N - 2], A[N - 2, N - 1] = 0.70, 0.30
        A[N - 1, N - 1] = 1.0
        pi = np.zeros(N)
        pi[0], pi[1] = 0.9, 0.1
        B = np.full((N, K), 1.0 / K)
        return HMM(A, B, pi, name, n_dir=n_dir)

    def init_emissions_from_data(self, sequences: Sequence[np.ndarray], floor=1e-2):
        """B inicial: histograma de simbolos de cada tramo (N tramos iguales por secuencia)."""
        H = np.zeros((self.N, self.K))
        for O in sequences:
            O = np.asarray(O)
            for j, chunk in enumerate(np.array_split(O, self.N)):
                for k in chunk:
                    H[j, k] += 1
        H = self._neighbor_smooth(H + 1e-9, 0.15)
        H += floor
        self.B = H / H.sum(axis=1, keepdims=True)

    def _neighbor_smooth(self, B, e):
        """Las direcciones son angulos (circulares): se reparte masa a los sectores vecinos.
        Los simbolos extra (levantar el lapiz) no son angulos y no se suavizan."""
        d = self.n_dir
        out = B.copy()
        D = B[:, :d]
        out[:, :d] = (1 - 2 * e) * D + e * (np.roll(D, 1, axis=1) + np.roll(D, -1, axis=1))
        return out

    # --------------------------------------------------------------- forward
    def forward(self, O: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        alpha_1(i)   = pi_i b_i(O_1)
        alpha_t+1(j) = [ sum_i alpha_t(i) a_ij ] b_j(O_t+1)
        Version escalada: alpha_hat_t = alpha_t / c_t,  c_t = sum_i alpha_t(i) (sin normalizar previo)
        log P(O|lambda) = sum_t log c_t
        """
        O = np.asarray(O, dtype=int)
        T = len(O)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)
        a = self.pi * self.B[:, O[0]]
        c[0] = max(a.sum(), _TINY)
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, O[t]]
            c[t] = max(a.sum(), _TINY)
            alpha[t] = a / c[t]
        return alpha, c

    def backward(self, O: Sequence[int], c: np.ndarray) -> np.ndarray:
        """beta_T(i)=1 ; beta_t(i) = sum_j a_ij b_j(O_t+1) beta_t+1(j)  (escalado con c_t+1)"""
        O = np.asarray(O, dtype=int)
        T = len(O)
        beta = np.ones((T, self.N))
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (self.B[:, O[t + 1]] * beta[t + 1]) / c[t + 1]
        return beta

    def log_likelihood(self, O: Sequence[int]) -> float:
        if len(O) == 0:
            return 0.0
        _, c = self.forward(O)
        return float(np.log(c).sum())

    # --------------------------------------------------------------- viterbi
    def viterbi(self, O: Sequence[int]) -> Tuple[np.ndarray, float]:
        """delta_t(j) = max_i [delta_t-1(i) a_ij] b_j(O_t) en dominio logaritmico."""
        O = np.asarray(O, dtype=int)
        T = len(O)
        with np.errstate(divide="ignore"):
            logA, logB, logpi = np.log(self.A), np.log(self.B), np.log(self.pi)
        delta = np.full((T, self.N), -np.inf)
        psi = np.zeros((T, self.N), dtype=int)
        delta[0] = logpi + logB[:, O[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA          # (i, j)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(self.N)] + logB[:, O[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, float(delta[-1].max())

    # ------------------------------------------------------------ Baum-Welch
    def fit(self, sequences: Sequence[np.ndarray], n_iter=30, tol=1e-3,
            floor=1e-4, neighbor=0.05, verbose=False) -> List[float]:
        """Baum-Welch (EM) para multiples secuencias de entrenamiento."""
        history = []
        S = len(sequences)
        for it in range(n_iter):
            N, K = self.N, self.K
            numA = np.zeros((N, N))
            denA = np.zeros(N)
            numBT = np.zeros((K, N))
            pi_acc = np.zeros(N)
            last_state_time = 0.0
            total = 0.0
            for O in sequences:
                O = np.asarray(O, dtype=int)
                T = len(O)
                alpha, c = self.forward(O)
                beta = self.backward(O, c)
                total += np.log(c).sum()
                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _TINY)
                if T > 1:
                    # xi_t(i,j) = alpha_t(i) a_ij b_j(O_t+1) beta_t+1(j) / c_t+1
                    numA += np.einsum("ti,ij,tj,t->ij", alpha[:-1], self.A,
                                      self.B[:, O[1:]].T * beta[1:], 1.0 / c[1:])
                    denA += gamma[:-1].sum(axis=0)
                np.add.at(numBT, O, gamma)
                pi_acc += gamma[0]
                last_state_time += gamma[:, N - 1].sum()
            history.append(total)
            if verbose:
                print(f"   iter {it:02d}  logL = {total:.2f}")
            # --- re-estimacion (M-step)
            A = self.A.copy()
            ok = denA > 1e-12
            A[ok] = numA[ok] / denA[ok][:, None]
            A /= np.maximum(A.sum(axis=1, keepdims=True), _TINY)
            B = numBT.T + floor
            B /= B.sum(axis=1, keepdims=True)
            B = self._neighbor_smooth(B, neighbor)
            pi = pi_acc / max(pi_acc.sum(), _TINY)
            self.A, self.B, self.pi = A, B, pi
            self.final_dwell = max(1.0, last_state_time / S)
            if it > 0 and abs(history[-1] - history[-2]) < tol * S:
                break
        return history

    # ------------------------------------------------ proyeccion de trayectoria
    def project_future(self, state: int, elapsed: int = 0, max_len: int = 250) -> List[int]:
        """
        Simbolos restantes {O_t+1..O_T} de la trayectoria aprendida por el modelo.
        Desde el estado actual se sigue la transicion mas probable hacia adelante
        hasta el estado final; en cada estado j se permanece su duracion esperada
        d_j = 1/(1-a_jj) y se emite el simbolo modal (media circular de b_j).
        """
        out: List[int] = []
        j = int(state)
        first = True
        while len(out) < max_len:
            if j == self.N - 1:
                dwell = self.final_dwell
            else:
                dwell = 1.0 / max(1e-3, 1.0 - self.A[j, j])
            dwell = min(dwell, DWELL_MAX)
            remaining = int(round(dwell - (elapsed if first else 0)))
            remaining = max(remaining, 0)
            sym = self._modal_symbol(j)
            out.extend([sym] * remaining)
            first = False
            if j == self.N - 1:
                break
            row = self.A[j].copy()
            row[j] = 0.0
            if row.sum() <= 0:
                break
            j = int(np.argmax(row))
        return out[:PROJ_MAX]

    def expected_elapsed(self, state: int, elapsed: int = 0) -> float:
        """Simbolos que un trazo 'tipico' habria gastado hasta estar en `state`
        (duracion esperada 1/(1-a_jj) de cada estado previo + lo transcurrido en el actual)."""
        d = 0.0
        for j in range(int(state)):
            d += 1.0 / max(1e-3, 1.0 - self.A[j, j])
        return d + elapsed

    def _modal_symbol(self, j: int) -> int:
        """Media circular de las direcciones del estado, o el simbolo extra (levantar el lapiz)
        si ese estado casi siempre lo emite."""
        d = self.n_dir
        if self.K > d and self.B[j, d:].max() > self.B[j, :d].sum():
            return d + int(np.argmax(self.B[j, d:]))
        ang = 2 * math.pi * np.arange(d) / d
        s = float((self.B[j, :d] * np.sin(ang)).sum())
        c = float((self.B[j, :d] * np.cos(ang)).sum())
        return int(round((math.atan2(s, c) % (2 * math.pi)) / (2 * math.pi) * d)) % d

    # ------------------------------------------------------------ persistencia
    def to_dict(self) -> dict:
        return {"N": self.N, "K": self.K, "n_dir": self.n_dir, "A": self.A.tolist(),
                "B": self.B.tolist(), "pi": self.pi.tolist(), "final_dwell": self.final_dwell}

    @staticmethod
    def from_dict(d: dict, name: str = "") -> "HMM":
        return HMM(d["A"], d["B"], d["pi"], name, d.get("final_dwell", 4.0), d.get("n_dir"))


class HMMClassifier:
    """Un HMM por caracter. Forward incremental (en lote) y posterior P(c | O_1:t)."""

    def __init__(self, models: Dict[str, HMM], temperature: float = 1.0):
        self.extra = None            # log-puntaje adicional por clase (rasgos geometricos)
        self.models = models
        self.names = list(models.keys())
        self.temperature = temperature
        M = len(self.names)
        Nm = max(m.N for m in models.values())
        K = next(iter(models.values())).K
        self.K = K
        self.A = np.zeros((M, Nm, Nm))
        self.B = np.zeros((M, Nm, K))
        self.pi = np.zeros((M, Nm))
        for i, n in enumerate(self.names):       # relleno con ceros: estados inexistentes
            m = models[n]
            self.A[i, :m.N, :m.N] = m.A
            self.B[i, :m.N, :] = m.B
            self.pi[i, :m.N] = m.pi
        self.reset()

    def set_extra(self, extra):
        """extra: vector log-puntaje por clase (orden self.names) o None."""
        self.extra = None if extra is None else np.asarray(extra, dtype=float)

    def reset(self):
        self.obs: List[int] = []
        self.alpha = None
        self.loglik = np.zeros(len(self.names))

    def update(self, o: int):
        if self.alpha is None:
            a = self.pi * self.B[:, :, o]
        else:
            a = np.einsum("mi,mij->mj", self.alpha, self.A) * self.B[:, :, o]
        c = np.maximum(a.sum(axis=1), _TINY)
        self.alpha = a / c[:, None]
        self.loglik += np.log(c)
        self.obs.append(int(o))

    def sync(self, symbols: Sequence[int]):
        """Procesa solo los simbolos nuevos; si el prefijo cambio, recalcula."""
        t = len(self.obs)
        if len(symbols) < t or list(symbols[:t]) != self.obs:
            self.reset()
            t = 0
        for o in symbols[t:]:
            self.update(int(o))

    def scores(self, final: bool = False, with_extra: bool = True) -> np.ndarray:
        """log P(O_1:t | lambda_c). Con final=True se exige que el trazo haya llegado a los
        ultimos estados del modelo:  log P(O_1:T, q_T = N | lambda_c).
        (Evita que 'a' se confunda con 'q', cuyo trazo comienza igual.)"""
        if not final or self.alpha is None:
            return self.loglik
        mass = np.array([self.alpha[i, self.models[n].N - 1] for i, n in enumerate(self.names)])
        sc = self.loglik + np.log(np.maximum(mass, 1e-12))
        if with_extra and self.extra is not None:     # solo con la letra terminada
            sc = sc + self.extra
        return sc

    def posterior(self, final: bool = False) -> np.ndarray:
        z = self.scores(final) / self.temperature
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()

    def ranking(self, final: bool = False) -> List[Tuple[str, float]]:
        p = self.posterior(final)
        order = np.argsort(-p)
        return [(self.names[i], float(p[i])) for i in order]

    def best(self, final: bool = False) -> str:
        return self.names[int(np.argmax(self.scores(final)))]

    def predict_projection(self) -> Tuple[str, List[int], float]:
        """c* = argmax P(O_1:t|lambda_c); estado actual por Viterbi; simbolos futuros y ESCALA.

        escala r = (simbolos observados) / (simbolos que un trazo tipico habria gastado hasta el
        mismo estado). r > 1 si la persona escribe mas grande que el modelo: la proyeccion se
        dibuja con paso r veces mayor para conservar la proporcion de lo ya escrito."""
        best = self.best()
        if not self.obs:
            return best, [], 1.0
        m = self.models[best]
        path, _ = m.viterbi(self.obs)
        state = int(path[-1])
        elapsed = int(np.sum(np.cumprod((path[::-1] == state).astype(int))))
        passed = m.expected_elapsed(state, elapsed)
        ok = len(self.obs) >= 8 and passed >= 4
        scale = float(np.clip(len(self.obs) / passed, SCALE_MIN, SCALE_MAX)) if ok else 1.0
        return best, m.project_future(state, elapsed), scale

    # ------------------------------------------------------------- carga
    @staticmethod
    def load(path: str, temperature: float = 1.0):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        models = {n: HMM.from_dict(md, n) for n, md in d["models"].items()}
        meta = {k: v for k, v in d.items() if k != "models"}
        return HMMClassifier(models, temperature), meta
