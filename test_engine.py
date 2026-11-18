"""Pruebas rapidas del motor HMM: Forward y Viterbi contra fuerza bruta, y Baum-Welch."""
import itertools
import numpy as np
from HMMEngine import HMM


def brute(m, O):
    tot, best, bp = 0.0, -1, None
    for q in itertools.product(range(m.N), repeat=len(O)):
        p = m.pi[q[0]] * m.B[q[0], O[0]]
        for t in range(1, len(O)):
            p *= m.A[q[t - 1], q[t]] * m.B[q[t], O[t]]
        tot += p
        if p > best:
            best, bp = p, q
    return tot, bp


rng = np.random.default_rng(0)
A = rng.dirichlet(np.ones(3), 3); B = rng.dirichlet(np.ones(4), 3); pi = rng.dirichlet(np.ones(3))
m = HMM(A, B, pi)
O = [0, 3, 1, 2, 2]
tot, bp = brute(m, O)
assert abs(np.exp(m.log_likelihood(O)) - tot) < 1e-12, "Forward != fuerza bruta"
path, _ = m.viterbi(O)
assert tuple(path) == bp, "Viterbi != fuerza bruta"
# Baum-Welch: la log-verosimilitud no debe decrecer
seqs = [rng.integers(0, 4, 12) for _ in range(20)]
m2 = HMM(rng.dirichlet(np.ones(3), 3), rng.dirichlet(np.ones(4), 3), rng.dirichlet(np.ones(3)))
h = m2.fit(seqs, n_iter=15, floor=0, neighbor=0)
assert all(b >= a - 1e-6 for a, b in zip(h, h[1:])), "EM decrece"
print("OK: Forward, Viterbi y Baum-Welch verificados")
