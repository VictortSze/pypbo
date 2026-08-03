"""Shared helpers for the PBO/CSCV experiments (part 2).

Provides:
  * an import shim so `import pypbo` works from the cloned repo (pypbo has
    no setup.py/pyproject and is not installed);
  * `cscv_pbo_fast`: a vectorised reimplementation of the CSCV/PBO
    algorithm exactly as stated in Bailey, Borwein, Lopez de Prado & Zhu,
    "The Probability of Backtest Overfitting" (JCF 2015), with pypbo's two
    concrete conventions (leading-row trim when T % S != 0, and
    w_bar = rank/(N+1)). It is validated run-for-run against the real
    `pypbo.pbo()` in test_pbo_null.py before being used for wide sweeps —
    the real thing is O(minutes) per run at S=16, the oracle is O(ms).
  * deterministic data generators (white noise, planted edge, AR(1)).

Both PBO threshold conventions are returned:
  pbo_le : fraction of logits <= 0   (pypbo's convention)
  pbo_lt : fraction of logits <  0   (purgedcv's convention)
The paper defines PBO = integral of f(lambda) for lambda <= 0; with a
continuous logit distribution the boundary has probability ~0 for even N,
but for odd N the value lambda == 0 occurs with probability ~1/N per
combination, so the two conventions genuinely differ (see reconcile test).
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

# Make the cloned pypbo repo importable without installing it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_COMBO_CACHE: dict[int, np.ndarray] = {}


def combo_matrix(S: int) -> np.ndarray:
    """(C(S, S/2), S) 0/1 matrix; row c marks the blocks chosen in-sample
    by the c-th combination, in lexicographic order (itertools order,
    matching both pypbo and purgedcv)."""
    if S not in _COMBO_CACHE:
        combos = list(combinations(range(S), S // 2))
        m = np.zeros((len(combos), S), dtype=np.float64)
        for c, combo in enumerate(combos):
            m[c, list(combo)] = 1.0
        _COMBO_CACHE[S] = m
    return _COMBO_CACHE[S]


def cscv_pbo_fast(M: np.ndarray, S: int, ddof: int = 1):
    """CSCV/PBO per the paper, pypbo conventions, via block sufficient stats.

    Parameters
    ----------
    M : (T, N) matrix of returns, configurations in columns (pypbo layout).
    S : even number of contiguous blocks.
    ddof : delta dof for the Sharpe standard deviation (pypbo uses 1).

    Returns
    -------
    dict with keys: logits (C,), pbo_le, pbo_lt, n_combos, n_zero_logits.
    """
    if S % 2:
        raise ValueError("S must be even")
    T, N = M.shape
    M = M[T % S :]  # pypbo trims the LEADING T % S rows
    T = M.shape[0]
    sub = T // S
    blocks = M.reshape(S, sub, N)

    s1 = blocks.sum(axis=1)          # (S, N) per-block sums
    s2 = (blocks ** 2).sum(axis=1)   # (S, N) per-block sums of squares
    mask = combo_matrix(S)           # (C, S)
    n_half = sub * (S // 2)

    is_s1 = mask @ s1
    is_s2 = mask @ s2
    oos_s1 = s1.sum(axis=0) - is_s1
    oos_s2 = s2.sum(axis=0) - is_s2

    def sharpe(sum1, sum2, n):
        mean = sum1 / n
        var = (sum2 - n * mean ** 2) / (n - ddof)
        return mean / np.sqrt(var)

    r_is = sharpe(is_s1, is_s2, n_half)    # (C, N) in-sample metric
    r_oos = sharpe(oos_s1, oos_s2, n_half)  # (C, N) out-of-sample metric

    winner = np.argmax(r_is, axis=1)        # IS-best config per combination
    w_oos = r_oos[np.arange(len(r_oos)), winner]
    less = (r_oos < w_oos[:, None]).sum(axis=1)
    equal = (r_oos == w_oos[:, None]).sum(axis=1)  # includes the winner itself
    rank = less + (equal + 1) / 2.0          # scipy rankdata 'average'
    w_bar = rank / (N + 1)                   # paper / pypbo convention
    logits = np.log(w_bar / (1.0 - w_bar))

    return {
        "logits": logits,
        "pbo_le": float(np.mean(logits <= 0.0)),
        "pbo_lt": float(np.mean(logits < 0.0)),
        "n_combos": len(logits),
        "n_zero_logits": int(np.sum(logits == 0.0)),
    }


def sharpe_metric_columns(J: np.ndarray) -> np.ndarray:
    """Column-wise Sharpe (ddof=1), the metric handed to the real pypbo.pbo().
    Identical maths to `cscv_pbo_fast` and to purgedcv's default `sharpe`."""
    return np.mean(J, axis=0) / np.std(J, axis=0, ddof=1)


def white_noise(T: int, N: int, seed) -> np.ndarray:
    """(T, N) iid N(0, 0.01^2) returns: true Sharpe exactly 0 by construction."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(T, N))


def planted_edge(T: int, N: int, n_skill: int, sharpe_per_obs: float, seed) -> np.ndarray:
    """White noise, but the FIRST `n_skill` columns get a constant drift of
    `sharpe_per_obs * sigma` per period: their true per-observation Sharpe is
    `sharpe_per_obs`; all other columns have true Sharpe 0."""
    m = white_noise(T, N, seed)
    m[:, :n_skill] += sharpe_per_obs * 0.01
    return m


def ar1_noise(T: int, N: int, phi: float, seed) -> np.ndarray:
    """(T, N) columns of independent stationary AR(1) with parameter `phi`,
    mean 0, unconditional sigma 0.01: true Sharpe 0, known autocorrelation."""
    rng = np.random.default_rng(seed)
    innov_sd = 0.01 * np.sqrt(1.0 - phi ** 2)
    eps = rng.normal(0.0, innov_sd, size=(T + 100, N))
    x = np.empty_like(eps)
    x[0] = rng.normal(0.0, 0.01, size=N)  # start in stationarity
    for t in range(1, len(eps)):
        x[t] = phi * x[t - 1] + eps[t]
    return x[100:]  # burn-in


def expected_combos(S: int) -> int:
    return comb(S, S // 2)
