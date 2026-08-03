"""STEP 3 — The null test: PBO on pure white noise should be ~ 0.5.

Theory (Bailey, Borwein, Lopez de Prado & Zhu 2015): if no configuration
has any true edge and returns are iid, the in-sample winner's out-of-sample
relative rank is uniform — picking the IS best is a coin flip OOS. Hence
for EVEN N,  E[PBO] = P(rank <= N/2) = 0.5 exactly, under either logit
threshold convention (lambda == 0 cannot occur without ties).

For ODD N the discreteness bites: rank (N+1)/2 gives w_bar = 1/2, logit
exactly 0, with probability 1/N per combination. So the paper-level
prediction is
    E[PBO with lambda <= 0] = 0.5 + 1/(2N)     (pypbo's convention)
    E[PBO with lambda <  0] = 0.5 - 1/(2N)     (purgedcv's convention)
— tested below as a sharp property.

Methodology note (disclosed prominently): the wide distributional sweep
uses `cscv_pbo_fast`, a vectorised replication of the CSCV algorithm as
stated in the paper with pypbo's conventions. It is validated in THIS file,
run-for-run (identical logit multisets and PBO values), against the real
`pypbo.pbo()` — including one full S=16 run whose combination count answers
the 12870-vs-12780 question. The real pypbo takes minutes per S=16 run,
which makes a 480-run distribution study infeasible directly.

Run:  python -m pytest experiments/test_pbo_null.py -v -s
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from _cscv_common import (
    cscv_pbo_fast,
    expected_combos,
    sharpe_metric_columns,
    white_noise,
)

import pypbo

S_MAIN = 16
N_GRID = [10, 50, 100, 200]
T_GRID = [500, 1000, 2000]
N_SEEDS = 40
BASE_SEED = 20260802


# ---------------------------------------------------------------------------
# Validation: the fast oracle reproduces the real pypbo.pbo() exactly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "S,N,T,seed",
    [
        (8, 10, 400, 0),
        (8, 50, 400, 1),
        (8, 9, 405, 2),     # odd N and T % S != 0 (trim path)
        (16, 10, 500, 3),   # full S=16, T % 16 = 4 exercises the trim
    ],
)
def test_oracle_matches_real_pypbo(S, N, T, seed):
    M = white_noise(T, N, seed)
    res = pypbo.pbo(
        M, S=S, metric_func=sharpe_metric_columns, threshold=0,
        n_jobs=1, plot=False, verbose=False,
    )
    fast = cscv_pbo_fast(M, S=S)

    # Combination count: C(S, S/2). For S=16 this asserts the code produces
    # 12870 = C(16,8); the 12780 sometimes seen in print is a typo.
    assert len(res.Cs) == expected_combos(S) == fast["n_combos"]
    if S == 16:
        assert len(res.Cs) == 12870

    # Identical logit multisets and identical PBO (pypbo counts lambda <= 0).
    assert np.allclose(np.sort(np.asarray(res.logits)), np.sort(fast["logits"]),
                       rtol=0, atol=1e-10)
    assert res.pbo == pytest.approx(fast["pbo_le"], abs=1e-12)
    print(f"\nS={S} N={N} T={T}: pypbo PBO={res.pbo:.6f} == oracle "
          f"{fast['pbo_le']:.6f} over {fast['n_combos']} combos")


# ---------------------------------------------------------------------------
# The null distribution, S=16, across the (N, T) grid and many seeds.
# ---------------------------------------------------------------------------

def _null_distribution(N: int, T: int, n_seeds: int) -> np.ndarray:
    out = np.empty(n_seeds)
    for r in range(n_seeds):
        M = white_noise(T, N, seed=[BASE_SEED, N, T, r])
        out[r] = cscv_pbo_fast(M, S=S_MAIN)["pbo_le"]
    return out


def test_null_pbo_distribution():
    """E[PBO] = 0.5 for even N under iid noise. Reports the full per-cell
    distribution; asserts cell means near 0.5 and the grand mean tightly so."""
    print(f"\nPBO under the null (S={S_MAIN}, {N_SEEDS} seeds per cell, "
          f"lambda<=0 convention):")
    print(f"{'N':>4} {'T':>5} {'mean':>7} {'sd':>6} {'min':>6} {'q25':>6} "
          f"{'med':>6} {'q75':>6} {'max':>6}")
    all_means = []
    cell_stats = {}
    for N in N_GRID:
        for T in T_GRID:
            d = _null_distribution(N, T, N_SEEDS)
            cell_stats[(N, T)] = d
            all_means.append(d.mean())
            q25, med, q75 = np.percentile(d, [25, 50, 75])
            print(f"{N:>4} {T:>5} {d.mean():>7.3f} {d.std(ddof=1):>6.3f} "
                  f"{d.min():>6.3f} {q25:>6.3f} {med:>6.3f} {q75:>6.3f} "
                  f"{d.max():>6.3f}")
            assert 0.0 <= d.min() and d.max() <= 1.0

    # Per-cell: the mean over 40 seeds should sit near 0.5. Per-run PBO is
    # noisy (the 12870 combos share data), so allow a generous band; the
    # POINT of this test is that no cell drifts far from the coin flip.
    worst = 0.0
    for (N, T), d in cell_stats.items():
        dev = abs(d.mean() - 0.5)
        worst = max(worst, dev)
        assert dev < 0.12, f"N={N} T={T}: mean PBO {d.mean():.3f} far from 0.5"

    # Grand mean over all 480 runs: tight.
    grand = float(np.mean(all_means))
    print(f"grand mean over {len(all_means) * N_SEEDS} runs: {grand:.4f} "
          f"(worst cell deviation {worst:.3f})")
    assert abs(grand - 0.5) < 0.03, f"grand mean {grand:.4f} deviates from 0.5"


def test_null_odd_n_discreteness():
    """Sharp paper-level property for odd N: the lambda == 0 atom has
    probability 1/N, so the two threshold conventions must bracket 0.5:
        E[PBO_le] ~ 0.5 + 1/(2N),   E[PBO_lt] ~ 0.5 - 1/(2N)."""
    N, T, n_seeds = 9, 800, 60
    le, lt, zero_frac = [], [], []
    for r in range(n_seeds):
        M = white_noise(T, N, seed=[BASE_SEED, 999, r])
        f = cscv_pbo_fast(M, S=S_MAIN)
        le.append(f["pbo_le"])
        lt.append(f["pbo_lt"])
        zero_frac.append(f["n_zero_logits"] / f["n_combos"])
    m_le, m_lt, m_zero = np.mean(le), np.mean(lt), np.mean(zero_frac)
    print(f"\nodd N={N}: mean PBO(<=0)={m_le:.4f} (predict ~{0.5 + 1/(2*N):.4f}), "
          f"mean PBO(<0)={m_lt:.4f} (predict ~{0.5 - 1/(2*N):.4f}), "
          f"mean P(lambda=0)={m_zero:.4f} (predict ~{1/N:.4f})")
    assert m_le > 0.5 > m_lt, "conventions failed to bracket 0.5"
    assert m_le == pytest.approx(0.5 + 1 / (2 * N), abs=0.05)
    assert m_lt == pytest.approx(0.5 - 1 / (2 * N), abs=0.05)
    assert m_zero == pytest.approx(1 / N, abs=0.04)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
