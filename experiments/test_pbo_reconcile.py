"""STEP 5 — Cross-implementation check: pypbo.pbo() vs purgedcv's
probability_of_backtest_overfitting on identical inputs.

Both implement CSCV from Bailey, Borwein, Lopez de Prado & Zhu (2015).
Read side by side, the implementations differ in exactly three places:

  D1. Zero-logit threshold. pypbo counts lambda <= 0 as overfitting
      (pbo.py line 250: ``1.0 if lam <= 0``); purgedcv counts strictly
      lambda < 0 (_pbo.py: ``np.mean(logit_arr < 0)``). The paper writes
      PBO as the integral of f(lambda) up to 0, which does not
      discriminate at an atom; for odd N the atom P(lambda = 0) ~ 1/N is
      real, so the two conventions genuinely diverge by n_zero/n_combos.

  D2. Remainder handling when T % S != 0. pypbo DROPS the first T % S
      rows (pbo.py lines 141-144: ``M = M[residual:]``), then uses S
      equal blocks; purgedcv keeps all rows and gives the first T % S
      blocks one extra row (_pbo.py `_contiguous_blocks`). Different
      data -> different logits. Neither contradicts the paper (which
      assumes S | T); they simply resolve the unstated case differently.

  D3. Input orientation. pypbo takes (T, N) with configs in columns;
      purgedcv takes (n_configs, n_obs) with configs in rows.

  Same in both: contiguous equal-ish blocks in time order, lexicographic
  C(S, S/2) enumeration (pypbo picks the IS half, purgedcv picks the OOS
  half — over the full enumeration these generate the same set of
  IS/OOS pairs), argmax IS winner, scipy average ranks,
  w_bar = rank/(N+1), natural-log logit.

Predicted reconciliation, TESTED below on identical inputs with the same
ddof=1 Sharpe metric and T divisible by S:

    sorted(logits_pypbo) == sorted(logits_purgedcv)          (exactly)
    pbo_pypbo - pbo_purgedcv == n_zero_logits / n_combos     (exactly)
    => identical PBO whenever no logit is exactly 0
       (generic for even N; atom of size ~1/N for odd N)

Run:  python -m pytest experiments/test_pbo_reconcile.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

from _cscv_common import (
    cscv_pbo_fast,
    planted_edge,
    sharpe_metric_columns,
    white_noise,
)

import pypbo
from purgedcv import probability_of_backtest_overfitting


def run_both(M: np.ndarray, S: int):
    """Run pypbo (configs in columns) and purgedcv (configs in rows) on the
    same matrix with the same ddof=1 Sharpe metric."""
    res_py = pypbo.pbo(M, S=S, metric_func=sharpe_metric_columns, threshold=0,
                       n_jobs=1, plot=False, verbose=False)
    res_pc = probability_of_backtest_overfitting(M.T, n_splits=S)
    return res_py, res_pc


@pytest.mark.parametrize(
    "name,S,make",
    [
        ("null_N10", 8, lambda: white_noise(400, 10, 101)),
        ("null_N50", 8, lambda: white_noise(960, 50, 102)),
        ("edge_N50", 8, lambda: planted_edge(960, 50, 5, 0.10, 103)),
        ("null_N10_S16", 16, lambda: white_noise(496, 10, 104)),
    ],
)
def test_identical_inputs_reconcile_exactly(name, S, make):
    """Aligned case (T % S == 0, even N): logit multisets identical and the
    PBO difference is exactly the zero-logit mass."""
    M = make()
    assert M.shape[0] % S == 0
    res_py, res_pc = run_both(M, S)

    logits_py = np.sort(np.asarray(res_py.logits))
    logits_pc = np.sort(np.asarray(res_pc.logits))
    assert res_pc.n_combos == len(res_py.Cs)
    assert np.allclose(logits_py, logits_pc, rtol=0, atol=1e-9), (
        f"{name}: logit multisets differ between implementations"
    )

    n_zero = int(np.sum(np.isclose(logits_pc, 0.0, atol=1e-12)))
    predicted_gap = n_zero / res_pc.n_combos
    gap = res_py.pbo - res_pc.pbo
    print(f"\n{name}: S={S} combos={res_pc.n_combos}  "
          f"pypbo PBO={res_py.pbo:.6f}  purgedcv PBO={res_pc.pbo:.6f}  "
          f"zero-logits={n_zero}  gap={gap:.6f}")
    assert gap == pytest.approx(predicted_gap, abs=1e-12), (
        f"{name}: PBO gap {gap} != zero-logit mass {predicted_gap}"
    )
    if n_zero == 0:
        assert res_py.pbo == pytest.approx(res_pc.pbo, abs=1e-12)


def test_s16_full_agreement_and_combo_count():
    """Full S=16 cross-check on one matrix: both enumerate 12870 = C(16,8)
    combinations (not 12780) and agree on every logit."""
    M = white_noise(496, 10, 104)
    res_py, res_pc = run_both(M, 16)
    assert len(res_py.Cs) == 12870
    assert res_pc.n_combos == 12870
    assert np.allclose(np.sort(np.asarray(res_py.logits)),
                       np.sort(np.asarray(res_pc.logits)), rtol=0, atol=1e-9)


def test_odd_n_zero_logit_divergence():
    """D1 made concrete: odd N puts an atom at lambda = 0. Find a seed whose
    run contains zero logits and show pypbo > purgedcv by exactly that mass."""
    S, N, T = 8, 9, 400
    for seed in range(10):
        M = white_noise(T, N, seed=[555, seed])
        fast = cscv_pbo_fast(M, S=S)
        if fast["n_zero_logits"] > 0:
            break
    else:
        pytest.fail("no zero logits in 10 seeds; with N=9 the atom has "
                    "probability ~1/9 per combo — this should not happen")

    res_py, res_pc = run_both(M, S)
    n_zero = int(np.sum(np.isclose(np.asarray(res_pc.logits), 0.0, atol=1e-12)))
    assert n_zero == fast["n_zero_logits"] > 0
    gap = res_py.pbo - res_pc.pbo
    print(f"\nodd-N divergence: seed={seed} zero-logits={n_zero}/70  "
          f"pypbo={res_py.pbo:.4f}  purgedcv={res_pc.pbo:.4f}  "
          f"gap={gap:.4f} == {n_zero}/70")
    assert gap == pytest.approx(n_zero / 70, abs=1e-12)
    assert res_py.pbo > res_pc.pbo


def test_remainder_partition_divergence():
    """D2 made concrete: T % S != 0 makes the implementations analyse
    different data (pypbo drops the first T % S rows; purgedcv keeps them
    and uses unequal blocks), so their logits legitimately differ."""
    S, N, T = 8, 20, 403  # T % S = 3
    diffs = []
    for seed in range(5):
        M = white_noise(T, N, seed=[777, seed])
        res_py, res_pc = run_both(M, S)
        same = np.allclose(np.sort(np.asarray(res_py.logits)),
                           np.sort(np.asarray(res_pc.logits)), rtol=0, atol=1e-9)
        diffs.append((res_py.pbo, res_pc.pbo, same))

    print("\nT=403, S=8 (remainder 3): pypbo-vs-purgedcv PBO per seed:")
    n_diverged = 0
    for i, (a, b, same) in enumerate(diffs):
        print(f"  seed {i}: pypbo={a:.4f}  purgedcv={b:.4f}  "
              f"logits_identical={same}  |diff|={abs(a - b):.4f}")
        if not same:
            n_diverged += 1
    # The partitions differ, so at least most seeds must yield different
    # logit multisets. (Equality could only occur by measure-zero accident.)
    assert n_diverged >= 4, (
        "expected the two partition conventions to produce different logits"
    )
    # Sanity: trimming 3 of 403 rows moves PBO a little, not wildly.
    for a, b, _ in diffs:
        assert abs(a - b) < 0.35


def test_dropped_rows_equivalence():
    """Confirm the D2 diagnosis mechanically: feeding purgedcv the matrix
    with the first T % S rows ALREADY removed (plus even N, aligned blocks)
    reproduces pypbo's logits exactly."""
    S, N, T = 8, 20, 403
    M = white_noise(T, N, seed=[777, 0])
    res_py = pypbo.pbo(M, S=S, metric_func=sharpe_metric_columns, threshold=0,
                       n_jobs=1, plot=False, verbose=False)
    res_pc = probability_of_backtest_overfitting(M[T % S:].T, n_splits=S)
    assert np.allclose(np.sort(np.asarray(res_py.logits)),
                       np.sort(np.asarray(res_pc.logits)), rtol=0, atol=1e-9)
    print("\nafter manually trimming the leading T % S rows, purgedcv "
          "reproduces pypbo exactly -> divergence fully explained by D2")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
