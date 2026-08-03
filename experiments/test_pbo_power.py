"""STEP 4 — The planted-edge (power) test: PBO must fall as real edge grows.

Setup: N = 50 configurations, T = 1000 observations, S = 16. The first 5
columns receive a genuine constant drift giving them a known true
per-observation Sharpe (the "edge"); the remaining 45 are pure noise.

Theory: with zero edge this is the null (PBO ~ 0.5). As the edge grows, the
in-sample winner is increasingly one of the truly skilled configurations,
which also outranks the noise out-of-sample, so logits go positive and PBO
must fall well below 0.5 — approaching 0 when the edge dominates sampling
noise. The transition should occur roughly where the true edge matches the
scale of the maximum in-sample noise Sharpe over N trials on T/2 obs
(around sqrt(2 ln N / (T/2)) ~ 0.13 per obs here) — this is the same
extreme-value scale that drives the DSR's expected-maximum term.

The sweep uses the fast oracle validated in test_pbo_null.py; one sweep
point is re-validated against the real pypbo.pbo() here as well.

Produces: experiments/pbo_power_curve.png

Run:  python -m pytest experiments/test_pbo_power.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _cscv_common import cscv_pbo_fast, planted_edge, sharpe_metric_columns

import pypbo

S = 16
N = 50
T = 1000
N_SKILL = 5
EDGES = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]  # true per-obs Sharpe
N_SEEDS = 30
BASE_SEED = 47


def _power_point(edge: float, n_seeds: int = N_SEEDS) -> np.ndarray:
    vals = np.empty(n_seeds)
    for r in range(n_seeds):
        M = planted_edge(T, N, N_SKILL, edge, seed=[BASE_SEED, int(edge * 1000), r])
        vals[r] = cscv_pbo_fast(M, S=S)["pbo_le"]
    return vals


def test_power_point_matches_real_pypbo():
    """Spot-validate the oracle on planted-edge data against real pypbo."""
    M = planted_edge(T, N, N_SKILL, 0.10, seed=[BASE_SEED, 100, 0])
    res = pypbo.pbo(M, S=8, metric_func=sharpe_metric_columns, threshold=0,
                    n_jobs=1, plot=False, verbose=False)
    fast = cscv_pbo_fast(M, S=8)
    assert np.allclose(np.sort(np.asarray(res.logits)), np.sort(fast["logits"]),
                       rtol=0, atol=1e-10)
    assert res.pbo == pytest.approx(fast["pbo_le"], abs=1e-12)


def test_pbo_power_curve():
    """PBO ~ 0.5 at zero edge, well below 0.5 at strong edge, and the
    transition is monotone (compared between well-separated points)."""
    means, sds, all_vals = [], [], {}
    print(f"\nPBO vs planted edge (S={S}, N={N}, {N_SKILL} skilled cols, "
          f"T={T}, {N_SEEDS} seeds):")
    print(f"{'edge/obs':>9} {'~SR_ann':>8} {'meanPBO':>8} {'sd':>6}")
    for e in EDGES:
        vals = _power_point(e)
        all_vals[e] = vals
        means.append(vals.mean())
        sds.append(vals.std(ddof=1))
        print(f"{e:>9.3f} {e * np.sqrt(252):>8.2f} {vals.mean():>8.3f} "
              f"{vals.std(ddof=1):>6.3f}")

    means_arr = np.asarray(means)

    # Zero edge is the null: mean near 0.5.
    se0 = all_vals[0.0].std(ddof=1) / np.sqrt(N_SEEDS)
    assert abs(means_arr[0] - 0.5) < max(0.1, 4 * se0), (
        f"zero-edge PBO {means_arr[0]:.3f} not consistent with 0.5"
    )

    # Strong edge: PBO must be far below the coin flip.
    assert means_arr[-1] < 0.15, (
        f"PBO at edge {EDGES[-1]} is {means_arr[-1]:.3f}; theory demands << 0.5"
    )

    # Monotone decrease between well-separated sweep points (allowing
    # sampling noise between neighbours).
    assert means_arr[3] < means_arr[0] - 0.05   # 0.075 vs 0.0
    assert means_arr[5] < means_arr[3] - 0.05   # 0.15  vs 0.075
    assert means_arr[-1] <= means_arr[5]        # 0.30  vs 0.15

    _plot(means_arr, np.asarray(sds))


def _plot(means: np.ndarray, sds: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    se = sds / np.sqrt(N_SEEDS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(EDGES, means, yerr=2 * se, marker="o", ms=4, lw=1.4,
                capsize=3, color="#3B6EF2", label="mean PBO ± 2 SE")
    ax.axhline(0.5, color="#95A1B4", lw=0.9, ls="--", label="null (no edge)")
    ax.set_xlabel("true per-observation Sharpe of the 5 skilled configs")
    ax.set_ylabel("PBO (S=16, mean over 30 seeds)")
    ax.set_title(f"CSCV power curve: PBO vs planted edge (N={N}, T={T})")
    sec = ax.secondary_xaxis(
        "top", functions=(lambda x: x * np.sqrt(252), lambda x: x / np.sqrt(252))
    )
    sec.set_xlabel("annualised Sharpe equivalent (252 obs/yr)")
    ax.legend()
    fig.tight_layout()
    out = Path(__file__).parent / "pbo_power_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"plot written to {out}")
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
