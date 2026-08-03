"""STEP 6 — Documented limitation: CSCV under autocorrelated returns.

pypbo's own docstring (pbo.py): "Not suitable for time series with strong
auto-correlation, especially when S is large."

EMPIRICAL CORRECTION TO THE NAIVE READING (found while building this test,
reported rather than papered over): a first version predicted that any
strong AR(1) coefficient drags mean PBO below 0.5. That is NOT what the
data show. At phi <= 0.95 with T=1600 and S<=16 (block length 100 >> the
correlation length l = 1/(1-phi) = 20) the mean PBO over 30 seeds is
statistically indistinguishable from 0.5 (wobbles ~0.43-0.54, SE ~0.035).
The operative quantity is the RATIO of correlation length to block length,
l / (T/S):

    l << T/S  ->  IS and OOS halves are nearly independent, PBO ~ 0.5;
    l >~ T/S  ->  interleaved IS/OOS halves share the same slow level, a
                  configuration's IS and OOS Sharpes become strongly
                  coupled, the IS winner persists OOS with no true edge,
                  and PBO collapses toward 0 — CSCV becomes confidently
                  over-optimistic under a TRUE null. Note the failure is
                  not extra noise: the seed-to-seed spread SHRINKS as the
                  collapse completes (sd -> ~0), a systematic false
                  reassurance.

This is exactly the docstring's warning made quantitative, including
"especially when S is large": at fixed phi and T, raising S shortens the
blocks and raises l/(T/S).

Probed regimes (mean PBO, 30 seeds, N=50 exchangeable AR(1) columns,
true Sharpe 0):
    phi=0.99, S=16, T=1600 (l=100, block=100) -> ~0.12
    phi=0.95, S=16, T=320  (l=20,  block=20)  -> ~0.11
    phi=0.99, S=16, T=320  (l=100, block=20)  -> ~0.00
    phi=0.99, S=4,  T=1600 (l=100, block=400) -> ~0.32  (larger S worse)

Produces: experiments/pbo_autocorr.png

Run:  python -m pytest experiments/test_pbo_autocorr.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _cscv_common import ar1_noise, cscv_pbo_fast, sharpe_metric_columns

import pypbo

N = 50
T_MAIN = 1600
PHIS = [0.0, 0.3, 0.6, 0.9, 0.95, 0.99]
S_GRID = [4, 8, 16]
N_SEEDS = 30
BASE_SEED = 90210

# Short-sample cells that push l/(T/S) >= 1 at moderate phi.
SHORT_CELLS = [(0.0, 16, 320), (0.95, 16, 320), (0.99, 16, 320)]


def _cell(phi: float, S: int, T: int) -> np.ndarray:
    vals = np.empty(N_SEEDS)
    for r in range(N_SEEDS):
        M = ar1_noise(T, N, phi, seed=[BASE_SEED, int(phi * 1000), S, T, r])
        vals[r] = cscv_pbo_fast(M, S=S)["pbo_le"]
    return vals


def corr_length(phi: float) -> float:
    return 1.0 if phi == 0.0 else 1.0 / (1.0 - phi)


def test_oracle_matches_real_pypbo_on_ar1():
    """The oracle validation extends to autocorrelated inputs."""
    M = ar1_noise(400, 10, 0.9, seed=[BASE_SEED, 1])
    res = pypbo.pbo(M, S=8, metric_func=sharpe_metric_columns, threshold=0,
                    n_jobs=1, plot=False, verbose=False)
    fast = cscv_pbo_fast(M, S=8)
    assert np.allclose(np.sort(np.asarray(res.logits)), np.sort(fast["logits"]),
                       rtol=0, atol=1e-10)
    assert res.pbo == pytest.approx(fast["pbo_le"], abs=1e-12)


def test_pbo_under_autocorrelation():
    stats: dict[tuple[float, int, int], tuple[float, float]] = {}

    print(f"\nmean(sd) PBO under AR(1) null, N={N}, {N_SEEDS} seeds/cell, "
          f"T={T_MAIN}:")
    print(f"{'phi':>6} {'l':>6} " + "  ".join(f"S={s:<11}" for s in S_GRID))
    for phi in PHIS:
        row = []
        for S in S_GRID:
            v = _cell(phi, S, T_MAIN)
            stats[(phi, S, T_MAIN)] = (v.mean(), v.std(ddof=1))
            row.append(f"{v.mean():.3f} ({v.std(ddof=1):.3f})")
        print(f"{phi:>6.2f} {corr_length(phi):>6.0f} " + "  ".join(row))

    print(f"short samples, T=320 (block length {320 // 16} at S=16):")
    for phi, S, T in SHORT_CELLS:
        v = _cell(phi, S, T)
        stats[(phi, S, T)] = (v.mean(), v.std(ddof=1))
        print(f"  phi={phi:<5} S={S:<3} T={T}: mean={v.mean():.3f} "
              f"sd={v.std(ddof=1):.3f}")

    def mean(phi, S, T=T_MAIN):
        return stats[(phi, S, T)][0]

    # --- Benign regime: iid data give ~0.5 for every S. ---
    for S in S_GRID:
        assert abs(mean(0.0, S) - 0.5) < 0.10, (
            f"phi=0, S={S}: mean PBO {mean(0.0, S):.3f} should be ~0.5"
        )
    assert abs(mean(0.0, 16, 320) - 0.5) < 0.10

    # --- Benign regime: block >> correlation length -> NO strong bias.
    # (The empirical finding that corrected the naive prediction.) ---
    for phi in (0.3, 0.6, 0.9, 0.95):
        for S in S_GRID:
            assert abs(mean(phi, S) - 0.5) < 0.15, (
                f"phi={phi}, S={S}: block {T_MAIN // S} >> l "
                f"{corr_length(phi):.0f}, expected mean PBO near 0.5, got "
                f"{mean(phi, S):.3f}"
            )

    # --- Breakdown regime: correlation length >= block length. ---
    assert mean(0.99, 16) < 0.30, (
        f"phi=0.99, S=16 (l = block = 100): expected collapse toward 0, "
        f"got {mean(0.99, 16):.3f}"
    )
    assert mean(0.95, 16, 320) < 0.30
    assert mean(0.99, 16, 320) < 0.10

    # --- 'Especially when S is large': at phi=0.99, S=16 much worse than
    # S=4 (shorter blocks, higher l/(T/S)). ---
    assert mean(0.99, 16) < mean(0.99, 4) - 0.05, (
        f"expected S=16 ({mean(0.99, 16):.3f}) clearly below "
        f"S=4 ({mean(0.99, 4):.3f}) at phi=0.99"
    )

    # --- The collapse is a confident lie, not noise: seed-to-seed sd in
    # the deep-breakdown cell is far below the iid cell's sd. ---
    sd_breakdown = stats[(0.99, 16, 320)][1]
    sd_iid = stats[(0.0, 16, 320)][1]
    assert sd_breakdown < sd_iid / 3, (
        f"expected the collapsed regime to be low-variance "
        f"(sd {sd_breakdown:.3f} vs iid {sd_iid:.3f})"
    )

    _plot(stats)


def _plot(stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    colors = {4: "#8A9BB8", 8: "#D96A3B", 16: "#3B6EF2"}
    for S in S_GRID:
        y = [stats[(phi, S, T_MAIN)][0] for phi in PHIS]
        yerr = [2 * stats[(phi, S, T_MAIN)][1] / np.sqrt(N_SEEDS) for phi in PHIS]
        ax1.errorbar(PHIS, y, yerr=yerr, marker="o", ms=4, lw=1.4, capsize=3,
                     color=colors[S], label=f"S = {S}")
    ax1.axhline(0.5, color="#95A1B4", lw=0.9, ls="--")
    ax1.set_xlabel("AR(1) coefficient (T = 1600, no true edge)")
    ax1.set_ylabel(f"PBO (mean over {N_SEEDS} seeds)")
    ax1.set_title("Flat near 0.5 while blocks >> corr. length;\n"
                  "collapses at phi = 0.99 (l = block)")
    ax1.legend()

    pts = []
    for (phi, S, T), (m, _) in stats.items():
        block = (T - T % S) / S
        pts.append((corr_length(phi) / block, m, S))
    pts.sort()
    for S in S_GRID:
        xs = [x for x, _, s in pts if s == S]
        ys = [y for _, y, s in pts if s == S]
        ax2.plot(xs, ys, "o", ms=5, color=colors[S], label=f"S = {S}")
    ax2.set_xscale("log")
    ax2.axhline(0.5, color="#95A1B4", lw=0.9, ls="--")
    ax2.axvline(1.0, color="#C24949", lw=0.9, ls=":",
                label="corr. length = block length")
    ax2.set_xlabel("correlation length / block length  (log)")
    ax2.set_ylabel("PBO")
    ax2.set_title("The operative ratio: PBO under a true null\n"
                  "collapses once l/(T/S) approaches 1")
    ax2.legend()

    fig.tight_layout()
    out = Path(__file__).parent / "pbo_autocorr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"plot written to {out}")
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
