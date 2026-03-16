"""
Multi-radius Schelling model experiment (r=1..6).

Runs FSS variance scaling for Chebyshev neighborhoods of radius 1-6
to study how the satisfaction spectrum size |F_k| affects criticality.

Usage:
  python benchmarking/multiradius_experiment.py               # all radii 3-6
  python benchmarking/multiradius_experiment.py --radius 3     # single radius

Output: outputs/data/multiradius_R{r}.npz per radius
"""

import argparse
import numpy as np
import sys, os, time
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.schelling import EMPTY, TYPE_A, TYPE_B

# ── Parameters ──────────────────────────────────────────────────────
RHO = 0.9
F_A = 0.5
MAX_STEPS = 500
SEED = 42

# Sweep parameters
N_T_SWEEP = 30
T_MIN, T_MAX = 0.10, 0.70
T_SWEEP = np.linspace(T_MIN, T_MAX, N_T_SWEEP)
L_SWEEP = 40
N_TRIALS_SWEEP = 30

# FSS parameters
L_FSS = [20, 40, 80]
N_TRIALS_FSS = 50


# ── Neighborhood offsets ────────────────────────────────────────────

def chebyshev_offsets(radius):
    offsets = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) <= radius:
                offsets.append((dr, dc))
    return offsets


def neighbor_count(radius):
    return (2 * radius + 1) ** 2 - 1


def satisfaction_spectrum(max_k):
    vals = set()
    for k in range(1, max_k + 1):
        for j in range(0, k + 1):
            vals.add(j / k)
    return sorted(vals)


# ── Vectorized Schelling with configurable neighborhood ─────────────

def compute_satisfaction_map(grid, size, offsets):
    """Compute satisfaction for all cells using given neighborhood offsets."""
    occupied_mask = grid != EMPTY
    same_count = np.zeros(grid.shape, dtype=np.float64)
    occ_count = np.zeros(grid.shape, dtype=np.float64)

    for dr, dc in offsets:
        shifted = np.roll(np.roll(grid, -dr, axis=0), -dc, axis=1)
        nbr_occ = shifted != EMPTY
        nbr_same = (shifted == grid) & nbr_occ & occupied_mask
        occ_count += nbr_occ
        same_count += nbr_same

    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(occ_count > 0, same_count / occ_count, 1.0)
    sat[~occupied_mask] = 0.0
    return sat


def init_grid(size, rho, f_A, rng):
    n_cells = size * size
    n_occ = int(n_cells * rho)
    n_a = int(n_occ * f_A)
    n_b = n_occ - n_a
    cells = np.array([TYPE_A]*n_a + [TYPE_B]*n_b + [EMPTY]*(n_cells - n_occ),
                      dtype=np.int8)
    rng.shuffle(cells)
    return cells.reshape(size, size)


def step(grid, size, T, offsets, rng):
    sat = compute_satisfaction_map(grid, size, offsets)
    occupied_mask = grid != EMPTY
    unsat_mask = occupied_mask & (sat < T)
    unsat_coords = np.argwhere(unsat_mask)

    empty_cells = list(zip(*np.where(grid == EMPTY)))
    if len(unsat_coords) == 0 or len(empty_cells) == 0:
        return 0

    rng.shuffle(unsat_coords)
    moved = 0
    for rc in unsat_coords:
        if not empty_cells:
            break
        r, c = rc[0], rc[1]
        idx = rng.integers(len(empty_cells))
        nr, nc = empty_cells[idx]
        grid[nr, nc] = grid[r, c]
        grid[r, c] = EMPTY
        empty_cells[idx] = (r, c)
        moved += 1
    return moved


def run_model(size, rho, f_A, T, offsets, rng, max_steps=MAX_STEPS):
    grid = init_grid(size, rho, f_A, rng)
    for _ in range(max_steps):
        if step(grid, size, T, offsets, rng) == 0:
            break
    return grid


def segregation_index(grid, offsets, f_A=F_A):
    sat = compute_satisfaction_map(grid, grid.shape[0], offsets)
    occ = grid != EMPTY
    if occ.sum() == 0:
        return 0.0
    mean_sat = float(sat[occ].mean())
    expected = f_A**2 + (1 - f_A)**2
    if expected >= 1.0:
        return 0.0
    return max(0.0, (mean_sat - expected) / (1.0 - expected))


# ── Single trial functions (for joblib) ─────────────────────────────

def _sweep_trial(T, offsets, trial, L=L_SWEEP):
    """Run one trial at a given T, return segregation index."""
    rng = np.random.default_rng(SEED + trial + int(round(T * 10000)))
    grid = run_model(L, RHO, F_A, T, offsets, rng)
    return segregation_index(grid, offsets)


def _fss_trial(L, T_fss, offsets, trial):
    """Run one FSS trial, return segregation index."""
    rng = np.random.default_rng(SEED + trial + 77777)
    grid = run_model(L, RHO, F_A, T_fss, offsets, rng)
    return segregation_index(grid, offsets)


# ── Main experiment for a single radius ─────────────────────────────

def run_radius(radius):
    """Run full experiment for a single radius: sweep + FSS + alpha fit."""
    k = neighbor_count(radius)
    offsets = chebyshev_offsets(radius)
    F_k = satisfaction_spectrum(k)
    F_k_size = len(F_k)

    print(f"\n{'='*60}")
    print(f"RADIUS {radius}: k={k} neighbors, |F_k|={F_k_size}")
    print(f"{'='*60}")

    # ── S(T) sweep ──────────────────────────────────────────────
    print(f"\nSweeping S(T) at L={L_SWEEP}, {N_TRIALS_SWEEP} trials, {N_T_SWEEP} T-values...")
    t0 = time.time()

    seg_means = []
    seg_stds = []
    for T in T_SWEEP:
        segs = Parallel(n_jobs=-1)(
            delayed(_sweep_trial)(T, offsets, trial) for trial in range(N_TRIALS_SWEEP)
        )
        seg_means.append(np.mean(segs))
        seg_stds.append(np.std(segs))

    seg_means = np.array(seg_means)
    seg_stds = np.array(seg_stds)
    print(f"  Sweep done in {time.time()-t0:.0f}s")

    # ── Find T_c via max |dS/dT| ───────────────────────────────
    dS = np.diff(seg_means) / np.diff(T_SWEEP)
    idx = np.argmax(dS)
    T_c = (T_SWEEP[idx] + T_SWEEP[idx + 1]) / 2
    print(f"  T_c = {T_c:.4f}")

    # ── FSS variance scaling ────────────────────────────────────
    T_fss = round(T_c * 100) / 100
    print(f"\nFSS variance scaling at T={T_fss:.2f}, L={L_FSS}...")

    variances = []
    for L in L_FSS:
        t1 = time.time()
        segs = Parallel(n_jobs=-1)(
            delayed(_fss_trial)(L, T_fss, offsets, trial) for trial in range(N_TRIALS_FSS)
        )
        v = np.var(segs)
        variances.append(v)
        print(f"  L={L}: Var(S)={v:.6f} ({time.time()-t1:.0f}s)")

    # ── Fit alpha ───────────────────────────────────────────────
    log_L = np.log(np.array(L_FSS, dtype=float))
    log_V = np.log(np.array(variances) + 1e-15)
    alpha, _ = np.polyfit(log_L, log_V, 1)
    print(f"  alpha = {alpha:.4f}")

    # ── Save ────────────────────────────────────────────────────
    os.makedirs('outputs/data', exist_ok=True)
    outpath = f'outputs/data/multiradius_R{radius}.npz'
    np.savez(outpath,
             radius=radius,
             k=k,
             T_c=T_c,
             variances=np.array(variances),
             alpha=alpha,
             L_fss=np.array(L_FSS),
             F_k_size=F_k_size,
             seg_mean=seg_means,
             seg_std=seg_stds,
             T_sweep=T_SWEEP)
    print(f"  Saved {outpath}")

    return {
        'radius': radius,
        'k': k,
        'T_c': T_c,
        'alpha': alpha,
        'F_k_size': F_k_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-radius Schelling experiment")
    parser.add_argument('--radius', type=int, default=None,
                        help='Run only this radius (1-6). Default: run 3-6.')
    args = parser.parse_args()

    if args.radius is not None:
        assert 1 <= args.radius <= 6, f"Radius must be 1-6, got {args.radius}"
        results = [run_radius(args.radius)]
    else:
        results = []
        for r in [3, 4, 5, 6]:
            results.append(run_radius(r))

    # ── Summary table ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'radius':>6} {'k':>5} {'|F_k|':>6} {'T_c':>8} {'alpha':>8}")
    print(f"{'-'*6:>6} {'-'*5:>5} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8}")
    for r in results:
        print(f"{r['radius']:>6d} {r['k']:>5d} {r['F_k_size']:>6d} "
              f"{r['T_c']:>8.4f} {r['alpha']:>8.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
