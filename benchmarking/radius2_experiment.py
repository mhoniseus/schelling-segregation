"""
Radius-2 neighborhood Schelling model experiment.

Tests whether the finer satisfaction spectrum F_24 (radius-2, 24 neighbors)
produces more critical-like FSS behavior than F_8 (radius-1, 8 neighbors).

Usage:
  python benchmarking/radius2_experiment.py            # full sweep + all FSS sizes
  python benchmarking/radius2_experiment.py --fss-L 160  # FSS for L=160 only (no sweep)

Output: outputs/data/radius2_results.npz  (full run)
        outputs/data/radius2_fss_L{L}.npz (single-L run)
"""

import argparse
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.schelling import EMPTY, TYPE_A, TYPE_B

# ── Parameters ──────────────────────────────────────────────────────
RHO = 0.9
F_A = 0.5
N_TRIALS = 30
MAX_STEPS = 500
SEED = 42

# Sweep parameters
T_SWEEP = np.arange(0.10, 0.91, 0.01)
L_SWEEP = 40

# FSS parameters
L_FSS = [20, 40, 80, 160, 320]
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

OFFSETS_R1 = chebyshev_offsets(1)  # 8
OFFSETS_R2 = chebyshev_offsets(2)  # 24


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


def run_fss_single_L(L, Tc_R1, Tc_R2):
    """Run FSS variance computation for a single lattice size L."""
    os.makedirs('outputs/data', exist_ok=True)
    partial = {}

    for label, offsets, Tc in [("R1", OFFSETS_R1, Tc_R1), ("R2", OFFSETS_R2, Tc_R2)]:
        T_fss = round(Tc * 100) / 100
        print(f"\nFSS variance scaling {label} at T={T_fss:.2f}, L={L}...")

        t0 = time.time()
        segs = []
        for trial in range(N_TRIALS_FSS):
            rng = np.random.default_rng(SEED + trial + 77777)
            grid = run_model(L, RHO, F_A, T_fss, offsets, rng)
            segs.append(segregation_index(grid, offsets))
        v = np.var(segs)
        partial[f"var_{label}_L{L}"] = v
        print(f"  L={L}: Var(S)={v:.6f} ({time.time()-t0:.0f}s)")

    partial['L'] = L
    outpath = f'outputs/data/radius2_fss_L{L}.npz'
    np.savez(outpath, **partial)
    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Radius-2 Schelling experiment")
    parser.add_argument('--fss-L', type=int, default=None,
                        help='Run FSS for a single lattice size L (requires sweep results)')
    parser.add_argument('--sweep-only', action='store_true',
                        help='Run only the S(T) sweep and save Tc values')
    args = parser.parse_args()

    os.makedirs('outputs/data', exist_ok=True)

    F8 = satisfaction_spectrum(8)
    F24 = satisfaction_spectrum(24)
    print(f"|F_8|  = {len(F8)}")
    print(f"|F_24| = {len(F24)}")

    # ── Single-L FSS mode (for CI parallelization) ─────────────────
    if args.fss_L is not None:
        L = args.fss_L
        assert L in L_FSS, f"L={L} not in L_FSS={L_FSS}"
        # Load Tc values from the sweep results
        sweep = np.load('outputs/data/radius2_sweep.npz')
        Tc_R1 = float(sweep['Tc_R1'])
        Tc_R2 = float(sweep['Tc_R2'])
        print(f"Loaded Tc: R1={Tc_R1:.3f}, R2={Tc_R2:.3f}")
        run_fss_single_L(L, Tc_R1, Tc_R2)
        return

    # ── Full run (backward-compatible) ─────────────────────────────
    results = {}

    # ── Sweep S(T) for both radii ──────────────────────────────────
    for label, offsets in [("R1", OFFSETS_R1), ("R2", OFFSETS_R2)]:
        print(f"\nSweeping {label} (L={L_SWEEP}, {N_TRIALS} trials)...")
        t0 = time.time()
        seg_means = []
        seg_stds = []
        for T in T_SWEEP:
            segs = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(SEED + trial + int(round(T * 10000)))
                grid = run_model(L_SWEEP, RHO, F_A, T, offsets, rng)
                segs.append(segregation_index(grid, offsets))
            seg_means.append(np.mean(segs))
            seg_stds.append(np.std(segs))
        results[f"seg_mean_{label}"] = np.array(seg_means)
        results[f"seg_std_{label}"] = np.array(seg_stds)
        print(f"  Done in {time.time()-t0:.0f}s")

    # ── Find Tc ────────────────────────────────────────────────────
    for label in ["R1", "R2"]:
        seg = results[f"seg_mean_{label}"]
        dS = np.diff(seg) / np.diff(T_SWEEP)
        idx = np.argmax(dS)
        Tc = (T_SWEEP[idx] + T_SWEEP[idx + 1]) / 2
        results[f"Tc_{label}"] = Tc
        print(f"T_c ({label}) = {Tc:.3f}")

    # Always save sweep data (used by --fss-L in CI)
    sweep_data = {
        'T_sweep': T_SWEEP,
        'Tc_R1': results['Tc_R1'],
        'Tc_R2': results['Tc_R2'],
        'seg_mean_R1': results['seg_mean_R1'],
        'seg_std_R1': results['seg_std_R1'],
        'seg_mean_R2': results['seg_mean_R2'],
        'seg_std_R2': results['seg_std_R2'],
    }
    np.savez('outputs/data/radius2_sweep.npz', **sweep_data)
    print("Saved outputs/data/radius2_sweep.npz")

    if args.sweep_only:
        return

    # ── FSS variance scaling for both radii ────────────────────────
    for label, offsets in [("R1", OFFSETS_R1), ("R2", OFFSETS_R2)]:
        Tc = results[f"Tc_{label}"]
        T_fss = round(Tc * 100) / 100
        print(f"\nFSS variance scaling {label} at T={T_fss:.2f}...")

        variances = []
        for L in L_FSS:
            t0 = time.time()
            segs = []
            for trial in range(N_TRIALS_FSS):
                rng = np.random.default_rng(SEED + trial + 77777)
                grid = run_model(L, RHO, F_A, T_fss, offsets, rng)
                segs.append(segregation_index(grid, offsets))
            v = np.var(segs)
            variances.append(v)
            print(f"  L={L}: Var(S)={v:.6f} ({time.time()-t0:.0f}s)")

        log_L = np.log(np.array(L_FSS, dtype=float))
        log_V = np.log(np.array(variances) + 1e-15)
        alpha, _ = np.polyfit(log_L, log_V, 1)
        results[f"var_{label}"] = np.array(variances)
        results[f"alpha_{label}"] = alpha
        print(f"  Variance exponent alpha ({label}) = {alpha:.3f}")

    # ── Save ───────────────────────────────────────────────────────
    results['T_sweep'] = T_SWEEP
    results['L_fss'] = np.array(L_FSS)
    results['F8_size'] = len(F8)
    results['F24_size'] = len(F24)
    np.savez('outputs/data/radius2_results.npz', **results)

    # ── Summary ────────────────────────────────────────────────────
    print(f"""
{'='*60}
RADIUS-2 SCHELLING TEST RESULTS
{'='*60}

Satisfaction spectrum: |F_8| = {len(F8)}, |F_24| = {len(F24)}

Transition: T_c(R1) = {results['Tc_R1']:.3f}, T_c(R2) = {results['Tc_R2']:.3f}

Variance scaling:
  R1: alpha = {results['alpha_R1']:.3f}
  R2: alpha = {results['alpha_R2']:.3f}
  (trivial averaging: -2.0; critical: > -2.0)

Interpretation:
  Radius-2 has {len(F24)} satisfaction thresholds vs {len(F8)} for radius-1.
  If alpha(R2) > alpha(R1) (less negative), the finer spectrum produces
  more critical-like fluctuations, supporting the continuous-limit conjecture.
{'='*60}
""")


if __name__ == '__main__':
    main()
