"""
CI worker for publication-scale Schelling benchmarks.

Each GitHub Actions job runs this script with a JOB_ID env var.
The job ID maps to a specific (size, tolerance_chunk) or experiment type.

Job allocation (20 jobs):
  0:     L=20,  all 50 tol points, 50 trials
  1:     L=40,  all 50 tol points, 50 trials
  2:     L=80,  all 50 tol points, 50 trials
  3-5:   L=160, ~17 tol points each, 50 trials
  6-15:  L=320, 5 tol points each, 50 trials
  16:    Heterogeneous comparison (L=50)
  17:    Heterogeneous Tc extraction (L=50)
  18:    Trajectory sweep + null model (L=50)
  19:    Tolerance sweep (L=50) + convergence curves + beta exponent (L=320)

Outputs: outputs/chunks/chunk_{JOB_ID}.npz
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.schelling import (
    SchellingModel, HeterogeneousSchellingModel,
    segregation_index, interface_density,
)
from src.phase_diagram import extract_critical_point
from src.spatial_analysis import (
    multiscalar_trajectory, trajectory_statistics, null_model_trajectory,
)

N_WORKERS = max(1, multiprocessing.cpu_count())
OUTDIR = "outputs/chunks"

# Publication parameters
N_TRIALS = 50
DENSITY = 0.9
# Wide range covering transition + frozen regime for proper sigmoid fit
TOL_RANGE = np.linspace(0.10, 0.70, 50)
SIZES = [20, 40, 80, 160, 320]
BASE_SEED = 42

# Per-agent tolerance noise (sigma of Gaussian perturbation).
# Breaks discrete k/8 lattice thresholds, producing a smooth continuous
# transition suitable for finite-size scaling analysis.
TOLERANCE_NOISE = 0.02

# max_steps consistent across sizes for fair FSS comparison
def _max_steps_for_size(size):
    return 1000


def _run_trial(size, density, tolerance, max_steps, seed, tolerance_noise=0.0):
    """Run one SchellingModel trial, return segregation index."""
    m = SchellingModel(size=size, density=density, tolerance=tolerance,
                       tolerance_noise=tolerance_noise, seed=seed)
    m.run(max_steps=max_steps)
    return segregation_index(m.grid)


def _run_trial_full(size, density, tolerance, max_steps, seed):
    """Run one trial, return (seg, interface_density)."""
    m = SchellingModel(size=size, density=density, tolerance=tolerance, seed=seed)
    m.run(max_steps=max_steps)
    return segregation_index(m.grid), interface_density(m.grid)


def _run_het_trial(size, density, alpha, beta_param, max_steps, seed):
    """Run one HeterogeneousSchellingModel trial."""
    m = HeterogeneousSchellingModel(
        size=size, density=density, alpha=alpha, beta=beta_param, seed=seed,
    )
    m.run(max_steps=max_steps)
    return segregation_index(m.grid)


def _run_trajectory_trial(size, density, tolerance, max_steps, seed, target_type=1):
    """Run one trial, compute D(r) trajectory."""
    m = SchellingModel(size=size, density=density, tolerance=tolerance, seed=seed)
    m.run(max_steps=max_steps)
    r, d = multiscalar_trajectory(m.grid, target_type)
    stats = trajectory_statistics(r, d)
    return r, d, stats


def _run_convergence_trial(size, density, tolerance, max_steps, seed):
    """Run one trial, return moved_history and satisfaction_history."""
    m = SchellingModel(size=size, density=density, tolerance=tolerance, seed=seed)
    result = m.run(max_steps=max_steps)
    return result["moved_history"], result["satisfaction_history"]


def run_seg_sweep(size, tol_indices, n_trials, max_steps, base_seed):
    """Run segregation sweep for a given size and tolerance slice.

    Returns raw S values: shape (n_tol_points, n_trials).
    """
    tols = TOL_RANGE[tol_indices[0]:tol_indices[1]]
    rng = np.random.default_rng(base_seed + size * 1000)

    # Pre-generate all seeds (must match across chunks for same size)
    all_seeds = {}
    for idx in range(len(TOL_RANGE)):
        all_seeds[idx] = [int(rng.integers(0, 2**31)) for _ in range(n_trials)]

    # Build tasks for our chunk
    tasks = []
    local_indices = list(range(tol_indices[0], tol_indices[1]))
    for idx in local_indices:
        tol = TOL_RANGE[idx]
        for trial in range(n_trials):
            seed = all_seeds[idx][trial]
            tasks.append((size, DENSITY, float(tol), max_steps, seed, idx, trial))

    print(f"  Running {len(tasks)} sims for L={size}, "
          f"tol[{tol_indices[0]}:{tol_indices[1]}] ({len(local_indices)} points)..."
          f" (noise={TOLERANCE_NOISE})")
    t0 = time.time()

    n_tols = len(local_indices)
    raw_S = np.zeros((n_tols, n_trials))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t in tasks:
            sz, dens, tol, ms, seed, tol_idx, trial_idx = t
            fut = pool.submit(_run_trial, sz, dens, tol, ms, seed, TOLERANCE_NOISE)
            futures[fut] = (tol_idx - tol_indices[0], trial_idx)

        done = 0
        for fut in as_completed(futures):
            local_tol_idx, trial_idx = futures[fut]
            raw_S[local_tol_idx, trial_idx] = fut.result()
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(tasks)} done ({rate:.1f} sims/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(tasks)/elapsed:.1f} sims/s)")
    return raw_S, local_indices


def run_tolerance_sweep_full(n_trials=50):
    """Full tolerance sweep with interface density (for Figure 1)."""
    tolerances = np.linspace(0.10, 0.90, 40)
    rng = np.random.default_rng(0)

    tasks = []
    for i, tol in enumerate(tolerances):
        for trial in range(n_trials):
            seed = int(rng.integers(0, 2**31))
            tasks.append((50, DENSITY, float(tol), 5000, seed, i, trial))

    print(f"  Tolerance sweep: {len(tasks)} sims...")
    t0 = time.time()

    seg_raw = np.zeros((len(tolerances), n_trials))
    iface_raw = np.zeros((len(tolerances), n_trials))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t in tasks:
            sz, dens, tol, ms, seed, tol_idx, trial_idx = t
            fut = pool.submit(_run_trial_full, sz, dens, tol, ms, seed)
            futures[fut] = (tol_idx, trial_idx)

        for fut in as_completed(futures):
            tol_idx, trial_idx = futures[fut]
            seg, iface = fut.result()
            seg_raw[tol_idx, trial_idx] = seg
            iface_raw[tol_idx, trial_idx] = iface

    print(f"  Done in {time.time()-t0:.1f}s")
    return tolerances, seg_raw, iface_raw


def run_het_comparison():
    """Heterogeneous comparison for Figure 12."""
    mean_tol_range = np.linspace(0.15, 0.70, 25)
    concentrations = [2.0, 5.0, 20.0]
    n_trials = 30

    rng = np.random.default_rng(7)

    # Homogeneous tasks
    hom_tasks = []
    for i, tol in enumerate(mean_tol_range):
        for trial in range(n_trials):
            seed = int(rng.integers(0, 2**31))
            hom_tasks.append((50, DENSITY, float(tol), 5000, seed, i, trial))

    # Heterogeneous tasks
    het_tasks = []
    for kappa in concentrations:
        for i, tol in enumerate(mean_tol_range):
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                continue
            for trial in range(n_trials):
                seed = int(rng.integers(0, 2**31))
                het_tasks.append((50, DENSITY, float(alpha), float(beta_param),
                                  5000, seed, kappa, i, trial))

    print(f"  Het comparison: {len(hom_tasks) + len(het_tasks)} sims...")
    t0 = time.time()

    hom_raw = np.zeros((len(mean_tol_range), n_trials))
    het_raw = {}
    for kappa in concentrations:
        het_raw[kappa] = np.zeros((len(mean_tol_range), n_trials))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # Submit homogeneous
        hom_futures = {}
        for t in hom_tasks:
            sz, dens, tol, ms, seed, tol_idx, trial_idx = t
            fut = pool.submit(_run_trial, sz, dens, tol, ms, seed)
            hom_futures[fut] = (tol_idx, trial_idx)

        # Submit heterogeneous
        het_futures = {}
        for t in het_tasks:
            sz, dens, alpha, beta_p, ms, seed, kappa, tol_idx, trial_idx = t
            fut = pool.submit(_run_het_trial, sz, dens, alpha, beta_p, ms, seed)
            het_futures[fut] = (kappa, tol_idx, trial_idx)

        for fut in as_completed(hom_futures):
            tol_idx, trial_idx = hom_futures[fut]
            hom_raw[tol_idx, trial_idx] = fut.result()

        for fut in as_completed(het_futures):
            kappa, tol_idx, trial_idx = het_futures[fut]
            het_raw[kappa][tol_idx, trial_idx] = fut.result()

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {"tolerances": mean_tol_range, "hom_raw": hom_raw}
    for kappa in concentrations:
        save_dict[f"het_raw_k{kappa:.1f}"] = het_raw[kappa]
    return save_dict


def run_het_tc():
    """Heterogeneous Tc extraction for all kappa values."""
    concentrations = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    tolerance_range = np.linspace(0.15, 0.70, 30)
    n_trials = 30

    rng = np.random.default_rng(77)

    tasks = []
    for kappa in concentrations:
        for i, tol in enumerate(tolerance_range):
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                continue
            for trial in range(n_trials):
                seed = int(rng.integers(0, 2**31))
                tasks.append((50, DENSITY, float(alpha), float(beta_param),
                              5000, seed, kappa, i, trial))

    print(f"  Het Tc: {len(tasks)} sims...")
    t0 = time.time()

    raw = {}
    for kappa in concentrations:
        raw[kappa] = np.zeros((len(tolerance_range), n_trials))

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t in tasks:
            sz, dens, alpha, beta_p, ms, seed, kappa, tol_idx, trial_idx = t
            fut = pool.submit(_run_het_trial, sz, dens, alpha, beta_p, ms, seed)
            futures[fut] = (kappa, tol_idx, trial_idx)

        done = 0
        for fut in as_completed(futures):
            kappa, tol_idx, trial_idx = futures[fut]
            raw[kappa][tol_idx, trial_idx] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(tasks)}")

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {
        "concentrations": np.array(concentrations),
        "tolerances": tolerance_range,
    }
    for kappa in concentrations:
        save_dict[f"raw_k{kappa:.1f}"] = raw[kappa]
    return save_dict


def run_trajectories():
    """Trajectory sweep + null model."""
    tolerances = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])
    n_trials = 20

    rng = np.random.default_rng(0)
    tasks = []
    for i, tol in enumerate(tolerances):
        for trial in range(n_trials):
            seed = int(rng.integers(0, 2**31))
            tasks.append((50, DENSITY, float(tol), 5000, seed))

    print(f"  Trajectories: {len(tasks)} sims...")
    t0 = time.time()

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t in tasks:
            fut = pool.submit(_run_trajectory_trial, *t)
            futures[fut] = round(t[2], 6)

        for fut in as_completed(futures):
            tol = futures[fut]
            r, d, stats = fut.result()
            raw.setdefault(tol, []).append((r, d, stats))

    print(f"  Sims done in {time.time()-t0:.1f}s")

    # Null model
    print("  Null model (500 samples)...")
    radii_null, mean_null, std_null = null_model_trajectory(
        size=50, density=DENSITY, n_samples=500, max_radius=12, seed=99,
    )

    radii = raw[round(float(tolerances[0]), 6)][0][0]
    save_dict = {
        "tolerances": tolerances, "radii": radii,
        "radii_null": radii_null, "mean_null": mean_null, "std_null": std_null,
    }
    for tol in tolerances:
        entries = raw[round(float(tol), 6)]
        all_D = np.array([e[1] for e in entries])
        save_dict[f"mean_D_T{tol:.2f}"] = np.mean(all_D, axis=0)
        save_dict[f"std_D_T{tol:.2f}"] = np.std(all_D, axis=0)
        all_stats = [e[2] for e in entries]
        for k in all_stats[0]:
            vals = [s[k] for s in all_stats]
            save_dict[f"stat_{k}_T{tol:.2f}"] = float(np.mean(vals))

    return save_dict


def run_convergence_and_beta():
    """Convergence curves + beta exponent using L=320."""
    # Convergence curves
    conv_tolerances = [0.30, 0.45, 0.60]
    n_conv = 20
    rng = np.random.default_rng(42)

    tasks = []
    for tol in conv_tolerances:
        for _ in range(n_conv):
            seed = int(rng.integers(0, 2**31))
            tasks.append((50, DENSITY, float(tol), 5000, seed))

    print(f"  Convergence curves: {len(tasks)} sims...")
    t0 = time.time()
    conv_raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_convergence_trial, *t): round(t[2], 6) for t in tasks}
        for fut in as_completed(futures):
            tol = futures[fut]
            moved, sat = fut.result()
            conv_raw.setdefault(tol, []).append((moved, sat))

    conv_dict = {}
    for tol in conv_tolerances:
        entries = conv_raw[round(float(tol), 6)]
        all_moved = [e[0] for e in entries]
        all_sat = [e[1] for e in entries]
        max_len = max(len(c) for c in all_moved)
        for lst in [all_moved, all_sat]:
            for i in range(len(lst)):
                lst[i] = lst[i] + [lst[i][-1]] * (max_len - len(lst[i]))
        conv_dict[f"moved_T{tol:.2f}"] = np.mean(all_moved, axis=0)
        conv_dict[f"sat_T{tol:.2f}"] = np.mean(all_sat, axis=0)

    print(f"  Convergence done in {time.time()-t0:.1f}s")

    # Tolerance sweep (Figure 1)
    print("  Tolerance sweep...")
    sweep_tols, sweep_seg, sweep_iface = run_tolerance_sweep_full(n_trials=50)
    conv_dict["sweep_tolerances"] = sweep_tols
    conv_dict["sweep_seg_raw"] = sweep_seg
    conv_dict["sweep_iface_raw"] = sweep_iface

    return conv_dict


# Job dispatch
JOB_MAP = {
    # (size, tol_start, tol_end) for seg sweep jobs
    0:  (20,  0, 50),
    1:  (40,  0, 50),
    2:  (80,  0, 50),
    3:  (160, 0, 17),
    4:  (160, 17, 34),
    5:  (160, 34, 50),
    6:  (320, 0, 5),
    7:  (320, 5, 10),
    8:  (320, 10, 15),
    9:  (320, 15, 20),
    10: (320, 20, 25),
    11: (320, 25, 30),
    12: (320, 30, 35),
    13: (320, 35, 40),
    14: (320, 40, 45),
    15: (320, 45, 50),
}


if __name__ == "__main__":
    job_id = int(os.environ.get("JOB_ID", sys.argv[1] if len(sys.argv) > 1 else "0"))
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"=== CI Worker: Job {job_id} ({N_WORKERS} cores) ===\n")
    t_total = time.time()

    if job_id in JOB_MAP:
        # Segregation sweep job
        size, tol_start, tol_end = JOB_MAP[job_id]
        max_steps = _max_steps_for_size(size)
        raw_S, local_indices = run_seg_sweep(
            size, (tol_start, tol_end), N_TRIALS, max_steps, BASE_SEED,
        )
        np.savez(
            os.path.join(OUTDIR, f"chunk_{job_id}.npz"),
            job_type="seg_sweep",
            size=size,
            tol_start=tol_start,
            tol_end=tol_end,
            raw_S=raw_S,
            local_indices=np.array(local_indices),
            tol_range=TOL_RANGE,
        )

    elif job_id == 16:
        save_dict = run_het_comparison()
        save_dict["job_type"] = "het_comparison"
        np.savez(os.path.join(OUTDIR, f"chunk_{job_id}.npz"), **save_dict)

    elif job_id == 17:
        save_dict = run_het_tc()
        save_dict["job_type"] = "het_tc"
        np.savez(os.path.join(OUTDIR, f"chunk_{job_id}.npz"), **save_dict)

    elif job_id == 18:
        save_dict = run_trajectories()
        save_dict["job_type"] = "trajectories"
        np.savez(os.path.join(OUTDIR, f"chunk_{job_id}.npz"), **save_dict)

    elif job_id == 19:
        save_dict = run_convergence_and_beta()
        save_dict["job_type"] = "convergence_sweep"
        np.savez(os.path.join(OUTDIR, f"chunk_{job_id}.npz"), **save_dict)

    else:
        print(f"Unknown job_id: {job_id}")
        sys.exit(1)

    print(f"\n=== Job {job_id} finished in {time.time()-t_total:.1f}s ===")
