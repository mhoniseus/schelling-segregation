"""
Parallel benchmark runner - uses all CPU cores to run trials concurrently.

Wraps bench_schelling.py functions but parallelizes across (size, tolerance) pairs
using concurrent.futures.ProcessPoolExecutor.

Auteur : Mouhssine Rifaki
"""

import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.schelling import (
    SchellingModel, HeterogeneousSchellingModel,
    segregation_index, interface_density,
)
from src.phase_diagram import (
    extract_critical_point, _sigmoid,
    critical_exponents, susceptibility_exponent,
)
from src.spatial_analysis import (
    multiscalar_trajectory, trajectory_statistics,
    null_model_trajectory,
)

OUTDIR = "outputs/data"
N_WORKERS = max(1, multiprocessing.cpu_count() - 2)  # leave 2 cores free


# Worker functions (must be top-level for pickling)

def _run_single_trial(size, density, tolerance, max_steps, seed):
    """Run one SchellingModel trial, return segregation index."""
    m = SchellingModel(size=size, density=density, tolerance=tolerance, seed=seed)
    m.run(max_steps=max_steps)
    return segregation_index(m.grid)


def _run_single_trial_full(size, density, tolerance, max_steps, seed):
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


# Parallel sweep helpers

def _parallel_seg_sweep(sizes, tolerance_range, density, n_trials, max_steps, base_seed):
    """Run S(T) for multiple sizes in parallel. Returns dict L -> (mean, std)."""
    rng = np.random.default_rng(base_seed)
    # Build all tasks: (size, tol, seed)
    tasks = []
    for sz in sizes:
        for tol in tolerance_range:
            for _ in range(n_trials):
                tasks.append((sz, density, float(tol), max_steps, int(rng.integers(0, 2**31))))

    # Run all tasks in parallel
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            task_args = futures[fut]
            sz, _, tol, _, _ = task_args
            key = (sz, round(tol, 6))
            if key not in results:
                results[key] = []
            results[key].append(fut.result())

    # Aggregate
    data = {}
    for sz in sizes:
        means = np.zeros(len(tolerance_range))
        stds = np.zeros(len(tolerance_range))
        for i, tol in enumerate(tolerance_range):
            vals = results[(sz, round(float(tol), 6))]
            means[i] = np.mean(vals)
            stds[i] = np.std(vals) / np.sqrt(len(vals))
        data[sz] = (means, stds)
    return data


# Benchmark functions

def run_tolerance_sweep(n_trials=20):
    """Tolerance sweep with parallel trials."""
    tolerances = np.linspace(0.10, 0.75, 30)
    rng = np.random.default_rng(0)
    tasks = []
    for tol in tolerances:
        for _ in range(n_trials):
            tasks.append((50, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    seg_results = {}
    iface_results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_trial_full, *t): t for t in tasks}
        for fut in as_completed(futures):
            tol = round(futures[fut][2], 6)
            seg, iface = fut.result()
            seg_results.setdefault(tol, []).append(seg)
            iface_results.setdefault(tol, []).append(iface)

    means = np.array([np.mean(seg_results[round(float(t), 6)]) for t in tolerances])
    stds = np.array([np.std(seg_results[round(float(t), 6)]) / np.sqrt(n_trials) for t in tolerances])
    iface_means = np.array([np.mean(iface_results[round(float(t), 6)]) for t in tolerances])
    iface_stds = np.array([np.std(iface_results[round(float(t), 6)]) / np.sqrt(n_trials) for t in tolerances])

    cp = extract_critical_point(tolerances, means, stds)
    np.savez(
        os.path.join(OUTDIR, "tolerance_sweep.npz"),
        tolerances=tolerances, means=means, stds=stds,
        interface_density=iface_means, interface_density_std=iface_stds,
        T_c=cp["T_c"], T_c_err=cp["T_c_err"],
        width=cp["width"], width_err=cp["width_err"],
        A=cp["A"], B=cp["B"],
        frozen_idx=cp.get("frozen_idx", len(tolerances)),
    )
    print(f"  Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}, width = {cp['width']:.4f}")
    return cp


def run_finite_size_scaling():
    """FSS with parallel trials across all (L, T) pairs."""
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.20, 0.70, 25)

    print("  Finite-size scaling (parallel)...")
    t0 = time.time()
    data = _parallel_seg_sweep(sizes, tolerance_range, 0.9, 10, 1000, 42)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Extract Tc(L)
    T_c_dict = {}
    for sz in sizes:
        mean_seg, std_seg = data[sz]
        cp = extract_critical_point(tolerance_range, mean_seg, std_seg)
        T_c_dict[sz] = (cp["T_c"], cp["T_c_err"])

    L_arr = np.array(sizes, dtype=float)
    Tc_arr = np.array([T_c_dict[sz][0] for sz in sizes])
    Tc_err_arr = np.array([T_c_dict[sz][1] for sz in sizes])
    ce = critical_exponents(L_arr, Tc_arr, Tc_err_arr)
    print(f"  Tc(inf) = {ce['T_c_inf']:.4f} +/- {ce['T_c_inf_err']:.4f}, nu = {ce['nu']:.2f} +/- {ce['nu_err']:.2f}")

    # Scaling collapse
    from src.phase_diagram import scaling_collapse
    sc = scaling_collapse(sizes, tolerance_range, data, ce["T_c_inf"])
    print(f"  Best collapse nu = {sc['best_nu']:.2f}")

    save_dict = {
        "sizes": L_arr, "tolerances": tolerance_range,
        "Tc_values": Tc_arr, "Tc_errors": Tc_err_arr,
        "Tc_inf": ce["T_c_inf"], "Tc_inf_err": ce["T_c_inf_err"],
        "nu": ce["nu"], "nu_err": ce["nu_err"],
        "collapse_best_nu": sc["best_nu"],
        "collapse_nu_range": sc["nu_range"], "collapse_quality": sc["quality"],
    }
    for sz in sizes:
        mean_seg, std_seg = data[sz]
        save_dict[f"seg_mean_L{sz}"] = mean_seg
        save_dict[f"seg_std_L{sz}"] = std_seg
        save_dict[f"collapse_x_L{sz}"] = sc["collapsed_x"][sz]
        save_dict[f"collapse_y_L{sz}"] = sc["collapsed_y"][sz]

    np.savez(os.path.join(OUTDIR, "finite_size_scaling.npz"), **save_dict)
    return data, ce, sc


def run_binder_cumulant():
    """Binder cumulant with parallel trials."""
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.25, 0.60, 25)
    n_trials = 15

    print("  Binder cumulant (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(123)
    tasks = []
    for sz in sizes:
        for tol in tolerance_range:
            for _ in range(n_trials):
                tasks.append((sz, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            sz, _, tol, _, _ = futures[fut]
            key = (sz, round(tol, 6))
            raw.setdefault(key, []).append(fut.result())

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {"sizes": np.array(sizes, dtype=float), "tolerances": tolerance_range}
    boot_rng = np.random.default_rng(999)

    for sz in sizes:
        u4_vals = np.zeros(len(tolerance_range))
        u4_errs = np.zeros(len(tolerance_range))
        for i, tol in enumerate(tolerance_range):
            samples = np.array(raw[(sz, round(float(tol), 6))])
            s2 = np.mean(samples**2)
            s4 = np.mean(samples**4)
            u4_vals[i] = 1.0 - s4 / (3.0 * s2**2) if s2 > 0 else 2.0/3.0
            # Bootstrap
            u4_boot = np.zeros(200)
            for b in range(200):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                s2b, s4b = np.mean(sb**2), np.mean(sb**4)
                u4_boot[b] = 1.0 - s4b / (3.0 * s2b**2) if s2b > 0 else 2.0/3.0
            u4_errs[i] = np.std(u4_boot)
        save_dict[f"U4_L{sz}"] = u4_vals
        save_dict[f"U4_err_L{sz}"] = u4_errs

    np.savez(os.path.join(OUTDIR, "binder_cumulant.npz"), **save_dict)


def run_susceptibility():
    """Susceptibility with parallel trials."""
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.25, 0.60, 25)
    n_trials = 15

    print("  Susceptibility (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(456)
    tasks = []
    for sz in sizes:
        for tol in tolerance_range:
            for _ in range(n_trials):
                tasks.append((sz, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            sz, _, tol, _, _ = futures[fut]
            key = (sz, round(tol, 6))
            raw.setdefault(key, []).append(fut.result())

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {"sizes": np.array(sizes, dtype=float), "tolerances": tolerance_range}
    boot_rng = np.random.default_rng(888)

    for sz in sizes:
        chi_vals = np.zeros(len(tolerance_range))
        chi_errs = np.zeros(len(tolerance_range))
        for i, tol in enumerate(tolerance_range):
            samples = np.array(raw[(sz, round(float(tol), 6))])
            chi_vals[i] = sz**2 * (np.mean(samples**2) - np.mean(samples)**2)
            u4_boot = np.zeros(200)
            for b in range(200):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                u4_boot[b] = sz**2 * (np.mean(sb**2) - np.mean(sb)**2)
            chi_errs[i] = np.std(u4_boot)
        save_dict[f"chi_L{sz}"] = chi_vals
        save_dict[f"chi_err_L{sz}"] = chi_errs
        peak_idx = np.argmax(chi_vals)
        save_dict[f"chi_peak_T_L{sz}"] = float(tolerance_range[peak_idx])
        save_dict[f"chi_peak_val_L{sz}"] = float(chi_vals[peak_idx])

    L_arr = np.array(sizes, dtype=float)
    chi_peaks = np.array([float(save_dict[f"chi_peak_val_L{sz}"]) for sz in sizes])
    ge = susceptibility_exponent(L_arr, chi_peaks)
    save_dict["gamma_over_nu"] = ge["gamma_over_nu"]
    save_dict["gamma_over_nu_err"] = ge["gamma_over_nu_err"]
    print(f"  gamma/nu = {ge['gamma_over_nu']:.3f} +/- {ge['gamma_over_nu_err']:.3f}")

    np.savez(os.path.join(OUTDIR, "susceptibility.npz"), **save_dict)


def run_order_parameter_exponent(T_c_inf):
    """Beta exponent with parallel trials."""
    tolerance_range = np.linspace(T_c_inf + 0.01, T_c_inf + 0.20, 15)
    n_trials = 20
    sz = 80

    print("  Order parameter exponent beta (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(789)
    tasks = []
    for tol in tolerance_range:
        for _ in range(n_trials):
            tasks.append((sz, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            _, _, tol, _, _ = futures[fut]
            key = round(tol, 6)
            raw.setdefault(key, []).append(fut.result())

    print(f"  Done in {time.time()-t0:.1f}s")

    mean_S = np.array([np.mean(raw[round(float(tol), 6)]) for tol in tolerance_range])
    dT = tolerance_range - T_c_inf
    valid = (mean_S > 0.01) & (dT > 0)

    beta, beta_err = np.nan, np.nan
    if valid.sum() >= 3:
        log_dT = np.log(dT[valid])
        log_S = np.log(mean_S[valid])
        try:
            coeffs, cov = np.polyfit(log_dT, log_S, deg=1, cov=True)
            beta = float(coeffs[0])
            beta_err = float(np.sqrt(cov[0, 0]))
        except (np.linalg.LinAlgError, ValueError):
            pass

    print(f"  beta = {beta:.3f} +/- {beta_err:.3f}")
    np.savez(
        os.path.join(OUTDIR, "order_parameter_exponent.npz"),
        beta=beta, beta_err=beta_err,
        T_fit=tolerance_range, S_fit=mean_S, T_c=T_c_inf,
    )


def run_trajectory_sweep():
    """Trajectory sweep with parallel trials."""
    tolerances = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])
    n_trials = 10

    print("  Trajectory sweep (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(0)
    tasks = []
    for tol in tolerances:
        for _ in range(n_trials):
            tasks.append((50, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_trajectory_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            _, _, tol, _, _ = futures[fut]
            key = round(tol, 6)
            r, d, stats = fut.result()
            raw.setdefault(key, []).append((r, d, stats))

    print(f"  Sims done in {time.time()-t0:.1f}s")

    # Null model (fast, no parallelism needed)
    print("  Null model trajectory (200 samples)...")
    radii_null, mean_null, std_null = null_model_trajectory(
        size=50, density=0.9, n_samples=200, max_radius=12, seed=99,
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

    np.savez(os.path.join(OUTDIR, "trajectory_sweep.npz"), **save_dict)


def run_heterogeneous_comparison():
    """Heterogeneous comparison with parallel trials."""
    mean_tol_range = np.linspace(0.15, 0.70, 20)
    concentrations = [2.0, 5.0, 20.0]
    n_trials = 10

    print("  Heterogeneous comparison (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(7)

    # Homogeneous
    hom_tasks = []
    for tol in mean_tol_range:
        for _ in range(n_trials):
            hom_tasks.append((50, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    # Heterogeneous
    het_tasks = []
    for kappa in concentrations:
        for tol in mean_tol_range:
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                continue
            for _ in range(n_trials):
                het_tasks.append((50, 0.9, float(alpha), float(beta_param), 1000,
                                  int(rng.integers(0, 2**31)), kappa, float(tol)))

    hom_raw = {}
    het_raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        hom_futures = {pool.submit(_run_single_trial, *t): t for t in hom_tasks}
        het_futures = {}
        for t in het_tasks:
            sz, dens, alpha, beta_p, ms, seed, kappa, tol = t
            fut = pool.submit(_run_het_trial, sz, dens, alpha, beta_p, ms, seed)
            het_futures[fut] = (kappa, tol)

        for fut in as_completed(hom_futures):
            tol = round(hom_futures[fut][2], 6)
            hom_raw.setdefault(tol, []).append(fut.result())

        for fut in as_completed(het_futures):
            kappa, tol = het_futures[fut]
            het_raw.setdefault((kappa, round(tol, 6)), []).append(fut.result())

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {"tolerances": mean_tol_range}
    hom_mean = np.array([np.mean(hom_raw[round(float(t), 6)]) for t in mean_tol_range])
    hom_std = np.array([np.std(hom_raw[round(float(t), 6)]) / np.sqrt(n_trials) for t in mean_tol_range])
    save_dict["hom_mean"] = hom_mean
    save_dict["hom_std"] = hom_std

    for kappa in concentrations:
        het_mean = np.zeros(len(mean_tol_range))
        het_std = np.zeros(len(mean_tol_range))
        for i, tol in enumerate(mean_tol_range):
            key = (kappa, round(float(tol), 6))
            if key in het_raw:
                vals = het_raw[key]
                het_mean[i] = np.mean(vals)
                het_std[i] = np.std(vals) / np.sqrt(len(vals))
        save_dict[f"het_mean_k{kappa:.1f}"] = het_mean
        save_dict[f"het_std_k{kappa:.1f}"] = het_std

    np.savez(os.path.join(OUTDIR, "heterogeneous_comparison.npz"), **save_dict)


def run_heterogeneous_tc():
    """Tc(kappa) extraction with parallel trials."""
    concentrations = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    tolerance_range = np.linspace(0.15, 0.70, 25)
    n_trials = 10

    print("  Heterogeneous Tc extraction (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(77)

    tasks = []
    for kappa in concentrations:
        for tol in tolerance_range:
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                continue
            for _ in range(n_trials):
                tasks.append((50, 0.9, float(alpha), float(beta_param), 1000,
                              int(rng.integers(0, 2**31)), kappa, float(tol)))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {}
        for t in tasks:
            sz, dens, alpha, beta_p, ms, seed, kappa, tol = t
            fut = pool.submit(_run_het_trial, sz, dens, alpha, beta_p, ms, seed)
            futures[fut] = (kappa, tol)

        for fut in as_completed(futures):
            kappa, tol = futures[fut]
            raw.setdefault((kappa, round(tol, 6)), []).append(fut.result())

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {
        "concentrations": np.array(concentrations),
        "tolerances": tolerance_range,
    }
    for kappa in concentrations:
        mean_seg = np.zeros(len(tolerance_range))
        std_seg = np.zeros(len(tolerance_range))
        for i, tol in enumerate(tolerance_range):
            key = (kappa, round(float(tol), 6))
            if key in raw:
                vals = raw[key]
                mean_seg[i] = np.mean(vals)
                std_seg[i] = np.std(vals) / np.sqrt(len(vals))

        save_dict[f"seg_mean_k{kappa:.1f}"] = mean_seg
        save_dict[f"seg_std_k{kappa:.1f}"] = std_seg

        cp = extract_critical_point(tolerance_range, mean_seg, std_seg)
        save_dict[f"Tc_k{kappa:.1f}"] = cp["T_c"]
        save_dict[f"Tc_err_k{kappa:.1f}"] = cp["T_c_err"]
        print(f"  kappa={kappa:.0f}: Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}")

    np.savez(os.path.join(OUTDIR, "heterogeneous_tc.npz"), **save_dict)


def run_convergence_curves():
    """Convergence curves with parallel trials."""
    tolerances = [0.30, 0.45, 0.60]
    n_trials = 10

    print("  Convergence curves (parallel)...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    tasks = []
    for tol in tolerances:
        for _ in range(n_trials):
            tasks.append((50, 0.9, float(tol), 1000, int(rng.integers(0, 2**31))))

    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_convergence_trial, *t): t for t in tasks}
        for fut in as_completed(futures):
            tol = round(futures[fut][2], 6)
            moved, sat = fut.result()
            raw.setdefault(tol, []).append((moved, sat))

    print(f"  Done in {time.time()-t0:.1f}s")

    save_dict = {}
    for tol in tolerances:
        entries = raw[round(float(tol), 6)]
        all_moved = [e[0] for e in entries]
        all_sat = [e[1] for e in entries]
        max_len = max(len(c) for c in all_moved)
        for lst in [all_moved, all_sat]:
            for i in range(len(lst)):
                lst[i] = lst[i] + [lst[i][-1]] * (max_len - len(lst[i]))
        save_dict[f"moved_T{tol:.2f}"] = np.mean(all_moved, axis=0)
        save_dict[f"sat_T{tol:.2f}"] = np.mean(all_sat, axis=0)

    np.savez(os.path.join(OUTDIR, "convergence_curves.npz"), **save_dict)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"=== Parallel Schelling benchmarks ({N_WORKERS} workers) ===\n")

    print("1. Tolerance sweep (L=50, n=20)...")
    run_tolerance_sweep()

    print("\n2. Finite-size scaling (L=20,30,40,60,80, n=10)...")
    data, ce, sc = run_finite_size_scaling()

    print("\n3. Binder cumulant (L=20,30,40,60,80, n=15)...")
    run_binder_cumulant()

    print("\n4. Susceptibility (L=20,30,40,60,80, n=15)...")
    run_susceptibility()

    print("\n5. Order parameter exponent beta (L=80, n=20)...")
    run_order_parameter_exponent(ce["T_c_inf"])

    print("\n6. Trajectoires multi-echelles (L=50, n=10)...")
    run_trajectory_sweep()

    print("\n7. Comparaison heterogene (L=50, n=10)...")
    run_heterogeneous_comparison()

    print("\n8. Tc(kappa) heterogene (L=50, n=10, 8 kappa values)...")
    run_heterogeneous_tc()

    print("\n9. Courbes de convergence (L=50, n=10)...")
    run_convergence_curves()

    print("\n=== Termine ===")
