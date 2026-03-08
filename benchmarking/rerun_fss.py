"""Rerun FSS, Binder, susceptibility, beta with more trials for better statistics."""
import time, sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.schelling import SchellingModel, segregation_index
from src.phase_diagram import extract_critical_point, critical_exponents, scaling_collapse, susceptibility_exponent

N_WORKERS = max(1, multiprocessing.cpu_count() - 2)
OUTDIR = "outputs/data"


def _run_trial(size, density, tolerance, max_steps, seed):
    m = SchellingModel(size=size, density=density, tolerance=tolerance, seed=seed)
    m.run(max_steps=max_steps)
    return segregation_index(m.grid)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    sizes = [20, 30, 40, 60, 80]
    tol_range = np.linspace(0.20, 0.65, 30)
    n_trials = 25
    rng = np.random.default_rng(42)

    # Build tasks
    tasks = []
    for sz in sizes:
        for tol in tol_range:
            for _ in range(n_trials):
                tasks.append((sz, 0.9, float(tol), 2000, int(rng.integers(0, 2**31))))

    print(f"Running {len(tasks)} simulations across {N_WORKERS} cores...")
    t0 = time.time()
    raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_trial, *t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            sz, _, tol, _, _ = futures[fut]
            key = (sz, round(tol, 6))
            raw.setdefault(key, []).append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} done...")

    print(f"All sims done in {time.time()-t0:.1f}s")

    # FSS
    data = {}
    for sz in sizes:
        means = np.array([np.mean(raw[(sz, round(float(t), 6))]) for t in tol_range])
        stds = np.array([np.std(raw[(sz, round(float(t), 6))]) / np.sqrt(n_trials) for t in tol_range])
        data[sz] = (means, stds)

    T_c_dict = {}
    for sz in sizes:
        cp = extract_critical_point(tol_range, data[sz][0], data[sz][1])
        T_c_dict[sz] = (cp["T_c"], cp["T_c_err"])
        print(f"L={sz}: Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}")

    L_arr = np.array(sizes, dtype=float)
    Tc_arr = np.array([T_c_dict[sz][0] for sz in sizes])
    Tc_err_arr = np.array([T_c_dict[sz][1] for sz in sizes])
    ce = critical_exponents(L_arr, Tc_arr, Tc_err_arr)
    print(f"Tc(inf) = {ce['T_c_inf']:.4f} +/- {ce['T_c_inf_err']:.4f}")
    print(f"nu = {ce['nu']:.2f} +/- {ce['nu_err']:.2f}")

    sc = scaling_collapse(sizes, tol_range, data, ce["T_c_inf"])
    print(f"Best collapse nu = {sc['best_nu']:.2f}")

    save_dict = {
        "sizes": L_arr, "tolerances": tol_range,
        "Tc_values": Tc_arr, "Tc_errors": Tc_err_arr,
        "Tc_inf": ce["T_c_inf"], "Tc_inf_err": ce["T_c_inf_err"],
        "nu": ce["nu"], "nu_err": ce["nu_err"],
        "collapse_best_nu": sc["best_nu"],
        "collapse_nu_range": sc["nu_range"], "collapse_quality": sc["quality"],
    }
    for sz in sizes:
        save_dict[f"seg_mean_L{sz}"] = data[sz][0]
        save_dict[f"seg_std_L{sz}"] = data[sz][1]
        save_dict[f"collapse_x_L{sz}"] = sc["collapsed_x"][sz]
        save_dict[f"collapse_y_L{sz}"] = sc["collapsed_y"][sz]
    np.savez(os.path.join(OUTDIR, "finite_size_scaling.npz"), **save_dict)

    # Binder
    boot_rng = np.random.default_rng(999)
    binder_dict = {"sizes": L_arr, "tolerances": tol_range}
    for sz in sizes:
        u4_vals = np.zeros(len(tol_range))
        u4_errs = np.zeros(len(tol_range))
        for i, tol in enumerate(tol_range):
            samples = np.array(raw[(sz, round(float(tol), 6))])
            s2, s4 = np.mean(samples**2), np.mean(samples**4)
            u4_vals[i] = 1.0 - s4 / (3.0 * s2**2) if s2 > 0 else 2.0 / 3.0
            u4_boot = np.zeros(200)
            for b in range(200):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                s2b, s4b = np.mean(sb**2), np.mean(sb**4)
                u4_boot[b] = 1.0 - s4b / (3.0 * s2b**2) if s2b > 0 else 2.0 / 3.0
            u4_errs[i] = np.std(u4_boot)
        binder_dict[f"U4_L{sz}"] = u4_vals
        binder_dict[f"U4_err_L{sz}"] = u4_errs
    np.savez(os.path.join(OUTDIR, "binder_cumulant.npz"), **binder_dict)

    # Susceptibility
    chi_dict = {"sizes": L_arr, "tolerances": tol_range}
    for sz in sizes:
        chi_vals = np.zeros(len(tol_range))
        chi_errs = np.zeros(len(tol_range))
        for i, tol in enumerate(tol_range):
            samples = np.array(raw[(sz, round(float(tol), 6))])
            chi_vals[i] = sz**2 * (np.mean(samples**2) - np.mean(samples)**2)
            chi_boot = np.zeros(200)
            for b in range(200):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                chi_boot[b] = sz**2 * (np.mean(sb**2) - np.mean(sb)**2)
            chi_errs[i] = np.std(chi_boot)
        chi_dict[f"chi_L{sz}"] = chi_vals
        chi_dict[f"chi_err_L{sz}"] = chi_errs
        peak_idx = np.argmax(chi_vals)
        chi_dict[f"chi_peak_T_L{sz}"] = float(tol_range[peak_idx])
        chi_dict[f"chi_peak_val_L{sz}"] = float(chi_vals[peak_idx])

    chi_peaks = np.array([float(chi_dict[f"chi_peak_val_L{sz}"]) for sz in sizes])
    ge = susceptibility_exponent(L_arr, chi_peaks)
    chi_dict["gamma_over_nu"] = ge["gamma_over_nu"]
    chi_dict["gamma_over_nu_err"] = ge["gamma_over_nu_err"]
    print(f"gamma/nu = {ge['gamma_over_nu']:.3f} +/- {ge['gamma_over_nu_err']:.3f}")
    np.savez(os.path.join(OUTDIR, "susceptibility.npz"), **chi_dict)

    # Beta exponent
    T_c_inf = ce["T_c_inf"]
    beta_tol = np.linspace(T_c_inf + 0.01, T_c_inf + 0.20, 15)
    rng2 = np.random.default_rng(789)
    beta_tasks = []
    for tol in beta_tol:
        for _ in range(20):
            beta_tasks.append((80, 0.9, float(tol), 2000, int(rng2.integers(0, 2**31))))

    print(f"Running {len(beta_tasks)} beta sims...")
    t0 = time.time()
    beta_raw = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_trial, *t): t for t in beta_tasks}
        for fut in as_completed(futures):
            tol = round(futures[fut][2], 6)
            beta_raw.setdefault(tol, []).append(fut.result())
    print(f"Done in {time.time()-t0:.1f}s")

    mean_S = np.array([np.mean(beta_raw[round(float(t), 6)]) for t in beta_tol])
    dT = beta_tol - T_c_inf
    valid = (mean_S > 0.01) & (dT > 0)
    beta, beta_err = np.nan, np.nan
    if valid.sum() >= 3:
        try:
            coeffs, cov = np.polyfit(np.log(dT[valid]), np.log(mean_S[valid]), deg=1, cov=True)
            beta, beta_err = float(coeffs[0]), float(np.sqrt(cov[0, 0]))
        except Exception:
            pass
    print(f"beta = {beta:.3f} +/- {beta_err:.3f}")
    np.savez(os.path.join(OUTDIR, "order_parameter_exponent.npz"),
             beta=beta, beta_err=beta_err, T_fit=beta_tol, S_fit=mean_S, T_c=T_c_inf)

    print("\n=== Critical analysis complete ===")
