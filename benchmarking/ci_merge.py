"""
Merge CI worker chunks into final publication-quality .npz files.

Reads outputs/chunks/chunk_*.npz and produces the same .npz files
as bench_schelling.py / bench_parallel.py in outputs/data/.

Run locally after downloading artifacts, or as a final CI job.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.phase_diagram import (
    extract_critical_point, critical_exponents, scaling_collapse,
    susceptibility_exponent, variance_scaling, detect_discrete_transitions,
)

CHUNK_DIR = "outputs/chunks"
OUTDIR = "outputs/data"
SIZES = [20, 40, 80, 160, 320]
TOL_RANGE = np.linspace(0.10, 0.70, 50)
N_TRIALS = 50


def load_chunks():
    """Load all chunk files and organize by type."""
    chunks = {}
    for fname in sorted(os.listdir(CHUNK_DIR)):
        if not fname.endswith(".npz"):
            continue
        job_id = int(fname.replace("chunk_", "").replace(".npz", ""))
        data = dict(np.load(os.path.join(CHUNK_DIR, fname), allow_pickle=True))
        chunks[job_id] = data
    return chunks


def merge_seg_sweeps(chunks):
    """Merge segregation sweep chunks into per-size raw data."""
    # Collect raw S values per size
    raw_by_size = {}
    for job_id, data in chunks.items():
        if str(data.get("job_type", "")) != "seg_sweep":
            continue
        size = int(data["size"])
        tol_start = int(data["tol_start"])
        tol_end = int(data["tol_end"])
        raw_S = data["raw_S"]

        if size not in raw_by_size:
            raw_by_size[size] = np.full((len(TOL_RANGE), N_TRIALS), np.nan)

        raw_by_size[size][tol_start:tol_end, :] = raw_S

    return raw_by_size


def compute_fss(raw_by_size):
    """Compute FSS, Binder, susceptibility, beta from raw S values."""
    print("Computing FSS...")

    # --- Finite-size scaling ---
    T_c_dict = {}
    seg_data = {}
    jump_data = {}
    for sz in SIZES:
        if sz not in raw_by_size:
            print(f"  WARNING: No data for L={sz}")
            continue
        raw = raw_by_size[sz]
        means = np.nanmean(raw, axis=1)
        n_valid = np.sum(~np.isnan(raw), axis=1)
        stds = np.nanstd(raw, axis=1) / np.sqrt(np.maximum(n_valid, 1))
        seg_data[sz] = (means, stds)

        cp = extract_critical_point(TOL_RANGE, means, stds)
        T_c_dict[sz] = (cp["T_c"], cp["T_c_err"])
        jump_data[sz] = cp.get("jumps", [])
        print(f"  L={sz}: Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}")
        if cp.get("jumps"):
            n_jumps = len(cp["jumps"])
            print(f"    {n_jumps} discrete jumps detected"
                  f" (largest: delta_S = {cp['jumps'][0]['delta_S']:.3f})")

    available_sizes = [sz for sz in SIZES if sz in seg_data]
    L_arr = np.array(available_sizes, dtype=float)
    Tc_arr = np.array([T_c_dict[sz][0] for sz in available_sizes])
    Tc_err_arr = np.array([T_c_dict[sz][1] for sz in available_sizes])

    ce = critical_exponents(L_arr, Tc_arr, Tc_err_arr)
    discrete = ce.get("discrete_transition", False)
    print(f"  Tc(inf) = {ce['T_c_inf']:.4f} +/- {ce['T_c_inf_err']:.4f}")
    if discrete:
        print("  Discrete lattice transition detected (Tc size-independent)")
    else:
        print(f"  nu = {ce['nu']:.2f} +/- {ce['nu_err']:.2f}")

    # Variance scaling: how fluctuations decay with system size
    vs = variance_scaling(L_arr, TOL_RANGE, raw_by_size, ce["T_c_inf"])
    if not np.isnan(vs.get("alpha", np.nan)):
        print(f"  Var(S) ~ L^(-{vs['alpha']:.2f} +/- {vs['alpha_err']:.2f})")
    if not np.isnan(vs.get("nu_eff", np.nan)):
        print(f"  nu_eff (steepness) = {vs['nu_eff']:.2f} +/- {vs['nu_eff_err']:.2f}")

    sc = scaling_collapse(available_sizes, TOL_RANGE, seg_data, ce["T_c_inf"])
    print(f"  Best collapse nu = {sc['best_nu']:.2f}")

    # Save FSS
    fss_dict = {
        "sizes": L_arr, "tolerances": TOL_RANGE,
        "Tc_values": Tc_arr, "Tc_errors": Tc_err_arr,
        "Tc_inf": ce["T_c_inf"], "Tc_inf_err": ce["T_c_inf_err"],
        "nu": ce.get("nu", np.nan), "nu_err": ce.get("nu_err", np.nan),
        "discrete_transition": discrete,
        "alpha": vs.get("alpha", np.nan),
        "alpha_err": vs.get("alpha_err", np.nan),
        "nu_eff": vs.get("nu_eff", np.nan),
        "nu_eff_err": vs.get("nu_eff_err", np.nan),
        "var_at_Tc": vs.get("var_at_Tc", np.array([])),
        "collapse_best_nu": sc["best_nu"],
        "collapse_nu_range": sc["nu_range"], "collapse_quality": sc["quality"],
    }
    for sz in available_sizes:
        fss_dict[f"seg_mean_L{sz}"] = seg_data[sz][0]
        fss_dict[f"seg_std_L{sz}"] = seg_data[sz][1]
        fss_dict[f"collapse_x_L{sz}"] = sc["collapsed_x"][sz]
        fss_dict[f"collapse_y_L{sz}"] = sc["collapsed_y"][sz]
        # Save jump info
        if sz in jump_data and jump_data[sz]:
            n_jumps = len(jump_data[sz])
            fss_dict[f"n_jumps_L{sz}"] = n_jumps
            fss_dict[f"jump_T_L{sz}"] = np.array([j["T_jump"] for j in jump_data[sz]])
            fss_dict[f"jump_dS_L{sz}"] = np.array([j["delta_S"] for j in jump_data[sz]])
    np.savez(os.path.join(OUTDIR, "finite_size_scaling.npz"), **fss_dict)

    # --- Binder cumulant ---
    print("\nComputing Binder cumulant...")
    boot_rng = np.random.default_rng(999)
    binder_dict = {"sizes": L_arr, "tolerances": TOL_RANGE}
    for sz in available_sizes:
        raw = raw_by_size[sz]
        u4_vals = np.zeros(len(TOL_RANGE))
        u4_errs = np.zeros(len(TOL_RANGE))
        for i in range(len(TOL_RANGE)):
            samples = raw[i, :]
            samples = samples[~np.isnan(samples)]
            if len(samples) < 2:
                u4_vals[i] = 2.0 / 3.0
                u4_errs[i] = 0.0
                continue
            s2, s4 = np.mean(samples**2), np.mean(samples**4)
            u4_vals[i] = 1.0 - s4 / (3.0 * s2**2) if s2 > 0 else 2.0 / 3.0
            # Bootstrap
            u4_boot = np.zeros(500)
            for b in range(500):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                s2b, s4b = np.mean(sb**2), np.mean(sb**4)
                u4_boot[b] = 1.0 - s4b / (3.0 * s2b**2) if s2b > 0 else 2.0 / 3.0
            u4_errs[i] = np.std(u4_boot)
        binder_dict[f"U4_L{sz}"] = u4_vals
        binder_dict[f"U4_err_L{sz}"] = u4_errs
    np.savez(os.path.join(OUTDIR, "binder_cumulant.npz"), **binder_dict)

    # --- Susceptibility ---
    print("Computing susceptibility...")
    chi_dict = {"sizes": L_arr, "tolerances": TOL_RANGE}
    for sz in available_sizes:
        raw = raw_by_size[sz]
        chi_vals = np.zeros(len(TOL_RANGE))
        chi_errs = np.zeros(len(TOL_RANGE))
        for i in range(len(TOL_RANGE)):
            samples = raw[i, :]
            samples = samples[~np.isnan(samples)]
            if len(samples) < 2:
                continue
            chi_vals[i] = sz**2 * (np.mean(samples**2) - np.mean(samples)**2)
            # Bootstrap
            chi_boot = np.zeros(500)
            for b in range(500):
                idx = boot_rng.integers(0, len(samples), size=len(samples))
                sb = samples[idx]
                chi_boot[b] = sz**2 * (np.mean(sb**2) - np.mean(sb)**2)
            chi_errs[i] = np.std(chi_boot)
        chi_dict[f"chi_L{sz}"] = chi_vals
        chi_dict[f"chi_err_L{sz}"] = chi_errs
        peak_idx = np.argmax(chi_vals)
        chi_dict[f"chi_peak_T_L{sz}"] = float(TOL_RANGE[peak_idx])
        chi_dict[f"chi_peak_val_L{sz}"] = float(chi_vals[peak_idx])

    chi_peaks = np.array([float(chi_dict[f"chi_peak_val_L{sz}"]) for sz in available_sizes])
    ge = susceptibility_exponent(L_arr, chi_peaks)
    chi_dict["gamma_over_nu"] = ge["gamma_over_nu"]
    chi_dict["gamma_over_nu_err"] = ge["gamma_over_nu_err"]
    print(f"  gamma/nu = {ge['gamma_over_nu']:.3f} +/- {ge['gamma_over_nu_err']:.3f}")
    np.savez(os.path.join(OUTDIR, "susceptibility.npz"), **chi_dict)

    # --- Order parameter exponent beta ---
    print("Computing beta exponent...")
    T_c_inf = ce["T_c_inf"]
    # Use largest available size
    largest = max(available_sizes)
    raw = raw_by_size[largest]
    # Find tol points where T > Tc + 0.01
    beta_mask = TOL_RANGE > T_c_inf + 0.01
    beta_tols = TOL_RANGE[beta_mask]
    if len(beta_tols) >= 3:
        mean_S = np.array([np.nanmean(raw[i, :]) for i in range(len(TOL_RANGE)) if beta_mask[i]])
        dT = beta_tols - T_c_inf
        valid = (mean_S > 0.01) & (dT > 0)
        beta, beta_err = np.nan, np.nan
        if valid.sum() >= 3:
            try:
                coeffs, cov = np.polyfit(np.log(dT[valid]), np.log(mean_S[valid]), deg=1, cov=True)
                beta = float(coeffs[0])
                beta_err = float(np.sqrt(cov[0, 0]))
            except Exception:
                pass
        print(f"  beta = {beta:.3f} +/- {beta_err:.3f} (from L={largest})")
        np.savez(os.path.join(OUTDIR, "order_parameter_exponent.npz"),
                 beta=beta, beta_err=beta_err, T_fit=beta_tols, S_fit=mean_S, T_c=T_c_inf)
    else:
        print("  WARNING: Not enough points for beta estimation")

    return ce


def merge_het_comparison(chunks):
    """Merge heterogeneous comparison chunk."""
    print("\nMerging heterogeneous comparison...")
    for job_id, data in chunks.items():
        if str(data.get("job_type", "")) != "het_comparison":
            continue

        tols = data["tolerances"]
        hom_raw = data["hom_raw"]
        n_trials = hom_raw.shape[1]

        save_dict = {"tolerances": tols}
        save_dict["hom_mean"] = np.mean(hom_raw, axis=1)
        save_dict["hom_std"] = np.std(hom_raw, axis=1) / np.sqrt(n_trials)

        for key in data:
            if key.startswith("het_raw_k"):
                kappa_str = key.replace("het_raw_k", "")
                het_data = data[key]
                save_dict[f"het_mean_k{kappa_str}"] = np.mean(het_data, axis=1)
                save_dict[f"het_std_k{kappa_str}"] = np.std(het_data, axis=1) / np.sqrt(n_trials)

        np.savez(os.path.join(OUTDIR, "heterogeneous_comparison.npz"), **save_dict)
        print("  Done")
        return


def merge_het_tc(chunks):
    """Merge heterogeneous Tc chunk."""
    print("Merging heterogeneous Tc...")
    for job_id, data in chunks.items():
        if str(data.get("job_type", "")) != "het_tc":
            continue

        concentrations = data["concentrations"]
        tols = data["tolerances"]

        save_dict = {"concentrations": concentrations, "tolerances": tols}

        for kappa in concentrations:
            kappa = float(kappa)
            key = f"raw_k{kappa:.1f}"
            if key not in data:
                continue
            raw = data[key]
            n_trials = raw.shape[1]
            mean_seg = np.mean(raw, axis=1)
            std_seg = np.std(raw, axis=1) / np.sqrt(n_trials)

            save_dict[f"seg_mean_k{kappa:.1f}"] = mean_seg
            save_dict[f"seg_std_k{kappa:.1f}"] = std_seg

            cp = extract_critical_point(tols, mean_seg, std_seg)
            save_dict[f"Tc_k{kappa:.1f}"] = cp["T_c"]
            save_dict[f"Tc_err_k{kappa:.1f}"] = cp["T_c_err"]
            print(f"  kappa={kappa:.0f}: Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}")

        np.savez(os.path.join(OUTDIR, "heterogeneous_tc.npz"), **save_dict)
        return


def merge_trajectories(chunks):
    """Merge trajectory sweep chunk."""
    print("Merging trajectories...")
    for job_id, data in chunks.items():
        if str(data.get("job_type", "")) != "trajectories":
            continue
        # Already in final format, just copy
        save_dict = {}
        for key in data:
            if key != "job_type":
                save_dict[key] = data[key]
        np.savez(os.path.join(OUTDIR, "trajectory_sweep.npz"), **save_dict)
        print("  Done")
        return


def merge_convergence_sweep(chunks):
    """Merge convergence + tolerance sweep chunk."""
    print("Merging convergence curves + tolerance sweep...")
    for job_id, data in chunks.items():
        if str(data.get("job_type", "")) != "convergence_sweep":
            continue

        # Convergence curves
        conv_dict = {}
        for key in data:
            if key.startswith("moved_T") or key.startswith("sat_T"):
                conv_dict[key] = data[key]
        np.savez(os.path.join(OUTDIR, "convergence_curves.npz"), **conv_dict)

        # Tolerance sweep
        if "sweep_tolerances" in data:
            tols = data["sweep_tolerances"]
            seg_raw = data["sweep_seg_raw"]
            iface_raw = data["sweep_iface_raw"]
            n_trials = seg_raw.shape[1]

            means = np.mean(seg_raw, axis=1)
            stds = np.std(seg_raw, axis=1) / np.sqrt(n_trials)
            iface_means = np.mean(iface_raw, axis=1)
            iface_stds = np.std(iface_raw, axis=1) / np.sqrt(n_trials)

            cp = extract_critical_point(tols, means, stds)
            print(f"  Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}")

            np.savez(
                os.path.join(OUTDIR, "tolerance_sweep.npz"),
                tolerances=tols, means=means, stds=stds,
                interface_density=iface_means, interface_density_std=iface_stds,
                T_c=cp["T_c"], T_c_err=cp["T_c_err"],
                width=cp["width"], width_err=cp["width_err"],
                A=cp["A"], B=cp["B"],
                frozen_idx=cp.get("frozen_idx", len(tols)),
            )

        print("  Done")
        return


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    print("=== Merging CI chunks ===\n")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks: {sorted(chunks.keys())}\n")

    # Merge segregation sweeps → FSS + Binder + susceptibility + beta
    raw_by_size = merge_seg_sweeps(chunks)
    ce = compute_fss(raw_by_size)

    # Merge other experiments
    merge_het_comparison(chunks)
    merge_het_tc(chunks)
    merge_trajectories(chunks)
    merge_convergence_sweep(chunks)

    print("\n=== All merges complete ===")
    print(f"Output files in {OUTDIR}/:")
    for f in sorted(os.listdir(OUTDIR)):
        if f.endswith(".npz"):
            size = os.path.getsize(os.path.join(OUTDIR, f))
            print(f"  {f} ({size/1024:.1f} KB)")
