"""
Evaluation quantitative du modele de Schelling : balayages de tolerance,
analyse de taille finie, cumulant de Binder, trajectoires multi-echelles,
comparaison heterogene, et extraction de Tc(kappa).

Produit des resultats NumPy dans outputs/data/ pour la generation de figures.

Auteur : Mouhssine Rifaki
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.schelling import (
    SchellingModel, HeterogeneousSchellingModel,
    segregation_index, interface_density,
)
from src.phase_diagram import (
    extract_critical_point,
    finite_size_scaling,
    critical_exponents,
    scaling_collapse,
    compare_homogeneous_heterogeneous,
    binder_cumulant,
    heterogeneous_critical_points,
    susceptibility,
    order_parameter_exponent,
    susceptibility_exponent,
)
from src.spatial_analysis import (
    multiscalar_trajectory,
    trajectory_statistics,
    null_model_trajectory,
    systematic_trajectory_sweep,
)

OUTDIR = "outputs/data"


def run_tolerance_sweep(n_trials=20):
    """Balayage fin de la tolerance avec extraction de Tc.

    Parameters actually used: L=50, n_trials=20, max_steps=1000, 40 points.
    max_steps=1000 is sufficient: convergence typically occurs in <500 steps
    for the segregated regime and <200 for the mixed regime.
    """
    tolerances = np.linspace(0.10, 0.75, 30)
    means = np.zeros(len(tolerances))
    stds = np.zeros(len(tolerances))
    iface = np.zeros(len(tolerances))
    iface_stds = np.zeros(len(tolerances))
    rng = np.random.default_rng(0)

    for i, tol in enumerate(tolerances):
        seg_vals = []
        iface_vals = []
        for _ in range(n_trials):
            seed = rng.integers(0, 2**31)
            model = SchellingModel(size=50, density=0.9, tolerance=tol, seed=seed)
            model.run(max_steps=1000)
            seg_vals.append(segregation_index(model.grid))
            iface_vals.append(interface_density(model.grid))
        means[i] = np.mean(seg_vals)
        stds[i] = np.std(seg_vals) / np.sqrt(n_trials)
        iface[i] = np.mean(iface_vals)
        iface_stds[i] = np.std(iface_vals) / np.sqrt(n_trials)

    cp = extract_critical_point(tolerances, means, stds)

    np.savez(
        os.path.join(OUTDIR, "tolerance_sweep.npz"),
        tolerances=tolerances, means=means, stds=stds,
        interface_density=iface, interface_density_std=iface_stds,
        T_c=cp["T_c"], T_c_err=cp["T_c_err"],
        width=cp["width"], width_err=cp["width_err"],
        A=cp["A"], B=cp["B"],
        frozen_idx=cp.get("frozen_idx", len(tolerances)),
    )
    print(f"  Tc = {cp['T_c']:.4f} +/- {cp['T_c_err']:.4f}, width = {cp['width']:.4f}")
    return cp


def run_finite_size_scaling():
    """Balayage S(T) pour L = 20, 30, 40, 60, 80 et extraction de Tc(L).

    Parameters actually used: sizes=[20,30,40,60,80], n_trials=15,
    max_steps=3000, 35 tolerance points focused near Tc.
    """
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.20, 0.70, 25)

    print("  Finite-size scaling...")
    t0 = time.time()
    fss = finite_size_scaling(
        sizes, tolerance_range,
        density=0.9, n_trials=10, max_steps=1000, seed=42,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    # Extract Tc(L) arrays
    L_arr = np.array(sizes, dtype=float)
    Tc_arr = np.array([fss["T_c"][sz][0] for sz in sizes])
    Tc_err_arr = np.array([fss["T_c"][sz][1] for sz in sizes])

    # Fit critical exponents
    ce = critical_exponents(L_arr, Tc_arr, Tc_err_arr)
    print(f"  Tc(inf) = {ce['T_c_inf']:.4f} +/- {ce['T_c_inf_err']:.4f}, nu = {ce['nu']:.2f} +/- {ce['nu_err']:.2f}")

    # Scaling collapse
    sc = scaling_collapse(
        sizes, tolerance_range, fss["segregation"], ce["T_c_inf"],
    )
    print(f"  Best collapse nu = {sc['best_nu']:.2f}")

    # Save all data
    save_dict = {
        "sizes": L_arr,
        "tolerances": tolerance_range,
        "Tc_values": Tc_arr,
        "Tc_errors": Tc_err_arr,
        "Tc_inf": ce["T_c_inf"],
        "Tc_inf_err": ce["T_c_inf_err"],
        "nu": ce["nu"],
        "nu_err": ce["nu_err"],
        "collapse_best_nu": sc["best_nu"],
        "collapse_nu_range": sc["nu_range"],
        "collapse_quality": sc["quality"],
    }
    for sz in sizes:
        mean_seg, std_seg = fss["segregation"][sz]
        save_dict[f"seg_mean_L{sz}"] = mean_seg
        save_dict[f"seg_std_L{sz}"] = std_seg
        save_dict[f"collapse_x_L{sz}"] = sc["collapsed_x"][sz]
        save_dict[f"collapse_y_L{sz}"] = sc["collapsed_y"][sz]

    np.savez(os.path.join(OUTDIR, "finite_size_scaling.npz"), **save_dict)
    return fss, ce, sc


def run_binder_cumulant():
    """Binder cumulant U_4 for multiple sizes to find Tc crossing.

    Parameters: sizes=[20,30,40,60,80], n_trials=20, max_steps=3000,
    35 tolerance points near the transition.
    """
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.25, 0.60, 25)

    print("  Binder cumulant...")
    t0 = time.time()
    bc = binder_cumulant(
        sizes, tolerance_range,
        density=0.9, n_trials=15, max_steps=1000, seed=123,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    save_dict = {
        "sizes": np.array(sizes, dtype=float),
        "tolerances": tolerance_range,
    }
    for sz in sizes:
        save_dict[f"U4_L{sz}"] = bc["U4"][sz]
        save_dict[f"U4_err_L{sz}"] = bc["U4_err"][sz]

    np.savez(os.path.join(OUTDIR, "binder_cumulant.npz"), **save_dict)
    return bc


def run_trajectory_sweep():
    """Trajectoires D(r) systematiques pour differentes tolerances.

    Parameters: L=50, n_trials=10, max_steps=2000.
    """
    tolerances = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70])

    print("  Trajectory sweep...")
    t0 = time.time()
    sweep = systematic_trajectory_sweep(
        tolerances, size=50, density=0.9, n_trials=10, max_steps=1000, seed=0,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    # Null model
    print("  Null model trajectory...")
    radii_null, mean_null, std_null = null_model_trajectory(
        size=50, density=0.9, n_samples=200, max_radius=12, seed=99,
    )

    save_dict = {
        "tolerances": tolerances,
        "radii": sweep["radii"],
        "radii_null": radii_null,
        "mean_null": mean_null,
        "std_null": std_null,
    }
    for tol in tolerances:
        mean_D, std_D = sweep["trajectories"][tol]
        save_dict[f"mean_D_T{tol:.2f}"] = mean_D
        save_dict[f"std_D_T{tol:.2f}"] = std_D
        stats = sweep["statistics"][tol]
        for k, v in stats.items():
            save_dict[f"stat_{k}_T{tol:.2f}"] = v

    np.savez(os.path.join(OUTDIR, "trajectory_sweep.npz"), **save_dict)
    return sweep


def run_heterogeneous_comparison():
    """Comparaison homogene vs heterogene.

    Parameters: L=50, n_trials=10, max_steps=2000, kappa in {2,5,10,20,50}.
    """
    mean_tol_range = np.linspace(0.15, 0.70, 20)
    concentrations = [2.0, 5.0, 20.0]

    print("  Heterogeneous comparison...")
    t0 = time.time()
    comp = compare_homogeneous_heterogeneous(
        mean_tol_range, concentrations=concentrations,
        size=50, density=0.9, n_trials=10, max_steps=1000, seed=7,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    save_dict = {
        "tolerances": mean_tol_range,
        "hom_mean": comp["homogeneous"][0],
        "hom_std": comp["homogeneous"][1],
    }
    for kappa in concentrations:
        save_dict[f"het_mean_k{kappa:.1f}"] = comp["heterogeneous"][kappa][0]
        save_dict[f"het_std_k{kappa:.1f}"] = comp["heterogeneous"][kappa][1]

    np.savez(os.path.join(OUTDIR, "heterogeneous_comparison.npz"), **save_dict)
    return comp


def run_heterogeneous_tc():
    """Extract Tc(kappa) for heterogeneous model.

    Parameters: L=50, n_trials=15, max_steps=2000, kappa in {2,5,10,20,50}.
    """
    concentrations = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    tolerance_range = np.linspace(0.15, 0.70, 25)

    print("  Heterogeneous Tc extraction...")
    t0 = time.time()
    hcp = heterogeneous_critical_points(
        concentrations, tolerance_range,
        size=50, density=0.9, n_trials=10, max_steps=1000, seed=77,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    save_dict = {
        "concentrations": np.array(concentrations),
        "tolerances": tolerance_range,
    }
    for kappa in concentrations:
        tc_val, tc_err = hcp["T_c"][kappa]
        save_dict[f"Tc_k{kappa:.1f}"] = tc_val
        save_dict[f"Tc_err_k{kappa:.1f}"] = tc_err
        mean_seg, std_seg = hcp["segregation"][kappa]
        save_dict[f"seg_mean_k{kappa:.1f}"] = mean_seg
        save_dict[f"seg_std_k{kappa:.1f}"] = std_seg

    np.savez(os.path.join(OUTDIR, "heterogeneous_tc.npz"), **save_dict)

    for kappa in concentrations:
        tc_val, tc_err = hcp["T_c"][kappa]
        print(f"  kappa={kappa:.0f}: Tc = {tc_val:.4f} +/- {tc_err:.4f}")

    return hcp


def run_susceptibility():
    """Susceptibility chi = L^2 * (<S^2> - <S>^2) for multiple sizes.

    Parameters: sizes=[20,30,40,60,80], n_trials=20, max_steps=3000,
    35 tolerance points near the transition.
    """
    sizes = [20, 30, 40, 60, 80]
    tolerance_range = np.linspace(0.25, 0.60, 25)

    print("  Susceptibility...")
    t0 = time.time()
    chi = susceptibility(
        sizes, tolerance_range,
        density=0.9, n_trials=15, max_steps=1000, seed=456,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")

    save_dict = {
        "sizes": np.array(sizes, dtype=float),
        "tolerances": tolerance_range,
    }
    for sz in sizes:
        save_dict[f"chi_L{sz}"] = chi["chi"][sz]
        save_dict[f"chi_err_L{sz}"] = chi["chi_err"][sz]
        save_dict[f"chi_peak_T_L{sz}"] = chi["chi_peak_T"][sz]
        save_dict[f"chi_peak_val_L{sz}"] = chi["chi_peak_val"][sz]

    # Fit gamma/nu from peak scaling
    L_arr = np.array(sizes, dtype=float)
    chi_peaks = np.array([chi["chi_peak_val"][sz] for sz in sizes])
    ge = susceptibility_exponent(L_arr, chi_peaks)
    save_dict["gamma_over_nu"] = ge["gamma_over_nu"]
    save_dict["gamma_over_nu_err"] = ge["gamma_over_nu_err"]
    print(f"  gamma/nu = {ge['gamma_over_nu']:.3f} +/- {ge['gamma_over_nu_err']:.3f}")

    np.savez(os.path.join(OUTDIR, "susceptibility.npz"), **save_dict)
    return chi, ge


def run_order_parameter_exponent(T_c_inf):
    """Estimate the order parameter exponent beta from <S> ~ (T - Tc)^beta.

    Parameters: L=80, n_trials=20, max_steps=3000.
    """
    tolerance_range = np.linspace(T_c_inf + 0.01, T_c_inf + 0.20, 15)

    print("  Order parameter exponent (beta)...")
    t0 = time.time()
    ope = order_parameter_exponent(
        sizes=[80], tolerance_range=tolerance_range,
        T_c=T_c_inf, density=0.9, n_trials=20, max_steps=1000, seed=789,
    )
    elapsed = time.time() - t0
    print(f"  Termine en {elapsed:.1f}s")
    print(f"  beta = {ope['beta']:.3f} +/- {ope['beta_err']:.3f}")

    np.savez(
        os.path.join(OUTDIR, "order_parameter_exponent.npz"),
        beta=ope["beta"], beta_err=ope["beta_err"],
        T_fit=ope["T_fit"], S_fit=ope["S_fit"],
        T_c=T_c_inf,
    )
    return ope


def run_convergence_curves():
    """Courbes de convergence detaillees pour 3 tolerances.

    Parameters: L=50, n_trials=10, max_steps=2000.
    """
    tolerances = [0.30, 0.45, 0.60]
    n_trials = 10
    rng = np.random.default_rng(42)

    save_dict = {}
    for tol in tolerances:
        all_seg = []
        all_sat = []
        for _ in range(n_trials):
            seed = rng.integers(0, 2**31)
            model = SchellingModel(size=50, density=0.9, tolerance=tol, seed=seed)
            result = model.run(max_steps=1000)
            all_seg.append(result["moved_history"])
            all_sat.append(result["satisfaction_history"])

        # Pad to same length
        max_len = max(len(c) for c in all_seg)
        for lst in [all_seg, all_sat]:
            for i in range(len(lst)):
                lst[i] = lst[i] + [lst[i][-1]] * (max_len - len(lst[i]))

        save_dict[f"moved_T{tol:.2f}"] = np.mean(all_seg, axis=0)
        save_dict[f"sat_T{tol:.2f}"] = np.mean(all_sat, axis=0)

    np.savez(os.path.join(OUTDIR, "convergence_curves.npz"), **save_dict)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    print("=== Benchmarking du modele de Schelling ===\n")

    print("1. Balayage de tolerance (L=50, n=20)...")
    run_tolerance_sweep()

    print("\n2. Finite-size scaling (L=20,30,40,60,80, n=15)...")
    fss, ce, sc = run_finite_size_scaling()

    print("\n3. Binder cumulant (L=20,30,40,60,80, n=20)...")
    run_binder_cumulant()

    print("\n4. Susceptibility (L=20,30,40,60,80, n=20)...")
    run_susceptibility()

    print("\n5. Order parameter exponent beta (L=80, n=20)...")
    run_order_parameter_exponent(ce["T_c_inf"])

    print("\n6. Trajectoires multi-echelles (L=50, n=10)...")
    run_trajectory_sweep()

    print("\n7. Comparaison heterogene (L=50, n=10)...")
    run_heterogeneous_comparison()

    print("\n8. Tc(kappa) heterogene (L=50, n=15, 8 kappa values)...")
    run_heterogeneous_tc()

    print("\n9. Courbes de convergence (L=50, n=10)...")
    run_convergence_curves()

    print("\n=== Termine ===")
