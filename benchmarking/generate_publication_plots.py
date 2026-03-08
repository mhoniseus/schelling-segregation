"""
Generate all publication-quality figures from merged benchmark data.

Reads outputs/data/*.npz and produces figures/*.png.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "serif",
})

DATADIR = "outputs/data"
FIGDIR = "figures"
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c",
           "#e67e22", "#2c3e50"]


# ── 1. Tolerance sweep with sigmoid + frozen regime ──────────────────────────

def plot_tolerance_sweep():
    d = np.load(os.path.join(DATADIR, "tolerance_sweep.npz"))
    tols, means, stds = d["tolerances"], d["means"], d["stds"]
    Tc, Tc_err = float(d["T_c"]), float(d["T_c_err"])
    width = float(d["width"])
    A, B = float(d["A"]), float(d["B"])
    frozen_idx = int(d["frozen_idx"])

    # Sigmoid fit only in monotone region
    t_fit = np.linspace(tols[0], tols[min(frozen_idx, len(tols)-1)], 300)
    s_fit = B + A / 2 * (1 + np.tanh((t_fit - Tc) / width))

    fig, ax = plt.subplots()
    # Data before frozen regime
    ax.errorbar(tols[:frozen_idx], means[:frozen_idx], yerr=stds[:frozen_idx],
                fmt="o", color="#e74c3c", markersize=4, capsize=2,
                label=f"Data ($L=50$, $n=50$)")
    # Frozen regime in different color
    if frozen_idx < len(tols):
        ax.errorbar(tols[frozen_idx:], means[frozen_idx:], yerr=stds[frozen_idx:],
                    fmt="s", color="#95a5a6", markersize=4, capsize=2,
                    label="Frozen regime")
    ax.plot(t_fit, s_fit, "k-", linewidth=1.5, label="Sigmoid fit")
    ax.axvline(Tc, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvspan(Tc - Tc_err, Tc + Tc_err, alpha=0.15, color="#f39c12",
               label=f"$T_c = {Tc:.3f} \\pm {Tc_err:.3f}$")
    ax.set_xlabel("Tolerance threshold $T$")
    ax.set_ylabel("Segregation index $S$")
    ax.set_title("Segregation transition with sigmoid fit")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(os.path.join(FIGDIR, "tolerance_sweep.png"))
    plt.close(fig)
    print("  tolerance_sweep.png")


# ── 2. Tolerance sweep + interface density (dual axis) ───────────────────────

def plot_tolerance_with_interface():
    d = np.load(os.path.join(DATADIR, "tolerance_sweep.npz"))
    tols = d["tolerances"]
    means, stds = d["means"], d["stds"]
    iface = d["interface_density"]
    iface_std = d["interface_density_std"]
    Tc = float(d["T_c"])
    frozen_idx = int(d["frozen_idx"])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.errorbar(tols[:frozen_idx], means[:frozen_idx], yerr=stds[:frozen_idx],
                 fmt="o-", color="#e74c3c", markersize=3, capsize=1.5,
                 linewidth=1.5, label="$S(T)$")
    if frozen_idx < len(tols):
        ax1.errorbar(tols[frozen_idx:], means[frozen_idx:], yerr=stds[frozen_idx:],
                     fmt="s", color="#e74c3c", markersize=3, capsize=1.5, alpha=0.4)

    ax2.errorbar(tols[:frozen_idx], iface[:frozen_idx], yerr=iface_std[:frozen_idx],
                 fmt="^-", color="#3498db", markersize=3, capsize=1.5,
                 linewidth=1.5, label="$\\rho_I(T)$")
    if frozen_idx < len(tols):
        ax2.errorbar(tols[frozen_idx:], iface[frozen_idx:], yerr=iface_std[frozen_idx:],
                     fmt="^", color="#3498db", markersize=3, capsize=1.5, alpha=0.4)

    ax1.axvline(Tc, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"$T_c = {Tc:.3f}$")
    ax1.set_xlabel("Tolerance threshold $T$")
    ax1.set_ylabel("Segregation index $S$", color="#e74c3c")
    ax2.set_ylabel("Interface density $\\rho_I$", color="#3498db")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax1.set_title("Segregation index and interface density ($L=50$)")
    fig.savefig(os.path.join(FIGDIR, "tolerance_interface.png"))
    plt.close(fig)
    print("  tolerance_interface.png")


# ── 3. S(T) for different sizes with discrete thresholds ─────────────────────

def plot_fss_curves():
    d = np.load(os.path.join(DATADIR, "finite_size_scaling.npz"), allow_pickle=True)
    sizes = d["sizes"]
    tols = d["tolerances"]
    discrete = bool(d.get("discrete_transition", False))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left panel: S(T) curves
    ax = axes[0]
    for i, sz in enumerate(sizes):
        sz = int(sz)
        mean = d[f"seg_mean_L{sz}"]
        std = d[f"seg_std_L{sz}"]
        ax.errorbar(tols, mean, yerr=std, fmt="o-", color=COLORS[i],
                    markersize=3, capsize=1.5, linewidth=1.2,
                    label=f"$L = {sz}$")

    # Annotate discrete thresholds (k/8 for Moore neighborhood)
    if discrete:
        for k in range(1, 8):
            thresh = k / 8.0
            if tols[0] < thresh < tols[-1]:
                ax.axvline(thresh, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
                ax.text(thresh, 1.0, f"$\\frac{{{k}}}{{8}}$",
                        ha="center", va="bottom", fontsize=8, color="gray",
                        transform=ax.get_xaxis_transform())

    Tc_inf = float(d["Tc_inf"])
    ax.axvline(Tc_inf, color="#f39c12", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"$T_c = {Tc_inf:.3f}$")

    ax.set_xlabel("Tolerance threshold $T$")
    ax.set_ylabel("Segregation index $S$")
    ax.set_title("$S(T)$ for different system sizes ($\\rho=0.9$, $n = 50$)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Right panel: variance scaling at Tc
    ax2 = axes[1]
    var_at_Tc = d.get("var_at_Tc", None)
    if var_at_Tc is not None and len(var_at_Tc) > 0:
        plot_sizes = sizes[:len(var_at_Tc)]
        ax2.loglog(plot_sizes, var_at_Tc, "s-", color="#e74c3c",
                   markersize=8, linewidth=2)
        alpha = float(d.get("alpha", np.nan))
        alpha_err = float(d.get("alpha_err", np.nan))
        if not np.isnan(alpha) and len(plot_sizes) >= 2:
            L_fit = np.linspace(plot_sizes.min(), plot_sizes.max(), 100)
            coeffs = np.polyfit(np.log(plot_sizes), np.log(var_at_Tc), 1)
            ax2.loglog(L_fit, np.exp(np.polyval(coeffs, np.log(L_fit))),
                       "k--", linewidth=1.5,
                       label=f"$\\mathrm{{Var}}(S) \\sim L^{{-{alpha:.2f} \\pm {alpha_err:.2f}}}$")
            ax2.legend(fontsize=10)
    ax2.set_xlabel("System size $L$")
    ax2.set_ylabel("$\\mathrm{Var}(S)$ at $T_c$")
    ax2.set_title("Fluctuation decay at the transition")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "finite_size_curves.png"))
    plt.close(fig)
    print("  finite_size_curves.png")


# ── 4. Binder cumulant ───────────────────────────────────────────────────────

def plot_binder():
    d = np.load(os.path.join(DATADIR, "binder_cumulant.npz"))
    sizes = d["sizes"]
    tols = d["tolerances"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: full range
    ax = axes[0]
    for i, sz in enumerate(sizes):
        sz = int(sz)
        u4 = d[f"U4_L{sz}"]
        u4_err = d[f"U4_err_L{sz}"]
        ax.errorbar(tols, u4, yerr=u4_err, fmt="o-", color=COLORS[i],
                    markersize=3, capsize=1.5, linewidth=1.2,
                    label=f"$L = {sz}$")
    ax.axhline(2/3, color="gray", linestyle=":", alpha=0.5, label="$U_4 = 2/3$")
    ax.set_xlabel("Tolerance threshold $T$")
    ax.set_ylabel("Binder cumulant $U_4$")
    ax.set_title("Binder cumulant $U_4(T)$")
    ax.legend(fontsize=8)

    # Right: zoom near crossing region
    ax2 = axes[1]
    # Find crossing in the transition region (exclude trivial convergence at high T)
    # Look for minimal spread where U4 is NOT near 2/3 for all sizes
    u4_all = np.array([d[f"U4_L{int(sz)}"] for sz in sizes])
    u4_mean = np.mean(u4_all, axis=0)
    spread = np.std(u4_all, axis=0)
    # Mask: only consider points where mean U4 is away from 2/3
    # (i.e., in the transition region, not the saturated regime)
    transition_mask = np.abs(u4_mean - 2/3) > 0.005
    if np.any(transition_mask):
        masked_spread = np.where(transition_mask, spread, np.inf)
        cross_idx = np.argmin(masked_spread)
    else:
        cross_idx = np.argmin(spread)
    cross_T = float(tols[cross_idx])

    zoom_mask = (tols > cross_T - 0.08) & (tols < cross_T + 0.08)
    for i, sz in enumerate(sizes):
        sz = int(sz)
        u4 = d[f"U4_L{sz}"]
        u4_err = d[f"U4_err_L{sz}"]
        ax2.errorbar(tols[zoom_mask], u4[zoom_mask], yerr=u4_err[zoom_mask],
                     fmt="o-", color=COLORS[i], markersize=5, capsize=2,
                     linewidth=1.5, label=f"$L = {sz}$")
    ax2.axvline(cross_T, color="#f39c12", linestyle="--", linewidth=1.5,
                label=f"Crossing $\\approx {cross_T:.3f}$")
    ax2.axhline(2/3, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Tolerance threshold $T$")
    ax2.set_ylabel("$U_4$")
    ax2.set_title("Binder cumulant crossing (zoom)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "binder_cumulant.png"))
    plt.close(fig)
    print("  binder_cumulant.png")


# ── 5. Susceptibility with peak scaling ──────────────────────────────────────

def plot_susceptibility():
    d = np.load(os.path.join(DATADIR, "susceptibility.npz"))
    sizes = d["sizes"]
    tols = d["tolerances"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: chi(T) for all sizes
    ax = axes[0]
    peak_sizes = []
    peak_vals = []
    for i, sz in enumerate(sizes):
        sz = int(sz)
        chi = d[f"chi_L{sz}"]
        chi_err = d[f"chi_err_L{sz}"]
        ax.errorbar(tols, chi, yerr=chi_err, fmt="o-", color=COLORS[i],
                    markersize=3, capsize=1.5, linewidth=1.2,
                    label=f"$L = {sz}$")
        peak_key = f"chi_peak_val_L{sz}"
        if peak_key in d:
            peak_sizes.append(float(sz))
            peak_vals.append(float(d[peak_key]))
    ax.set_xlabel("Tolerance threshold $T$")
    ax.set_ylabel("$\\chi = L^2(\\langle S^2 \\rangle - \\langle S \\rangle^2)$")
    ax.set_title("Susceptibility $\\chi(T)$")
    ax.legend(fontsize=8)

    # Right: peak scaling chi_max vs L
    ax2 = axes[1]
    if len(peak_sizes) >= 2:
        peak_sizes = np.array(peak_sizes)
        peak_vals = np.array(peak_vals)
        ax2.loglog(peak_sizes, peak_vals, "s-", color="#9b59b6",
                   markersize=8, linewidth=2)
        gamma_over_nu = float(d.get("gamma_over_nu", np.nan))
        gamma_over_nu_err = float(d.get("gamma_over_nu_err", np.nan))
        if not np.isnan(gamma_over_nu):
            L_fit = np.linspace(peak_sizes.min(), peak_sizes.max(), 100)
            coeffs = np.polyfit(np.log(peak_sizes), np.log(peak_vals), 1)
            ax2.loglog(L_fit, np.exp(np.polyval(coeffs, np.log(L_fit))),
                       "k--", linewidth=1.5,
                       label=f"$\\chi_{{\\max}} \\sim L^{{{gamma_over_nu:.2f} \\pm {gamma_over_nu_err:.2f}}}$")
            ax2.legend(fontsize=10)
    ax2.set_xlabel("System size $L$")
    ax2.set_ylabel("$\\chi_{\\max}$")
    ax2.set_title("Susceptibility peak scaling")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "susceptibility.png"))
    plt.close(fig)
    print("  susceptibility.png")


# ── 6. Order parameter exponent ──────────────────────────────────────────────

def plot_order_parameter():
    f = os.path.join(DATADIR, "order_parameter_exponent.npz")
    if not os.path.exists(f):
        print("  SKIP order_parameter_exponent (no data)")
        return
    d = np.load(f)
    beta_exp = float(d["beta"])
    beta_err = float(d["beta_err"])
    T_fit = d["T_fit"]
    S_fit = d["S_fit"]
    Tc = float(d["T_c"])

    fig, ax = plt.subplots()
    dT = T_fit - Tc
    valid = (dT > 0) & (S_fit > 0.01)
    ax.loglog(dT[valid], S_fit[valid], "o", color="#e74c3c", markersize=5)
    if valid.sum() >= 2:
        x_line = np.linspace(dT[valid].min(), dT[valid].max(), 100)
        coeffs = np.polyfit(np.log(dT[valid]), np.log(S_fit[valid]), 1)
        ax.loglog(x_line, np.exp(np.polyval(coeffs, np.log(x_line))),
                  "k--", linewidth=1.5,
                  label=f"$\\beta = {beta_exp:.3f} \\pm {beta_err:.3f}$")
    ax.set_xlabel("$T - T_c$")
    ax.set_ylabel("$\\langle S \\rangle$")
    ax.set_title(f"Order parameter exponent ($L = 320$)")
    ax.legend(fontsize=11)
    fig.savefig(os.path.join(FIGDIR, "order_parameter_exponent.png"))
    plt.close(fig)
    print("  order_parameter_exponent.png")


# ── 7. Heterogeneous comparison ──────────────────────────────────────────────

def plot_het_comparison():
    f = os.path.join(DATADIR, "heterogeneous_comparison.npz")
    if not os.path.exists(f):
        print("  SKIP heterogeneous_comparison (no data)")
        return
    d = np.load(f)
    tols = d["tolerances"]

    fig, ax = plt.subplots()
    ax.errorbar(tols, d["hom_mean"], yerr=d["hom_std"], fmt="o-",
                color="black", markersize=3, capsize=1.5, linewidth=1.5,
                label="Homogeneous ($\\kappa \\to \\infty$)")

    het_keys = sorted([k for k in d.files if k.startswith("het_mean_k")])
    for i, key in enumerate(het_keys):
        kappa = key.replace("het_mean_k", "")
        err_key = f"het_std_k{kappa}"
        ax.errorbar(tols, d[key], yerr=d[err_key], fmt="o-",
                    color=COLORS[i % len(COLORS)], markersize=3,
                    capsize=1.5, linewidth=1.2,
                    label=f"$\\kappa = {kappa}$")

    ax.set_xlabel("Mean tolerance $\\bar{T}$")
    ax.set_ylabel("Segregation index $S$")
    ax.set_title("Homogeneous vs heterogeneous tolerance ($L=50$, $n=50$)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(os.path.join(FIGDIR, "heterogeneous_comparison.png"))
    plt.close(fig)
    print("  heterogeneous_comparison.png")


# ── 8. Convergence curves ────────────────────────────────────────────────────

def plot_convergence():
    f = os.path.join(DATADIR, "convergence_curves.npz")
    if not os.path.exists(f):
        print("  SKIP convergence_curves (no data)")
        return
    d = np.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for key in sorted(d.files):
        if key.startswith("moved_T"):
            T_str = key.replace("moved_T", "")
            moved = d[key]
            n_agents = moved[0] if moved[0] > 0 else 1
            frac = moved / n_agents
            ax1.plot(range(len(frac)), frac, linewidth=1.5,
                     label=f"$T = {T_str}$")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Fraction of agents moved")
    ax1.set_title("Convergence: agents moved per step")
    ax1.legend()
    ax1.set_yscale("log")

    for key in sorted(d.files):
        if key.startswith("sat_T"):
            T_str = key.replace("sat_T", "")
            sat = d[key]
            ax2.plot(range(len(sat)), sat, linewidth=1.5,
                     label=f"$T = {T_str}$")

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Mean satisfaction")
    ax2.set_title("Convergence: mean satisfaction")
    ax2.legend()

    fig.savefig(os.path.join(FIGDIR, "convergence_curves.png"))
    plt.close(fig)
    print("  convergence_curves.png")


# ── 9. Multiscalar trajectories ──────────────────────────────────────────────

def plot_trajectories():
    f = os.path.join(DATADIR, "trajectory_sweep.npz")
    if not os.path.exists(f):
        print("  SKIP trajectory_sweep (no data)")
        return
    d = np.load(f, allow_pickle=True)
    radii = d["radii"]

    fig, ax = plt.subplots()
    keys = sorted([k for k in d.files if k.startswith("mean_D_T")])
    for i, key in enumerate(keys):
        T_str = key.replace("mean_D_T", "")
        D_mean = d[key]
        std_key = f"std_D_T{T_str}"
        D_std = d[std_key] if std_key in d else None
        ax.plot(radii, D_mean, "o-", color=COLORS[i % len(COLORS)],
                markersize=4, linewidth=1.5, label=f"$T = {T_str}$")
        if D_std is not None:
            ax.fill_between(radii, D_mean - D_std, D_mean + D_std,
                            color=COLORS[i % len(COLORS)], alpha=0.1)

    if "mean_null" in d:
        D_null = d["mean_null"]
        ax.plot(radii, D_null, "k--", linewidth=2, alpha=0.5, label="Null model")

    ax.set_xlabel("Scale $r$")
    ax.set_ylabel("Dissimilarity $D(r)$")
    ax.set_title("Multiscalar dissimilarity trajectories ($L=50$, $n=10$)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIGDIR, "multiscalar_trajectories.png"))
    plt.close(fig)
    print("  multiscalar_trajectories.png")


# ── 10. Trajectory statistics ────────────────────────────────────────────────

def plot_trajectory_statistics():
    f = os.path.join(DATADIR, "trajectory_sweep.npz")
    if not os.path.exists(f):
        print("  SKIP trajectory_statistics (no data)")
        return
    d = np.load(f, allow_pickle=True)
    tols = d["tolerances"]

    areas, slopes, char_lengths = [], [], []
    for t in tols:
        t_str = f"{t:.2f}"
        areas.append(float(d[f"stat_area_T{t_str}"]))
        slopes.append(float(d[f"stat_slope_T{t_str}"]))
        char_lengths.append(float(d[f"stat_characteristic_length_T{t_str}"]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax1.plot(tols, areas, "o-", color="#e74c3c", markersize=6, linewidth=2)
    ax1.set_xlabel("Tolerance $T$")
    ax1.set_ylabel("Area $\\int D(r)\\,dr$")
    ax1.set_title("Total segregation")

    ax2.plot(tols, slopes, "s-", color="#3498db", markersize=6, linewidth=2)
    ax2.set_xlabel("Tolerance $T$")
    ax2.set_ylabel("Log-log slope")
    ax2.set_title("Decay rate")

    ax3.plot(tols, char_lengths, "^-", color="#2ecc71", markersize=6, linewidth=2)
    ax3.set_xlabel("Tolerance $T$")
    ax3.set_ylabel("$r_{1/2}$")
    ax3.set_title("Characteristic length")

    fig.suptitle("Trajectory summary statistics ($L=50$)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "trajectory_statistics.png"))
    plt.close(fig)
    print("  trajectory_statistics.png")


# ── 11. Heterogeneous Tc(kappa) ──────────────────────────────────────────────

def plot_het_tc():
    f = os.path.join(DATADIR, "heterogeneous_tc.npz")
    if not os.path.exists(f):
        print("  SKIP heterogeneous_tc (no data)")
        return
    d = np.load(f)
    concentrations = d["concentrations"]

    kappas, tcs, tc_errs = [], [], []
    for kappa in concentrations:
        kappa = float(kappa)
        tc_key = f"Tc_k{kappa:.1f}"
        tc_err_key = f"Tc_err_k{kappa:.1f}"
        if tc_key in d:
            kappas.append(kappa)
            tcs.append(float(d[tc_key]))
            tc_errs.append(float(d[tc_err_key]))

    if not kappas:
        print("  SKIP heterogeneous_tc (no Tc values)")
        return

    fig, ax = plt.subplots()
    ax.errorbar(kappas, tcs, yerr=tc_errs, fmt="s-", color="#9b59b6",
                markersize=8, capsize=4, linewidth=2)
    ax.set_xlabel("Concentration $\\kappa$")
    ax.set_ylabel("$T_c(\\kappa)$")
    ax.set_title("Critical point vs tolerance dispersion ($L=50$)")
    ax.set_xscale("log")
    fig.savefig(os.path.join(FIGDIR, "heterogeneous_tc.png"))
    plt.close(fig)
    print("  heterogeneous_tc.png")


# ── 12. Phase diagram heatmap (T, rho) ───────────────────────────────────────

def plot_phase_diagram():
    """Generate phase diagram by running quick simulations."""
    from src.schelling import SchellingModel, segregation_index

    print("  Computing phase diagram (this may take a moment)...")
    T_range = np.linspace(0.10, 0.70, 25)
    rho_range = np.linspace(0.50, 0.95, 20)
    S_map = np.zeros((len(rho_range), len(T_range)))

    for i, rho in enumerate(rho_range):
        for j, T in enumerate(T_range):
            vals = []
            for trial in range(5):
                m = SchellingModel(size=40, density=rho, tolerance=T,
                                   seed=1000*i + 100*j + trial)
                m.run(max_steps=1000)
                vals.append(segregation_index(m.grid))
            S_map[i, j] = np.mean(vals)

    fig, ax = plt.subplots()
    im = ax.imshow(S_map, origin="lower", aspect="auto",
                   extent=[T_range[0], T_range[-1], rho_range[0], rho_range[-1]],
                   cmap="RdYlBu_r", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, label="Segregation index $S$")

    # Contour lines
    cs = ax.contour(T_range, rho_range, S_map, levels=[0.3, 0.5, 0.7],
                    colors=["white"], linewidths=1.5, linestyles="--")
    ax.clabel(cs, fmt="$S=%.1f$", fontsize=9)

    ax.set_xlabel("Tolerance threshold $T$")
    ax.set_ylabel("Density $\\rho$")
    ax.set_title("Phase diagram in $(T, \\rho)$ plane ($L=40$, $n=5$)")
    fig.savefig(os.path.join(FIGDIR, "phase_diagram.png"))
    plt.close(fig)
    print("  phase_diagram.png")


# ── 13. Scaling collapse ─────────────────────────────────────────────────────

def plot_scaling_collapse():
    f = os.path.join(DATADIR, "finite_size_scaling.npz")
    if not os.path.exists(f):
        print("  SKIP scaling_collapse (no data)")
        return
    d = np.load(f, allow_pickle=True)
    sizes = d["sizes"]
    best_nu = float(d.get("collapse_best_nu", 1.0))
    Tc_inf = float(d["Tc_inf"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # Left: scaling collapse with best nu
    ax = axes[0]
    for i, sz in enumerate(sizes):
        sz = int(sz)
        key_x = f"collapse_x_L{sz}"
        key_y = f"collapse_y_L{sz}"
        if key_x in d and key_y in d:
            ax.plot(d[key_x], d[key_y], "o-", color=COLORS[i],
                    markersize=3, linewidth=1.2, label=f"$L = {sz}$")
    ax.set_xlabel(f"$(T - T_c) \\cdot L^{{1/\\nu}}$ ($\\nu = {best_nu:.2f}$)")
    ax.set_ylabel("$S$")
    ax.set_title("Scaling collapse")
    ax.legend(fontsize=8)

    # Right: collapse quality vs nu
    nu_range = d.get("collapse_nu_range", None)
    quality = d.get("collapse_quality", None)
    ax2 = axes[1]
    if nu_range is not None and quality is not None:
        ax2.semilogy(nu_range, quality, "b-", linewidth=2)
        ax2.axvline(best_nu, color="#f39c12", linestyle="--", linewidth=1.5,
                    label=f"$\\nu_{{opt}} = {best_nu:.2f}$")
        ax2.set_xlabel("Trial $\\nu$")
        ax2.set_ylabel("Residual variance")
        ax2.set_title("Collapse quality")
        ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "scaling_collapse.png"))
    plt.close(fig)
    print("  scaling_collapse.png")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    print("Generating publication figures...")

    plot_tolerance_sweep()
    plot_tolerance_with_interface()
    plot_fss_curves()
    plot_binder()
    plot_susceptibility()
    plot_order_parameter()
    plot_scaling_collapse()
    plot_het_comparison()
    plot_het_tc()
    plot_convergence()
    plot_trajectories()
    plot_trajectory_statistics()
    plot_phase_diagram()

    print("\nAll figures saved to figures/")
