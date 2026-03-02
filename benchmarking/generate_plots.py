"""
Génération rapide de graphiques de diagnostic à partir des données de benchmark.

Lit depuis outputs/data/ et écrit les graphiques PNG dans outputs/plots/.
Pour les figures de qualité publication, utilisez report/generate_figures.py.

Auteur : Mouhssine Rifaki
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

DATADIR = "outputs/data"
PLOTDIR = "outputs/plots"


def plot_tolerance_sweep():
    """Ségrégation en fonction de la tolérance avec fit."""
    d = np.load(os.path.join(DATADIR, "tolerance_sweep.npz"))
    tols = d["tolerances"]
    means = d["means"]
    stds = d["stds"]
    Tc = float(d["T_c"])

    fig, ax = plt.subplots()
    ax.errorbar(tols, means, yerr=stds, fmt="o-", color="#e74c3c",
                linewidth=2, capsize=3)
    ax.axvline(Tc, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"$T_c = {Tc:.3f}$")
    ax.set_xlabel("Seuil de tolérance")
    ax.set_ylabel("Indice de ségrégation")
    ax.set_title("Ségrégation en fonction de la tolérance ($L=50$, $n=20$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, "tolerance_sweep.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé tolerance_sweep.png")


def plot_finite_size():
    """Tc(L) pour l'analyse de taille finie."""
    d = np.load(os.path.join(DATADIR, "finite_size_scaling.npz"))
    sizes = d["sizes"]
    Tc_vals = d["Tc_values"]
    Tc_errs = d["Tc_errors"]
    Tc_inf = float(d["Tc_inf"])

    fig, ax = plt.subplots()
    ax.errorbar(sizes, Tc_vals, yerr=Tc_errs, fmt="s-", color="#3498db",
                linewidth=2, capsize=4, markersize=8)
    ax.axhline(Tc_inf, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"$T_c^\\infty = {Tc_inf:.3f}$")
    ax.set_xlabel("Taille du système ($L$)")
    ax.set_ylabel("$T_c(L)$")
    ax.set_title("Point critique vs taille du système")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTDIR, "finite_size.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé finite_size.png")


if __name__ == "__main__":
    os.makedirs(PLOTDIR, exist_ok=True)
    print("Génération des graphiques de diagnostic...")

    if os.path.exists(os.path.join(DATADIR, "tolerance_sweep.npz")):
        plot_tolerance_sweep()
    if os.path.exists(os.path.join(DATADIR, "finite_size_scaling.npz")):
        plot_finite_size()

    print("Terminé.")
