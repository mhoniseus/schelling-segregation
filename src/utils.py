"""
Utilitaires de visualisation et de génération de données.

Auteur : Mouhssine Rifaki
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Tuple


def _validate_unit_interval(value: float, name: str) -> None:
    """Raise ValueError if value is not in [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


# Palette de couleurs : vide = blanc, type A = bleu, type B = orange
GRID_CMAP = ListedColormap(["#ffffff", "#3498db", "#e67e22"])


def plot_grid(
    grid: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> plt.Figure:
    """Affiche une grille de Schelling avec les types d'agents codés par couleur.

    Paramètres
    ----------
    grid : np.ndarray
        Tableau 2D avec les valeurs 0 (vide), 1 (type A), 2 (type B).
    title : str
        Titre du graphique.
    ax : matplotlib Axes ou None
        Axes existants sur lesquels tracer. Si None, une nouvelle figure est créée.
    figsize : tuple
        Taille de la figure (utilisé uniquement quand ax est None).

    Retourne
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(grid, cmap=GRID_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_grid_sequence(
    grids: list,
    titles: Optional[list] = None,
    figsize_per: Tuple[float, float] = (4, 4),
) -> plt.Figure:
    """Affiche une séquence de grilles côte à côte."""
    n = len(grids)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]

    for i, (g, ax) in enumerate(zip(grids, axes)):
        ax.imshow(g, cmap=GRID_CMAP, vmin=0, vmax=2, interpolation="nearest")
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_trajectory(
    radii: np.ndarray,
    dissimilarities: np.ndarray,
    label: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 5),
    loglog: bool = False,
) -> plt.Figure:
    """Affiche une trajectoire de ségrégation multi-échelles.

    Paramètres
    ----------
    radii : np.ndarray
        Rayons d'observation.
    dissimilarities : np.ndarray
        Valeurs de dissimilarité à chaque rayon.
    label : str
        Étiquette de la courbe.
    ax : matplotlib Axes ou None
        Axes existants. Si None, crée une nouvelle figure.
    figsize : tuple
        Taille de la figure.
    loglog : bool
        Si True, utilise des axes log-log.

    Retourne
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if loglog:
        ax.loglog(radii, dissimilarities, "o-", label=label, linewidth=1.5, markersize=4)
    else:
        ax.plot(radii, dissimilarities, "o-", label=label, linewidth=1.5, markersize=4)

    ax.set_xlabel("Rayon d'observation $r$")
    ax.set_ylabel("Dissimilarité $D(r)$")
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend()

    return fig


def plot_phase_diagram(
    tolerances: np.ndarray,
    densities: np.ndarray,
    segregation_map: np.ndarray,
    boundary: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Affiche un diagramme de phase tolérance-densité sous forme de carte de chaleur.

    Paramètres
    ----------
    tolerances : np.ndarray
        Valeurs de tolérance (axe y).
    densities : np.ndarray
        Valeurs de densité (axe x).
    segregation_map : np.ndarray
        Tableau 2D de forme (len(tolerances), len(densities)).
    boundary : liste de tuples (tol, den) ou None
        Frontière de phase à superposer.
    ax : matplotlib Axes ou None
    figsize : tuple

    Retourne
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        densities, tolerances, segregation_map,
        cmap="RdYlBu_r", vmin=0, vmax=1, shading="auto",
    )
    fig.colorbar(im, ax=ax, label="Indice de ségrégation")

    if boundary is not None and len(boundary) > 0:
        bx = [b[1] for b in boundary]
        by = [b[0] for b in boundary]
        ax.plot(bx, by, "k--", linewidth=2, label="Frontière de phase")
        ax.legend()

    ax.set_xlabel("Densité")
    ax.set_ylabel("Tolérance")
    ax.set_title("Diagramme de phase")
    return fig


def plot_convergence(
    moved_history: list,
    satisfaction_history: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Affiche les diagnostics de convergence : agents déplacés et satisfaction au cours du temps.

    Paramètres
    ----------
    moved_history : liste d'entiers
        Nombre d'agents déplacés à chaque étape.
    satisfaction_history : liste de flottants ou None
        Satisfaction moyenne à chaque étape.
    ax : matplotlib Axes ou None
    figsize : tuple

    Retourne
    -------
    matplotlib.figure.Figure
    """
    if satisfaction_history is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        else:
            fig = ax.figure
            ax1 = ax
        ax2 = None

    steps = range(1, len(moved_history) + 1)
    ax1.plot(steps, moved_history, "b-", linewidth=1)
    ax1.set_xlabel("Étape")
    ax1.set_ylabel("Agents déplacés")
    ax1.set_title("Convergence")
    ax1.grid(True, alpha=0.3)

    if ax2 is not None and satisfaction_history is not None:
        ax2.plot(steps, satisfaction_history, "g-", linewidth=1)
        ax2.set_xlabel("Étape")
        ax2.set_ylabel("Satisfaction moyenne")
        ax2.set_title("Satisfaction au cours du temps")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_checkerboard(size: int = 10, empty_frac: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    """Génère une grille en damier parfaitement alternée.

    Cela représente une configuration maximalement mélangée (ségrégation nulle).

    Paramètres
    ----------
    size : int
        Longueur du côté de la grille.
    empty_frac : float
        Fraction de cellules à laisser vides (choisies au hasard).
    seed : int ou None
        Graine aléatoire.

    Retourne
    -------
    np.ndarray de forme (size, size)
    """
    _validate_unit_interval(empty_frac, "empty_frac")

    grid = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            grid[r, c] = 1 + ((r + c) % 2)

    if empty_frac > 0:
        rng = np.random.default_rng(seed)
        n_empty = int(size * size * empty_frac)
        flat = grid.ravel()
        indices = rng.choice(size * size, size=n_empty, replace=False)
        flat[indices] = 0
        grid = flat.reshape(size, size)

    return grid


def generate_random_grid(
    size: int = 30,
    density: float = 0.9,
    fraction_a: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Génère une grille aléatoire avec la densité et le ratio de types spécifiés.

    Paramètres
    ----------
    size : int
        Longueur du côté de la grille.
    density : float
        Fraction de cellules occupées.
    fraction_a : float
        Fraction des cellules occupées qui sont de type A.
    seed : int ou None
        Graine aléatoire.

    Retourne
    -------
    np.ndarray de forme (size, size)
    """
    _validate_unit_interval(density, "density")
    _validate_unit_interval(fraction_a, "fraction_a")

    rng = np.random.default_rng(seed)
    n_cells = size * size
    n_occ = int(n_cells * density)
    n_a = int(n_occ * fraction_a)
    n_b = n_occ - n_a

    cells = np.array([1] * n_a + [2] * n_b + [0] * (n_cells - n_occ))
    rng.shuffle(cells)
    return cells.reshape(size, size)


def generate_clustered_grid(
    size: int = 30,
    density: float = 0.9,
    n_clusters: int = 4,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Génère une grille avec des agents regroupés (partiellement ségrégués).

    Les agents sont placés en grappes spatiales du même type en utilisant une
    affectation de type Voronoï : des centres aléatoires sont choisis pour chaque type,
    et les cellules sont affectées au centre le plus proche du type approprié.

    Paramètres
    ----------
    size : int
        Longueur du côté de la grille.
    density : float
        Fraction de cellules occupées.
    n_clusters : int
        Nombre de centres de grappes par type.
    seed : int ou None
        Graine aléatoire.

    Retourne
    -------
    np.ndarray de forme (size, size)
    """
    _validate_unit_interval(density, "density")

    rng = np.random.default_rng(seed)

    # Générer les centres des grappes
    centres_a = rng.uniform(0, size, size=(n_clusters, 2))
    centres_b = rng.uniform(0, size, size=(n_clusters, 2))

    grid = np.zeros((size, size), dtype=int)
    coords = np.array([(r, c) for r in range(size) for c in range(size)], dtype=float)

    # Affecter chaque cellule au centre le plus proche
    dist_a = np.min([np.linalg.norm(coords - ca, axis=1) for ca in centres_a], axis=0)
    dist_b = np.min([np.linalg.norm(coords - cb, axis=1) for cb in centres_b], axis=0)

    types = np.where(dist_a <= dist_b, 1, 2)
    grid = types.reshape(size, size)

    # Retirer des cellules pour correspondre à la densité
    n_empty = int(size * size * (1 - density))
    if n_empty > 0:
        indices = rng.choice(size * size, size=n_empty, replace=False)
        flat = grid.ravel()
        flat[indices] = 0
        grid = flat.reshape(size, size)

    return grid
