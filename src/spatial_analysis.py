"""
Calcul de trajectoires multi-échelles et mesures de dissimilarité spatiale.

Implémente le cadre d'analyse spatiale de Randon-Furling et al. pour
mesurer la ségrégation à travers les échelles : du local (niveau du voisinage)
au global (ville entière), capturant comment l'hétérogénéité spatiale change
avec le rayon d'observation.

Auteur : Mouhssine Rifaki
"""

import numpy as np
from scipy.ndimage import uniform_filter
from typing import Dict, Tuple, Optional


def _fraction_map(grid: np.ndarray, target_type: int) -> np.ndarray:
    """Carte indicatrice binaire : 1 où grid == target_type, 0 ailleurs."""
    return (grid == target_type).astype(float)


def _population_map(grid: np.ndarray, empty_val: int = 0) -> np.ndarray:
    """Carte indicatrice binaire : 1 où une cellule est occupée, 0 où elle est vide."""
    return (grid != empty_val).astype(float)


def local_fraction(
    grid: np.ndarray,
    target_type: int,
    radius: int,
    empty_val: int = 0,
) -> np.ndarray:
    """Calcule la fraction locale de target_type dans une fenêtre carrée de rayon donné.

    Pour chaque cellule, compte le nombre d'agents de target_type et divise par
    le nombre total d'agents occupés dans une fenêtre (2*radius+1) x (2*radius+1).
    Utilise des conditions aux limites périodiques (enroulement).

    Paramètres
    ----------
    grid : np.ndarray
        La grille 2D des types d'agents.
    target_type : int
        Le type d'agent pour lequel calculer la fraction.
    radius : int
        Demi-côté de la fenêtre d'observation carrée (taille de fenêtre = 2*radius + 1).
    empty_val : int
        Valeur représentant les cellules vides.

    Retourne
    -------
    np.ndarray
        Carte de fraction locale (même forme que la grille). Les cellules sans voisins
        occupés dans la fenêtre obtiennent la valeur 0.
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    indicator = _fraction_map(grid, target_type)
    population = _population_map(grid, empty_val)

    window_size = 2 * radius + 1

    # Tuiler la grille pour gérer les limites périodiques
    tiled_ind = np.tile(indicator, (3, 3))
    tiled_pop = np.tile(population, (3, 3))

    count_target = uniform_filter(tiled_ind, size=window_size, mode="constant", cval=0.0)
    count_total = uniform_filter(tiled_pop, size=window_size, mode="constant", cval=0.0)

    h, w = grid.shape
    count_target = count_target[h:2*h, w:2*w] * (window_size ** 2)
    count_total = count_total[h:2*h, w:2*w] * (window_size ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(count_total > 0, count_target / count_total, 0.0)

    return result


def spatial_dissimilarity(
    grid: np.ndarray,
    target_type: int,
    radius: int,
    empty_val: int = 0,
) -> float:
    """Indice de dissimilarité spatiale à une échelle donnée (rayon).

    Basé sur l'indice de dissimilarité de Duncan-Duncan adapté à un
    contexte spatial. Mesure l'écart de la composition locale par rapport
    à la composition globale à une échelle spatiale donnée.

    D(r) = (1 / (2 * T * P * (1 - P))) * sum_i |t_i * p_i(r) - T * P|

    où T est la population totale, P est la fraction globale de target_type,
    t_i est le nombre de population locale à la cellule i, et p_i(r) est la
    fraction locale de target_type à l'échelle r.

    Paramètres
    ----------
    grid : np.ndarray
        La grille 2D.
    target_type : int
        Type d'agent d'intérêt.
    radius : int
        Échelle d'observation.
    empty_val : int
        Valeur des cellules vides.

    Retourne
    -------
    float
        Indice de dissimilarité dans [0, 1]. 0 signifie un mélange uniforme à cette échelle,
        1 signifie une ségrégation complète.
    """
    population = _population_map(grid, empty_val)
    total_pop = population.sum()
    if total_pop == 0:
        return 0.0

    indicator = _fraction_map(grid, target_type)
    global_frac = indicator.sum() / total_pop

    if global_frac == 0.0 or global_frac == 1.0:
        return 0.0

    loc_frac = local_fraction(grid, target_type, radius, empty_val)

    # Calculer uniquement sur les cellules occupées
    occupied_mask = population > 0
    deviations = np.abs(loc_frac[occupied_mask] - global_frac)

    D = deviations.sum() / (2.0 * total_pop * global_frac * (1.0 - global_frac))
    return float(np.clip(D, 0.0, 1.0))


def exposure_index(
    grid: np.ndarray,
    type_from: int,
    type_to: int,
    radius: int,
    empty_val: int = 0,
) -> float:
    """Indice d'exposition (ou d'interaction) spatiale à une échelle donnée.

    Mesure la fraction locale moyenne de type_to expérimentée par les agents
    de type_from, au rayon d'observation r.

    E(from, to, r) = (1 / N_from) * sum_{i in from} p_to(i, r)

    Paramètres
    ----------
    grid : np.ndarray
        La grille 2D.
    type_from : int
        Type d'agent dont on prend la perspective.
    type_to : int
        Type d'agent mesuré dans l'environnement.
    radius : int
        Échelle d'observation.
    empty_val : int
        Valeur des cellules vides.

    Retourne
    -------
    float
        Indice d'exposition dans [0, 1].
    """
    from_mask = grid == type_from
    n_from = from_mask.sum()
    if n_from == 0:
        return 0.0

    loc_frac_to = local_fraction(grid, type_to, radius, empty_val)
    return float(loc_frac_to[from_mask].mean())


def multiscalar_trajectory(
    grid: np.ndarray,
    target_type: int,
    radii: Optional[np.ndarray] = None,
    max_radius: Optional[int] = None,
    empty_val: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule la trajectoire de ségrégation multi-échelles.

    La trajectoire est une courbe (r, D(r)) où r est le rayon d'observation et
    D(r) est la dissimilarité spatiale à cette échelle. Cela capture comment
    la ségrégation varie de l'échelle locale à globale, comme décrit dans
    Randon-Furling et al.

    Une décroissance abrupte indique que la ségrégation est un phénomène local ;
    une trajectoire plate indique que la ségrégation persiste à toutes les échelles.

    Paramètres
    ----------
    grid : np.ndarray
        La grille 2D.
    target_type : int
        Type d'agent d'intérêt.
    radii : np.ndarray ou None
        Liste explicite de rayons à évaluer. Si None, les rayons sont générés
        automatiquement de 1 à max_radius.
    max_radius : int ou None
        Rayon maximal (par défaut size // 4).
    empty_val : int
        Valeur des cellules vides.

    Retourne
    -------
    radii : np.ndarray
        Les rayons d'observation.
    dissimilarities : np.ndarray
        D(r) à chaque rayon.
    """
    size = grid.shape[0]
    if radii is None:
        if max_radius is None:
            max_radius = size // 4
        radii = np.arange(1, max_radius + 1)

    dissimilarities = np.array([
        spatial_dissimilarity(grid, target_type, int(r), empty_val)
        for r in radii
    ])

    return radii, dissimilarities


def trajectory_area(radii: np.ndarray, dissimilarities: np.ndarray) -> float:
    """Aire sous la trajectoire multi-échelles (règle des trapèzes).

    Une aire plus grande indique plus de ségrégation à travers les échelles. Cela sert
    de résumé en un seul nombre de la structure spatiale.
    """
    if len(radii) < 2:
        return 0.0
    return float(np.trapezoid(dissimilarities, radii))


def trajectory_statistics(
    radii: np.ndarray,
    dissimilarities: np.ndarray,
) -> Dict[str, float]:
    """Résumé quantitatif complet d'une trajectoire multi-échelles.

    Calcule l'aire, la pente log-log, la longueur caractéristique (rayon
    où D(r) tombe à 50% de D(1)), et les dissimilarités locale et globale.

    Retourne
    -------
    dict avec area, slope, characteristic_length, D_local, D_global
    """
    area = trajectory_area(radii, dissimilarities)
    slope = trajectory_slope(radii, dissimilarities)

    D_local = float(dissimilarities[0]) if len(dissimilarities) > 0 else 0.0
    D_global = float(dissimilarities[-1]) if len(dissimilarities) > 0 else 0.0

    # Longueur caractéristique : r où D(r) = 0.5 * D(1)
    threshold = 0.5 * D_local
    char_length = float(radii[-1])  # défaut : la ségrégation persiste
    if D_local > 0:
        below = np.where(dissimilarities <= threshold)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # Interpolation linéaire
                r0, r1 = float(radii[idx - 1]), float(radii[idx])
                d0, d1 = dissimilarities[idx - 1], dissimilarities[idx]
                if d0 != d1:
                    char_length = r0 + (threshold - d0) * (r1 - r0) / (d1 - d0)
                else:
                    char_length = r0
            else:
                char_length = float(radii[0])

    return {
        "area": area,
        "slope": slope,
        "characteristic_length": char_length,
        "D_local": D_local,
        "D_global": D_global,
    }


def null_model_trajectory(
    size: int,
    density: float = 0.9,
    n_samples: int = 50,
    target_type: int = 1,
    max_radius: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trajectoire D(r) moyenne sur des grilles aléatoires (modèle nul).

    Sert de référence : la ségrégation attendue par pur hasard à chaque échelle.

    Retourne
    -------
    (radii, mean_D, std_D) : les radii, la moyenne et l'écart-type de D(r)
    """
    from .schelling import TYPE_A, TYPE_B, EMPTY
    rng = np.random.default_rng(seed)

    all_D = []
    radii = None
    for _ in range(n_samples):
        n_cells = size * size
        n_occ = int(n_cells * density)
        n_a = n_occ // 2
        n_b = n_occ - n_a
        cells = np.array([TYPE_A] * n_a + [TYPE_B] * n_b + [EMPTY] * (n_cells - n_occ))
        rng.shuffle(cells)
        grid = cells.reshape(size, size)

        r, d = multiscalar_trajectory(grid, target_type, max_radius=max_radius)
        if radii is None:
            radii = r
        all_D.append(d)

    all_D = np.array(all_D)
    return radii, np.mean(all_D, axis=0), np.std(all_D, axis=0)


def systematic_trajectory_sweep(
    tolerances: np.ndarray,
    size: int = 50,
    density: float = 0.9,
    n_trials: int = 10,
    max_steps: int = 2000,
    target_type: int = 1,
    seed: Optional[int] = None,
) -> Dict:
    """Trajectoires D(r) systématiques en fonction de la tolérance.

    Pour chaque tolérance, exécute n_trials simulations et moyenne les
    trajectoires et les statistiques.

    Retourne
    -------
    dict avec tolerances, radii, trajectories (dict T -> (mean_D, std_D)),
         statistics (dict T -> dict de statistiques moyennées)
    """
    from .schelling import SchellingModel, segregation_index
    rng = np.random.default_rng(seed)

    result = {
        "tolerances": tolerances,
        "radii": None,
        "trajectories": {},
        "statistics": {},
    }

    for tol in tolerances:
        all_D = []
        all_stats = []
        for _ in range(n_trials):
            ts = rng.integers(0, 2**31)
            model = SchellingModel(size=size, density=density, tolerance=tol, seed=ts)
            model.run(max_steps=max_steps)

            r, d = multiscalar_trajectory(model.grid, target_type)
            all_D.append(d)
            all_stats.append(trajectory_statistics(r, d))

            if result["radii"] is None:
                result["radii"] = r

        all_D = np.array(all_D)
        result["trajectories"][tol] = (np.mean(all_D, axis=0), np.std(all_D, axis=0))

        # Moyenner les statistiques
        avg_stats = {}
        for key in all_stats[0]:
            vals = [s[key] for s in all_stats]
            avg_stats[key] = float(np.mean(vals))
            avg_stats[key + "_std"] = float(np.std(vals))
        result["statistics"][tol] = avg_stats

    return result


def heterogeneous_trajectory_sweep(
    mean_tolerances: np.ndarray,
    concentrations: list,
    size: int = 50,
    density: float = 0.9,
    n_trials: int = 10,
    max_steps: int = 2000,
    target_type: int = 1,
    seed: int = None,
) -> Dict:
    """Compare D(r) trajectories between homogeneous and heterogeneous models.

    For each mean tolerance, runs both the homogeneous model and the
    heterogeneous model (at each kappa) and computes multiscalar trajectories.
    This reveals how tolerance heterogeneity changes the spatial structure
    of segregation beyond what the scalar index S captures.

    Returns
    -------
    dict with tolerances, radii, homogeneous (dict T -> (mean_D, std_D)),
         heterogeneous (dict (T, kappa) -> (mean_D, std_D))
    """
    from .schelling import SchellingModel, HeterogeneousSchellingModel
    rng = np.random.default_rng(seed)

    result = {
        "tolerances": mean_tolerances,
        "radii": None,
        "homogeneous": {},
        "heterogeneous": {},
    }

    for tol in mean_tolerances:
        # Homogeneous
        all_D_hom = []
        for _ in range(n_trials):
            ts = rng.integers(0, 2**31)
            m = SchellingModel(size=size, density=density, tolerance=tol, seed=ts)
            m.run(max_steps=max_steps)
            r, d = multiscalar_trajectory(m.grid, target_type)
            all_D_hom.append(d)
            if result["radii"] is None:
                result["radii"] = r
        all_D_hom = np.array(all_D_hom)
        result["homogeneous"][tol] = (np.mean(all_D_hom, axis=0), np.std(all_D_hom, axis=0))

        # Heterogeneous for each kappa
        for kappa in concentrations:
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                continue
            all_D_het = []
            for _ in range(n_trials):
                ts = rng.integers(0, 2**31)
                m = HeterogeneousSchellingModel(
                    size=size, density=density, alpha=alpha, beta=beta_param, seed=ts,
                )
                m.run(max_steps=max_steps)
                r, d = multiscalar_trajectory(m.grid, target_type)
                all_D_het.append(d)
            all_D_het = np.array(all_D_het)
            result["heterogeneous"][(tol, kappa)] = (
                np.mean(all_D_het, axis=0), np.std(all_D_het, axis=0)
            )

    return result


def trajectory_slope(radii: np.ndarray, dissimilarities: np.ndarray) -> float:
    """Pente de l'ajustement log-log de la trajectoire.

    Une pente plus raide (plus négative) indique une décroissance plus rapide de la ségrégation
    avec l'échelle, pointant vers un motif plus local.
    """
    if len(radii) < 2:
        return 0.0

    # Filtrer les valeurs nulles ou négatives pour le logarithme
    valid = (radii > 0) & (dissimilarities > 0)
    if valid.sum() < 2:
        return 0.0

    log_r = np.log(radii[valid].astype(float))
    log_d = np.log(dissimilarities[valid])

    # Régression linéaire dans l'espace log-log
    coeffs = np.polyfit(log_r, log_d, deg=1)
    return float(coeffs[0])
