"""
Utilitaires de balayage de paramètres pour cartographier les transitions de phase dans le modèle de Schelling.

Fournit des fonctions pour balayer la tolérance, la densité et la taille du système afin
d'identifier les frontières de phase (transition d'états mélangés à ségrégués).
Basé sur l'analyse du diagramme de phase de Gauvin, Vannimenus et Nadal (2009).

Auteur : Mouhssine Rifaki
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple, Optional, List, Dict
from .schelling import SchellingModel, HeterogeneousSchellingModel, segregation_index, interface_density


def parameter_sweep(
    tolerances: np.ndarray,
    densities: np.ndarray,
    size: int = 30,
    max_steps: int = 200,
    n_trials: int = 3,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Balayage de la tolérance et de la densité pour construire un diagramme de phase.

    Pour chaque paire (tolérance, densité), exécute le modèle de Schelling et
    enregistre l'indice de ségrégation final. Plusieurs essais sont moyennés
    pour réduire le bruit.

    Paramètres
    ----------
    tolerances : np.ndarray
        Tableau 1D des valeurs de tolérance à balayer.
    densities : np.ndarray
        Tableau 1D des valeurs de densité à balayer.
    size : int
        Longueur du côté de la grille.
    max_steps : int
        Nombre maximal d'étapes de simulation par exécution.
    n_trials : int
        Nombre d'exécutions indépendantes par paire de paramètres.
    seed : int ou None
        Graine aléatoire de base.

    Retourne
    -------
    dict avec les clés :
        tolerances, densities : tableaux 1D (les paramètres d'entrée)
        segregation : tableau 2D de forme (len(tolerances), len(densities))
        satisfaction : tableau 2D de même forme
        convergence_steps : tableau 2D de même forme
    """
    rng = np.random.default_rng(seed)
    n_tol = len(tolerances)
    n_den = len(densities)

    seg_map = np.zeros((n_tol, n_den))
    sat_map = np.zeros((n_tol, n_den))
    steps_map = np.zeros((n_tol, n_den))

    for i, tol in enumerate(tolerances):
        for j, den in enumerate(densities):
            seg_vals = []
            sat_vals = []
            step_vals = []

            for trial in range(n_trials):
                trial_seed = rng.integers(0, 2**31)
                model = SchellingModel(
                    size=size,
                    density=den,
                    tolerance=tol,
                    seed=trial_seed,
                )
                result = model.run(max_steps=max_steps, record_every=max_steps)

                seg_vals.append(segregation_index(model.grid))
                sat_vals.append(model.mean_satisfaction())
                step_vals.append(result["steps"])

            seg_map[i, j] = np.mean(seg_vals)
            sat_map[i, j] = np.mean(sat_vals)
            steps_map[i, j] = np.mean(step_vals)

    return {
        "tolerances": tolerances,
        "densities": densities,
        "segregation": seg_map,
        "satisfaction": sat_map,
        "convergence_steps": steps_map,
    }


def phase_boundary(
    tolerances: np.ndarray,
    densities: np.ndarray,
    segregation_map: np.ndarray,
    threshold: float = 0.3,
) -> List[Tuple[float, float]]:
    """Extraction de la frontière de phase à partir d'une carte de ségrégation.

    La frontière est définie comme l'ensemble des points (tolérance, densité) où
    l'indice de ségrégation traverse le seuil donné. Utilise une interpolation
    linéaire le long de l'axe de tolérance.

    Paramètres
    ----------
    tolerances : np.ndarray
        Valeurs de tolérance (lignes de segregation_map).
    densities : np.ndarray
        Valeurs de densité (colonnes de segregation_map).
    segregation_map : np.ndarray
        Tableau 2D d'indices de ségrégation, de forme (n_tol, n_den).
    threshold : float
        Niveau de ségrégation définissant la frontière.

    Retourne
    -------
    liste de tuples (tolérance, densité) formant la frontière de phase.
    """
    boundary = []

    for j, den in enumerate(densities):
        seg_col = segregation_map[:, j]

        # Trouver les croisements
        for i in range(len(seg_col) - 1):
            if (seg_col[i] - threshold) * (seg_col[i + 1] - threshold) < 0:
                # Interpolation linéaire
                t1, t2 = tolerances[i], tolerances[i + 1]
                s1, s2 = seg_col[i], seg_col[i + 1]
                t_cross = t1 + (threshold - s1) * (t2 - t1) / (s2 - s1 + 1e-12)
                boundary.append((float(t_cross), float(den)))

    return boundary


def convergence_sweep(
    tolerances: np.ndarray,
    size: int = 30,
    density: float = 0.9,
    max_steps: int = 500,
    n_trials: int = 3,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Balayage de la tolérance pour étudier le temps de convergence et l'état final.

    Paramètres
    ----------
    tolerances : np.ndarray
        Tableau 1D des valeurs de tolérance.
    size : int
        Longueur du côté de la grille.
    density : float
        Densité fixée pour le balayage.
    max_steps : int
        Nombre maximal d'étapes autorisées.
    n_trials : int
        Nombre d'exécutions par valeur de tolérance.
    seed : int ou None
        Graine aléatoire de base.

    Retourne
    -------
    dict avec les clés :
        tolerances : tableau 1D
        mean_steps : tableau 1D (étapes de convergence moyennes)
        mean_segregation : tableau 1D
        mean_satisfaction : tableau 1D
        fraction_converged : tableau 1D (fraction des essais atteignant l'équilibre avant max_steps)
    """
    rng = np.random.default_rng(seed)

    mean_steps = np.zeros(len(tolerances))
    mean_seg = np.zeros(len(tolerances))
    mean_sat = np.zeros(len(tolerances))
    frac_conv = np.zeros(len(tolerances))

    for i, tol in enumerate(tolerances):
        steps_list = []
        seg_list = []
        sat_list = []
        conv_list = []

        for trial in range(n_trials):
            trial_seed = rng.integers(0, 2**31)
            model = SchellingModel(
                size=size,
                density=density,
                tolerance=tol,
                seed=trial_seed,
            )
            result = model.run(max_steps=max_steps)

            steps_list.append(result["steps"])
            seg_list.append(segregation_index(model.grid))
            sat_list.append(model.mean_satisfaction())
            conv_list.append(1.0 if result["steps"] < max_steps else 0.0)

        mean_steps[i] = np.mean(steps_list)
        mean_seg[i] = np.mean(seg_list)
        mean_sat[i] = np.mean(sat_list)
        frac_conv[i] = np.mean(conv_list)

    return {
        "tolerances": tolerances,
        "mean_steps": mean_steps,
        "mean_segregation": mean_seg,
        "mean_satisfaction": mean_sat,
        "fraction_converged": frac_conv,
    }


def size_scaling(
    sizes: List[int],
    tolerance: float = 0.5,
    density: float = 0.9,
    max_steps: int = 300,
    n_trials: int = 3,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Étude de la dépendance de la ségrégation à la taille du système (effets de taille finie).

    Paramètres
    ----------
    sizes : liste d'entiers
        Tailles de grille à tester.
    tolerance : float
        Tolérance fixée.
    density : float
        Densité fixée.
    max_steps : int
        Nombre maximal d'étapes.
    n_trials : int
        Nombre d'exécutions par taille.
    seed : int ou None
        Graine aléatoire de base.

    Retourne
    -------
    dict avec les clés : sizes, segregation, convergence_steps
    """
    rng = np.random.default_rng(seed)

    seg_arr = np.zeros(len(sizes))
    steps_arr = np.zeros(len(sizes))

    for i, sz in enumerate(sizes):
        seg_list = []
        steps_list = []

        for trial in range(n_trials):
            trial_seed = rng.integers(0, 2**31)
            model = SchellingModel(
                size=sz,
                density=density,
                tolerance=tolerance,
                seed=trial_seed,
            )
            result = model.run(max_steps=max_steps)
            seg_list.append(segregation_index(model.grid))
            steps_list.append(result["steps"])

        seg_arr[i] = np.mean(seg_list)
        steps_arr[i] = np.mean(steps_list)

    return {
        "sizes": np.array(sizes),
        "segregation": seg_arr,
        "convergence_steps": steps_arr,
    }


# ---------------------------------------------------------------------------
# Analyse quantitative de la transition de phase
# ---------------------------------------------------------------------------

def binder_cumulant(
    sizes: List[int],
    tolerance_range: np.ndarray,
    density: float = 0.9,
    n_trials: int = 30,
    max_steps: int = 5000,
    seed: Optional[int] = None,
) -> Dict:
    """Compute the Binder cumulant U_4 = 1 - <S^4> / (3 <S^2>^2) for each (L, T).

    The Binder cumulant is a dimensionless ratio that is size-independent at T_c.
    Curves for different L cross at the critical point, providing a model-free
    estimate of T_c without any fitting assumptions.

    Returns
    -------
    dict with sizes, tolerances, U4 (dict L -> array of U4 values),
         U4_err (dict L -> standard error)
    """
    rng = np.random.default_rng(seed)
    result = {
        "sizes": sizes,
        "tolerances": tolerance_range,
        "U4": {},
        "U4_err": {},
    }

    for sz in sizes:
        u4_vals = np.zeros(len(tolerance_range))
        u4_errs = np.zeros(len(tolerance_range))

        for i, tol in enumerate(tolerance_range):
            s_samples = np.zeros(n_trials)
            for t in range(n_trials):
                trial_seed = rng.integers(0, 2**31)
                model = SchellingModel(
                    size=sz, density=density, tolerance=tol, seed=trial_seed,
                )
                model.run(max_steps=max_steps)
                s_samples[t] = segregation_index(model.grid)

            s2 = np.mean(s_samples**2)
            s4 = np.mean(s_samples**4)
            if s2 > 0:
                u4_vals[i] = 1.0 - s4 / (3.0 * s2**2)
            else:
                u4_vals[i] = 2.0 / 3.0  # trivial limit

            # Bootstrap error estimate
            n_boot = 200
            u4_boot = np.zeros(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n_trials, size=n_trials)
                s_boot = s_samples[idx]
                s2_b = np.mean(s_boot**2)
                s4_b = np.mean(s_boot**4)
                if s2_b > 0:
                    u4_boot[b] = 1.0 - s4_b / (3.0 * s2_b**2)
                else:
                    u4_boot[b] = 2.0 / 3.0
            u4_errs[i] = np.std(u4_boot)

        result["U4"][sz] = u4_vals.copy()
        result["U4_err"][sz] = u4_errs.copy()

    return result


def _sigmoid(T: np.ndarray, B: float, A: float, Tc: float, w: float) -> np.ndarray:
    """Sigmoide : S(T) = B + A * (1 + tanh((T - Tc) / w)) / 2."""
    return B + A * (1.0 + np.tanh((T - Tc) / w)) / 2.0


def _detect_frozen_regime(tolerances: np.ndarray, segregation_values: np.ndarray) -> int:
    """Detect where the frozen regime begins (S starts decreasing after peak).

    The frozen regime occurs at high tolerance when agents cannot find any
    satisfying position: the system gets stuck in a disordered state and S
    drops. Physically, this corresponds to T above the percolation threshold
    of satisfiable configurations.

    Returns the index of the last point before the frozen regime.
    If no frozen regime is detected, returns len(tolerances).
    """
    if len(segregation_values) < 5:
        return len(tolerances)

    peak_idx = np.argmax(segregation_values)
    peak_val = segregation_values[peak_idx]

    if peak_idx >= len(tolerances) - 3:
        return len(tolerances)

    post_peak = segregation_values[peak_idx:]

    cumulative_drop = peak_val - np.min(post_peak)
    if cumulative_drop < 0.15 * peak_val:
        return len(tolerances)

    for i in range(len(post_peak) - 2):
        if (post_peak[i] > post_peak[i+1] > post_peak[i+2] and
                post_peak[i] - post_peak[i+2] > 0.05):
            return peak_idx + i + 1

    threshold = 0.85 * peak_val
    below = np.where(post_peak < threshold)[0]
    if len(below) > 0:
        return peak_idx + below[0]

    return len(tolerances)


def detect_discrete_transitions(
    tolerances: np.ndarray,
    segregation_values: np.ndarray,
    seg_errors: Optional[np.ndarray] = None,
    min_jump: float = 0.05,
) -> List[Dict[str, float]]:
    """Detect discrete jumps in S(T) caused by lattice neighbor thresholds.

    In the Schelling model with Moore neighborhood (8 neighbors), agents
    compare their fraction of same-type neighbors against a tolerance T.
    Since the fraction takes discrete values k/n_occ (where n_occ <= 8),
    the equilibrium segregation shows discontinuous jumps at rational
    tolerance thresholds (1/8, 2/8, 3/8, ...).

    This function identifies these jumps by looking for large changes in
    the gradient of S(T), rather than fitting a smooth sigmoid.

    Parameters
    ----------
    tolerances : np.ndarray
        Tolerance values.
    segregation_values : np.ndarray
        Mean segregation at each tolerance.
    seg_errors : np.ndarray, optional
        Standard errors.
    min_jump : float
        Minimum jump magnitude to report.

    Returns
    -------
    List of dicts, each with:
        T_jump: midpoint tolerance of the jump
        T_low, T_high: tolerance range bracketing the jump
        S_before, S_after: segregation levels before and after
        delta_S: magnitude of the jump
    Sorted by delta_S descending (largest jump first).
    """
    if len(tolerances) < 3:
        return []

    dS = np.diff(segregation_values)
    dt = np.diff(tolerances)
    grad = dS / dt

    jumps = []
    i = 0
    while i < len(grad):
        if abs(dS[i]) >= min_jump:
            # Find the extent of this jump (consecutive large-gradient region)
            j = i
            while j < len(grad) - 1 and abs(dS[j + 1]) >= min_jump * 0.5:
                j += 1
            T_low = float(tolerances[i])
            T_high = float(tolerances[j + 1])
            S_before = float(segregation_values[i])
            S_after = float(segregation_values[j + 1])
            jumps.append({
                "T_jump": float((T_low + T_high) / 2),
                "T_low": T_low,
                "T_high": T_high,
                "S_before": S_before,
                "S_after": S_after,
                "delta_S": float(abs(S_after - S_before)),
            })
            i = j + 1
        else:
            i += 1

    jumps.sort(key=lambda x: x["delta_S"], reverse=True)
    return jumps


def extract_critical_point(
    tolerances: np.ndarray,
    segregation_values: np.ndarray,
    seg_errors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Extract the primary transition point from S(T).

    Uses a hybrid approach: first detects discrete jumps characteristic of
    the Schelling model's lattice thresholds, then falls back to sigmoid
    fitting if the transition is smooth enough.

    The Schelling model on a Moore-neighborhood lattice exhibits a staircase
    S(T) with jumps at rational tolerance thresholds (k/8). The "critical
    point" T_c is defined as the midpoint of the largest jump.

    Parameters
    ----------
    tolerances : np.ndarray
        Tolerance values.
    segregation_values : np.ndarray
        Mean segregation at each tolerance.
    seg_errors : np.ndarray, optional
        Standard errors.

    Returns
    -------
    dict with T_c, T_c_err, width, width_err, A, B, residual, frozen_idx,
         jumps (list of detected discrete transitions)
    """
    frozen_idx = _detect_frozen_regime(tolerances, segregation_values)
    fit_tol = tolerances[:frozen_idx]
    fit_seg = segregation_values[:frozen_idx]
    fit_err = seg_errors[:frozen_idx] if seg_errors is not None else None

    # Detect discrete jumps
    jumps = detect_discrete_transitions(fit_tol, fit_seg, fit_err)

    if jumps and jumps[0]["delta_S"] > 0.08:
        # Discrete transition: use the largest jump
        main_jump = jumps[0]
        T_c = main_jump["T_jump"]
        T_c_err = (main_jump["T_high"] - main_jump["T_low"]) / 2
        width = main_jump["T_high"] - main_jump["T_low"]
        A = main_jump["delta_S"]
        B = main_jump["S_before"]
        return {
            "T_c": float(T_c),
            "T_c_err": float(T_c_err),
            "width": float(width),
            "width_err": float(T_c_err),
            "A": float(A),
            "B": float(B),
            "residual": 0.0,
            "frozen_idx": frozen_idx,
            "jumps": jumps,
        }

    # Fallback: sigmoid fit for smooth transitions (e.g. heterogeneous model)
    if len(fit_tol) < 4:
        grad = np.gradient(segregation_values, tolerances)
        idx = np.argmax(np.abs(grad))
        return {
            "T_c": float(tolerances[idx]),
            "T_c_err": float(tolerances[1] - tolerances[0]),
            "width": 0.05, "width_err": np.nan,
            "A": float(np.ptp(segregation_values)),
            "B": float(np.min(segregation_values)),
            "residual": np.nan, "frozen_idx": frozen_idx, "jumps": jumps,
        }

    p0 = [0.0, 1.0, np.median(fit_tol), 0.05]
    bounds = ([-0.5, 0.0, fit_tol.min(), 0.001],
              [0.5, 2.0, fit_tol.max(), 0.5])
    sigma = fit_err if fit_err is not None else None
    try:
        popt, pcov = curve_fit(
            _sigmoid, fit_tol, fit_seg,
            p0=p0, bounds=bounds, sigma=sigma, absolute_sigma=True, maxfev=10000,
        )
    except RuntimeError:
        grad = np.gradient(fit_seg, fit_tol)
        idx = np.argmax(np.abs(grad))
        return {
            "T_c": float(fit_tol[idx]),
            "T_c_err": float(fit_tol[1] - fit_tol[0]),
            "width": 0.05, "width_err": np.nan,
            "A": float(np.ptp(fit_seg)),
            "B": float(np.min(fit_seg)),
            "residual": np.nan, "frozen_idx": frozen_idx, "jumps": jumps,
        }

    perr = np.sqrt(np.diag(pcov))
    fitted = _sigmoid(fit_tol, *popt)
    residual = float(np.mean((fit_seg - fitted) ** 2))

    return {
        "T_c": float(popt[2]),
        "T_c_err": float(perr[2]),
        "width": float(abs(popt[3])),
        "width_err": float(perr[3]),
        "A": float(popt[1]),
        "B": float(popt[0]),
        "residual": residual,
        "frozen_idx": frozen_idx,
        "jumps": jumps,
    }


def finite_size_scaling(
    sizes: List[int],
    tolerance_range: np.ndarray,
    density: float = 0.9,
    n_trials: int = 20,
    max_steps: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Balayage S(T) pour plusieurs tailles L et extraction de Tc(L).

    Pour chaque taille L, effectue un balayage en tolérance avec n_trials
    essais et ajuste S(T) pour extraire Tc(L). Permet ensuite l'analyse
    d'échelle de taille finie.

    Retourne
    -------
    dict avec sizes, tolerances, segregation (dict L -> (mean, std)),
         T_c (dict L -> (val, err))
    """
    rng = np.random.default_rng(seed)
    result = {
        "sizes": sizes,
        "tolerances": tolerance_range,
        "segregation": {},
        "T_c": {},
    }

    for sz in sizes:
        mean_seg = np.zeros(len(tolerance_range))
        std_seg = np.zeros(len(tolerance_range))

        for i, tol in enumerate(tolerance_range):
            seg_vals = []
            for _ in range(n_trials):
                trial_seed = rng.integers(0, 2**31)
                model = SchellingModel(
                    size=sz, density=density, tolerance=tol, seed=trial_seed,
                )
                model.run(max_steps=max_steps)
                seg_vals.append(segregation_index(model.grid))
            mean_seg[i] = np.mean(seg_vals)
            std_seg[i] = np.std(seg_vals) / np.sqrt(n_trials)

        result["segregation"][sz] = (mean_seg.copy(), std_seg.copy())

        cp = extract_critical_point(tolerance_range, mean_seg, std_seg)
        result["T_c"][sz] = (cp["T_c"], cp["T_c_err"])

    return result


def susceptibility(
    sizes: List[int],
    tolerance_range: np.ndarray,
    density: float = 0.9,
    n_trials: int = 30,
    max_steps: int = 5000,
    seed: Optional[int] = None,
) -> Dict:
    """Compute the susceptibility chi = L^2 * (<S^2> - <S>^2) for each (L, T).

    The susceptibility diverges at the critical point in the thermodynamic limit,
    with a peak height that scales as L^(gamma/nu). The peak position gives
    another estimate of T_c(L).

    Returns
    -------
    dict with sizes, tolerances, chi (dict L -> array), chi_err (dict L -> array),
         chi_peak_T (dict L -> float), chi_peak_val (dict L -> float)
    """
    rng = np.random.default_rng(seed)
    result = {
        "sizes": sizes,
        "tolerances": tolerance_range,
        "chi": {},
        "chi_err": {},
        "chi_peak_T": {},
        "chi_peak_val": {},
    }

    for sz in sizes:
        chi_vals = np.zeros(len(tolerance_range))
        chi_errs = np.zeros(len(tolerance_range))

        for i, tol in enumerate(tolerance_range):
            s_samples = np.zeros(n_trials)
            for t in range(n_trials):
                trial_seed = rng.integers(0, 2**31)
                model = SchellingModel(
                    size=sz, density=density, tolerance=tol, seed=trial_seed,
                )
                model.run(max_steps=max_steps)
                s_samples[t] = segregation_index(model.grid)

            chi_vals[i] = sz**2 * (np.mean(s_samples**2) - np.mean(s_samples)**2)

            # Bootstrap error
            n_boot = 200
            chi_boot = np.zeros(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n_trials, size=n_trials)
                s_b = s_samples[idx]
                chi_boot[b] = sz**2 * (np.mean(s_b**2) - np.mean(s_b)**2)
            chi_errs[i] = np.std(chi_boot)

        result["chi"][sz] = chi_vals.copy()
        result["chi_err"][sz] = chi_errs.copy()

        peak_idx = np.argmax(chi_vals)
        result["chi_peak_T"][sz] = float(tolerance_range[peak_idx])
        result["chi_peak_val"][sz] = float(chi_vals[peak_idx])

    return result


def order_parameter_exponent(
    sizes: List[int],
    tolerance_range: np.ndarray,
    T_c: float,
    density: float = 0.9,
    n_trials: int = 30,
    max_steps: int = 5000,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate the order parameter exponent beta from <S> ~ (Tc - T)^beta.

    Measures <S> at several T < Tc and fits a power law to extract beta.
    Only uses points in the ordered phase sufficiently close to Tc.

    Returns
    -------
    dict with beta, beta_err, T_fit, S_fit (arrays used for the fit)
    """
    rng = np.random.default_rng(seed)

    # Select points in the ordered phase (T > Tc for Schelling, where higher T = more segregated)
    T_above = tolerance_range[tolerance_range > T_c + 0.01]
    if len(T_above) < 3:
        return {"beta": np.nan, "beta_err": np.nan, "T_fit": np.array([]), "S_fit": np.array([])}

    # Use the first size (largest for best statistics)
    sz = max(sizes)
    mean_S = np.zeros(len(T_above))
    for i, tol in enumerate(T_above):
        s_vals = []
        for _ in range(n_trials):
            trial_seed = rng.integers(0, 2**31)
            model = SchellingModel(size=sz, density=density, tolerance=tol, seed=trial_seed)
            model.run(max_steps=max_steps)
            s_vals.append(segregation_index(model.grid))
        mean_S[i] = np.mean(s_vals)

    # Fit S = a * (T - Tc)^beta in log-log space
    dT = T_above - T_c
    valid = (mean_S > 0.01) & (dT > 0)
    if valid.sum() < 3:
        return {"beta": np.nan, "beta_err": np.nan, "T_fit": T_above, "S_fit": mean_S}

    log_dT = np.log(dT[valid])
    log_S = np.log(mean_S[valid])

    try:
        coeffs, cov = np.polyfit(log_dT, log_S, deg=1, cov=True)
        beta = float(coeffs[0])
        beta_err = float(np.sqrt(cov[0, 0]))
    except (np.linalg.LinAlgError, ValueError):
        beta = np.nan
        beta_err = np.nan

    return {"beta": beta, "beta_err": beta_err, "T_fit": T_above, "S_fit": mean_S}


def susceptibility_exponent(
    sizes: np.ndarray,
    chi_peak_values: np.ndarray,
    chi_peak_errors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate gamma/nu from the scaling of susceptibility peak: chi_max ~ L^(gamma/nu).

    Fits log(chi_max) vs log(L) to extract the ratio gamma/nu.

    Returns
    -------
    dict with gamma_over_nu, gamma_over_nu_err
    """
    valid = chi_peak_values > 0
    if valid.sum() < 2:
        return {"gamma_over_nu": np.nan, "gamma_over_nu_err": np.nan}

    log_L = np.log(sizes[valid].astype(float))
    log_chi = np.log(chi_peak_values[valid])

    try:
        coeffs, cov = np.polyfit(log_L, log_chi, deg=1, cov=True)
        return {
            "gamma_over_nu": float(coeffs[0]),
            "gamma_over_nu_err": float(np.sqrt(cov[0, 0])),
        }
    except (np.linalg.LinAlgError, ValueError):
        return {"gamma_over_nu": np.nan, "gamma_over_nu_err": np.nan}


def critical_exponents(
    sizes: np.ndarray,
    T_c_values: np.ndarray,
    T_c_errors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Estimate Tc(inf) and the correlation length exponent nu from Tc(L).

    For the Schelling model, the transition is discrete (lattice thresholds),
    so Tc(L) converges rapidly to a size-independent value. When Tc values
    are essentially constant across sizes (std < mean error), we report the
    weighted mean as Tc_inf and flag the transition as discrete.

    For models with genuine size-dependent Tc (e.g. heterogeneous variants),
    fits Tc(L) = Tc_inf + a * L^(-1/nu).

    Returns
    -------
    dict with T_c_inf, T_c_inf_err, nu, nu_err, discrete_transition (bool)
    """
    # Check if Tc values are essentially size-independent (discrete transition)
    tc_spread = np.ptp(T_c_values)
    tc_mean_err = np.mean(T_c_errors) if T_c_errors is not None else 0.01

    if tc_spread < 3 * tc_mean_err or tc_spread < 0.02:
        # Tc is size-independent: discrete lattice transition
        if T_c_errors is not None and np.all(T_c_errors > 0):
            weights = 1.0 / T_c_errors**2
            tc_inf = float(np.average(T_c_values, weights=weights))
            tc_inf_err = float(1.0 / np.sqrt(np.sum(weights)))
        else:
            tc_inf = float(np.mean(T_c_values))
            tc_inf_err = float(np.std(T_c_values) / np.sqrt(len(T_c_values)))
        return {
            "T_c_inf": tc_inf,
            "T_c_inf_err": tc_inf_err,
            "nu": np.inf,
            "nu_err": np.nan,
            "discrete_transition": True,
        }

    # Genuine FSS: fit Tc(L) = Tc_inf + a * L^(-1/nu)
    def _tc_model(L, Tc_inf, a, inv_nu):
        return Tc_inf + a * L ** (-inv_nu)

    p0 = [T_c_values[-1], 0.5, 1.0]
    bounds = ([0.0, -10.0, 0.1], [1.0, 10.0, 5.0])

    sigma = T_c_errors if T_c_errors is not None else None
    try:
        popt, pcov = curve_fit(
            _tc_model, sizes.astype(float), T_c_values,
            p0=p0, bounds=bounds, sigma=sigma, absolute_sigma=True, maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        return {
            "T_c_inf": float(popt[0]),
            "T_c_inf_err": float(perr[0]),
            "nu": float(1.0 / popt[2]),
            "nu_err": float(perr[2] / popt[2]**2),
            "discrete_transition": False,
        }
    except RuntimeError:
        return {
            "T_c_inf": float(T_c_values[-1]),
            "T_c_inf_err": np.nan,
            "nu": np.nan,
            "nu_err": np.nan,
            "discrete_transition": False,
        }


def variance_scaling(
    sizes: np.ndarray,
    tolerances: np.ndarray,
    raw_data: Dict[int, np.ndarray],
    T_c: float,
) -> Dict[str, float]:
    """Measure how fluctuations at the transition scale with system size.

    For the discrete Schelling transition, the traditional nu exponent is
    meaningless (Tc doesn't shift with L). Instead, we measure how the
    variance of S at the transition point scales with L:
        Var(S) ~ L^(-alpha)
    This captures how the transition sharpens with system size.

    Also measures the steepness of the main jump: dS/dT at Tc scales
    as L^(1/nu_eff), providing an effective finite-size exponent.

    Parameters
    ----------
    sizes : array of system sizes
    tolerances : tolerance values
    raw_data : dict mapping size -> (n_tol, n_trials) array of raw S values
    T_c : critical tolerance

    Returns
    -------
    dict with alpha, alpha_err (variance decay exponent),
         nu_eff, nu_eff_err (effective steepness exponent),
         var_at_Tc (array of variances for each size)
    """
    # Find the tolerance index closest to Tc
    tc_idx = np.argmin(np.abs(tolerances - T_c))

    # Measure variance at Tc for each size
    var_at_tc = []
    steep_at_tc = []
    valid_sizes = []

    for sz in sizes:
        sz = int(sz)
        if sz not in raw_data:
            continue
        raw = raw_data[sz]
        # Variance of S at the transition
        samples = raw[tc_idx, :]
        samples = samples[~np.isnan(samples)]
        if len(samples) < 2:
            continue
        var_at_tc.append(np.var(samples))

        # Steepness: gradient of mean S at Tc
        means = np.nanmean(raw, axis=1)
        grad = np.gradient(means, tolerances)
        steep_at_tc.append(abs(grad[tc_idx]))
        valid_sizes.append(float(sz))

    valid_sizes = np.array(valid_sizes)
    var_at_tc = np.array(var_at_tc)
    steep_at_tc = np.array(steep_at_tc)

    result = {"var_at_Tc": var_at_tc, "sizes_used": valid_sizes}

    # Fit Var ~ L^(-alpha)
    if len(valid_sizes) >= 3 and np.all(var_at_tc > 0):
        try:
            coeffs, cov = np.polyfit(
                np.log(valid_sizes), np.log(var_at_tc), 1, cov=True)
            result["alpha"] = float(-coeffs[0])
            result["alpha_err"] = float(np.sqrt(cov[0, 0]))
        except (np.linalg.LinAlgError, ValueError):
            result["alpha"] = np.nan
            result["alpha_err"] = np.nan
    else:
        result["alpha"] = np.nan
        result["alpha_err"] = np.nan

    # Fit steepness ~ L^(1/nu_eff)
    if len(valid_sizes) >= 3 and np.all(steep_at_tc > 0):
        try:
            coeffs, cov = np.polyfit(
                np.log(valid_sizes), np.log(steep_at_tc), 1, cov=True)
            result["nu_eff"] = float(1.0 / coeffs[0]) if coeffs[0] > 0 else np.nan
            result["nu_eff_err"] = (float(np.sqrt(cov[0, 0]) / coeffs[0]**2)
                                    if coeffs[0] > 0 else np.nan)
        except (np.linalg.LinAlgError, ValueError):
            result["nu_eff"] = np.nan
            result["nu_eff_err"] = np.nan
    else:
        result["nu_eff"] = np.nan
        result["nu_eff_err"] = np.nan

    return result


def scaling_collapse(
    sizes: List[int],
    tolerances: np.ndarray,
    segregation_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    T_c_inf: float,
    nu_range: Optional[np.ndarray] = None,
) -> Dict:
    """Tentative de collapse d'échelle : S vs (T - Tc) * L^(1/nu).

    Cherche la valeur de nu qui minimise la dispersion entre les courbes
    rescalées pour différentes tailles L.

    Retourne
    -------
    dict avec best_nu, nu_range, quality (variance résiduelle), collapsed_x, collapsed_y
    """
    if nu_range is None:
        nu_range = np.linspace(0.3, 3.0, 50)

    quality = np.zeros(len(nu_range))

    for k, nu in enumerate(nu_range):
        # Construire les courbes rescalées
        all_x = []
        all_y = []
        for sz in sizes:
            mean_seg = segregation_data[sz][0]
            x_scaled = (tolerances - T_c_inf) * sz ** (1.0 / nu)
            all_x.append(x_scaled)
            all_y.append(mean_seg)

        # Interpoler sur une grille commune
        x_min = max(x.min() for x in all_x)
        x_max = min(x.max() for x in all_x)
        if x_min >= x_max:
            quality[k] = np.inf
            continue

        x_common = np.linspace(x_min, x_max, 100)
        interpolated = []
        for x, y in zip(all_x, all_y):
            interpolated.append(np.interp(x_common, x, y))

        interpolated = np.array(interpolated)
        # Variance inter-courbes à chaque point rescalé
        quality[k] = np.mean(np.var(interpolated, axis=0))

    best_idx = np.argmin(quality)
    best_nu = float(nu_range[best_idx])

    # Construire le collapse final
    collapsed_x = {}
    collapsed_y = {}
    for sz in sizes:
        mean_seg = segregation_data[sz][0]
        collapsed_x[sz] = (tolerances - T_c_inf) * sz ** (1.0 / best_nu)
        collapsed_y[sz] = mean_seg

    return {
        "best_nu": best_nu,
        "nu_range": nu_range,
        "quality": quality,
        "collapsed_x": collapsed_x,
        "collapsed_y": collapsed_y,
    }


def compare_homogeneous_heterogeneous(
    mean_tolerance_range: np.ndarray,
    concentrations: List[float] = [2.0, 5.0, 20.0],
    size: int = 50,
    density: float = 0.9,
    n_trials: int = 10,
    max_steps: int = 2000,
    seed: Optional[int] = None,
) -> Dict:
    """Compare la transition de phase entre modèle homogène et hétérogène.

    Pour le modèle hétérogène, la tolérance de chaque agent est tirée d'une
    distribution Beta(kappa * T_mean, kappa * (1 - T_mean)) où kappa contrôle
    la concentration autour de la moyenne.

    Retourne
    -------
    dict avec tolerances, homogeneous (mean, std),
         heterogeneous: dict kappa -> (mean, std)
    """
    rng = np.random.default_rng(seed)
    result = {
        "tolerances": mean_tolerance_range,
        "homogeneous": None,
        "heterogeneous": {},
    }

    # Homogène
    hom_mean = np.zeros(len(mean_tolerance_range))
    hom_std = np.zeros(len(mean_tolerance_range))
    for i, tol in enumerate(mean_tolerance_range):
        vals = []
        for _ in range(n_trials):
            ts = rng.integers(0, 2**31)
            m = SchellingModel(size=size, density=density, tolerance=tol, seed=ts)
            m.run(max_steps=max_steps)
            vals.append(segregation_index(m.grid))
        hom_mean[i] = np.mean(vals)
        hom_std[i] = np.std(vals) / np.sqrt(n_trials)
    result["homogeneous"] = (hom_mean, hom_std)

    # Hétérogène pour chaque concentration
    for kappa in concentrations:
        het_mean = np.zeros(len(mean_tolerance_range))
        het_std = np.zeros(len(mean_tolerance_range))
        for i, tol in enumerate(mean_tolerance_range):
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                het_mean[i] = 0.0
                het_std[i] = 0.0
                continue
            vals = []
            for _ in range(n_trials):
                ts = rng.integers(0, 2**31)
                m = HeterogeneousSchellingModel(
                    size=size, density=density, alpha=alpha, beta=beta_param, seed=ts,
                )
                m.run(max_steps=max_steps)
                vals.append(segregation_index(m.grid))
            het_mean[i] = np.mean(vals)
            het_std[i] = np.std(vals) / np.sqrt(n_trials)
        result["heterogeneous"][kappa] = (het_mean, het_std)

    return result


def heterogeneous_critical_points(
    concentrations: List[float],
    tolerance_range: np.ndarray,
    size: int = 50,
    density: float = 0.9,
    n_trials: int = 20,
    max_steps: int = 3000,
    seed: Optional[int] = None,
) -> Dict:
    """Extract T_c as a function of kappa for the heterogeneous model.

    For each concentration kappa, sweeps mean tolerance and fits a sigmoid
    to extract T_c(kappa). This quantifies how tolerance heterogeneity
    shifts the critical point.

    Returns
    -------
    dict with concentrations, tolerances, T_c (dict kappa -> (val, err)),
         segregation (dict kappa -> (mean, std))
    """
    rng = np.random.default_rng(seed)
    result = {
        "concentrations": concentrations,
        "tolerances": tolerance_range,
        "T_c": {},
        "segregation": {},
    }

    for kappa in concentrations:
        mean_seg = np.zeros(len(tolerance_range))
        std_seg = np.zeros(len(tolerance_range))

        for i, tol in enumerate(tolerance_range):
            alpha = kappa * tol
            beta_param = kappa * (1.0 - tol)
            if alpha <= 0 or beta_param <= 0:
                mean_seg[i] = 0.0
                std_seg[i] = 0.0
                continue
            vals = []
            for _ in range(n_trials):
                ts = rng.integers(0, 2**31)
                m = HeterogeneousSchellingModel(
                    size=size, density=density, alpha=alpha, beta=beta_param, seed=ts,
                )
                m.run(max_steps=max_steps)
                vals.append(segregation_index(m.grid))
            mean_seg[i] = np.mean(vals)
            std_seg[i] = np.std(vals) / np.sqrt(n_trials)

        result["segregation"][kappa] = (mean_seg.copy(), std_seg.copy())
        cp = extract_critical_point(tolerance_range, mean_seg, std_seg)
        result["T_c"][kappa] = (cp["T_c"], cp["T_c_err"])

    return result
