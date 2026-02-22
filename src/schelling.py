"""
Modèle de ségrégation de Schelling sur une grille 2D.

Implémente le modèle classique de Schelling (1971) où des agents de deux types
occupent des cellules sur une grille et se déplacent lorsque la fraction de voisins
du même type tombe en dessous d'un seuil de tolérance.

Auteur : Mouhssine Rifaki
"""

import numpy as np
from typing import Tuple, Optional

from src.utils import _validate_unit_interval


# États des cellules
EMPTY = 0
TYPE_A = 1
TYPE_B = 2

# Moore neighborhood offsets (8-connected)
_MOORE_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]


def _cell_satisfaction(grid: np.ndarray, r: int, c: int) -> float:
    """Compute satisfaction of the agent at (r, c) on a periodic grid.

    Returns the fraction of occupied Moore neighbours sharing the agent's type.
    Returns 1.0 for isolated agents (no occupied neighbours), 0.0 for empty cells.
    """
    size = grid.shape[0]
    agent_type = grid[r, c]
    if agent_type == EMPTY:
        return 0.0

    same = 0
    occupied = 0
    for dr, dc in _MOORE_OFFSETS:
        nr = (r + dr) % size
        nc = (c + dc) % size
        neighbour = grid[nr, nc]
        if neighbour != EMPTY:
            occupied += 1
            if neighbour == agent_type:
                same += 1

    if occupied == 0:
        return 1.0
    return same / occupied


def _vectorized_satisfaction_map(grid: np.ndarray) -> np.ndarray:
    """Compute satisfaction for all cells simultaneously using np.roll.

    Returns an (L, L) array where each occupied cell contains its satisfaction
    (fraction of same-type occupied Moore neighbors). Empty cells get 0.0,
    isolated agents (no occupied neighbors) get 1.0.
    """
    occupied_mask = grid != EMPTY
    same_count = np.zeros(grid.shape, dtype=np.float64)
    occupied_count = np.zeros(grid.shape, dtype=np.float64)

    for dr, dc in _MOORE_OFFSETS:
        shifted = np.roll(np.roll(grid, -dr, axis=0), -dc, axis=1)
        neighbor_occupied = shifted != EMPTY
        neighbor_same = (shifted == grid) & neighbor_occupied & occupied_mask
        occupied_count += neighbor_occupied
        same_count += neighbor_same

    with np.errstate(divide='ignore', invalid='ignore'):
        sat = np.where(occupied_count > 0, same_count / occupied_count, 1.0)
    sat[~occupied_mask] = 0.0
    return sat


class SchellingModel:
    """Modèle de ségrégation de Schelling sur grille.

    Paramètres
    ----------
    size : int
        Longueur du côté de la grille carrée.
    density : float
        Fraction de cellules occupées (entre 0 et 1).
    fraction_a : float
        Fraction des cellules occupées qui sont de type A.
    tolerance : float
        Fraction minimale de voisins du même type pour qu'un agent soit satisfait.
    seed : int ou None
        Graine aléatoire pour la reproductibilité.
    """

    def __init__(
        self,
        size: int = 50,
        density: float = 0.9,
        fraction_a: float = 0.5,
        tolerance: float = 0.5,
        tolerance_noise: float = 0.0,
        seed: Optional[int] = None,
    ):
        if size < 1:
            raise ValueError(f"size must be >= 1, got {size}")
        _validate_unit_interval(density, "density")
        _validate_unit_interval(fraction_a, "fraction_a")
        _validate_unit_interval(tolerance, "tolerance")

        self.size = size
        self.density = density
        self.fraction_a = fraction_a
        self.tolerance = tolerance
        self.tolerance_noise = tolerance_noise
        self.rng = np.random.default_rng(seed)

        self.grid = self._initialize_grid()
        self.step_count = 0
        self.history = []

        # Per-agent tolerance: T_i = clip(tolerance + N(0, noise), 0, 1)
        if tolerance_noise > 0:
            noise = self.rng.normal(0, tolerance_noise, size=(size, size))
            self.tolerance_map = np.clip(tolerance + noise, 0.0, 1.0)
            self.tolerance_map[self.grid == EMPTY] = 0.0
        else:
            self.tolerance_map = None

    def _initialize_grid(self) -> np.ndarray:
        """Placement aléatoire des agents sur la grille."""
        n_cells = self.size * self.size
        n_occupied = int(n_cells * self.density)
        n_a = int(n_occupied * self.fraction_a)
        n_b = n_occupied - n_a

        cells = np.array(
            [TYPE_A] * n_a + [TYPE_B] * n_b + [EMPTY] * (n_cells - n_occupied)
        )
        self.rng.shuffle(cells)
        return cells.reshape(self.size, self.size)

    def _get_neighbours(self, r: int, c: int) -> np.ndarray:
        """Retourne les valeurs des 8 voisins de Moore (avec conditions aux limites périodiques)."""
        neighbours = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = (r + dr) % self.size
                nc = (c + dc) % self.size
                neighbours.append(self.grid[nr, nc])
        return np.array(neighbours)

    def agent_satisfaction(self, r: int, c: int) -> float:
        """Calcule la satisfaction de l'agent à la position (r, c).

        La satisfaction est la fraction de voisins occupés qui partagent le
        type de l'agent. Retourne 1.0 si l'agent n'a aucun voisin occupé
        (convention : les agents isolés sont satisfaits).
        """
        return _cell_satisfaction(self.grid, r, c)

    def is_satisfied(self, r: int, c: int) -> bool:
        """Vérifie si l'agent à (r, c) atteint le seuil de tolérance."""
        if self.grid[r, c] == EMPTY:
            return True
        tol = self.tolerance_map[r, c] if self.tolerance_map is not None else self.tolerance
        return self.agent_satisfaction(r, c) >= tol

    def satisfaction_map(self) -> np.ndarray:
        """Calcule la satisfaction pour chaque cellule de la grille.

        Retourne un tableau (size, size) où les cellules vides ont la valeur 0.
        """
        return _vectorized_satisfaction_map(self.grid)

    def mean_satisfaction(self) -> float:
        """Satisfaction moyenne sur toutes les cellules occupées."""
        smap = self.satisfaction_map()
        occupied_mask = self.grid != EMPTY
        if occupied_mask.sum() == 0:
            return 0.0
        return float(smap[occupied_mask].mean())

    def fraction_satisfied(self) -> float:
        """Fraction des agents occupés qui sont satisfaits."""
        smap = self.satisfaction_map()
        occupied_mask = self.grid != EMPTY
        total = occupied_mask.sum()
        if total == 0:
            return 0.0
        if self.tolerance_map is not None:
            return float((smap[occupied_mask] >= self.tolerance_map[occupied_mask]).sum() / total)
        return float((smap[occupied_mask] >= self.tolerance).sum() / total)

    def step(self) -> int:
        """Exécute une étape de la dynamique.

        Chaque agent insatisfait se déplace vers une cellule vide aléatoire.
        Retourne le nombre d'agents qui se sont déplacés.
        """
        smap = _vectorized_satisfaction_map(self.grid)
        occupied_mask = self.grid != EMPTY
        if self.tolerance_map is not None:
            unsatisfied_mask = occupied_mask & (smap < self.tolerance_map)
        else:
            unsatisfied_mask = occupied_mask & (smap < self.tolerance)
        unsatisfied_coords = np.argwhere(unsatisfied_mask)

        empty_cells = list(zip(*np.where(self.grid == EMPTY)))

        if len(unsatisfied_coords) == 0 or len(empty_cells) == 0:
            return 0

        # Shuffle unsatisfied agents
        self.rng.shuffle(unsatisfied_coords)

        moved = 0
        for rc in unsatisfied_coords:
            if len(empty_cells) == 0:
                break
            r, c = rc[0], rc[1]

            idx = self.rng.integers(len(empty_cells))
            nr, nc = empty_cells[idx]

            self.grid[nr, nc] = self.grid[r, c]
            self.grid[r, c] = EMPTY

            # Move tolerance with the agent
            if self.tolerance_map is not None:
                self.tolerance_map[nr, nc] = self.tolerance_map[r, c]
                self.tolerance_map[r, c] = 0.0

            empty_cells[idx] = (r, c)
            moved += 1

        self.step_count += 1
        return moved

    def run(
        self,
        max_steps: int = 2000,
        record_every: int = 1,
        convergence_window: int = 20,
        convergence_threshold: float = 0.001,
    ) -> dict:
        """Exécute la simulation jusqu'à l'équilibre ou max_steps.

        La convergence est détectée soit quand aucun agent ne bouge, soit quand
        l'indice de ségrégation se stabilise (écart-type sur les dernières
        convergence_window étapes inférieur à convergence_threshold).

        Paramètres
        ----------
        max_steps : int
            Nombre maximal d'étapes.
        record_every : int
            Enregistre l'état de la grille toutes les N étapes.
        convergence_window : int
            Taille de la fenêtre glissante pour détecter la stabilisation.
        convergence_threshold : float
            Seuil d'écart-type de la ségrégation en dessous duquel on considère
            le système comme convergé.

        Retourne
        -------
        dict avec les clés : steps, moved_history, satisfaction_history, grids,
              converged, final_segregation
        """
        moved_history = []
        satisfaction_history = []
        seg_buffer = []
        grids = [self.grid.copy()]
        converged = False

        for s in range(max_steps):
            moved = self.step()
            moved_history.append(moved)
            satisfaction_history.append(self.mean_satisfaction())
            seg_buffer.append(segregation_index(self.grid))

            if s % record_every == 0:
                grids.append(self.grid.copy())

            if moved == 0:
                converged = True
                break

            if len(seg_buffer) >= convergence_window:
                recent = seg_buffer[-convergence_window:]
                if np.std(recent) < convergence_threshold:
                    converged = True
                    break

        return {
            "steps": len(moved_history),
            "moved_history": moved_history,
            "satisfaction_history": satisfaction_history,
            "grids": grids,
            "converged": converged,
            "final_segregation": segregation_index(self.grid),
        }


def satisfaction_score(grid: np.ndarray, tolerance: float = 0.5) -> float:
    """Calcule la satisfaction moyenne sur tous les agents de la grille.

    Fonction autonome qui ne nécessite pas d'instance de SchellingModel.
    """
    smap = _vectorized_satisfaction_map(grid)
    occupied_mask = grid != EMPTY
    if occupied_mask.sum() == 0:
        return 0.0
    return float(smap[occupied_mask].mean())


class HeterogeneousSchellingModel(SchellingModel):
    """Modèle de Schelling avec tolérance hétérogène.

    Chaque agent tire sa tolérance individuellement depuis une distribution
    Beta(alpha, beta). Cela modélise une population où les préférences de
    voisinage varient d'un individu à l'autre.

    Paramètres
    ----------
    alpha, beta : float
        Paramètres de la distribution Beta. La moyenne est alpha/(alpha+beta).
        Un alpha+beta grand concentre les tolérances autour de la moyenne ;
        un alpha+beta petit les étale.
    """

    def __init__(
        self,
        size: int = 50,
        density: float = 0.9,
        fraction_a: float = 0.5,
        alpha: float = 2.0,
        beta: float = 2.0,
        seed: Optional[int] = None,
    ):
        mean_tol = alpha / (alpha + beta)
        super().__init__(size=size, density=density, fraction_a=fraction_a,
                         tolerance=mean_tol, seed=seed)
        self.alpha = alpha
        self.beta = beta
        self.tolerance_map = self.rng.beta(alpha, beta, size=(size, size))
        # Les cellules vides n'ont pas de tolérance
        self.tolerance_map[self.grid == EMPTY] = 0.0

    def is_satisfied(self, r: int, c: int) -> bool:
        if self.grid[r, c] == EMPTY:
            return True
        return self.agent_satisfaction(r, c) >= self.tolerance_map[r, c]

    def step(self) -> int:
        smap = _vectorized_satisfaction_map(self.grid)
        occupied_mask = self.grid != EMPTY
        unsatisfied_mask = occupied_mask & (smap < self.tolerance_map)
        unsatisfied_coords = np.argwhere(unsatisfied_mask)

        empty_cells = list(zip(*np.where(self.grid == EMPTY)))
        if len(unsatisfied_coords) == 0 or len(empty_cells) == 0:
            return 0

        self.rng.shuffle(unsatisfied_coords)
        moved = 0
        for rc in unsatisfied_coords:
            if len(empty_cells) == 0:
                break
            r, c = rc[0], rc[1]
            idx = self.rng.integers(len(empty_cells))
            nr, nc = empty_cells[idx]

            self.grid[nr, nc] = self.grid[r, c]
            self.grid[r, c] = EMPTY
            self.tolerance_map[nr, nc] = self.tolerance_map[r, c]
            self.tolerance_map[r, c] = 0.0

            empty_cells[idx] = (r, c)
            moved += 1

        self.step_count += 1
        return moved


def interface_density(grid: np.ndarray) -> float:
    """Fraction of occupied neighbor pairs that are of different types.

    Standard order parameter in lattice segregation literature. Counts
    all occupied nearest-neighbor pairs (horizontal and vertical) where
    the two agents have different types, divided by total occupied pairs.
    Returns 0 for a fully segregated grid and approaches 0.5 for a
    well-mixed grid with equal populations.
    """
    occupied_mask = grid != EMPTY
    unlike = 0
    total = 0
    for dr, dc in [(0, 1), (1, 0)]:
        shifted = np.roll(np.roll(grid, -dr, axis=0), -dc, axis=1)
        both_occupied = occupied_mask & (shifted != EMPTY)
        total += both_occupied.sum()
        unlike += (both_occupied & (grid != shifted)).sum()
    if total == 0:
        return 0.0
    return float(unlike / total)


def segregation_index(grid: np.ndarray) -> float:
    """Calcule un indice de ségrégation basé sur la composition du voisinage.

    L'indice est (satisfaction_moyenne - satisfaction_attendue) / (1 - satisfaction_attendue),
    où la satisfaction_attendue est la fraction globale d'agents du même type.
    Cet indice varie de 0 (mélange aléatoire) à 1 (ségrégation complète).
    """
    size = grid.shape[0]
    occupied = grid[grid != EMPTY]
    if len(occupied) == 0:
        return 0.0

    n_a = np.sum(occupied == TYPE_A)
    n_b = np.sum(occupied == TYPE_B)
    n_total = n_a + n_b
    frac_a = n_a / n_total
    frac_b = n_b / n_total

    # Sous mélange aléatoire, fraction attendue du même type
    expected = frac_a**2 + frac_b**2

    mean_sat = satisfaction_score(grid)

    if expected >= 1.0:
        return 0.0

    return max(0.0, (mean_sat - expected) / (1.0 - expected))
