"""Tests pour le modèle de ségrégation de Schelling."""

import numpy as np
import pytest
from src.schelling import (
    SchellingModel,
    HeterogeneousSchellingModel,
    satisfaction_score,
    segregation_index,
    interface_density,
    _vectorized_satisfaction_map,
    TYPE_A,
    TYPE_B,
    EMPTY,
)


class TestModelInitialization:
    """Tests pour le constructeur de SchellingModel et l'initialisation de la grille."""

    def test_grid_shape(self, small_model):
        """La grille doit avoir les bonnes dimensions."""
        assert small_model.grid.shape == (10, 10)

    def test_density(self):
        """La fraction occupée doit correspondre à la densité demandée."""
        model = SchellingModel(size=20, density=0.7, seed=0)
        occupied = (model.grid != EMPTY).sum()
        expected = int(20 * 20 * 0.7)
        assert occupied == expected

    def test_type_ratio(self):
        """Le ratio du type A par rapport au total occupé doit correspondre à fraction_a."""
        model = SchellingModel(size=20, density=0.8, fraction_a=0.6, seed=0)
        occupied = model.grid[model.grid != EMPTY]
        n_a = (occupied == TYPE_A).sum()
        n_total = len(occupied)
        expected_a = int(int(400 * 0.8) * 0.6)
        assert n_a == expected_a

    def test_deterministic_with_seed(self):
        """La même graine doit produire des grilles identiques."""
        m1 = SchellingModel(size=15, density=0.9, seed=123)
        m2 = SchellingModel(size=15, density=0.9, seed=123)
        assert np.array_equal(m1.grid, m2.grid)

    def test_different_seeds_differ(self):
        """Des graines différentes doivent (presque certainement) produire des grilles différentes."""
        m1 = SchellingModel(size=15, density=0.9, seed=0)
        m2 = SchellingModel(size=15, density=0.9, seed=1)
        assert not np.array_equal(m1.grid, m2.grid)

    def test_empty_grid(self):
        """Une densité de 0 doit produire une grille entièrement vide."""
        model = SchellingModel(size=5, density=0.0, seed=0)
        assert (model.grid == EMPTY).all()


class TestSatisfaction:
    """Tests pour le calcul de la satisfaction."""

    def test_checkerboard_moderate_satisfaction(self, checkerboard_grid):
        """Le damier a la moitié des voisins du même type (diagonales), donc satisfaction = 0.5."""
        model = SchellingModel(size=6, density=1.0, tolerance=0.5, seed=0)
        model.grid = checkerboard_grid
        mean_sat = model.mean_satisfaction()
        assert np.isclose(mean_sat, 0.5, atol=0.01)

    def test_segregated_high_satisfaction(self, segregated_grid):
        """Une grille entièrement ségrégée doit avoir une satisfaction élevée."""
        model = SchellingModel(size=6, density=1.0, tolerance=0.5, seed=0)
        model.grid = segregated_grid
        mean_sat = model.mean_satisfaction()
        assert mean_sat > 0.7

    def test_satisfaction_range(self, small_model):
        """La satisfaction individuelle doit être entre 0 et 1."""
        smap = small_model.satisfaction_map()
        occupied_mask = small_model.grid != EMPTY
        assert np.all(smap[occupied_mask] >= 0.0)
        assert np.all(smap[occupied_mask] <= 1.0)

    def test_empty_cell_satisfaction(self, small_model):
        """Les cellules vides doivent avoir une satisfaction de 0."""
        smap = small_model.satisfaction_map()
        empty_mask = small_model.grid == EMPTY
        if empty_mask.any():
            assert np.all(smap[empty_mask] == 0.0)

    def test_isolated_agent_satisfied(self):
        """Un agent isolé (tous les voisins vides) doit avoir une satisfaction de 1."""
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 2] = TYPE_A
        model = SchellingModel(size=5, density=1.0, seed=0)
        model.grid = grid
        assert model.agent_satisfaction(2, 2) == 1.0


class TestDynamics:
    """Tests pour l'étape de simulation et l'exécution."""

    def test_step_returns_moved_count(self, small_model):
        """step() doit retourner un entier non négatif."""
        moved = small_model.step()
        assert isinstance(moved, (int, np.integer))
        assert moved >= 0

    def test_step_conserves_population(self, small_model):
        """Le nombre total d'agents ne doit pas changer après une étape."""
        before = (small_model.grid != EMPTY).sum()
        small_model.step()
        after = (small_model.grid != EMPTY).sum()
        assert before == after

    def test_step_conserves_types(self, small_model):
        """Le nombre de chaque type ne doit pas changer après une étape."""
        a_before = (small_model.grid == TYPE_A).sum()
        b_before = (small_model.grid == TYPE_B).sum()
        small_model.step()
        a_after = (small_model.grid == TYPE_A).sum()
        b_after = (small_model.grid == TYPE_B).sum()
        assert a_before == a_after
        assert b_before == b_after

    def test_run_returns_dict(self, small_model):
        """run() doit retourner un dict avec les clés attendues."""
        result = small_model.run(max_steps=10)
        expected_keys = {
            "steps", "moved_history", "satisfaction_history",
            "grids", "converged", "final_segregation",
        }
        assert set(result.keys()) == expected_keys

    def test_run_converged_flag(self):
        """La convergence doit être détectée quand aucun agent ne bouge."""
        model = SchellingModel(size=10, density=0.8, tolerance=0.0, seed=0)
        result = model.run(max_steps=100)
        assert result["converged"] is True

    def test_run_final_segregation(self):
        """final_segregation doit être un float dans [0, 1]."""
        model = SchellingModel(size=10, density=0.9, tolerance=0.5, seed=42)
        result = model.run(max_steps=50)
        assert 0.0 <= result["final_segregation"] <= 1.0

    def test_equilibrium_all_satisfied(self):
        """Une faible tolérance doit atteindre l'équilibre rapidement (tous les agents satisfaits)."""
        model = SchellingModel(size=10, density=0.8, tolerance=0.0, seed=0)
        result = model.run(max_steps=100)
        assert result["steps"] == 1
        assert result["moved_history"][0] == 0

    def test_convergence_increases_satisfaction(self):
        """L'exécution du modèle ne doit pas diminuer la satisfaction moyenne."""
        model = SchellingModel(size=15, density=0.9, tolerance=0.5, seed=42)
        sat_before = model.mean_satisfaction()
        model.run(max_steps=50)
        sat_after = model.mean_satisfaction()
        assert sat_after >= sat_before - 0.01


class TestHeterogeneousModel:
    """Tests pour le modèle de Schelling hétérogène."""

    def test_initialization(self):
        """Le modèle hétérogène doit s'initialiser correctement."""
        model = HeterogeneousSchellingModel(size=10, density=0.9, alpha=2.0, beta=2.0, seed=42)
        assert model.grid.shape == (10, 10)
        assert model.tolerance_map.shape == (10, 10)
        assert model.alpha == 2.0
        assert model.beta == 2.0

    def test_tolerance_map_range(self):
        """Les tolérances individuelles doivent être dans [0, 1]."""
        model = HeterogeneousSchellingModel(size=15, density=0.9, alpha=2.0, beta=2.0, seed=0)
        occupied = model.grid != EMPTY
        assert np.all(model.tolerance_map[occupied] >= 0.0)
        assert np.all(model.tolerance_map[occupied] <= 1.0)

    def test_empty_cells_zero_tolerance(self):
        """Les cellules vides doivent avoir une tolérance de 0."""
        model = HeterogeneousSchellingModel(size=10, density=0.8, alpha=2.0, beta=2.0, seed=0)
        empty = model.grid == EMPTY
        assert np.all(model.tolerance_map[empty] == 0.0)

    def test_step_conserves_population(self):
        """La population doit être conservée après une étape."""
        model = HeterogeneousSchellingModel(size=10, density=0.9, alpha=2.0, beta=2.0, seed=42)
        before = (model.grid != EMPTY).sum()
        model.step()
        after = (model.grid != EMPTY).sum()
        assert before == after

    def test_tolerance_moves_with_agent(self):
        """Quand un agent se déplace, sa tolérance doit le suivre."""
        model = HeterogeneousSchellingModel(size=10, density=0.8, alpha=5.0, beta=5.0, seed=42)
        # Record tolerances of occupied cells before step
        occupied_before = model.grid != EMPTY
        tol_values_before = sorted(model.tolerance_map[occupied_before].tolist())
        model.step()
        occupied_after = model.grid != EMPTY
        tol_values_after = sorted(model.tolerance_map[occupied_after].tolist())
        # Same set of tolerance values (just relocated)
        assert np.allclose(tol_values_before, tol_values_after, atol=1e-10)

    def test_run_returns_dict(self):
        """run() doit retourner un dict avec les clés attendues."""
        model = HeterogeneousSchellingModel(size=8, density=0.9, alpha=2.0, beta=2.0, seed=0)
        result = model.run(max_steps=20)
        assert "converged" in result
        assert "final_segregation" in result

    def test_high_concentration_behaves_like_homogeneous(self):
        """Concentration très élevée (kappa grand) doit approcher le modèle homogène."""
        # kappa=100 -> Beta(50, 50) concentré autour de 0.5
        model_het = HeterogeneousSchellingModel(
            size=15, density=0.9, alpha=50.0, beta=50.0, seed=42
        )
        model_hom = SchellingModel(size=15, density=0.9, tolerance=0.5, seed=42)
        model_het.run(max_steps=200)
        model_hom.run(max_steps=200)
        seg_het = segregation_index(model_het.grid)
        seg_hom = segregation_index(model_hom.grid)
        # Should be in the same ballpark (both produce segregation)
        assert abs(seg_het - seg_hom) < 0.3


class TestStandaloneFunctions:
    """Tests pour les fonctions autonomes satisfaction_score et segregation_index."""

    def test_satisfaction_score_matches_model(self, small_model):
        """La fonction autonome doit correspondre à la méthode du modèle."""
        model_sat = small_model.mean_satisfaction()
        func_sat = satisfaction_score(small_model.grid)
        assert np.isclose(model_sat, func_sat, atol=1e-10)

    def test_segregation_index_checkerboard(self, checkerboard_grid):
        """Le damier doit avoir un indice de ségrégation proche de 0."""
        idx = segregation_index(checkerboard_grid)
        assert idx < 0.1

    def test_segregation_index_segregated(self, segregated_grid):
        """Une grille ségrégée doit avoir un indice de ségrégation positif."""
        idx = segregation_index(segregated_grid)
        assert idx > 0.3

    def test_segregation_index_range(self, small_model):
        """L'indice de ségrégation doit être entre 0 et 1."""
        idx = segregation_index(small_model.grid)
        assert 0.0 <= idx <= 1.0


class TestInterfaceDensity:
    """Tests pour la densité d'interface."""

    def test_checkerboard_high_interface(self, checkerboard_grid):
        """Le damier doit avoir une densité d'interface maximale (toutes les paires sont différentes)."""
        rho = interface_density(checkerboard_grid)
        assert rho > 0.9

    def test_segregated_low_interface(self, segregated_grid):
        """Une grille ségrégée doit avoir une faible densité d'interface."""
        rho = interface_density(segregated_grid)
        assert rho < 0.2

    def test_interface_range(self, small_model):
        """La densité d'interface doit être entre 0 et 1."""
        rho = interface_density(small_model.grid)
        assert 0.0 <= rho <= 1.0

    def test_uniform_grid_zero_interface(self):
        """Une grille d'un seul type doit avoir une densité d'interface de 0."""
        grid = np.full((5, 5), TYPE_A, dtype=int)
        assert interface_density(grid) == 0.0

    def test_empty_grid_zero(self):
        """Une grille vide doit retourner 0."""
        grid = np.zeros((5, 5), dtype=int)
        assert interface_density(grid) == 0.0


class TestVectorizedSatisfaction:
    """Tests that the vectorized satisfaction matches the scalar version."""

    def test_matches_scalar(self):
        """Vectorized satisfaction must match cell-by-cell computation."""
        from src.schelling import _cell_satisfaction
        model = SchellingModel(size=12, density=0.85, tolerance=0.5, seed=99)
        grid = model.grid
        vec_map = _vectorized_satisfaction_map(grid)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                expected = _cell_satisfaction(grid, r, c)
                assert np.isclose(vec_map[r, c], expected, atol=1e-12), \
                    f"Mismatch at ({r},{c}): vec={vec_map[r,c]}, scalar={expected}"
