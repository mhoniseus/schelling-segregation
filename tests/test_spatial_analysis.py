"""Tests pour le calcul de trajectoires multi-échelles et les métriques de dissimilarité."""

import numpy as np
import pytest
from src.spatial_analysis import (
    local_fraction,
    spatial_dissimilarity,
    exposure_index,
    multiscalar_trajectory,
    trajectory_area,
    trajectory_slope,
    trajectory_statistics,
    null_model_trajectory,
)
from src.schelling import TYPE_A, TYPE_B, EMPTY


class TestLocalFraction:
    """Tests pour le calcul de la fraction locale."""

    def test_uniform_grid_fraction(self):
        """Sur une grille uniforme (tout type A), la fraction locale doit être 1 partout."""
        grid = np.ones((10, 10), dtype=int)  # tout TYPE_A
        frac = local_fraction(grid, TYPE_A, radius=2)
        assert np.allclose(frac, 1.0, atol=1e-6)

    def test_checkerboard_fraction(self, checkerboard_grid):
        """Sur un damier, la fraction locale doit être proche de 0.5."""
        frac = local_fraction(checkerboard_grid, TYPE_A, radius=3)
        assert np.all(frac > 0.2)
        assert np.all(frac < 0.8)

    def test_empty_grid(self):
        """Sur une grille entièrement vide, la fraction locale doit être 0."""
        grid = np.zeros((8, 8), dtype=int)
        frac = local_fraction(grid, TYPE_A, radius=2)
        assert np.allclose(frac, 0.0)

    def test_fraction_in_range(self, rng):
        """Les fractions locales doivent toujours être entre 0 et 1."""
        grid = rng.choice([EMPTY, TYPE_A, TYPE_B], size=(15, 15))
        frac = local_fraction(grid, TYPE_A, radius=3)
        assert np.all(frac >= 0.0)
        assert np.all(frac <= 1.0)


class TestSpatialDissimilarity:
    """Tests pour l'indice de dissimilarité spatiale."""

    def test_uniform_grid_zero_dissimilarity(self):
        """Une grille avec un seul type doit avoir une dissimilarité de 0."""
        grid = np.ones((10, 10), dtype=int)
        d = spatial_dissimilarity(grid, TYPE_A, radius=3)
        assert d == 0.0

    def test_checkerboard_low_dissimilarity(self, checkerboard_grid):
        """Un damier bien mélangé doit avoir une faible dissimilarité à grand rayon."""
        d = spatial_dissimilarity(checkerboard_grid, TYPE_A, radius=3)
        assert d < 0.5

    def test_segregated_high_dissimilarity(self, segregated_grid):
        """Une grille ségrégée doit avoir une forte dissimilarité à petit rayon."""
        d = spatial_dissimilarity(segregated_grid, TYPE_A, radius=1)
        assert d > 0.2

    def test_dissimilarity_decreases_with_radius(self, segregated_grid):
        """La dissimilarité doit généralement décroître avec un rayon plus grand."""
        d1 = spatial_dissimilarity(segregated_grid, TYPE_A, radius=1)
        d3 = spatial_dissimilarity(segregated_grid, TYPE_A, radius=3)
        assert d1 >= d3 - 0.1


class TestExposureIndex:
    """Tests pour l'indice d'exposition."""

    def test_self_exposure_uniform(self):
        """L'auto-exposition sur une grille uniforme doit être 1."""
        grid = np.ones((10, 10), dtype=int)
        e = exposure_index(grid, TYPE_A, TYPE_A, radius=2)
        assert np.isclose(e, 1.0, atol=1e-3)

    def test_cross_exposure_checkerboard(self, checkerboard_grid):
        """L'exposition croisée sur un damier doit être proche de 0.5."""
        e = exposure_index(checkerboard_grid, TYPE_A, TYPE_B, radius=3)
        assert 0.2 < e < 0.8

    def test_exposure_range(self, rng):
        """L'indice d'exposition doit être entre 0 et 1."""
        grid = rng.choice([TYPE_A, TYPE_B], size=(12, 12))
        e = exposure_index(grid, TYPE_A, TYPE_B, radius=2)
        assert 0.0 <= e <= 1.0


class TestMultiscalarTrajectory:
    """Tests pour le calcul de la trajectoire multi-échelles."""

    def test_output_shape(self, segregated_grid):
        """La trajectoire doit avoir des longueurs correspondantes."""
        radii, diss = multiscalar_trajectory(segregated_grid, TYPE_A, max_radius=3)
        assert len(radii) == len(diss)
        assert len(radii) == 3

    def test_trajectory_values_in_range(self, segregated_grid):
        """Les valeurs de dissimilarité doivent être entre 0 et 1."""
        radii, diss = multiscalar_trajectory(segregated_grid, TYPE_A, max_radius=3)
        assert np.all(diss >= 0.0)
        assert np.all(diss <= 1.0)

    def test_explicit_radii(self, checkerboard_grid):
        """Doit accepter un tableau explicite de rayons."""
        radii_in = np.array([1, 2])
        radii, diss = multiscalar_trajectory(checkerboard_grid, TYPE_A, radii=radii_in)
        assert np.array_equal(radii, radii_in)
        assert len(diss) == 2


class TestTrajectoryMetrics:
    """Tests pour l'aire et la pente de la trajectoire."""

    def test_area_positive(self):
        """L'aire sous une trajectoire positive doit être positive."""
        radii = np.array([1, 2, 3, 4])
        diss = np.array([0.8, 0.6, 0.4, 0.2])
        area = trajectory_area(radii, diss)
        assert area > 0

    def test_area_zero_flat(self):
        """L'aire sous une trajectoire plate à zéro doit être 0."""
        radii = np.array([1, 2, 3])
        diss = np.array([0.0, 0.0, 0.0])
        area = trajectory_area(radii, diss)
        assert area == 0.0

    def test_slope_negative_for_decay(self):
        """La pente doit être négative pour une trajectoire décroissante."""
        radii = np.array([1, 2, 4, 8])
        diss = np.array([0.8, 0.5, 0.25, 0.1])
        slope = trajectory_slope(radii, diss)
        assert slope < 0

    def test_slope_handles_short_array(self):
        """La pente avec moins de 2 points doit retourner 0."""
        slope = trajectory_slope(np.array([1]), np.array([0.5]))
        assert slope == 0.0


class TestTrajectoryStatistics:
    """Tests pour le résumé statistique d'une trajectoire."""

    def test_output_keys(self):
        """trajectory_statistics doit retourner les clés attendues."""
        radii = np.array([1, 2, 3, 4, 5])
        diss = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
        stats = trajectory_statistics(radii, diss)
        assert set(stats.keys()) == {
            "area", "slope", "characteristic_length", "D_local", "D_global",
        }

    def test_D_local_D_global(self):
        """D_local and D_global should match first and last values."""
        radii = np.array([1, 2, 3])
        diss = np.array([0.9, 0.5, 0.1])
        stats = trajectory_statistics(radii, diss)
        assert np.isclose(stats["D_local"], 0.9)
        assert np.isclose(stats["D_global"], 0.1)

    def test_characteristic_length_reasonable(self):
        """Characteristic length should be between first and last radius."""
        radii = np.array([1, 2, 3, 4, 5])
        diss = np.array([0.8, 0.6, 0.3, 0.15, 0.05])
        stats = trajectory_statistics(radii, diss)
        assert 1.0 <= stats["characteristic_length"] <= 5.0


class TestNullModelTrajectory:
    """Tests pour la trajectoire du modèle nul."""

    def test_output_shapes(self):
        """null_model_trajectory doit retourner des tableaux de même longueur."""
        radii, mean_D, std_D = null_model_trajectory(
            size=10, n_samples=3, max_radius=3, seed=0
        )
        assert len(radii) == len(mean_D) == len(std_D)
        assert len(radii) == 3

    def test_null_low_dissimilarity(self):
        """Random grids should have low dissimilarity (close to well-mixed)."""
        radii, mean_D, std_D = null_model_trajectory(
            size=15, n_samples=5, max_radius=3, seed=42
        )
        # Random placement should give D(r) close to 0
        assert np.all(mean_D < 0.3)

    def test_std_positive(self):
        """Standard deviation should be non-negative."""
        radii, mean_D, std_D = null_model_trajectory(
            size=10, n_samples=5, max_radius=3, seed=0
        )
        assert np.all(std_D >= 0.0)
