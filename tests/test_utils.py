"""Tests pour les fonctions utilitaires : génération de grilles et aides à la visualisation."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # backend non interactif pour les tests
import matplotlib.pyplot as plt

from src.utils import (
    generate_checkerboard,
    generate_random_grid,
    generate_clustered_grid,
    plot_grid,
    plot_trajectory,
)
from src.schelling import TYPE_A, TYPE_B, EMPTY


class TestGenerateCheckerboard:
    """Tests pour la génération de grille en damier."""

    def test_shape(self):
        """La sortie doit correspondre à la taille demandée."""
        grid = generate_checkerboard(size=12)
        assert grid.shape == (12, 12)

    def test_alternating_pattern(self):
        """Les cellules adjacentes doivent avoir des types différents (pas de cellules vides)."""
        grid = generate_checkerboard(size=8)
        for r in range(8):
            for c in range(7):
                assert grid[r, c] != grid[r, c + 1]

    def test_no_empty_by_default(self):
        """Le damier par défaut ne doit pas avoir de cellules vides."""
        grid = generate_checkerboard(size=10)
        assert (grid == EMPTY).sum() == 0

    def test_empty_fraction(self):
        """Avec empty_frac > 0, certaines cellules doivent être vides."""
        grid = generate_checkerboard(size=10, empty_frac=0.2, seed=0)
        n_empty = (grid == EMPTY).sum()
        assert n_empty == int(100 * 0.2)

    def test_only_valid_values(self):
        """La grille ne doit contenir que EMPTY, TYPE_A, TYPE_B."""
        grid = generate_checkerboard(size=10, empty_frac=0.1, seed=0)
        unique = set(np.unique(grid))
        assert unique.issubset({EMPTY, TYPE_A, TYPE_B})


class TestGenerateRandomGrid:
    """Tests pour la génération de grille aléatoire."""

    def test_shape(self):
        """La sortie doit correspondre à la taille demandée."""
        grid = generate_random_grid(size=20, seed=0)
        assert grid.shape == (20, 20)

    def test_density(self):
        """La fraction occupée doit correspondre à la densité demandée."""
        grid = generate_random_grid(size=20, density=0.75, seed=0)
        occupied = (grid != EMPTY).sum()
        expected = int(400 * 0.75)
        assert occupied == expected

    def test_type_ratio(self):
        """La fraction de type A parmi les cellules occupées doit correspondre à fraction_a."""
        grid = generate_random_grid(size=20, density=0.8, fraction_a=0.6, seed=0)
        occupied = grid[grid != EMPTY]
        n_a = (occupied == TYPE_A).sum()
        expected = int(int(400 * 0.8) * 0.6)
        assert n_a == expected

    def test_deterministic_with_seed(self):
        """La même graine doit produire des grilles identiques."""
        g1 = generate_random_grid(size=15, seed=42)
        g2 = generate_random_grid(size=15, seed=42)
        assert np.array_equal(g1, g2)


class TestGenerateClusteredGrid:
    """Tests pour la génération de grille avec grappes."""

    def test_shape(self):
        """La sortie doit correspondre à la taille demandée."""
        grid = generate_clustered_grid(size=20, seed=0)
        assert grid.shape == (20, 20)

    def test_has_both_types(self):
        """La grille doit contenir à la fois le type A et le type B."""
        grid = generate_clustered_grid(size=20, density=0.9, seed=0)
        assert (grid == TYPE_A).sum() > 0
        assert (grid == TYPE_B).sum() > 0

    def test_density_approximate(self):
        """La fraction occupée doit être approximativement correcte."""
        grid = generate_clustered_grid(size=20, density=0.7, seed=0)
        actual_density = (grid != EMPTY).sum() / 400.0
        assert abs(actual_density - 0.7) < 0.05


class TestPlotGrid:
    """Tests pour la visualisation de grilles."""

    def test_returns_figure(self):
        """plot_grid doit retourner une Figure matplotlib."""
        grid = generate_checkerboard(size=5)
        fig = plot_grid(grid, title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_axes(self):
        """plot_grid doit fonctionner avec des axes fournis."""
        fig, ax = plt.subplots()
        grid = generate_random_grid(size=5, seed=0)
        result_fig = plot_grid(grid, ax=ax)
        assert result_fig is fig
        plt.close(fig)


class TestPlotTrajectory:
    """Tests pour la visualisation de trajectoires."""

    def test_returns_figure(self):
        """plot_trajectory doit retourner une Figure matplotlib."""
        radii = np.array([1, 2, 3, 4])
        diss = np.array([0.8, 0.6, 0.4, 0.2])
        fig = plot_trajectory(radii, diss, label="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_loglog_mode(self):
        """plot_trajectory doit fonctionner en mode log-log."""
        radii = np.array([1, 2, 4, 8])
        diss = np.array([0.8, 0.5, 0.3, 0.1])
        fig = plot_trajectory(radii, diss, loglog=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
