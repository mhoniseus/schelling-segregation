"""Fixtures partagées pour les tests de ségrégation de Schelling."""

import numpy as np
import pytest
from src.schelling import SchellingModel, TYPE_A, TYPE_B, EMPTY


@pytest.fixture
def rng():
    """Générateur aléatoire déterministe."""
    return np.random.default_rng(3407)


@pytest.fixture
def small_model():
    """Un petit modèle de Schelling 10x10 pour des tests rapides."""
    return SchellingModel(size=10, density=0.8, tolerance=0.5, seed=42)


@pytest.fixture
def checkerboard_grid():
    """Grille 6x6 parfaitement alternée (mélange maximal)."""
    grid = np.zeros((6, 6), dtype=int)
    for r in range(6):
        for c in range(6):
            grid[r, c] = 1 + ((r + c) % 2)
    return grid


@pytest.fixture
def segregated_grid():
    """Grille 6x6 entièrement ségrégée : moitié supérieure type A, moitié inférieure type B."""
    grid = np.zeros((6, 6), dtype=int)
    grid[:3, :] = TYPE_A
    grid[3:, :] = TYPE_B
    return grid


@pytest.fixture
def empty_border_grid():
    """Grille 8x8 avec des agents au centre et une bordure vide."""
    grid = np.zeros((8, 8), dtype=int)
    grid[2:6, 2:6] = TYPE_A
    grid[2:4, 4:6] = TYPE_B
    return grid
