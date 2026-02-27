"""Tests for phase diagram parameter sweeps, boundary extraction, and critical point analysis."""

import numpy as np
import pytest
from src.phase_diagram import (
    parameter_sweep,
    phase_boundary,
    convergence_sweep,
    size_scaling,
    extract_critical_point,
    _sigmoid,
    _detect_frozen_regime,
    binder_cumulant,
)
from src.schelling import TYPE_A, TYPE_B, EMPTY


class TestParameterSweep:
    """Tests for the 2D parameter sweep."""

    def test_output_keys(self):
        """parameter_sweep must return a dict with the expected keys."""
        tols = np.array([0.3, 0.5])
        dens = np.array([0.8, 0.9])
        result = parameter_sweep(tols, dens, size=8, max_steps=10, n_trials=1, seed=0)
        assert set(result.keys()) == {
            "tolerances", "densities", "segregation", "satisfaction", "convergence_steps"
        }

    def test_output_shapes(self):
        """Segregation map must have shape (n_tol, n_den)."""
        tols = np.array([0.3, 0.5, 0.7])
        dens = np.array([0.7, 0.8, 0.9])
        result = parameter_sweep(tols, dens, size=8, max_steps=10, n_trials=1, seed=0)
        assert result["segregation"].shape == (3, 3)
        assert result["satisfaction"].shape == (3, 3)
        assert result["convergence_steps"].shape == (3, 3)

    def test_segregation_values_in_range(self):
        """All segregation values must be in [0, 1]."""
        tols = np.array([0.3, 0.5, 0.7])
        dens = np.array([0.7, 0.9])
        result = parameter_sweep(tols, dens, size=10, max_steps=30, n_trials=2, seed=42)
        assert np.all(result["segregation"] >= 0.0)
        assert np.all(result["segregation"] <= 1.0)

    def test_satisfaction_values_in_range(self):
        """All satisfaction values must be in [0, 1]."""
        tols = np.array([0.3, 0.7])
        dens = np.array([0.8])
        result = parameter_sweep(tols, dens, size=10, max_steps=30, n_trials=1, seed=0)
        assert np.all(result["satisfaction"] >= 0.0)
        assert np.all(result["satisfaction"] <= 1.0)

    def test_convergence_steps_positive(self):
        """Convergence steps must be positive integers."""
        tols = np.array([0.3, 0.5])
        dens = np.array([0.8])
        result = parameter_sweep(tols, dens, size=8, max_steps=20, n_trials=1, seed=0)
        assert np.all(result["convergence_steps"] >= 1)

    def test_deterministic_with_seed(self):
        """Same seed must produce identical results."""
        tols = np.array([0.3, 0.5])
        dens = np.array([0.8])
        r1 = parameter_sweep(tols, dens, size=8, max_steps=20, n_trials=2, seed=123)
        r2 = parameter_sweep(tols, dens, size=8, max_steps=20, n_trials=2, seed=123)
        assert np.array_equal(r1["segregation"], r2["segregation"])

    def test_low_tolerance_low_segregation(self):
        """Very low tolerance (everyone is easily satisfied) should produce low segregation."""
        tols = np.array([0.05])
        dens = np.array([0.9])
        result = parameter_sweep(tols, dens, size=15, max_steps=50, n_trials=2, seed=0)
        assert result["segregation"][0, 0] < 0.3


class TestPhaseBoundary:
    """Tests for phase boundary extraction."""

    def test_boundary_on_crossing(self):
        """Boundary should be found where segregation crosses the threshold."""
        tols = np.array([0.2, 0.4, 0.6, 0.8])
        dens = np.array([0.9])
        seg_map = np.array([[0.1], [0.35], [0.6], [0.9]])
        boundary = phase_boundary(tols, dens, seg_map, threshold=0.3)
        assert len(boundary) >= 1
        tol_cross = boundary[0][0]
        assert 0.2 <= tol_cross <= 0.4

    def test_no_boundary_below_threshold(self):
        """No boundary if all values are below the threshold."""
        tols = np.array([0.2, 0.4, 0.6])
        dens = np.array([0.9])
        seg_map = np.array([[0.05], [0.1], [0.15]])
        boundary = phase_boundary(tols, dens, seg_map, threshold=0.5)
        assert len(boundary) == 0

    def test_no_boundary_above_threshold(self):
        """No boundary if all values are above the threshold."""
        tols = np.array([0.2, 0.4, 0.6])
        dens = np.array([0.9])
        seg_map = np.array([[0.7], [0.8], [0.9]])
        boundary = phase_boundary(tols, dens, seg_map, threshold=0.3)
        assert len(boundary) == 0

    def test_multiple_densities(self):
        """Boundary should be extracted for each density column independently."""
        tols = np.array([0.2, 0.4, 0.6])
        dens = np.array([0.7, 0.9])
        seg_map = np.array([
            [0.1, 0.2],
            [0.5, 0.6],
            [0.8, 0.9],
        ])
        boundary = phase_boundary(tols, dens, seg_map, threshold=0.4)
        assert len(boundary) == 2

    def test_boundary_returns_tuples(self):
        """Each boundary point must be a (tolerance, density) tuple of floats."""
        tols = np.array([0.3, 0.5])
        dens = np.array([0.8])
        seg_map = np.array([[0.1], [0.6]])
        boundary = phase_boundary(tols, dens, seg_map, threshold=0.3)
        assert len(boundary) >= 1
        for point in boundary:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)


class TestConvergenceSweep:
    """Tests for convergence sweep across tolerances."""

    def test_output_keys(self):
        """convergence_sweep must return a dict with the expected keys."""
        tols = np.array([0.3, 0.5])
        result = convergence_sweep(tols, size=8, max_steps=20, n_trials=1, seed=0)
        assert set(result.keys()) == {
            "tolerances", "mean_steps", "mean_segregation",
            "mean_satisfaction", "fraction_converged",
        }

    def test_output_lengths(self):
        """All output arrays must match the number of tolerances."""
        tols = np.array([0.3, 0.5, 0.7])
        result = convergence_sweep(tols, size=8, max_steps=20, n_trials=1, seed=0)
        for key in ["mean_steps", "mean_segregation", "mean_satisfaction", "fraction_converged"]:
            assert len(result[key]) == 3

    def test_fraction_converged_range(self):
        """Fraction converged must be in [0, 1]."""
        tols = np.array([0.1, 0.5, 0.9])
        result = convergence_sweep(tols, size=10, max_steps=50, n_trials=2, seed=0)
        assert np.all(result["fraction_converged"] >= 0.0)
        assert np.all(result["fraction_converged"] <= 1.0)

    def test_zero_tolerance_converges_fast(self):
        """Tolerance=0 means everyone is satisfied immediately, so steps should be minimal."""
        tols = np.array([0.0])
        result = convergence_sweep(tols, size=10, max_steps=100, n_trials=2, seed=0)
        assert result["mean_steps"][0] <= 2
        assert result["fraction_converged"][0] == 1.0


class TestSizeScaling:
    """Tests for finite-size scaling analysis."""

    def test_output_keys(self):
        """size_scaling must return a dict with the expected keys."""
        result = size_scaling([5, 8], tolerance=0.5, max_steps=20, n_trials=1, seed=0)
        assert set(result.keys()) == {"sizes", "segregation", "convergence_steps"}

    def test_output_lengths(self):
        """All output arrays must match the number of sizes."""
        sizes = [5, 8, 10]
        result = size_scaling(sizes, tolerance=0.5, max_steps=20, n_trials=1, seed=0)
        assert len(result["sizes"]) == 3
        assert len(result["segregation"]) == 3
        assert len(result["convergence_steps"]) == 3

    def test_segregation_in_range(self):
        """Segregation values must be in [0, 1]."""
        result = size_scaling([8, 12], tolerance=0.5, max_steps=30, n_trials=2, seed=42)
        assert np.all(result["segregation"] >= 0.0)
        assert np.all(result["segregation"] <= 1.0)

    def test_deterministic_with_seed(self):
        """Same seed must produce identical results."""
        r1 = size_scaling([6, 10], tolerance=0.5, max_steps=20, n_trials=2, seed=99)
        r2 = size_scaling([6, 10], tolerance=0.5, max_steps=20, n_trials=2, seed=99)
        assert np.array_equal(r1["segregation"], r2["segregation"])


class TestSigmoid:
    """Tests for the sigmoid fitting function."""

    def test_sigmoid_shape(self):
        """Sigmoid should produce values between B and B+A."""
        T = np.linspace(0, 1, 50)
        y = _sigmoid(T, B=0.0, A=1.0, Tc=0.5, w=0.05)
        assert np.all(y >= -0.01)
        assert np.all(y <= 1.01)

    def test_sigmoid_midpoint(self):
        """At T=Tc, sigmoid should equal B + A/2."""
        y = _sigmoid(np.array([0.5]), B=0.1, A=0.8, Tc=0.5, w=0.05)
        assert np.isclose(y[0], 0.1 + 0.8 / 2, atol=1e-6)


class TestExtractCriticalPoint:
    """Tests for critical point extraction via sigmoid fit."""

    def test_synthetic_sigmoid(self):
        """Should recover T_c from synthetic sigmoid data."""
        T = np.linspace(0.1, 0.9, 40)
        true_Tc = 0.45
        y = _sigmoid(T, B=0.05, A=0.9, Tc=true_Tc, w=0.04)
        # Add small noise
        rng = np.random.default_rng(0)
        y_noisy = y + rng.normal(0, 0.01, len(y))
        result = extract_critical_point(T, y_noisy)
        assert abs(result["T_c"] - true_Tc) < 0.05
        assert result["T_c_err"] < 0.1
        assert result["width"] > 0

    def test_flat_data_fallback(self):
        """Flat data should not crash, should use fallback."""
        T = np.linspace(0.1, 0.9, 10)
        y = np.ones(10) * 0.5
        result = extract_critical_point(T, y)
        assert "T_c" in result
        assert "T_c_err" in result

    def test_frozen_regime_excluded(self):
        """Sigmoid fit should handle non-monotonic data (frozen regime)."""
        T = np.linspace(0.1, 0.9, 20)
        # Sigmoid up to 0.7, then drops (frozen regime)
        y = _sigmoid(T, B=0.0, A=1.0, Tc=0.4, w=0.04)
        frozen_mask = T > 0.7
        y[frozen_mask] = y[frozen_mask] * np.linspace(1.0, 0.3, frozen_mask.sum())
        result = extract_critical_point(T, y)
        assert abs(result["T_c"] - 0.4) < 0.1
        assert "frozen_idx" in result


class TestFrozenRegimeDetection:
    """Tests for frozen regime detection."""

    def test_monotonic_no_freeze(self):
        """Monotonic data should report no frozen regime."""
        T = np.linspace(0.1, 0.9, 20)
        S = np.linspace(0.0, 1.0, 20)
        idx = _detect_frozen_regime(T, S)
        assert idx == len(T)

    def test_peak_then_drop(self):
        """Non-monotonic data should detect frozen regime."""
        T = np.linspace(0.1, 0.9, 20)
        S = np.concatenate([np.linspace(0.0, 0.9, 14), np.linspace(0.85, 0.4, 6)])
        idx = _detect_frozen_regime(T, S)
        assert idx < len(T)


class TestBinderCumulant:
    """Tests for Binder cumulant computation."""

    def test_output_structure(self):
        """Binder cumulant must return dict with U4 per size."""
        sizes = [6, 8]
        T = np.array([0.3, 0.5])
        result = binder_cumulant(sizes, T, n_trials=3, max_steps=20, seed=0)
        assert set(result.keys()) == {"sizes", "tolerances", "U4", "U4_err"}
        for sz in sizes:
            assert sz in result["U4"]
            assert len(result["U4"][sz]) == len(T)

    def test_u4_range(self):
        """U4 should typically be between 0 and 2/3."""
        sizes = [8]
        T = np.array([0.3, 0.5, 0.7])
        result = binder_cumulant(sizes, T, n_trials=5, max_steps=30, seed=42)
        u4 = result["U4"][8]
        assert np.all(u4 >= -0.5)
        assert np.all(u4 <= 1.0)
