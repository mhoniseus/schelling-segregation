"""Ségrégation de Schelling, diagrammes de phase et analyse de complexité spatiale."""

from .schelling import (
    SchellingModel,
    HeterogeneousSchellingModel,
    satisfaction_score,
    segregation_index,
    TYPE_A,
    TYPE_B,
    EMPTY,
)
from .spatial_analysis import (
    multiscalar_trajectory,
    spatial_dissimilarity,
    exposure_index,
    trajectory_area,
    trajectory_slope,
    trajectory_statistics,
    null_model_trajectory,
    systematic_trajectory_sweep,
)
from .phase_diagram import (
    parameter_sweep,
    phase_boundary,
    convergence_sweep,
    size_scaling,
    extract_critical_point,
    finite_size_scaling,
    critical_exponents,
    scaling_collapse,
    compare_homogeneous_heterogeneous,
)
from .utils import plot_grid, plot_trajectory, generate_checkerboard, generate_random_grid
