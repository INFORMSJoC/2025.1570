"""
Robust Quasi-Concave Optimization package.
"""

from .core import (
    true_function,
    partial_derivatives,
    compute_max_lipschitz,
    generate_samples,
    create_mesh_grid,
    DEFAULT_X_MIN,
    DEFAULT_X_MAX,
    DEFAULT_N_FEATURES,
    DEFAULT_L,
    DEFAULT_ALPHA,
    DEFAULT_COST,
    DEFAULT_A,
    DEFAULT_B,
)

from .interpolation import (
    worst_case_interpolation,
    piecewise_constant_interpolation,
    concave_regression,
    milp_interpolation_problem,
    sos1_interpolation_problem,
    interpolation_mesh,
)

from .optimization import (
    robust_optimization,
    piecewise_constant_optimization,
    true_optimization,
    concave_regression_optimization,
    level_function_method_optimization,
)

from .visualization import (
    plot_true_function_3d,
    plot_true_function_contour,
    plot_interpolation_3d,
    plot_contour_comparison,
    plot_l1_error,
)

# Max-Min Utility Problem modules
from .maxmin_utility import (
    sigmoidal_utility,
    min_utility_function,
    is_feasible,
    generate_population,
    sample_feasible_region,
    generate_maxmin_samples,
    true_maxmin_optimization,
    compute_individual_utilities,
    compute_fairness_metrics,
    gini_coefficient,
    create_maxmin_mesh_grid,
    compute_lipschitz_estimate,
    DEFAULT_N_INDIVIDUALS,
    DEFAULT_KAPPA,
    DEFAULT_D0,
    DEFAULT_L_MAXMIN,
    DEFAULT_BOX_MIN,
    DEFAULT_BOX_MAX,
    DEFAULT_A_MAXMIN,
    DEFAULT_B_MAXMIN,
    DEFAULT_POP_MIN,
    DEFAULT_POP_MAX,
)

from .maxmin_optimization import (
    robust_optimization_maxmin,
    piecewise_constant_optimization_maxmin,
    concave_regression_optimization_maxmin,
    worst_case_interpolation_maxmin,
    piecewise_constant_interpolation_maxmin,
    concave_regression_maxmin,
    interpolation_mesh_maxmin,
)

from .maxmin_visualization import (
    plot_maxmin_true_function_3d,
    plot_maxmin_true_function_contour,
    plot_individual_utilities,
    plot_interpolation_comparison_maxmin,
    plot_l1_error_maxmin,
    plot_fairness_comparison,
    plot_utility_distribution,
    plot_solutions_on_map,
    plot_computational_efficiency,
)

__all__ = [
    # Core (Cobb-Douglas)
    'true_function',
    'partial_derivatives',
    'compute_max_lipschitz',
    'generate_samples',
    'create_mesh_grid',
    'DEFAULT_X_MIN',
    'DEFAULT_X_MAX',
    'DEFAULT_N_FEATURES',
    'DEFAULT_L',
    'DEFAULT_ALPHA',
    'DEFAULT_COST',
    'DEFAULT_A',
    'DEFAULT_B',
    # Interpolation
    'worst_case_interpolation',
    'piecewise_constant_interpolation',
    'concave_regression',
    'milp_interpolation_problem',
    'sos1_interpolation_problem',
    'interpolation_mesh',
    # Optimization
    'robust_optimization',
    'piecewise_constant_optimization',
    'true_optimization',
    'concave_regression_optimization',
    'level_function_method_optimization',
    # Visualization
    'plot_true_function_3d',
    'plot_true_function_contour',
    'plot_interpolation_3d',
    'plot_contour_comparison',
    'plot_l1_error',
    # Max-Min Utility
    'sigmoidal_utility',
    'min_utility_function',
    'is_feasible',
    'generate_population',
    'sample_feasible_region',
    'generate_maxmin_samples',
    'true_maxmin_optimization',
    'compute_individual_utilities',
    'compute_fairness_metrics',
    'gini_coefficient',
    'create_maxmin_mesh_grid',
    'compute_lipschitz_estimate',
    'DEFAULT_N_INDIVIDUALS',
    'DEFAULT_KAPPA',
    'DEFAULT_D0',
    'DEFAULT_L_MAXMIN',
    'DEFAULT_BOX_MIN',
    'DEFAULT_BOX_MAX',
    'DEFAULT_A_MAXMIN',
    'DEFAULT_B_MAXMIN',
    'DEFAULT_POP_MIN',
    'DEFAULT_POP_MAX',
    # Max-Min Optimization
    'robust_optimization_maxmin',
    'piecewise_constant_optimization_maxmin',
    'concave_regression_optimization_maxmin',
    'worst_case_interpolation_maxmin',
    'piecewise_constant_interpolation_maxmin',
    'concave_regression_maxmin',
    'interpolation_mesh_maxmin',
    # Max-Min Visualization
    'plot_maxmin_true_function_3d',
    'plot_maxmin_true_function_contour',
    'plot_individual_utilities',
    'plot_interpolation_comparison_maxmin',
    'plot_l1_error_maxmin',
    'plot_fairness_comparison',
    'plot_utility_distribution',
    'plot_solutions_on_map',
    'plot_computational_efficiency',
]
