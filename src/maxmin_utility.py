"""
Max-Min Utility Problem for Robust Quasi-Concave Optimization.

This module implements the max-min utility problem where we maximize 
the utility of the worst-off individual in a heterogeneous population.

Problem formulation:
    max_{x ∈ X} min_{i ∈ I} u_i(x)

where:
- x ∈ R^2 is the decision variable (location of a service facility)
- I = {1, 2, ..., I} is the index set of the population
- u_i(x) is the sigmoidal utility function for individual i
- X ⊆ R^2 is a convex, compact feasible region (box intersected with disk)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


# Default parameters for max-min utility problem
DEFAULT_N_INDIVIDUALS = 10
DEFAULT_KAPPA = 1.5   # Sigmoid steepness: higher = sharper transition, more "kinks" in min-utility
DEFAULT_D0 = 2.5      # Critical distance threshold: smaller = transition closer to individuals
DEFAULT_L_MAXMIN = 0.5  # Lipschitz constant (approx kappa/4 * sqrt(2) for 2D)

# Feasible region: X = {x ∈ R^2 : x_1, x_2 ∈ [0, 10], Ax <= b}
# The constraint x1 + x2 <= 8 ensures the global optimal (population centroid ~(5,5))
# is OUTSIDE the feasible region, which is key for QCO Lipschitz vs QCO Constant comparison
DEFAULT_BOX_MIN = 0.0
DEFAULT_BOX_MAX = 10.0

# Linear constraint: A @ x <= b
# x1 + x2 <= 8 makes the feasible region a triangle with vertices (0,0), (8,0), (0,8)
# The population centroid (around (5,5)) is outside this region since 5+5=10 > 8
import numpy as np
DEFAULT_A_MAXMIN = np.array([[1.0, 1.0]])  # Constraint matrix (1 constraint)
DEFAULT_B_MAXMIN = np.array([8.0])         # RHS of constraint (tighter to exclude optimal)

# Population location bounds - spread out more to create more "kinks" in min-utility
DEFAULT_POP_MIN = 2.0
DEFAULT_POP_MAX = 8.0

# Sampling extension: how much to extend beyond feasible region for sampling
DEFAULT_SAMPLE_EXTENSION = 3.0


def sigmoidal_utility(x: np.ndarray, p_i: np.ndarray, kappa: float = DEFAULT_KAPPA, 
                      d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute sigmoidal utility for individual i at location p_i.
    
    u_i(x) = 1 - 1 / (1 + exp(-κ(d(x, p_i) - d_0)))
    
    This is a decreasing function of distance - closer facilities provide higher utility.
    
    Args:
        x: (n_points, 2) array of facility locations
        p_i: (2,) array - location of individual i
        kappa: Steepness parameter of sigmoid
        d0: Critical distance threshold
        
    Returns:
        (n_points,) array of utility values in [0, 1]
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Euclidean distance
    distance = np.linalg.norm(x - p_i, axis=1)
    
    # Sigmoidal utility
    utility = 1.0 - 1.0 / (1.0 + np.exp(-kappa * (distance - d0)))
    
    return utility


def min_utility_function(x: np.ndarray, population: np.ndarray, 
                         kappa: float = DEFAULT_KAPPA, d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute the minimum utility across all individuals (true objective).
    
    f(x) = min_{i ∈ I} u_i(x)
    
    This function is quasiconcave since each u_i(x) is quasiconcave.
    
    Args:
        x: (n_points, 2) array of facility locations
        population: (n_individuals, 2) array of individual locations
        kappa: Steepness parameter
        d0: Critical distance threshold
        
    Returns:
        (n_points,) array of minimum utility values
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_points = x.shape[0]
    n_individuals = population.shape[0]
    
    # Compute utilities for all individuals
    utilities = np.zeros((n_points, n_individuals))
    for i in range(n_individuals):
        utilities[:, i] = sigmoidal_utility(x, population[i], kappa, d0)
    
    # Return minimum utility
    return utilities.min(axis=1)


def is_feasible(x: np.ndarray, box_min: float = DEFAULT_BOX_MIN, 
                box_max: float = DEFAULT_BOX_MAX,
                A: np.ndarray = None, b: np.ndarray = None) -> np.ndarray:
    """
    Check if points are in the feasible region.
    
    X = {x ∈ R^2 : x_1, x_2 ∈ [box_min, box_max], Ax <= b}
    
    Args:
        x: (n_points, 2) array of points
        A: (m, 2) constraint matrix
        b: (m,) constraint RHS
        
    Returns:
        (n_points,) boolean array
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
        
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Box constraint
    in_box = np.all((x >= box_min) & (x <= box_max), axis=1)
    
    # Linear constraint: A @ x <= b
    in_linear = np.all(x @ A.T <= b, axis=1)
    
    return in_box & in_linear


def generate_population(n_individuals: int = DEFAULT_N_INDIVIDUALS,
                        pop_min: float = DEFAULT_POP_MIN,
                        pop_max: float = DEFAULT_POP_MAX,
                        seed: int = None) -> np.ndarray:
    """
    Generate random population locations uniformly from [pop_min, pop_max]^2.
    
    Args:
        n_individuals: Number of individuals
        pop_min, pop_max: Bounds for population locations
        seed: Random seed
        
    Returns:
        (n_individuals, 2) array of population locations
    """
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.uniform(pop_min, pop_max, size=(n_individuals, 2))


def sample_feasible_region(n_samples: int, box_min: float = DEFAULT_BOX_MIN,
                           box_max: float = DEFAULT_BOX_MAX,
                           A: np.ndarray = None, b: np.ndarray = None,
                           seed: int = None,
                           extend_beyond: float = 0.0) -> np.ndarray:
    """
    Sample points uniformly from the feasible region X or an extended region.
    
    X = {x ∈ R^2 : x_1, x_2 ∈ [box_min, box_max], Ax <= b}
    
    When extend_beyond > 0, samples from a larger region [box_min, box_max + extend_beyond]^2
    without enforcing the linear constraint, allowing samples outside the feasible region.
    This is important for QCO methods to "see" the function behavior beyond the feasible region.
    
    Args:
        n_samples: Number of samples
        box_min, box_max: Box bounds
        A, b: Linear constraint Ax <= b
        seed: Random seed
        extend_beyond: If > 0, sample from extended region without linear constraint
        
    Returns:
        (n_samples, 2) array of sample points
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
        
    if seed is not None:
        np.random.seed(seed)
    
    if extend_beyond > 0:
        # Sample from extended region without enforcing constraints
        # This allows samples beyond the feasible region
        ext_max = box_max + extend_beyond
        return np.random.uniform(box_min, ext_max, size=(n_samples, 2))
    
    # Original behavior: rejection sampling for feasible region
    samples = []
    while len(samples) < n_samples:
        # Sample from box
        x = np.random.uniform(box_min, box_max, size=(n_samples * 2, 2))
        
        # Keep only feasible points (satisfying Ax <= b)
        feasible_mask = np.all(x @ A.T <= b, axis=1)
        samples.extend(x[feasible_mask].tolist())
    
    return np.array(samples[:n_samples])


def generate_maxmin_samples(n_samples: int, population: np.ndarray,
                            kappa: float = DEFAULT_KAPPA, d0: float = DEFAULT_D0,
                            box_min: float = DEFAULT_BOX_MIN,
                            box_max: float = DEFAULT_BOX_MAX,
                            A: np.ndarray = None, b: np.ndarray = None,
                            seed: int = None,
                            extend_beyond: float = DEFAULT_SAMPLE_EXTENSION) -> tuple:
    """
    Generate samples and evaluate true objective.
    
    By default, samples from an extended region beyond the feasible region.
    This allows QCO methods to "see" the function behavior outside the feasible region,
    which is crucial when the global optimum is outside the feasible region.
    
    Args:
        n_samples: Number of samples
        population: (n_individuals, 2) array of population locations
        kappa, d0: Sigmoid parameters
        box_min, box_max: Feasible region parameters
        A, b: Linear constraint Ax <= b
        seed: Random seed
        extend_beyond: How much to extend sampling region beyond box_max (default: 3.0)
        
    Returns:
        (X_samples, Y_values, y_sorted, idx_order)
    """
    X_samples = sample_feasible_region(n_samples, box_min, box_max, A, b, seed, 
                                       extend_beyond=extend_beyond)
    Y_values = min_utility_function(X_samples, population, kappa, d0)
    
    # Sort by descending value
    idx_order = list(np.argsort(Y_values)[::-1])
    y_sorted = [Y_values[i] for i in idx_order]
    
    return X_samples, Y_values, y_sorted, idx_order


def true_maxmin_optimization(population: np.ndarray, kappa: float = DEFAULT_KAPPA,
                             d0: float = DEFAULT_D0, box_min: float = DEFAULT_BOX_MIN,
                             box_max: float = DEFAULT_BOX_MAX,
                             A: np.ndarray = None, b: np.ndarray = None,
                             n_restarts: int = 20) -> tuple:
    """
    Solve the true max-min utility optimization problem.
    
    max_{x ∈ X} min_{i ∈ I} u_i(x)
    
    Since this is non-convex, we use multiple restarts with local optimization
    and also try differential evolution as a global optimizer.
    
    Args:
        population: (n_individuals, 2) array of individual locations
        kappa, d0: Sigmoid parameters
        box_min, box_max: Feasible region parameters
        A, b: Linear constraint Ax <= b
        n_restarts: Number of random restarts for local optimization
        
    Returns:
        (optimal_value, optimal_solution)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    def objective(x):
        """Negative min utility (for minimization)."""
        return -min_utility_function(x.reshape(1, -1), population, kappa, d0)[0]
    
    def linear_constraint(x):
        """Constraint: Ax <= b (return b - Ax >= 0)."""
        return b - A @ x
    
    bounds = [(box_min, box_max), (box_min, box_max)]
    constraints = [{'type': 'ineq', 'fun': linear_constraint}]
    
    best_val = -np.inf
    best_sol = None
    
    # Try differential evolution (global optimizer) with penalty
    try:
        def penalized_objective(x):
            if np.any(A @ x > b):
                return 1e6  # Large penalty for constraint violation
            return objective(x)
        
        result = differential_evolution(penalized_objective, bounds, seed=42, 
                                         maxiter=500, polish=True)
        if result.success and np.all(A @ result.x <= b + 1e-6) and -result.fun > best_val:
            best_val = -result.fun
            best_sol = result.x
    except Exception:
        pass
    
    # Multiple restarts with local optimizer
    np.random.seed(42)
    for _ in range(n_restarts):
        # Random starting point in feasible region
        while True:
            x0 = np.random.uniform(box_min, box_max, size=2)
            if np.all(A @ x0 <= b):
                break
        
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 500})
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_sol = result.x
        except Exception:
            continue
    
    # Also try centroid of population as starting point
    centroid = population.mean(axis=0)
    if np.all(centroid >= box_min) and np.all(centroid <= box_max) and np.all(A @ centroid <= b):
        try:
            result = minimize(objective, centroid, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 500})
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_sol = result.x
        except Exception:
            pass
    
    return best_val, best_sol


def sum_utility_function(x: np.ndarray, population: np.ndarray,
                         kappa: float = DEFAULT_KAPPA, d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute the sum of utilities across all individuals (utilitarian objective).

    g(x) = sum_{i ∈ I} u_i(x)

    Args:
        x: (n_points, 2) array of facility locations
        population: (n_individuals, 2) array of individual locations
        kappa: Steepness parameter
        d0: Critical distance threshold

    Returns:
        (n_points,) array of summed utility values
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)

    n_points = x.shape[0]
    n_individuals = population.shape[0]

    utilities = np.zeros((n_points, n_individuals))
    for i in range(n_individuals):
        utilities[:, i] = sigmoidal_utility(x, population[i], kappa, d0)

    return utilities.sum(axis=1)


def true_utilitarian_optimization(population: np.ndarray, kappa: float = DEFAULT_KAPPA,
                                  d0: float = DEFAULT_D0, box_min: float = DEFAULT_BOX_MIN,
                                  box_max: float = DEFAULT_BOX_MAX,
                                  A: np.ndarray = None, b: np.ndarray = None,
                                  n_restarts: int = 20) -> tuple:
    """
    Solve the utilitarian (sum-maximising) optimization problem.

    max_{x ∈ X} sum_{i ∈ I} u_i(x)

    Args:
        population: (n_individuals, 2) array of individual locations
        kappa, d0: Sigmoid parameters
        box_min, box_max: Feasible region parameters
        A, b: Linear constraint Ax <= b
        n_restarts: Number of random restarts for local optimization

    Returns:
        (optimal_value, optimal_solution)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN

    def objective(x):
        """Negative sum utility (for minimization)."""
        return -sum_utility_function(x.reshape(1, -1), population, kappa, d0)[0]

    def linear_constraint(x):
        return b - A @ x

    bounds = [(box_min, box_max), (box_min, box_max)]
    constraints = [{'type': 'ineq', 'fun': linear_constraint}]

    best_val = -np.inf
    best_sol = None

    # Differential evolution (global optimizer)
    try:
        def penalized_objective(x):
            if np.any(A @ x > b):
                return 1e6
            return objective(x)

        result = differential_evolution(penalized_objective, bounds, seed=42,
                                        maxiter=500, polish=True)
        if result.success and np.all(A @ result.x <= b + 1e-6) and -result.fun > best_val:
            best_val = -result.fun
            best_sol = result.x
    except Exception:
        pass

    # Multiple restarts with local optimizer
    np.random.seed(42)
    for _ in range(n_restarts):
        while True:
            x0 = np.random.uniform(box_min, box_max, size=2)
            if np.all(A @ x0 <= b):
                break
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 500})
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_sol = result.x
        except Exception:
            continue

    # Try centroid as starting point
    centroid = population.mean(axis=0)
    if np.all(centroid >= box_min) and np.all(centroid <= box_max) and np.all(A @ centroid <= b):
        try:
            result = minimize(objective, centroid, method='SLSQP', bounds=bounds,
                              constraints=constraints, options={'maxiter': 500})
            if result.success and -result.fun > best_val:
                best_val = -result.fun
                best_sol = result.x
        except Exception:
            pass

    return best_val, best_sol


def compute_individual_utilities(x: np.ndarray, population: np.ndarray,
                                 kappa: float = DEFAULT_KAPPA,
                                 d0: float = DEFAULT_D0) -> np.ndarray:
    """
    Compute utilities for all individuals at a given facility location.
    
    Args:
        x: (2,) facility location
        population: (n_individuals, 2) array of individual locations
        kappa, d0: Sigmoid parameters
        
    Returns:
        (n_individuals,) array of utilities
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_individuals = population.shape[0]
    utilities = np.zeros(n_individuals)
    
    for i in range(n_individuals):
        utilities[i] = sigmoidal_utility(x, population[i], kappa, d0)[0]
    
    return utilities


def compute_fairness_metrics(utilities: np.ndarray) -> dict:
    """
    Compute fairness metrics for a distribution of utilities.
    
    Args:
        utilities: (n_individuals,) array of utility values
        
    Returns:
        dict with min, mean, std, max, gini coefficient
    """
    return {
        'min_utility': np.min(utilities),
        'mean_utility': np.mean(utilities),
        'std_utility': np.std(utilities),
        'max_utility': np.max(utilities),
        'gini': gini_coefficient(utilities)
    }


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient (measure of inequality).
    
    0 = perfect equality, 1 = maximal inequality
    """
    sorted_values = np.sort(values)
    n = len(values)
    cumulative = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


def create_maxmin_mesh_grid(box_min: float = DEFAULT_BOX_MIN,
                            box_max: float = DEFAULT_BOX_MAX,
                            A: np.ndarray = None, b: np.ndarray = None,
                            n_grid: int = 50) -> tuple:
    """
    Create 2D mesh grid for visualization with feasibility mask.
    
    Returns:
        (X1, X2, X_grid, feasible_mask)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    x1 = np.linspace(box_min, box_max, n_grid)
    x2 = np.linspace(box_min, box_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    # Feasibility mask for linear constraint
    feasible_mask = np.all(X_grid @ A.T <= b, axis=1).reshape(X1.shape)
    
    return X1, X2, X_grid, feasible_mask


def compute_lipschitz_estimate(population: np.ndarray, kappa: float = DEFAULT_KAPPA,
                               d0: float = DEFAULT_D0, box_min: float = DEFAULT_BOX_MIN,
                               box_max: float = DEFAULT_BOX_MAX,
                               A: np.ndarray = None, b: np.ndarray = None,
                               n_grid: int = 50) -> float:
    """
    Estimate the Lipschitz constant of the min-utility function.
    
    The gradient of u_i(x) = 1 - 1/(1+exp(-kappa(d-d0))) with respect to x:
    
    du_i/dx = kappa * exp(-kappa(d-d0)) / (1+exp(-kappa(d-d0)))^2 * (x - p_i) / d
    
    The Lipschitz constant is the maximum L1 norm of the subgradient.
    
    Returns:
        Estimated Lipschitz constant
    """
    X1, X2, X_grid, feasible_mask = create_maxmin_mesh_grid(box_min, box_max, A, b, n_grid)
    
    # Only consider feasible points
    X_feasible = X_grid[feasible_mask.ravel()]
    
    max_grad_norm = 0.0
    
    for x in X_feasible:
        x = x.reshape(1, -1)
        
        # Find which individual has minimum utility at this point
        utilities = np.array([sigmoidal_utility(x, p_i, kappa, d0)[0] 
                             for p_i in population])
        min_idx = np.argmin(utilities)
        p_i = population[min_idx]
        
        # Compute gradient of u_{min_idx}(x)
        d = np.linalg.norm(x[0] - p_i)
        if d > 1e-8:  # Avoid division by zero
            exp_term = np.exp(-kappa * (d - d0))
            denom = (1.0 + exp_term) ** 2
            scalar = kappa * exp_term / denom
            grad = scalar * (x[0] - p_i) / d
            
            # L1 norm
            grad_norm = np.abs(grad).sum()
            max_grad_norm = max(max_grad_norm, grad_norm)
    
    return max_grad_norm
