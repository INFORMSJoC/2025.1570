"""
Core functions for Robust Quasi-Concave Optimization.

Contains the true objective function, parameter definitions, and utility functions.
"""

import numpy as np


# Default parameters
DEFAULT_X_MIN = 0.5
DEFAULT_X_MAX = 10.0
DEFAULT_N_FEATURES = 2
DEFAULT_L = 0.3
DEFAULT_ALPHA = np.array([1.0, 0.6, 0.4])
DEFAULT_COST = np.array([1.0, 1.0, 2.0])

# Default constraints
DEFAULT_A = np.array([[1.0, 2.0], [2.0, 0.5]])
DEFAULT_B = np.array([8.0, 10.0])


def true_function(x: np.ndarray, alpha: np.ndarray = None, cost_param: np.ndarray = None) -> np.ndarray:
    """
    Quasi-concave objective function: Ratio of Cobb-Douglas and linear cost.
    
    f(x) = alpha[0] * prod(x^alpha[1:]) / (cost_param[0] + sum(x * cost_param[1:]))
    
    Args:
        x: (n_points, n_features) array of input points
        alpha: (n_features+1,) array of Cobb-Douglas parameters
        cost_param: (n_features+1,) array of cost parameters
        
    Returns:
        (n_points,) array of function values
    """
    n_features = x.shape[1]
    if alpha is None:
        alpha = np.random.rand(n_features + 1)
        alpha = alpha / alpha.sum()
    if cost_param is None:
        cost_param = np.random.rand(n_features + 1)
    
    numerator = alpha[0] * np.prod(x ** alpha[1:], axis=1)
    denominator = cost_param[0] + np.sum(x * cost_param[1:], axis=1)
    return numerator / denominator


def partial_derivatives(x: np.ndarray, alpha: np.ndarray, cost_param: np.ndarray) -> np.ndarray:
    """
    Compute partial derivatives of the true function.
    
    Args:
        x: (n_points, n_features) array of input points
        alpha: Cobb-Douglas parameters
        cost_param: Cost parameters
        
    Returns:
        (n_points, n_features) array of gradients
    """
    N = alpha[0] * np.prod(x ** alpha[1:], axis=1)
    D = cost_param[0] + np.sum(x * cost_param[1:], axis=1)

    grads = np.zeros_like(x)
    for i in range(x.shape[1]):
        # dN/dxi
        dN_dxi = alpha[0] * alpha[i + 1] * (x[:, i] ** (alpha[i + 1] - 1))
        for j in range(x.shape[1]):
            if j != i:
                dN_dxi *= x[:, j] ** alpha[j + 1]
        dD_dxi = cost_param[i + 1]
        grads[:, i] = (dN_dxi * D - N * dD_dxi) / (D ** 2)
    return grads


def compute_max_lipschitz(x_min: float, x_max: float, alpha: np.ndarray, 
                          cost_param: np.ndarray, n_grid: int = 100) -> tuple:
    """
    Compute maximum L1 norm of gradient over the domain.
    
    Returns:
        (max_l1_norm, x_at_max, gradient_at_max)
    """
    n_features = len(alpha) - 1
    
    # Create grid
    grids = [np.linspace(x_min, x_max, n_grid) for _ in range(n_features)]
    mesh = np.meshgrid(*grids)
    X_grid = np.column_stack([m.ravel() for m in mesh])
    
    grads = partial_derivatives(X_grid, alpha, cost_param)
    l1_norm = np.abs(grads).sum(axis=1)
    
    max_l1_norm = np.max(l1_norm)
    idx_max = np.argmax(l1_norm)
    x_at_max = X_grid[idx_max]
    
    return max_l1_norm, x_at_max, grads[idx_max]


def generate_samples(n_samples: int, n_features: int, x_min: float, x_max: float,
                     alpha: np.ndarray, cost_param: np.ndarray, seed: int = None) -> tuple:
    """
    Generate random samples and their function values.
    
    Returns:
        (X_samples, Y_values, y_sorted, idx_order)
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_samples = np.random.uniform(x_min, x_max, size=(n_samples, n_features))
    Y_values = true_function(X_samples, alpha=alpha, cost_param=cost_param)
    
    # Sort by descending value
    idx_order = list(np.argsort(Y_values)[::-1])
    y_sorted = [Y_values[i] for i in idx_order]
    
    return X_samples, Y_values, y_sorted, idx_order


def create_mesh_grid(x_min: float, x_max: float, n_grid: int = 50) -> tuple:
    """
    Create 2D mesh grid for visualization.
    
    Returns:
        (X1, X2, X_grid) where X_grid is (n_grid^2, 2)
    """
    x1 = np.linspace(x_min, x_max, n_grid)
    x2 = np.linspace(x_min, x_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    return X1, X2, X_grid
