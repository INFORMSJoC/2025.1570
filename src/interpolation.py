"""
Interpolation methods for Robust Quasi-Concave Optimization.

Contains worst-case interpolation, piecewise constant interpolation, 
concave regression, and MILP formulation.
"""

import numpy as np
import cvxpy as cp
from sklearn.cluster import KMeans


def solve_lp_interpolation(x_new: np.ndarray, X: np.ndarray, sorted_vals: list,
                           idx_order: list, L: float, level: int) -> float:
    """
    Solve LP for worst-case interpolation at a given level.
    
    Args:
        x_new: Query point
        X: Sample points
        sorted_vals: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
        level: Number of points before x_new (1 to n_samples)
    
    Returns:
        Optimal value at x_new
    """
    n_features = X.shape[1]
    v_new = cp.Variable()
    s = cp.Variable(n_features, nonneg=False)

    constraints = [cp.sum(cp.abs(s)) <= L]
    for rank, j in enumerate(idx_order[:level]):
        constraints.append(v_new + s @ (X[j] - x_new) >= sorted_vals[rank])

    objective = cp.Minimize(v_new)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return max(v_new.value, 0.0)
    else:
        raise ValueError("Problem is not solved to optimality!")


def worst_case_interpolation(x_new: np.ndarray, X: np.ndarray, y_sorted: list,
                             idx_order: list, L: float) -> float:
    """
    Binary search to find the worst-case interpolation value.
    
    Args:
        x_new: Query point
        X: Sample points
        y_sorted: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
    
    Returns:
        Worst-case interpolation value
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        val = solve_lp_interpolation(x_new, X, y_sorted, idx_order, L, mid)
        if val >= y_sorted[mid]:
            right = mid
        else:
            left = mid + 1
    val = solve_lp_interpolation(x_new, X, y_sorted, idx_order, L, left)
    return min(y_sorted[left - 1], val)


def solve_constant_interpolation(x_new: np.ndarray, X: np.ndarray, sorted_vals: list,
                                 idx_order: list, level: int) -> float:
    """
    Solve LP for piecewise constant interpolation at a given level.
    """
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)

    constraints = [cp.sum(p) == 1]
    constraints += [v >= cp.sum(cp.abs(sum(p[i] * X[idx_order[i]] for i in range(level)) - x_new))]

    objective = cp.Minimize(v)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return v.value
    else:
        raise ValueError("Problem is not solved to optimality!")


def piecewise_constant_interpolation(x_new: np.ndarray, X: np.ndarray,
                                     y_sorted: list, idx_order: list) -> float:
    """
    Binary search for piecewise constant interpolation.
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        v = solve_constant_interpolation(x_new, X, y_sorted, idx_order, mid)
        if v <= 0:
            right = mid
        else:
            left = mid + 1
    return y_sorted[left] if left < len(y_sorted) else 0.0


def concave_regression(x_new: np.ndarray, X_samples: np.ndarray, y_samples: np.ndarray,
                       K: int = 10) -> float:
    """
    Fit concave regression (min of affine functions) and evaluate at x_new.
    
    Uses K-means clustering to fit piecewise linear concave approximation.
    """
    n, d = X_samples.shape
    K = min(K, n)
    
    kmeans = KMeans(n_clusters=K, n_init=5, random_state=0)
    labels = kmeans.fit_predict(X_samples)
    
    coefs = []
    intercepts = []
    for k in range(K):
        idx = (labels == k)
        if np.sum(idx) < d + 1:
            continue
        Xk = X_samples[idx]
        yk = y_samples[idx]
        # Fit affine: y = a^T x + b
        Xk_aug = np.hstack([Xk, np.ones((Xk.shape[0], 1))])
        sol, _, _, _ = np.linalg.lstsq(Xk_aug, yk, rcond=None)
        coefs.append(sol[:-1])
        intercepts.append(sol[-1])
    
    if len(coefs) == 0:
        return np.mean(y_samples)
    
    vals = [np.dot(a, x_new) + b for a, b in zip(coefs, intercepts)]
    return min(vals)


def milp_interpolation_problem(x_new: np.ndarray, X: np.ndarray, y_sorted: list,
                               idx_order: list, L: float, 
                               return_gradient: bool = False) -> tuple:
    """
    MILP formulation for interpolation problem using Big-M constraints.
    
    Args:
        return_gradient: If True, return (value, gradient), else return value only
    
    Returns:
        (f_x_value,) or (f_x_value, s_x_gradient) depending on return_gradient
    """
    n = X.shape[0]
    n_features = X.shape[1]
    BigM = 1000
    
    f_x = cp.Variable()
    s_x = cp.Variable(n_features, nonneg=False)
    a1 = cp.Variable(n)
    a2 = cp.Variable(n)
    b = cp.Variable(n, boolean=True)
    
    constraints = []
    constraints += [f_x >= 0, a1 >= 0, a2 >= 0]
    constraints += [cp.sum(cp.abs(s_x)) <= L]
    
    for rank, j in enumerate(idx_order):
        constraints += [
            f_x + a1[j] >= y_sorted[rank],
            s_x @ (X[j] - x_new) >= a1[j] - a2[j],
            a1[j] <= b[j] * BigM,
            a2[j] <= (1 - b[j]) * BigM
        ]
    
    obj = cp.Minimize(f_x)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)
    
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver failed with status {prob.status}")
    
    if return_gradient:
        return f_x.value, s_x.value
    else:
        return f_x.value


def interpolation_mesh(func, *args, x_min: float, x_max: float, n_grid: int = 50) -> tuple:
    """
    Evaluate interpolation function on a 2D mesh grid.
    
    Returns:
        (xx1, xx2, Z) mesh and values
    """
    xx1, xx2 = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(x_min, x_max, n_grid)
    )
    
    Z = np.zeros_like(xx1)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            x_query = np.array([xx1[i, j], xx2[i, j]])
            Z[i, j] = func(x_query, *args)
    
    return xx1, xx2, Z
