"""
Optimization methods for Robust Quasi-Concave Optimization.

Contains robust optimization, piecewise constant optimization, 
true optimization (GP), concave regression optimization, and level function method.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from .interpolation import milp_interpolation_problem, solve_lp_interpolation


def solve_lp_feasibility(X: np.ndarray, sorted_vals: list, idx_order: list,
                         L: float, level: int, A: np.ndarray, b: np.ndarray,
                         x_min: float, x_max: float) -> tuple:
    """
    Solve LP feasibility for robust optimization at a given level.
    
    Uses convex hull relaxation to find a candidate solution g. Returns the
    convex hull LP value (an upper bound), which is sufficient for binary search.
    
    For non-monotone functions, uses both upper and lower bound constraints.
    
    Returns:
        (lp_value, solution_x) where lp_value is the convex hull LP value (upper bound)
    """
    n_features = X.shape[1]
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)
    g = cp.Variable(n_features, nonneg=False)

    e_N = np.ones(n_features)
    
    # Compute shifted points for lower and upper bounds (non-monotone case)
    # Lower shift: theta_i - y_i/L (for constraint g >= ...)
    # Upper shift: theta_i + y_i/L (for constraint g <= ...)
    X_shifted_lower = np.array([
        X[idx_order[i]] - sorted_vals[i] * e_N / L for i in range(len(idx_order))
    ])
    X_shifted_upper = np.array([
        X[idx_order[i]] + sorted_vals[i] * e_N / L for i in range(len(idx_order))
    ])

    constraints = [cp.sum(p) == 1]
    constraints += [A @ g <= b]
    constraints += [g >= x_min, g <= x_max]
    # Lower bound: g >= sum(p_i * (theta_i - y_i/L)) + v/L
    constraints += [g >= sum(p[i] * X_shifted_lower[i] for i in range(level)) + v * e_N / L]
    # Upper bound: g <= sum(p_i * (theta_i + y_i/L)) - v/L (for non-monotone)
    constraints += [g <= sum(p[i] * X_shifted_upper[i] for i in range(level)) - v * e_N / L]

    objective = cp.Maximize(v)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Problem not solved: {prob.status}")
    
    # Return convex hull value (upper bound) - sufficient for binary search
    return v.value, g.value.copy()


def robust_optimization(X: np.ndarray, y_sorted: list, idx_order: list,
                        L: float, A: np.ndarray, b: np.ndarray,
                        x_min: float, x_max: float) -> tuple:
    """
    Binary search for robust optimization.
    
    Uses convex hull LP values (upper bounds) for binary search, then evaluates
    the true interpolation value at the final solution.
    
    Returns:
        (optimal_value, optimal_solution) where optimal_value is the TRUE interpolation value
    """
    from .interpolation import worst_case_interpolation
    
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        # Use convex hull value (upper bound) for binary search direction
        val, sol = solve_lp_feasibility(X, y_sorted, idx_order, L, mid, A, b, x_min, x_max)
        if val >= y_sorted[mid]:
            right = mid
        else:
            left = mid + 1


    # Get final solution from convex hull LP
    val, sol = solve_lp_feasibility(X, y_sorted, idx_order, L, left, A, b, x_min, x_max)
    
    # Value is min of LP value and the level threshold (y_sorted[left-1] is the achieved level)
    true_val = min(val, y_sorted[left - 1])
    
    return true_val, sol


def solve_constant_feasibility(X: np.ndarray, sorted_vals: list, idx_order: list,
                               level: int, A: np.ndarray, b: np.ndarray,
                               x_min: float, x_max: float) -> tuple:
    """
    Solve LP feasibility for piecewise constant optimization at a given level.
    """
    n_features = X.shape[1]
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)
    g = cp.Variable(n_features, nonneg=False)

    constraints = [cp.sum(p) == 1]
    constraints += [A @ g <= b]
    constraints += [g >= x_min, g <= x_max]
    constraints += [v >= cp.max(cp.abs(sum(p[i] * X[idx_order[i]] for i in range(level)) - g))]

    # Regularization term to break ties
    lambda_reg = 0 #1e-8
    objective = cp.Minimize(v + lambda_reg * cp.norm(g, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return v.value, g.value
    else:
        raise ValueError("Problem is not solved to optimality!")


def piecewise_constant_optimization(X: np.ndarray, y_sorted: list, idx_order: list,
                                    A: np.ndarray, b: np.ndarray,
                                    x_min: float, x_max: float) -> tuple:
    """
    Binary search for piecewise constant optimization.
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        v, sol = solve_constant_feasibility(X, y_sorted, idx_order, mid, A, b, x_min, x_max)
        if v <= 0:
            right = mid
        else:
            left = mid + 1
    v, sol = solve_constant_feasibility(X, y_sorted, idx_order, left, A, b, x_min, x_max)
    return y_sorted[left - 1], sol


def true_optimization(alpha: np.ndarray, cost_param: np.ndarray,
                      A: np.ndarray, b: np.ndarray,
                      x_min: float, x_max: float) -> tuple:
    """
    Solve the true optimization problem using geometric programming.
    
    Returns:
        (optimal_value, optimal_solution)
    """
    n_features = len(alpha) - 1
    x = cp.Variable(n_features, pos=True)
    t = cp.Variable(pos=True)
    
    constraints = [A @ x <= b]
    constraints += [x >= x_min, x <= x_max]
    constraints += [(cost_param[0] + cost_param[1:] @ x) * cp.inv_pos(t) <= 1]
    
    monomial = alpha[0] * cp.prod([cp.power(x[n], alpha[n + 1]) for n in range(n_features)]) \
               * cp.inv_pos(t)
    objective = cp.Maximize(monomial)

    prob = cp.Problem(objective, constraints)
    prob.solve(gp=True, verbose=False)
    
    return prob.value, x.value


def concave_regression_optimization(X_samples: np.ndarray, y_samples: np.ndarray,
                                    A: np.ndarray, b: np.ndarray,
                                    x_min: float, x_max: float, K: int = 10) -> tuple:
    """
    Fit concave regression and optimize over constraints.
    
    Returns:
        (optimal_value, optimal_solution)
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
        Xk_aug = np.hstack([Xk, np.ones((Xk.shape[0], 1))])
        sol, _, _, _ = np.linalg.lstsq(Xk_aug, yk, rcond=None)
        coefs.append(sol[:-1])
        intercepts.append(sol[-1])
    
    if len(coefs) == 0:
        mean_y = np.mean(y_samples)
        def fitted_func(x):
            return mean_y
    else:
        def fitted_func(x):
            vals = [np.dot(a, x) + inter for a, inter in zip(coefs, intercepts)]
            return min(vals)
    
    # Maximize fitted_func subject to A @ x <= b, x in [x_min, x_max]
    def obj(x):
        return -fitted_func(x)
    
    cons = [{'type': 'ineq', 'fun': lambda x: b - A @ x}]
    bounds = [(x_min, x_max)] * d
    x0 = np.mean(X_samples, axis=0)
    
    res = minimize(obj, x0, bounds=bounds, constraints=cons, method='SLSQP')
    return -res.fun, res.x


def level_function_method_optimization(X: np.ndarray, y_sorted: list, idx_order: list,
                                       L: float, A: np.ndarray, b: np.ndarray,
                                       x_min: float, x_max: float,
                                       epsilon: float = 1e-5, max_iters: int = 100) -> tuple:
    """
    Level function method (Xu 2001) for robust optimization.
    
    At each iteration, constructs a level function using subgradients,
    and maximizes the minimum to get the next iterate.
    
    Returns:
        (optimal_value, optimal_solution)
    """
    d = X.shape[1]
    
    def get_subgradient(x):
        _, s = milp_interpolation_problem(x, X, y_sorted, idx_order, L, return_gradient=True)
        return s
    
    x_list = [np.mean(X, axis=0)]
    s_list = [get_subgradient(x_list[0])]
    
    for _ in range(max_iters):
        x = cp.Variable(d)
        t = cp.Variable()
        
        constraints = [A @ x <= b]
        constraints += [x >= x_min, x <= x_max]
        constraints += [t <= s_list[j] @ (x - x_list[j]) for j in range(len(x_list))]
        
        obj = cp.Maximize(t)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Solver failed with status {prob.status}")
        
        x_next = x.value
        sigma_val = t.value
        
        if sigma_val <= epsilon:
            return sigma_val, x_next
        
        x_list.append(x_next)
        s_list.append(get_subgradient(x_next))
    
    return sigma_val, x_next
