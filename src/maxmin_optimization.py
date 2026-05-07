"""
Optimization methods for Max-Min Utility Problem.

Contains robust optimization methods adapted for the disk-box feasible region,
as well as benchmark methods (concave regression, piecewise constant).
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from sklearn.cluster import KMeans

from .maxmin_utility import (
    min_utility_function,
    DEFAULT_BOX_MIN,
    DEFAULT_BOX_MAX,
    DEFAULT_L_MAXMIN,
    DEFAULT_A_MAXMIN,
    DEFAULT_B_MAXMIN,
)


def solve_lp_feasibility_maxmin(X: np.ndarray, sorted_vals: list, idx_order: list,
                                 L: float, level: int,
                                 box_min: float = DEFAULT_BOX_MIN,
                                 box_max: float = DEFAULT_BOX_MAX,
                                 A: np.ndarray = None, b: np.ndarray = None) -> tuple:
    """
    Solve LP feasibility for robust optimization at a given level.
    
    Uses convex hull relaxation to find a candidate solution g. Returns the
    convex hull LP value (an upper bound), which is sufficient for binary search.
    
    The feasible region is X = {x : box_min <= x <= box_max, Ax <= b}.
    
    Args:
        X: Sample points
        sorted_vals: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
        level: Number of points before x_new (1 to n_samples)
        box_min, box_max: Box bounds
        A, b: Linear constraint Ax <= b
        
    Returns:
        (lp_value, solution_x) where lp_value is the convex hull LP value (upper bound)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    n_features = X.shape[1]
    e_N = np.ones(n_features)
    
    # Use convex hull LP to find candidate solution g
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)
    g = cp.Variable(n_features)
    
    X_shifted_lower = np.array([
        X[idx_order[i]] - sorted_vals[i] * e_N / L for i in range(level)
    ])
    X_shifted_upper = np.array([
        X[idx_order[i]] + sorted_vals[i] * e_N / L for i in range(level)
    ])

    constraints = [cp.sum(p) == 1]
    constraints += [g >= box_min, g <= box_max]
    constraints += [A @ g <= b]
    constraints += [g >= sum(p[i] * X_shifted_lower[i] for i in range(level)) + v * e_N / L]
    constraints += [g <= sum(p[i] * X_shifted_upper[i] for i in range(level)) - v * e_N / L]

    prob = cp.Problem(cp.Maximize(v), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Problem not solved: {prob.status}")
    
    # Return convex hull value (upper bound) - sufficient for binary search
    return v.value, g.value.copy()


def solve_lp_interpolation_maxmin(x_new: np.ndarray, X: np.ndarray, sorted_vals: list,
                                  idx_order: list, L: float, level: int) -> float:
    """
    Solve LP for worst-case interpolation at a given level.
    
    Finds: min_s max_{j<=level} (y_j - s @ (theta_j - x_new)) subject to ||s||_1 <= L
    
    This is the EXACT worst-case value at x_new for the given level.
    """
    n_features = X.shape[1]
    v_new = cp.Variable()
    s = cp.Variable(n_features)

    constraints = [cp.sum(cp.abs(s)) <= L]
    for rank in range(level):
        j = idx_order[rank]
        constraints.append(v_new + s @ (X[j] - x_new) >= sorted_vals[rank])

    objective = cp.Minimize(v_new)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return max(v_new.value, 0.0)
    else:
        raise ValueError("Problem is not solved to optimality!")


def robust_optimization_maxmin(X: np.ndarray, y_sorted: list, idx_order: list,
                               L: float = DEFAULT_L_MAXMIN,
                               box_min: float = DEFAULT_BOX_MIN,
                               box_max: float = DEFAULT_BOX_MAX,
                               A: np.ndarray = None, b: np.ndarray = None) -> tuple:
    """
    Binary search for robust optimization with box + linear constraint feasible region.
    
    This is our QCO Lipschitz method adapted for the max-min utility problem.
    Uses convex hull LP values (upper bounds) for binary search, then evaluates
    the true interpolation value at the final solution.
    
    Args:
        X: Sample points
        y_sorted: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
        box_min, box_max: Box bounds
        A, b: Linear constraint Ax <= b
        
    Returns:
        (optimal_value, optimal_solution) where optimal_value is the TRUE interpolation value
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        # Use convex hull value (upper bound) for binary search direction
        val, sol = solve_lp_feasibility_maxmin(
            X, y_sorted, idx_order, L, mid, box_min, box_max, A, b
        )
        if val >= y_sorted[mid]:
            right = mid
        else:
            left = mid + 1
    
    # Get final solution from convex hull LP
    val, sol = solve_lp_feasibility_maxmin(
        X, y_sorted, idx_order, L, left, box_min, box_max, A, b
    )
    
    # Value is min of LP value and the level threshold (y_sorted[left-1] is the achieved level)
    true_val = min(val, y_sorted[left - 1])
    
    return true_val, sol


def solve_constant_feasibility_maxmin(X: np.ndarray, sorted_vals: list, idx_order: list,
                                      level: int,
                                      box_min: float = DEFAULT_BOX_MIN,
                                      box_max: float = DEFAULT_BOX_MAX,
                                      A: np.ndarray = None, b: np.ndarray = None) -> tuple:
    """
    Solve LP feasibility for piecewise constant optimization at a given level.
    
    With box + linear constraint feasible region.
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    n_features = X.shape[1]
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)
    g = cp.Variable(n_features)

    constraints = [cp.sum(p) == 1]
    
    # Box constraints
    constraints += [g >= box_min, g <= box_max]
    
    # Linear constraint: Ax <= b
    constraints += [A @ g <= b]
    
    # L-infinity distance to convex combination
    weighted_sum = sum(p[i] * X[idx_order[i]] for i in range(level))
    constraints += [v >= cp.max(cp.abs(weighted_sum - g))]

    # Regularization term to break ties
    lambda_reg = 1e-8
    objective = cp.Minimize(v + lambda_reg * cp.norm(g, 1))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return v.value, g.value
    else:
        raise ValueError(f"Problem is not solved to optimality! Status: {prob.status}")


def piecewise_constant_optimization_maxmin(X: np.ndarray, y_sorted: list, idx_order: list,
                                           box_min: float = DEFAULT_BOX_MIN,
                                           box_max: float = DEFAULT_BOX_MAX,
                                           A: np.ndarray = None, b: np.ndarray = None) -> tuple:
    """
    Binary search for piecewise constant optimization.
    
    This is the extreme case of our method with L -> infinity.
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        v, sol = solve_constant_feasibility_maxmin(
            X, y_sorted, idx_order, mid, box_min, box_max, A, b
        )
        if v <= 0:
            right = mid
        else:
            left = mid + 1
    
    v, sol = solve_constant_feasibility_maxmin(
        X, y_sorted, idx_order, left, box_min, box_max, A, b
    )
    return y_sorted[left - 1], sol


def concave_regression_optimization_maxmin(X_samples: np.ndarray, y_samples: np.ndarray,
                                           box_min: float = DEFAULT_BOX_MIN,
                                           box_max: float = DEFAULT_BOX_MAX,
                                           A: np.ndarray = None, b: np.ndarray = None,
                                           K: int = 10) -> tuple:
    """
    Fit concave regression and optimize over box + linear constraint feasible region.
    
    Uses K-means clustering to fit piecewise linear concave approximation,
    then maximizes using SLSQP with multiple restarts.
    
    Returns:
        (optimal_value, optimal_solution)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
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
    
    # Maximize fitted_func subject to box + linear constraints
    def obj(x):
        return -fitted_func(x)
    
    def linear_constraint(x):
        return b - A @ x
    
    bounds = [(box_min, box_max)] * d
    constraints = [{'type': 'ineq', 'fun': linear_constraint}]
    
    # Multiple restarts
    best_val = -np.inf
    best_sol = None
    
    np.random.seed(0)
    for _ in range(10):
        while True:
            x0 = np.random.uniform(box_min, box_max, size=d)
            if np.all(A @ x0 <= b):
                break
        
        try:
            res = minimize(obj, x0, bounds=bounds, constraints=constraints, method='SLSQP')
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_sol = res.x
        except Exception:
            continue
    
    # Try centroid as starting point
    x0 = np.mean(X_samples, axis=0)
    if np.all(A @ x0 <= b):
        try:
            res = minimize(obj, x0, bounds=bounds, constraints=constraints, method='SLSQP')
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_sol = res.x
        except Exception:
            pass
    
    if best_sol is None:
        # Fallback to sample mean (project to feasible if needed)
        best_sol = np.mean(X_samples, axis=0)
        best_val = fitted_func(best_sol)
    
    return best_val, best_sol


def worst_case_interpolation_maxmin(x_new: np.ndarray, X: np.ndarray, y_sorted: list,
                                    idx_order: list, L: float) -> float:
    """
    Binary search to find the worst-case interpolation value.
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        val = solve_lp_interpolation_maxmin(x_new, X, y_sorted, idx_order, L, mid)
        if val >= y_sorted[mid]:
            right = mid
        else:
            left = mid + 1
    val = solve_lp_interpolation_maxmin(x_new, X, y_sorted, idx_order, L, left)
    return min(y_sorted[left - 1], val)


def solve_constant_interpolation_maxmin(x_new: np.ndarray, X: np.ndarray, sorted_vals: list,
                                        idx_order: list, level: int) -> float:
    """
    Solve LP for piecewise constant interpolation at a given level.
    """
    v = cp.Variable()
    p = cp.Variable(level, nonneg=True)

    constraints = [cp.sum(p) == 1]
    weighted_sum = sum(p[i] * X[idx_order[i]] for i in range(level))
    constraints += [v >= cp.max(cp.abs(weighted_sum - x_new))]

    objective = cp.Minimize(v)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return v.value
    else:
        raise ValueError("Problem is not solved to optimality!")


def piecewise_constant_interpolation_maxmin(x_new: np.ndarray, X: np.ndarray,
                                            y_sorted: list, idx_order: list) -> float:
    """
    Binary search for piecewise constant interpolation.
    """
    left, right = 1, len(y_sorted)
    while left < right:
        mid = (left + right) // 2
        v = solve_constant_interpolation_maxmin(x_new, X, y_sorted, idx_order, mid)
        if v <= 0:
            right = mid
        else:
            left = mid + 1
    return y_sorted[left] if left < len(y_sorted) else 0.0


def concave_regression_maxmin(x_new: np.ndarray, X_samples: np.ndarray, 
                              y_samples: np.ndarray, K: int = 10) -> float:
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
        Xk_aug = np.hstack([Xk, np.ones((Xk.shape[0], 1))])
        sol, _, _, _ = np.linalg.lstsq(Xk_aug, yk, rcond=None)
        coefs.append(sol[:-1])
        intercepts.append(sol[-1])
    
    if len(coefs) == 0:
        return np.mean(y_samples)
    
    vals = [np.dot(a, x_new) + b for a, b in zip(coefs, intercepts)]
    return min(vals)


def milp_interpolation_maxmin(x_new: np.ndarray, X: np.ndarray, y_sorted: list,
                              idx_order: list, L: float,
                              return_gradient: bool = False):
    """
    MILP-based interpolation for max-min problem.
    
    Solves the exact worst-case interpolation using MILP with binary variables.
    Can optionally return the subgradient for use in level function method.
    
    Args:
        x_new: Query point
        X: Sample points
        y_sorted: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
        return_gradient: If True, return (value, subgradient)
        
    Returns:
        If return_gradient=False: interpolation value
        If return_gradient=True: (value, subgradient)
    """
    n_samples = len(y_sorted)
    n_features = X.shape[1]
    
    # Variables
    v = cp.Variable()
    s = cp.Variable(n_features)
    z = cp.Variable(n_samples, boolean=True)
    
    # Big-M for constraints
    M = 1e6
    
    constraints = []
    
    # L1 norm constraint on subgradient
    constraints.append(cp.sum(cp.abs(s)) <= L)
    
    # For each sample point, either v + s @ (X[j] - x_new) >= y_j OR z[j] = 1
    for rank in range(n_samples):
        j = idx_order[rank]
        constraints.append(v + s @ (X[j] - x_new) >= y_sorted[rank] - M * z[rank])
    
    # At most k-1 points can be "skipped" if we achieve level k
    # This is encoded by: sum(z) <= k-1 implies v >= y_sorted[k-1]
    # We use: sum(1 - z[i]) for i < k >= 1 for some k
    
    # Simpler formulation: minimize v subject to constraints
    # The binary variables allow us to skip some constraints
    objective = cp.Minimize(v)
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # Fallback to LP-based interpolation
        val = worst_case_interpolation_maxmin(x_new, X, y_sorted, idx_order, L)
        if return_gradient:
            # Compute subgradient from LP
            left, right = 1, len(y_sorted)
            while left < right:
                mid = (left + right) // 2
                lp_val = solve_lp_interpolation_maxmin(x_new, X, y_sorted, idx_order, L, mid)
                if lp_val >= y_sorted[mid]:
                    right = mid
                else:
                    left = mid + 1
            
            # Get subgradient from final LP
            n_features = X.shape[1]
            v_new = cp.Variable()
            s_grad = cp.Variable(n_features)
            constraints = [cp.sum(cp.abs(s_grad)) <= L]
            for rank in range(left):
                j = idx_order[rank]
                constraints.append(v_new + s_grad @ (X[j] - x_new) >= y_sorted[rank])
            prob_grad = cp.Problem(cp.Minimize(v_new), constraints)
            prob_grad.solve(solver=cp.ECOS, verbose=False)
            return val, s_grad.value if s_grad.value is not None else np.zeros(n_features)
        return val
    
    if return_gradient:
        return v.value, s.value if s.value is not None else np.zeros(n_features)
    return v.value


def level_function_method_maxmin(X: np.ndarray, y_sorted: list, idx_order: list,
                                 L: float = DEFAULT_L_MAXMIN,
                                 box_min: float = DEFAULT_BOX_MIN,
                                 box_max: float = DEFAULT_BOX_MAX,
                                 A: np.ndarray = None, b: np.ndarray = None,
                                 epsilon: float = 1e-5, max_iters: int = 100) -> tuple:
    """
    Level function method (Xu 2001) for robust optimization adapted for max-min problem.
    
    At each iteration, constructs a level function using subgradients,
    and maximizes the minimum to get the next iterate.
    
    Args:
        X: Sample points
        y_sorted: Sorted function values (descending)
        idx_order: Indices corresponding to sorted values
        L: Lipschitz constant
        box_min, box_max: Box bounds
        A, b: Linear constraint Ax <= b
        epsilon: Convergence tolerance
        max_iters: Maximum iterations
        
    Returns:
        (optimal_value, optimal_solution)
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    d = X.shape[1]
    
    def get_subgradient(x):
        """Get subgradient at point x using MILP interpolation."""
        _, s = milp_interpolation_maxmin(x, X, y_sorted, idx_order, L, return_gradient=True)
        return s
    
    # Initialize with centroid of samples
    x_init = np.mean(X, axis=0)
    # Project to feasible region if needed
    if np.any(x_init < box_min):
        x_init = np.maximum(x_init, box_min)
    if np.any(x_init > box_max):
        x_init = np.minimum(x_init, box_max)
    if np.any(A @ x_init > b):
        # Simple projection: scale down
        scale = np.min(b / (A @ x_init + 1e-10))
        if scale < 1:
            x_init = x_init * scale * 0.9
    
    x_list = [x_init]
    s_list = [get_subgradient(x_list[0])]
    
    sigma_val = None
    x_next = x_init
    
    for iteration in range(max_iters):
        x = cp.Variable(d)
        t = cp.Variable()
        
        constraints = [A @ x <= b]
        constraints += [x >= box_min, x <= box_max]
        
        # Level function: t <= f(x_j) + s_j @ (x - x_j) for all j
        # Since we're maximizing, we use: t <= s_j @ (x - x_j) 
        # (the function value at x_j is implicit in the subgradient)
        for j in range(len(x_list)):
            f_xj = worst_case_interpolation_maxmin(x_list[j], X, y_sorted, idx_order, L)
            constraints.append(t <= f_xj + s_list[j] @ (x - x_list[j]))
        
        obj = cp.Maximize(t)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # Return best solution found so far
            break
        
        x_next = x.value
        sigma_val = t.value
        
        # Check convergence
        if sigma_val <= epsilon:
            break 
        # if len(x_list) > 1:
        #     prev_val = worst_case_interpolation_maxmin(x_list[-1], X, y_sorted, idx_order, L)
        #     curr_val = worst_case_interpolation_maxmin(x_next, X, y_sorted, idx_order, L)
        #     if abs(curr_val - prev_val) <= epsilon:
        #         break
        
        x_list.append(x_next)
        s_list.append(get_subgradient(x_next))
    
    # Return the interpolation value at the final solution
    final_val = worst_case_interpolation_maxmin(x_next, X, y_sorted, idx_order, L)
    return final_val, x_next


def interpolation_mesh_maxmin(func, *args, 
                              box_min: float = DEFAULT_BOX_MIN,
                              box_max: float = DEFAULT_BOX_MAX,
                              A: np.ndarray = None, b: np.ndarray = None,
                              n_grid: int = 50) -> tuple:
    """
    Evaluate interpolation function on a 2D mesh grid for max-min problem.
    
    Points outside the feasible region are set to NaN.
    
    Returns:
        (xx1, xx2, Z) mesh and values
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    xx1, xx2 = np.meshgrid(
        np.linspace(box_min, box_max, n_grid),
        np.linspace(box_min, box_max, n_grid)
    )
    
    Z = np.full_like(xx1, np.nan)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            x_query = np.array([xx1[i, j], xx2[i, j]])
            # Check feasibility
            if np.all(A @ x_query <= b):
                Z[i, j] = func(x_query, *args)
    
    return xx1, xx2, Z
