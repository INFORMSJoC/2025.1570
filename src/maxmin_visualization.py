"""
Visualization utilities for Max-Min Utility Problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

from .maxmin_utility import (
    min_utility_function,
    sigmoidal_utility,
    create_maxmin_mesh_grid,
    DEFAULT_BOX_MIN,
    DEFAULT_BOX_MAX,
    DEFAULT_A_MAXMIN,
    DEFAULT_B_MAXMIN,
    DEFAULT_KAPPA,
    DEFAULT_D0,
    DEFAULT_SAMPLE_EXTENSION,
)


def plot_maxmin_true_function_3d(population: np.ndarray,
                                  kappa: float = DEFAULT_KAPPA,
                                  d0: float = DEFAULT_D0,
                                  box_min: float = DEFAULT_BOX_MIN,
                                  box_max: float = DEFAULT_BOX_MAX,
                                  A: np.ndarray = None, b: np.ndarray = None,
                                  n_grid: int = 50,
                                  save_path: str = None):
    """Plot 3D surface of the true min-utility function over the full box region."""
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    # Create grid over the full box (no masking)
    x1 = np.linspace(box_min, box_max, n_grid)
    x2 = np.linspace(box_min, box_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    Z = min_utility_function(X_grid, population, kappa, d0).reshape(X1.shape)
    # No masking - show full function
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.9, 
                           edgecolor='none', antialiased=True)
    
    # Draw the constraint boundary on the surface (x1 + x2 = b)
    if A.shape[0] == 1:
        x_line = np.linspace(box_min, min(box_max, b[0]), 100)
        y_line = np.clip(b[0] - x_line, box_min, box_max)
        valid = (y_line >= box_min) & (y_line <= box_max) & (x_line >= box_min) & (x_line <= box_max)
        x_line = x_line[valid]
        y_line = y_line[valid]
        # Get Z values along the constraint line
        z_line = min_utility_function(np.column_stack([x_line, y_line]), population, kappa, d0)
        ax.plot(x_line, y_line, z_line, 'r-', linewidth=3, label=f'$x_1+x_2={b[0]:.0f}$')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('$\\min_i u_i(x)$', fontsize=12)
    ax.set_title('True Min-Utility Function', fontsize=14)
    ax.legend(loc='upper left')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Min utility')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_maxmin_true_function_contour(population: np.ndarray,
                                       kappa: float = DEFAULT_KAPPA,
                                       d0: float = DEFAULT_D0,
                                       box_min: float = DEFAULT_BOX_MIN,
                                       box_max: float = DEFAULT_BOX_MAX,
                                       A: np.ndarray = None, b: np.ndarray = None,
                                       n_grid: int = 100,
                                       show_population: bool = True,
                                       optimal_sol: np.ndarray = None,
                                       save_path: str = None,
                                       extension: float = 0.0):
    """Plot contour of the true min-utility function with feasible region boundary.
    
    Shows the function over the box region with the linear constraint boundary marked.
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    # Grid within the box [box_min, box_max]^2
    x1 = np.linspace(box_min, box_max, n_grid)
    x2 = np.linspace(box_min, box_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    Z = min_utility_function(X_grid, population, kappa, d0).reshape(X1.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Filled contour (full region, no masking)
    contour = ax.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Min utility $\\min_i u_i(x)$')
    
    # Contour lines
    cs = ax.contour(X1, X2, Z, levels=15, colors='white', 
                    linewidths=0.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    # Draw linear constraint boundary (e.g., x1 + x2 = b for single constraint)
    if A.shape[0] == 1:
        x_line = np.linspace(box_min, box_max, 100)
        y_line = (b[0] - A[0, 0] * x_line) / A[0, 1]
        valid = (y_line >= box_min) & (y_line <= box_max) & (x_line >= box_min) & (x_line <= box_max)
        ax.plot(x_line[valid], y_line[valid], 'r--', linewidth=2.5, 
                label=f'Constraint: $x_1 + x_2 = {b[0]:.0f}$')
    
    # Show population locations
    if show_population:
        ax.scatter(population[:, 0], population[:, 1], c='cyan', s=100, 
                   marker='o', edgecolors='black', linewidths=2,
                   label='Individual locations', zorder=5)
        for i, p in enumerate(population):
            ax.annotate(f'{i+1}', (p[0]+0.15, p[1]+0.15), fontsize=10, color='black')
    
    # Show optimal solution
    if optimal_sol is not None:
        ax.scatter([optimal_sol[0]], [optimal_sol[1]], c='gold', s=200, 
                   marker='*', edgecolors='black', linewidths=2,
                   label='Optimal', zorder=6)
    
    ax.set_xlim(box_min - 0.3, box_max + 0.3)
    ax.set_ylim(box_min - 0.3, box_max + 0.3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Min-Utility Function Contour', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_individual_utilities(x: np.ndarray, population: np.ndarray,
                               kappa: float = DEFAULT_KAPPA,
                               d0: float = DEFAULT_D0,
                               save_path: str = None):
    """Plot bar chart of utilities for all individuals at location x."""
    n_individuals = population.shape[0]
    utilities = np.array([sigmoidal_utility(x.reshape(1, -1), p, kappa, d0)[0] 
                          for p in population])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if u == utilities.min() else '#3498db' for u in utilities]
    bars = ax.bar(range(1, n_individuals + 1), utilities, color=colors, 
                  edgecolor='white', linewidth=1.5)
    
    ax.axhline(y=utilities.min(), color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Min = {utilities.min():.3f}')
    ax.axhline(y=utilities.mean(), color='#2ecc71', linestyle=':', linewidth=2,
               label=f'Mean = {utilities.mean():.3f}')
    
    ax.set_xlabel('Individual', fontsize=12)
    ax.set_ylabel('Utility', fontsize=12)
    ax.set_title(f'Individual Utilities at x = ({x[0]:.2f}, {x[1]:.2f})', fontsize=14)
    ax.set_xticks(range(1, n_individuals + 1))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_interpolation_comparison_maxmin(population: np.ndarray,
                                          X_samples: np.ndarray,
                                          y_sorted: list,
                                          idx_order: list,
                                          Y_values: np.ndarray,
                                          L: float,
                                          kappa: float = DEFAULT_KAPPA,
                                          d0: float = DEFAULT_D0,
                                          box_min: float = DEFAULT_BOX_MIN,
                                          box_max: float = DEFAULT_BOX_MAX,
                                          A: np.ndarray = None, b: np.ndarray = None,
                                          n_grid: int = 40,
                                          save_path: str = None,
                                          extension: float = 0.0):
    """Plot comparison of different interpolation methods.
    
    Shows the function and interpolations within the box region,
    with the linear constraint boundary clearly marked.
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    from .maxmin_optimization import (
        worst_case_interpolation_maxmin,
        piecewise_constant_interpolation_maxmin,
        concave_regression_maxmin,
    )
    
    # Grid within the box [box_min, box_max]^2
    xx1, xx2 = np.meshgrid(
        np.linspace(box_min, box_max, n_grid),
        np.linspace(box_min, box_max, n_grid)
    )
    
    # Compute interpolations on grid
    Z_QC = np.zeros_like(xx1)
    Z_PC = np.zeros_like(xx1)
    Z_CR = np.zeros_like(xx1)
    
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            x_query = np.array([xx1[i, j], xx2[i, j]])
            Z_QC[i, j] = worst_case_interpolation_maxmin(x_query, X_samples, y_sorted, idx_order, L)
            Z_PC[i, j] = piecewise_constant_interpolation_maxmin(x_query, X_samples, y_sorted, idx_order)
            Z_CR[i, j] = concave_regression_maxmin(x_query, X_samples, Y_values)
    
    # True function on grid
    X_grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    Z_true = min_utility_function(X_grid, population, kappa, d0).reshape(xx1.shape)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Common colorbar range
    vmin = np.nanmin([Z_true, Z_QC, Z_PC, Z_CR])
    vmax = np.nanmax([Z_true, Z_QC, Z_PC, Z_CR])
    
    titles = ['True Function', 'QCO Lipschitz (Ours)', 
              'Piecewise Constant', 'Concave Regression']
    Z_list = [Z_true, Z_QC, Z_PC, Z_CR]
    
    for ax, Z, title in zip(axes.flat, Z_list, titles):
        im = ax.contourf(xx1, xx2, Z, levels=20, cmap='viridis', 
                        vmin=vmin, vmax=vmax, alpha=0.9)
        ax.contour(xx1, xx2, Z, levels=10, colors='white', linewidths=0.5, alpha=0.5)
        
        # Draw linear constraint boundary
        if A.shape[0] == 1:
            x_line = np.linspace(box_min, box_max, 100)
            y_line = (b[0] - A[0, 0] * x_line) / A[0, 1]
            valid = (y_line >= box_min) & (y_line <= box_max) & (x_line >= box_min) & (x_line <= box_max)
            ax.plot(x_line[valid], y_line[valid], 'k--', linewidth=2)
        
        # Sample points (only those within box)
        in_box = (X_samples[:, 0] >= box_min) & (X_samples[:, 0] <= box_max) & \
                 (X_samples[:, 1] >= box_min) & (X_samples[:, 1] <= box_max)
        ax.scatter(X_samples[in_box, 0], X_samples[in_box, 1], c='red', s=15, 
                   alpha=0.6, marker='.', zorder=3)
        
        # Population
        ax.scatter(population[:, 0], population[:, 1], c='white', s=60,
                   marker='o', edgecolors='black', linewidths=1.5, zorder=4)
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(box_min - 0.3, box_max + 0.3)
        ax.set_ylim(box_min - 0.3, box_max + 0.3)
        ax.set_aspect('equal')
    
    # Add colorbar on the right side without overlapping
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Min utility')
    
    plt.suptitle(f'Interpolation Comparison (Feasible region: $x_1+x_2 \\leq {b[0]:.0f}$)', 
                 fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_l1_error_maxmin(df_l1, save_path: str = None):
    """Plot L1 error vs sample size for max-min utility problem."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    markers = {'L1_QC': 'd', 'L1_PC': 'o', 'L1_CR': 's'}
    colors = {'L1_QC': '#e74c3c', 'L1_PC': '#3498db', 'L1_CR': '#2ecc71'}
    labels = {'L1_QC': 'QCO Lipschitz (Ours)', 
              'L1_PC': 'Piecewise Constant', 
              'L1_CR': 'Concave Regression'}
    
    for col in ['L1_QC', 'L1_PC', 'L1_CR']:
        ax.plot(df_l1['n_samples'], df_l1[col], 
                marker=markers[col], color=colors[col],
                linewidth=2, markersize=8, label=labels[col])
    
    ax.set_xlabel('Sample size $J$', fontsize=12)
    ax.set_ylabel('$L_1$ error', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title('$L_1$ Error vs. Sample Size (Max-Min Utility)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_fairness_comparison(fairness_results: dict, save_path: str = None):
    """
    Plot fairness metrics comparison across methods.

    Panels: Min Utility, Mean Utility, Std Utility, Price of Fairness (POF).

    POF is defined as the relative reduction in total utility under the fair
    solution compared to the utilitarian optimum (Bertsimas et al., 2011):
        POF = (sum_utilitarian - sum_fair) / sum_utilitarian.
    A lower POF means less efficiency is sacrificed for fairness.

    Args:
        fairness_results: dict with method names as keys and dicts of metrics
            as values. The dict must contain keys 'min_utility', 'mean_utility',
            'std_utility', and (if available) 'pof'.
    """
    methods = list(fairness_results.keys())

    has_pof = all('pof' in fairness_results[m] for m in methods)

    if has_pof:
        metrics      = ['min_utility', 'mean_utility', 'std_utility', 'pof']
        metric_labels = [
            'Min Utility\n(Higher is Better)',
            'Mean Utility\n(Higher is Better)',
            'Std Utility\n(Lower is Better)',
            'Price of Fairness\n(Lower is Better)',
        ]
        fig, axes_grid = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes_grid.flat
    else:
        metrics      = ['min_utility', 'mean_utility', 'std_utility']
        metric_labels = [
            'Min Utility\n(Higher is Better)',
            'Mean Utility\n(Higher is Better)',
            'Std Utility\n(Lower is Better)',
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = [fairness_results[m][metric] for m in methods]
        bars = ax.bar(methods, values, color=colors[:len(methods)],
                      edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        # Extend y-axis top by 15 % so bar labels fit inside the axes
        ymax = ax.get_ylim()[1]
        ax.set_ylim(top=ymax * 1.15)
        ax.set_ylabel(label, fontsize=11)
        ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Fairness Metrics Comparison', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_utility_distribution(utilities_dict: dict, save_path: str = None):
    """
    Plot utility distributions across individuals for different methods.
    
    Args:
        utilities_dict: dict with method names as keys and utility arrays as values
    """
    methods = list(utilities_dict.keys())
    n_methods = len(methods)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    positions = np.arange(len(utilities_dict[methods[0]]))
    width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        utilities = utilities_dict[method]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(positions + offset, utilities, width, 
                      label=method, color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Individual', fontsize=12)
    ax.set_ylabel('Utility', fontsize=12)
    ax.set_title('Utility Distribution Across Individuals by Method', fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i+1) for i in positions])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_solutions_on_map(population: np.ndarray, solutions: dict,
                          kappa: float = DEFAULT_KAPPA,
                          d0: float = DEFAULT_D0,
                          box_min: float = DEFAULT_BOX_MIN,
                          box_max: float = DEFAULT_BOX_MAX,
                          A: np.ndarray = None, b: np.ndarray = None,
                          n_grid: int = 80,
                          save_path: str = None,
                          extension: float = 0.0):
    """
    Plot different solutions on the utility contour map.
    
    Shows the function within the box region with the linear constraint boundary marked.
    
    Args:
        population: (n_individuals, 2) array
        solutions: dict with method names as keys and (x, y) solutions as values
    """
    if A is None:
        A = DEFAULT_A_MAXMIN
    if b is None:
        b = DEFAULT_B_MAXMIN
    
    # Grid within the box [box_min, box_max]^2
    x1 = np.linspace(box_min, box_max, n_grid)
    x2 = np.linspace(box_min, box_max, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    
    Z = min_utility_function(X_grid, population, kappa, d0).reshape(X1.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour plot (full region, no masking)
    contour = ax.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.8)
    ax.contour(X1, X2, Z, levels=15, colors='white', linewidths=0.3, alpha=0.5)
    plt.colorbar(contour, ax=ax, label='Min utility')
    
    # Draw linear constraint boundary
    if A.shape[0] == 1:
        x_line = np.linspace(box_min, box_max, 100)
        y_line = (b[0] - A[0, 0] * x_line) / A[0, 1]
        valid = (y_line >= box_min) & (y_line <= box_max) & (x_line >= box_min) & (x_line <= box_max)
        ax.plot(x_line[valid], y_line[valid], 'k--', linewidth=2.5,
                label=f'$x_1 + x_2 = {b[0]:.0f}$')
    
    # Population
    ax.scatter(population[:, 0], population[:, 1], c='white', s=100,
               marker='o', edgecolors='black', linewidths=2, label='Individuals', zorder=4)
    
    # Solutions
    markers = ['*', 'd', 's', '^', 'p']
    colors = ['gold', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, (method, sol) in enumerate(solutions.items()):
        ax.scatter([sol[0]], [sol[1]], c=colors[i % len(colors)], 
                   s=300, marker=markers[i % len(markers)],
                   edgecolors='black', linewidths=2, label=method, zorder=5)
    
    ax.set_xlim(box_min - 0.3, box_max + 0.3)
    ax.set_ylim(box_min - 0.3, box_max + 0.3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title('Solution Comparison on Min-Utility Landscape', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_computational_efficiency(df_time, save_path: str = None):
    """Plot computational time comparison."""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    if 'time_binary' in df_time.columns:
        ax.plot(df_time['n_samples'], df_time['time_binary'], 
                marker='d', color='#e74c3c', linewidth=2, markersize=8,
                label='Binary Search (Ours)')
    
    if 'time_level' in df_time.columns:
        ax.plot(df_time['n_samples'], df_time['time_level'], 
                marker='s', color='#3498db', linewidth=2, markersize=8,
                label='Level Function Method')
    
    ax.set_xlabel('Sample size $J$', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title('Computational Efficiency Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
