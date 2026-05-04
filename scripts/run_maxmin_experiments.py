"""
Experiments for Max-Min Utility Problem with Robust Quasi-Concave Optimization.

Experiments:
1. Sensitivity to Sample Size (L1 error comparison)
2. Fairness Analysis (utility distribution comparison)
3. Computational Efficiency (binary search vs level function method)

Usage:
    python scripts/run_maxmin_experiments.py [--experiment NAME] [--no-plots] [--n-rep N]
    
    NAME can be: all, visualize, l1_error, fairness, computational
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.maxmin_utility import (
    generate_population,
    generate_maxmin_samples,
    min_utility_function,
    sum_utility_function,
    true_maxmin_optimization,
    true_utilitarian_optimization,
    compute_individual_utilities,
    compute_fairness_metrics,
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

from src.maxmin_optimization import (
    robust_optimization_maxmin,
    piecewise_constant_optimization_maxmin,
    concave_regression_optimization_maxmin,
    worst_case_interpolation_maxmin,
    piecewise_constant_interpolation_maxmin,
    concave_regression_maxmin,
    interpolation_mesh_maxmin,
    level_function_method_maxmin,
)

from src.maxmin_visualization import (
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


# ============================================================================
# Configuration
# ============================================================================

class MaxMinConfig:
    """Configuration for max-min utility experiments."""
    # Problem parameters
    n_individuals = DEFAULT_N_INDIVIDUALS  # I = 10
    kappa = DEFAULT_KAPPA  # kappa = 0.8
    d0 = DEFAULT_D0  # d_0 = 3.0
    L = DEFAULT_L_MAXMIN  # Lipschitz constant for QCO
    
    # Feasible region: X = {x : x_1, x_2 in [0, 10], Ax <= b}
    box_min = DEFAULT_BOX_MIN
    box_max = DEFAULT_BOX_MAX
    A = DEFAULT_A_MAXMIN  # Linear constraint matrix
    b = DEFAULT_B_MAXMIN  # Linear constraint RHS (x1 + x2 <= 15)
    
    # Population location bounds
    pop_min = DEFAULT_POP_MIN
    pop_max = DEFAULT_POP_MAX
    
    # Experiment settings
    seed = 42
    n_grid = 30  # Grid size for visualization/L1 error


def runtime(func, *args, **kwargs) -> float:
    """Measure runtime of a function call."""
    start = time.time()
    _ = func(*args, **kwargs)
    return time.time() - start


# ============================================================================
# Experiment 0: Visualization
# ============================================================================

def run_visualization(cfg: MaxMinConfig, show_plots: bool = True):
    """Visualize the max-min utility problem setup and interpolations."""
    print("\n" + "=" * 60)
    print("Visualization: Max-Min Utility Problem")
    print("=" * 60)
    
    # Generate population
    population = generate_population(cfg.n_individuals, cfg.pop_min, cfg.pop_max, 
                                     seed=cfg.seed)
    print(f"Generated {cfg.n_individuals} individuals at locations:")
    for i, p in enumerate(population):
        print(f"  Individual {i+1}: ({p[0]:.2f}, {p[1]:.2f})")
    
    # Estimate Lipschitz constant
    L_est = compute_lipschitz_estimate(population, cfg.kappa, cfg.d0, 
                                        cfg.box_min, cfg.box_max, cfg.A, cfg.b)
    print(f"\nEstimated Lipschitz constant: {L_est:.4f}")
    print(f"Using L = {cfg.L} for optimization")
    
    # Find true optimum
    print("\nFinding true optimal solution...")
    true_val, true_sol = true_maxmin_optimization(
        population, cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b
    )
    print(f"True optimal value: {true_val:.6f}")
    print(f"True optimal solution: ({true_sol[0]:.4f}, {true_sol[1]:.4f})")
    
    if not show_plots:
        print("Plots disabled.")
        return population, true_val, true_sol
    
    # Plot true function
    print("\nPlotting true min-utility function (3D)...")
    plot_maxmin_true_function_3d(population, cfg.kappa, cfg.d0, 
                                  cfg.box_min, cfg.box_max, cfg.A, cfg.b,
                                  save_path='results/maxmin_true_func_3d.png')
    
    print("Plotting true min-utility function (contour)...")
    plot_maxmin_true_function_contour(population, cfg.kappa, cfg.d0,
                                       cfg.box_min, cfg.box_max, cfg.A, cfg.b,
                                       optimal_sol=true_sol,
                                       save_path='results/maxmin_true_contour.png')
    
    # Generate samples and show interpolations
    n_samples = 128
    X_samples, Y_values, y_sorted, idx_order = generate_maxmin_samples(
        n_samples, population, cfg.kappa, cfg.d0,
        cfg.box_min, cfg.box_max, cfg.A, cfg.b, seed=cfg.seed
    )
    
    print(f"\nPlotting interpolation comparison (J={n_samples})...")
    plot_interpolation_comparison_maxmin(
        population, X_samples, y_sorted, idx_order, Y_values, cfg.L,
        cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b,
        n_grid=cfg.n_grid,
        save_path='results/maxmin_interpolation_comparison.png'
    )
    
    # Plot utility distribution at optimal
    print("Plotting utility distribution at optimal solution...")
    plot_individual_utilities(true_sol, population, cfg.kappa, cfg.d0,
                               save_path='results/maxmin_utilities_optimal.png')
    
    print("Visualization complete!")
    return population, true_val, true_sol


# ============================================================================
# Experiment 1: Sensitivity to Sample Size (L1 Error)
# ============================================================================

def run_l1_error_experiment(cfg: MaxMinConfig, n_rep: int = 20, show_plots: bool = True,
                            n_eval_points: int = 100):
    """
    Experiment 1: Compare function approximation quality across different sample sizes.
    
    Metrics: L1 error between approximated function and true objective function.
    Uses a fixed set of random evaluation points instead of grid for efficiency.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: Sensitivity to Sample Size (L1 Error)")
    print("=" * 60)
    
    sample_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    # Generate fixed random evaluation points in the box (same for all replications)
    np.random.seed(cfg.seed + 99999)  # Different seed for eval points
    X_eval = np.random.uniform(cfg.box_min, cfg.box_max, size=(n_eval_points, 2))
    print(f"  Using {n_eval_points} random evaluation points in [{cfg.box_min}, {cfg.box_max}]^2")
    
    # Aggregated results across replications
    all_results = []
    
    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Replication {rep + 1}/{n_rep}...")
        
        # Generate new population for each replication
        population = generate_population(cfg.n_individuals, cfg.pop_min, cfg.pop_max,
                                         seed=cfg.seed + rep)
        
        # True function values at evaluation points
        Z_true = min_utility_function(X_eval, population, cfg.kappa, cfg.d0)
        
        for n_samples in sample_sizes:
            # Generate samples
            X_samples, Y_values, y_sorted, idx_order = generate_maxmin_samples(
                n_samples, population, cfg.kappa, cfg.d0,
                cfg.box_min, cfg.box_max, cfg.A, cfg.b,
                seed=cfg.seed + rep * 1000 + n_samples
            )
            
            # Compute interpolations at evaluation points
            Z_QC = np.zeros(n_eval_points)
            Z_PC = np.zeros(n_eval_points)
            Z_CR = np.zeros(n_eval_points)
            
            for i in range(n_eval_points):
                x = X_eval[i]
                Z_QC[i] = worst_case_interpolation_maxmin(
                    x, X_samples, y_sorted, idx_order, cfg.L
                )
                Z_PC[i] = piecewise_constant_interpolation_maxmin(
                    x, X_samples, y_sorted, idx_order
                )
                Z_CR[i] = concave_regression_maxmin(x, X_samples, Y_values)
            
            # Compute L1 errors
            l1_qc = np.mean(np.abs(Z_QC - Z_true))
            l1_pc = np.mean(np.abs(Z_PC - Z_true))
            l1_cr = np.mean(np.abs(Z_CR - Z_true))
            
            all_results.append({
                'rep': rep,
                'n_samples': n_samples,
                'L1_QC': l1_qc,
                'L1_PC': l1_pc,
                'L1_CR': l1_cr
            })
    
    # Aggregate results
    df_all = pd.DataFrame(all_results)
    df_summary = df_all.groupby('n_samples').agg({
        'L1_QC': ['mean', 'std', 'max'],
        'L1_PC': ['mean', 'std', 'max'],
        'L1_CR': ['mean', 'std', 'max']
    }).reset_index()
    
    # Flatten column names
    df_summary.columns = ['n_samples', 
                          'L1_QC_mean', 'L1_QC_std', 'L1_QC_max',
                          'L1_PC_mean', 'L1_PC_std', 'L1_PC_max',
                          'L1_CR_mean', 'L1_CR_std', 'L1_CR_max']
    
    print("\nL1 Error Summary (mean ± std, max):")
    print("-" * 80)
    print(f"{'J':<8} {'QCO Lipschitz':<25} {'Piecewise Const':<25} {'Concave Reg':<25}")
    print("-" * 80)
    for _, row in df_summary.iterrows():
        print(f"{int(row['n_samples']):<8} "
              f"{row['L1_QC_mean']:.4f}±{row['L1_QC_std']:.4f} ({row['L1_QC_max']:.4f})   "
              f"{row['L1_PC_mean']:.4f}±{row['L1_PC_std']:.4f} ({row['L1_PC_max']:.4f})   "
              f"{row['L1_CR_mean']:.4f}±{row['L1_CR_std']:.4f} ({row['L1_CR_max']:.4f})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_summary.to_csv(f'df_maxmin_l1_error_{timestamp}.csv', index=False)
    print(f"\nResults saved to df_maxmin_l1_error_{timestamp}.csv")
    
    # Plot
    if show_plots:
        df_plot = pd.DataFrame({
            'n_samples': df_summary['n_samples'],
            'L1_QC': df_summary['L1_QC_mean'],
            'L1_PC': df_summary['L1_PC_mean'],
            'L1_CR': df_summary['L1_CR_mean']
        })
        plot_l1_error_maxmin(df_plot, save_path='results/maxmin_l1_error.png')
    
    return df_summary


# ============================================================================
# Experiment 2: Fairness Analysis
# ============================================================================

def run_fairness_experiment(cfg: MaxMinConfig, n_rep: int = 200, show_plots: bool = True):
    """
    Experiment 2: Analyze the distribution of utilities across individuals.
    
    Compare:
    1. Optimal (perfect knowledge)
    2. QCO Lipschitz (our method)
    3. Concave regression
    4. Piecewise constant
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Fairness Analysis")
    print("=" * 60)
    
    n_samples = 128  # Fixed sample size
    
    all_results = []
    
    for rep in range(n_rep):
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  Replication {rep + 1}/{n_rep}...")
        
        # Generate population
        population = generate_population(cfg.n_individuals, cfg.pop_min, cfg.pop_max,
                                         seed=cfg.seed + rep)
        
        # Generate samples
        X_samples, Y_values, y_sorted, idx_order = generate_maxmin_samples(
            n_samples, population, cfg.kappa, cfg.d0,
            cfg.box_min, cfg.box_max, cfg.A, cfg.b, seed=cfg.seed + rep * 1000
        )
        
        # Find true max-min optimum
        true_val, true_sol = true_maxmin_optimization(
            population, cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )

        # Find utilitarian optimum (maximises sum of utilities)
        util_sum_val, util_sol = true_utilitarian_optimization(
            population, cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        # Sum of utilities at the utilitarian solution (denominator for POF)
        sum_utilitarian = sum_utility_function(
            util_sol, population, cfg.kappa, cfg.d0
        )[0]
        
        # Find solutions for each method
        solutions = {}
        
        # 1. Optimal (perfect knowledge)
        solutions['Optimal'] = true_sol
        
        # 2. QCO Lipschitz (our method)
        qco_val, qco_sol = robust_optimization_maxmin(
            X_samples, y_sorted, idx_order, cfg.L,
            cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        solutions['QCO Lipschitz'] = qco_sol
        
        # 3. Concave regression
        cr_val, cr_sol = concave_regression_optimization_maxmin(
            X_samples, Y_values, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        solutions['Concave Reg'] = cr_sol
        
        # 4. Piecewise constant
        pc_val, pc_sol = piecewise_constant_optimization_maxmin(
            X_samples, y_sorted, idx_order, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        solutions['Piecewise Const'] = pc_sol
        
        # Compute utilities, fairness metrics, and POF for each solution
        for method, sol in solutions.items():
            utilities = compute_individual_utilities(sol, population, cfg.kappa, cfg.d0)
            metrics = compute_fairness_metrics(utilities)

            # Price of Fairness: relative reduction in total utility vs utilitarian solution
            sum_fair = utilities.sum()
            pof = (sum_utilitarian - sum_fair) / sum_utilitarian if sum_utilitarian > 0 else np.nan
            
            all_results.append({
                'rep': rep,
                'method': method,
                'min_utility': metrics['min_utility'],
                'mean_utility': metrics['mean_utility'],
                'std_utility': metrics['std_utility'],
                'max_utility': metrics['max_utility'],
                'gini': metrics['gini'],
                'pof': pof,
            })
    
    # Aggregate results
    df_all = pd.DataFrame(all_results)
    df_summary = df_all.groupby('method').agg({
        'min_utility': ['mean', 'std'],
        'mean_utility': ['mean', 'std'],
        'std_utility': ['mean', 'std'],
        'gini': ['mean', 'std'],
        'pof': ['mean', 'std'],
    }).reset_index()
    
    print("\nFairness Analysis Summary:")
    print("-" * 120)
    print(f"{'Method':<18} {'Min Utility':<20} {'Mean Utility':<20} "
          f"{'Std Utility':<20} {'Gini':<20} {'POF':<15}")
    print("-" * 120)
    
    for _, row in df_summary.iterrows():
        method = row[('method', '')]
        min_u = f"{row[('min_utility', 'mean')]:.4f}±{row[('min_utility', 'std')]:.4f}"
        mean_u = f"{row[('mean_utility', 'mean')]:.4f}±{row[('mean_utility', 'std')]:.4f}"
        std_u = f"{row[('std_utility', 'mean')]:.4f}±{row[('std_utility', 'std')]:.4f}"
        gini = f"{row[('gini', 'mean')]:.4f}±{row[('gini', 'std')]:.4f}"
        pof = f"{row[('pof', 'mean')]:.4f}±{row[('pof', 'std')]:.4f}"
        print(f"{method:<18} {min_u:<20} {mean_u:<20} {std_u:<20} {gini:<20} {pof:<15}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df_all.to_csv(f'df_maxmin_fairness_{timestamp}.csv', index=False)
    print(f"\nResults saved to df_maxmin_fairness_{timestamp}.csv")
    
    # Generate LaTeX table
    print("\nLaTeX Table:")
    print(generate_fairness_latex_table(df_summary))
    
    # Visualization for a single representative case
    if show_plots:
        # Generate one instance for visualization
        population = generate_population(cfg.n_individuals, cfg.pop_min, cfg.pop_max,
                                         seed=cfg.seed)
        X_samples, Y_values, y_sorted, idx_order = generate_maxmin_samples(
            n_samples, population, cfg.kappa, cfg.d0,
            cfg.box_min, cfg.box_max, cfg.A, cfg.b, seed=cfg.seed
        )
        
        true_val, true_sol = true_maxmin_optimization(
            population, cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        
        qco_val, qco_sol = robust_optimization_maxmin(
            X_samples, y_sorted, idx_order, cfg.L,
            cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        
        cr_val, cr_sol = concave_regression_optimization_maxmin(
            X_samples, Y_values, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        
        pc_val, pc_sol = piecewise_constant_optimization_maxmin(
            X_samples, y_sorted, idx_order, cfg.box_min, cfg.box_max, cfg.A, cfg.b
        )
        
        solutions = {
            'Optimal': true_sol,
            'QCO Lipschitz': qco_sol,
            'Concave Reg': cr_sol,
            'Piecewise Const': pc_sol
        }
        
        # Plot solutions on map
        plot_solutions_on_map(population, solutions, cfg.kappa, cfg.d0,
                               cfg.box_min, cfg.box_max, cfg.A, cfg.b,
                               save_path='results/maxmin_solutions_map.png')
        
        # Plot utility distributions
        utilities_dict = {}
        for method, sol in solutions.items():
            utilities_dict[method] = compute_individual_utilities(
                sol, population, cfg.kappa, cfg.d0
            )
        
        plot_utility_distribution(utilities_dict, 
                                   save_path='results/maxmin_utility_distribution.png')
        
        # Plot fairness comparison (use mean values)
        fairness_for_plot = {}
        for method in solutions.keys():
            method_data = df_all[df_all['method'] == method]
            fairness_for_plot[method] = {
                'min_utility': method_data['min_utility'].mean(),
                'mean_utility': method_data['mean_utility'].mean(),
                'std_utility': method_data['std_utility'].mean(),
                'pof': method_data['pof'].mean(),
            }
        
        plot_fairness_comparison(fairness_for_plot,
                                  save_path='results/maxmin_fairness_comparison.png')
    
    return df_summary


def generate_fairness_latex_table(df_summary):
    """Generate LaTeX table from fairness summary."""
    latex = []
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("Method & Min Utility $\\uparrow$ & Mean Utility $\\uparrow$ & "
                 "Std Utility $\\downarrow$ & Gini $\\downarrow$ & POF $\\downarrow$ \\\\")
    latex.append("\\midrule")
    
    for _, row in df_summary.iterrows():
        method = row[('method', '')]
        min_u = f"{row[('min_utility', 'mean')]:.3f}$\\pm${row[('min_utility', 'std')]:.3f}"
        mean_u = f"{row[('mean_utility', 'mean')]:.3f}$\\pm${row[('mean_utility', 'std')]:.3f}"
        std_u = f"{row[('std_utility', 'mean')]:.3f}$\\pm${row[('std_utility', 'std')]:.3f}"
        gini = f"{row[('gini', 'mean')]:.3f}$\\pm${row[('gini', 'std')]:.3f}"
        pof = f"{row[('pof', 'mean')]:.3f}$\\pm${row[('pof', 'std')]:.3f}"
        latex.append(f"{method} & {min_u} & {mean_u} & {std_u} & {gini} & {pof} \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    
    return "\n".join(latex)


# ============================================================================
# Experiment 3: Computational Efficiency
# ============================================================================

def run_computational_experiment(cfg: MaxMinConfig, n_rep: int = 50, show_plots: bool = True):
    """
    Experiment 3: Compare computational efficiency.
    
    Compare running time of:
    1. Binary search algorithm (our method)
    2. Level function method (baseline that requires MILP at each iteration)
    
    Uses the max-min utility problem with:
    - Smoother sigmoidal function (kappa=0.3, d0=5.0)
    - Global optimum OUTSIDE the feasible region (population centroid ~(5,5), but x1+x2<=8)
    - Tight Lipschitz constant (L=0.15)
    - Samples from extended region beyond feasible box
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Computational Efficiency")
    print("=" * 60)
    
    # Use smaller sample sizes for level function method (it's much slower)
    sample_sizes = [512]
    
    # Generate a fixed population
    population = generate_population(cfg.n_individuals, cfg.pop_min, cfg.pop_max,
                                     seed=cfg.seed)
    pop_centroid = population.mean(axis=0)
    
    print(f"Configuration for computational experiment:")
    print(f"  Function: Min-utility with sigmoidal (kappa={cfg.kappa}, d0={cfg.d0})")
    print(f"  Population centroid: ({pop_centroid[0]:.2f}, {pop_centroid[1]:.2f})")
    print(f"  Feasible region: [{cfg.box_min}, {cfg.box_max}]^2 with x1+x2 <= {cfg.b[0]}")
    print(f"  Population centroid in feasible? {pop_centroid[0] + pop_centroid[1] <= cfg.b[0]}")
    print(f"  Tight Lipschitz constant: L = {cfg.L}")
    
    # First, demonstrate that QCO Lipschitz and QCO Constant give different results
    print("\n--- Demonstrating difference between QCO Lipschitz and QCO Constant ---")
    n_demo = 64
    X_demo, Y_demo, y_sorted_demo, idx_order_demo = generate_maxmin_samples(
        n_demo, population, cfg.kappa, cfg.d0,
        cfg.box_min, cfg.box_max, cfg.A, cfg.b, seed=cfg.seed
    )
    
    # QCO Lipschitz (with tight L)
    qco_lip_val, qco_lip_sol = robust_optimization_maxmin(
        X_demo, y_sorted_demo, idx_order_demo, cfg.L,
        cfg.box_min, cfg.box_max, cfg.A, cfg.b
    )
    
    # QCO Constant (piecewise constant, equivalent to L -> infinity)
    qco_const_val, qco_const_sol = piecewise_constant_optimization_maxmin(
        X_demo, y_sorted_demo, idx_order_demo,
        cfg.box_min, cfg.box_max, cfg.A, cfg.b
    )
    
    # True optimal on feasible region
    true_val, true_sol = true_maxmin_optimization(
        population, cfg.kappa, cfg.d0, cfg.box_min, cfg.box_max, cfg.A, cfg.b
    )
    
    print(f"\n  With J={n_demo} samples:")
    print(f"  True optimal in feasible region: x* = ({true_sol[0]:.4f}, {true_sol[1]:.4f}), f(x*) = {true_val:.4f}")
    print(f"  QCO Lipschitz (L={cfg.L}):  x = ({qco_lip_sol[0]:.4f}, {qco_lip_sol[1]:.4f}), "
          f"interp_val = {qco_lip_val:.4f}, true_f = {min_utility_function(qco_lip_sol, population, cfg.kappa, cfg.d0)[0]:.4f}")
    print(f"  QCO Constant (L=inf): x = ({qco_const_sol[0]:.4f}, {qco_const_sol[1]:.4f}), "
          f"interp_val = {qco_const_val:.4f}, true_f = {min_utility_function(qco_const_sol, population, cfg.kappa, cfg.d0)[0]:.4f}")
    print(f"  Solution difference: ||x_Lip - x_Const||_2 = {np.linalg.norm(qco_lip_sol - qco_const_sol):.4f}")
    
    print("\n--- Timing Comparison: Binary Search vs Level Function Method ---")
    
    results = []
    
    for n_samples in sample_sizes:
        print(f"  n_samples = {n_samples}...")
        
        t_binary_total = 0.0
        t_level_total = 0.0
        
        # Use fewer reps for larger sample sizes (level function is slow)
        actual_n_rep = n_rep if n_samples <= 128 else max(5, n_rep // 5)
        
        for rep in range(actual_n_rep):
            # Generate samples (extended beyond feasible region by default)
            X_samples, Y_values, y_sorted, idx_order = generate_maxmin_samples(
                n_samples, population, cfg.kappa, cfg.d0,
                cfg.box_min, cfg.box_max, cfg.A, cfg.b,
                seed=cfg.seed + rep * 1000 + n_samples
            )
            
            # Binary search (robust optimization with tight Lipschitz)
            t_binary_total += runtime(
                robust_optimization_maxmin,
                X_samples, y_sorted, idx_order, cfg.L,
                cfg.box_min, cfg.box_max, cfg.A, cfg.b
            )
            
            # Level function method (uses MILP at each iteration)
            t_level_total += runtime(
                level_function_method_maxmin,
                X_samples, y_sorted, idx_order, cfg.L,
                cfg.box_min, cfg.box_max, cfg.A, cfg.b
            )
        
        results.append({
            'n_samples': n_samples,
            'time_binary': t_binary_total / actual_n_rep,
            'time_level': t_level_total / actual_n_rep,
        })
    
    df = pd.DataFrame(results)
    
    print("\nComputational Efficiency Summary:")
    print("-" * 60)
    print(f"{'J':<10} {'Binary Search (s)':<20} {'Level Function (s)':<20}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{int(row['n_samples']):<10} {row['time_binary']:<20.4f} {row['time_level']:<20.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'df_maxmin_computational_{timestamp}.csv', index=False)
    print(f"\nResults saved to df_maxmin_computational_{timestamp}.csv")
    
    
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run Max-Min Utility experiments for Robust Quasi-Concave Optimization'
    )
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'visualize', 'l1_error', 'fairness', 'computational'],
                        help='Which experiment to run')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--n-rep', type=int, default=200,
                        help='Number of replications for statistical experiments')
    
    args = parser.parse_args()
    
    cfg = MaxMinConfig()
    show_plots = not args.no_plots
    
    print("=" * 60)
    print("Max-Min Utility Problem Experiments")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Number of individuals: {cfg.n_individuals}")
    print(f"  Sigmoid parameters: kappa={cfg.kappa}, d_0={cfg.d0}")
    print(f"  Lipschitz constant: L={cfg.L}")
    print(f"  Feasible region: box [0,10]^2, x1+x2 <= {cfg.b[0]}")
    print(f"  Population bounds: [{cfg.pop_min}, {cfg.pop_max}]^2")
    print(f"  Random seed: {cfg.seed}")
    
    if args.experiment in ['all', 'visualize']:
        run_visualization(cfg, show_plots=show_plots)
    
    if args.experiment in ['all', 'l1_error']:
        run_l1_error_experiment(cfg, n_rep=args.n_rep, show_plots=show_plots)
    
    if args.experiment in ['all', 'fairness']:
        run_fairness_experiment(cfg, n_rep=args.n_rep, show_plots=show_plots)
    
    if args.experiment in ['all', 'computational']:
        run_computational_experiment(cfg, n_rep=min(50, args.n_rep), show_plots=show_plots)
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
