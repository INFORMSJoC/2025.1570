"""
Main script to run all experiments for Robust Quasi-Concave Optimization.

Experiments:
1. Visualization of true function and interpolations
2. L1 error comparison across sample sizes
3. Optimality gap comparison across methods
4. Scalability of interpolation methods
5. Scalability of robust optimization methods

Usage:
    python scripts/run_experiments.py [--experiment NAME] [--no-plots]
    
    NAME can be: all, visualize, l1_error, optimality_gap, scalability_interp, scalability_ro
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

from src import (
    true_function,
    compute_max_lipschitz,
    generate_samples,
    create_mesh_grid,
    worst_case_interpolation,
    piecewise_constant_interpolation,
    concave_regression,
    milp_interpolation_problem,
    interpolation_mesh,
    robust_optimization,
    piecewise_constant_optimization,
    true_optimization,
    concave_regression_optimization,
    level_function_method_optimization,
    plot_true_function_3d,
    plot_true_function_contour,
    plot_interpolation_3d,
    plot_contour_comparison,
    plot_l1_error,
    DEFAULT_X_MIN,
    DEFAULT_X_MAX,
    DEFAULT_L,
    DEFAULT_ALPHA,
    DEFAULT_COST,
    DEFAULT_A,
    DEFAULT_B,
)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Experiment configuration."""
    x_min = DEFAULT_X_MIN
    x_max = DEFAULT_X_MAX
    n_features = 2
    L = DEFAULT_L
    alpha = DEFAULT_ALPHA
    cost = DEFAULT_COST
    A = DEFAULT_A
    b = DEFAULT_B
    
    # Experiment settings
    n_samples_default = 200
    n_grid = 10 # 20
    seed = 0


def runtime(func, *args, **kwargs) -> float:
    """Measure runtime of a function call."""
    start = time.time()
    _ = func(*args, **kwargs)
    return time.time() - start


# ============================================================================
# Experiment 1: Visualization
# ============================================================================

def run_visualization(cfg: Config, show_plots: bool = True):
    """Run visualization experiments."""
    print("\n" + "=" * 60)
    print("Experiment 1: Visualization")
    print("=" * 60)
    
    # Compute Lipschitz constant
    max_l1, x_max_grad, grad_at_max = compute_max_lipschitz(
        cfg.x_min, cfg.x_max, cfg.alpha, cfg.cost
    )
    print(f"Max L1 norm of gradient: {max_l1:.6f} at x = {x_max_grad}")
    print(f"Gradient: {grad_at_max}")
    
    if not show_plots:
        print("Plots disabled. Skipping visualization.")
        return
    
    # Plot true function
    print("\nPlotting true function (3D surface)...")
    plot_true_function_3d(cfg.x_min, cfg.x_max, cfg.alpha, cfg.cost,
                          save_path='results/true_func.png')
    
    print("Plotting true function (contour)...")
    plot_true_function_contour(cfg.x_min, cfg.x_max, cfg.alpha, cfg.cost,
                               save_path='results/true_contour.png')
    
    # Generate samples
    X_samples, Y_values, y_sorted, idx_order = generate_samples(
        cfg.n_samples_default, cfg.n_features, cfg.x_min, cfg.x_max,
        cfg.alpha, cfg.cost, seed=cfg.seed
    )
    
    # Worst-case interpolation
    print("\nComputing worst-case (quasiconcave) interpolation...")
    args_qc = (X_samples, y_sorted, idx_order, cfg.L)
    xx1, xx2, Z_QC = interpolation_mesh(worst_case_interpolation, *args_qc,
                                        x_min=cfg.x_min, x_max=cfg.x_max, n_grid=cfg.n_grid)
    plot_interpolation_3d(xx1, xx2, Z_QC, X_samples, Y_values,
                          title='Quasiconcave Interpolation',
                          save_path='results/qco_int.png')
    
    # Piecewise constant interpolation
    print("Computing piecewise constant interpolation...")
    args_pc = (X_samples, y_sorted, idx_order)
    _, _, Z_PC = interpolation_mesh(piecewise_constant_interpolation, *args_pc,
                                    x_min=cfg.x_min, x_max=cfg.x_max, n_grid=cfg.n_grid)
    plot_interpolation_3d(xx1, xx2, Z_PC, X_samples, Y_values,
                          title='Piecewise Constant Interpolation',
                          save_path='results/const_int.png')
    
    # Concave regression
    print("Computing concave regression...")
    args_cr = (X_samples, Y_values)
    _, _, Z_CR = interpolation_mesh(concave_regression, *args_cr,
                                    x_min=cfg.x_min, x_max=cfg.x_max, n_grid=cfg.n_grid)
    plot_interpolation_3d(xx1, xx2, Z_CR, X_samples, Y_values,
                          title='Concave Regression',
                          save_path='results/co_reg.png')
    
    # Contour comparison
    print("Plotting contour comparison...")
    plot_contour_comparison(xx1, xx2, Z_QC, cfg.alpha, cfg.cost,
                            save_path='results/contour_compare.png')
    
    print("Visualization complete!")


# ============================================================================
# Experiment 2: L1 Error Comparison
# ============================================================================

def run_l1_error_experiment(cfg: Config, show_plots: bool = True):
    """Run L1 error comparison experiment."""
    print("\n" + "=" * 60)
    print("Experiment 2: L1 Error Comparison")
    print("=" * 60)
    
    np.random.seed(cfg.seed)
    sample_sizes = [20, 40, 80, 160, 320, 640, 1280, 2560]
    max_n_samples = max(sample_sizes)
    
    X_full = np.random.uniform(cfg.x_min, cfg.x_max, size=(max_n_samples, cfg.n_features))
    y_full = true_function(X_full, cfg.alpha, cfg.cost)
    
    results = []
    for n_samples in sample_sizes:
        print(f"  Processing n_samples = {n_samples}...")
        X_samples = X_full[:n_samples]
        y_samples = y_full[:n_samples]
        
        idx_order = list(np.argsort(y_samples)[::-1])
        y_sorted = [y_samples[i] for i in idx_order]
        
        # Compute interpolations
        xx1_pc, xx2_pc, Z_PC = interpolation_mesh(
            piecewise_constant_interpolation, X_samples, y_sorted, idx_order,
            x_min=cfg.x_min, x_max=cfg.x_max, n_grid=10
        )
        _, _, Z_CR = interpolation_mesh(
            concave_regression, X_samples, y_samples,
            x_min=cfg.x_min, x_max=cfg.x_max, n_grid=10
        )
        _, _, Z_QC = interpolation_mesh(
            worst_case_interpolation, X_samples, y_sorted, idx_order, cfg.L,
            x_min=cfg.x_min, x_max=cfg.x_max, n_grid=10
        )
        
        # True function on grid
        x_grid = np.column_stack([xx1_pc.ravel(), xx2_pc.ravel()])
        Z_true = true_function(x_grid, cfg.alpha, cfg.cost).reshape(xx1_pc.shape)
        
        # Compute L1 errors
        l1_pc = np.mean(np.abs(Z_PC - Z_true))
        l1_cr = np.mean(np.abs(Z_CR - Z_true))
        l1_qc = np.mean(np.abs(Z_QC - Z_true))
        
        results.append({
            'n_samples': n_samples,
            'L1_PC': l1_pc,
            'L1_CR': l1_cr,
            'L1_QC': l1_qc
        })
    
    df_l1 = pd.DataFrame(results)
    print("\nL1 Error Results:")
    print(df_l1.to_string(index=False))
    
    if show_plots:
        plot_l1_error(df_l1, save_path='results/L1.png')
    
    return df_l1


# ============================================================================
# Experiment 3: Optimality Gap Comparison
# ============================================================================

def run_optimality_gap_experiment(cfg: Config, n_samples_list=None, n_rep: int = 20):
    """Run optimality gap comparison experiment."""
    print("\n" + "=" * 60)
    print("Experiment 3: Optimality Gap Comparison")
    print("=" * 60)
    
    np.random.seed(cfg.seed)
    
    if n_samples_list is None:
        n_samples_list = [32, 64, 128, 256, 512, 1024]
    
    results = []
    max_n_samples = max(n_samples_list)
    
    # Get true optimum
    true_val, true_sol = true_optimization(cfg.alpha, cfg.cost, cfg.A, cfg.b, cfg.x_min, cfg.x_max)
    print(f"True optimal value: {true_val:.6f}")
    print(f"True optimal solution: {true_sol}")
    
    for rep in range(n_rep):
        print(f"  Repetition {rep + 1}/{n_rep}...")
        
        X_full = np.random.uniform(cfg.x_min, cfg.x_max, size=(max_n_samples, cfg.n_features))
        y_full = true_function(X_full, cfg.alpha, cfg.cost)
        
        for n_samples in n_samples_list:
            X_samples = X_full[:n_samples]
            y_samples = y_full[:n_samples]
            
            idx_order = list(np.argsort(y_samples)[::-1])
            y_sorted = [y_samples[i] for i in idx_order]
            
            # Robust optimization (quasiconcave)
            ro_val, ro_sol = robust_optimization(
                X_samples, y_sorted, idx_order, cfg.L, cfg.A, cfg.b, cfg.x_min, cfg.x_max
            )
            ro_true_val = true_function(np.array([ro_sol]), cfg.alpha, cfg.cost)[0]
            
            # Concave regression optimization
            cr_val, cr_sol = concave_regression_optimization(
                X_samples, y_samples, cfg.A, cfg.b, cfg.x_min, cfg.x_max
            )
            cr_true_val = true_function(np.array([cr_sol]), cfg.alpha, cfg.cost)[0]
            
            # Piecewise constant optimization
            pc_val, pc_sol = piecewise_constant_optimization(
                X_samples, y_sorted, idx_order, cfg.A, cfg.b, cfg.x_min, cfg.x_max
            )
            pc_true_val = true_function(np.array([pc_sol]), cfg.alpha, cfg.cost)[0]

            
            
            # Compute optimality gaps
            for method, val in [('qcv_ro', ro_true_val), ('constant', pc_true_val),
                                ('cv_reg', cr_true_val)]:
                results.append({
                    'n_samples': n_samples,
                    'method': method,
                    'opt_gap': (true_val - val) / true_val
                })
    
    # Aggregate results
    df_results = pd.DataFrame(results)
    summary_df = df_results.groupby(['n_samples', 'method'])['opt_gap'].agg(
        avg_opt_gap='mean',
        std_opt_gap='std',
        max_opt_gap='max'
    ).reset_index()
    
    print("\nOptimality Gap Summary:")
    print(summary_df.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'df_optimality_gap_{timestamp}.csv'
    summary_df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # Generate LaTeX table
    print("\nLaTeX Table:")
    print(generate_latex_table(summary_df))
    
    return summary_df


def generate_latex_table(summary_df):
    """Generate LaTeX table from summary DataFrame."""
    pivot_df = summary_df.pivot(index='method', columns='n_samples',
                                values=['avg_opt_gap', 'std_opt_gap', 'max_opt_gap'])
    
    n_samples_sorted = sorted(summary_df['n_samples'].unique())
    
    latex = []
    latex.append("\\begin{tabular}{l" + "c" * len(n_samples_sorted) + "}")
    latex.append("\\toprule")
    latex.append("Method & " + " & ".join([str(n) for n in n_samples_sorted]) + " \\\\")
    latex.append("\\midrule")
    
    for method in pivot_df.index:
        row = [method]
        for n in n_samples_sorted:
            try:
                avg = pivot_df.loc[method, ('avg_opt_gap', n)]
                std = pivot_df.loc[method, ('std_opt_gap', n)]
                maxv = pivot_df.loc[method, ('max_opt_gap', n)]
                if np.isnan(avg):
                    cell = "-"
                else:
                    cell = f"{avg*100:.1f} ({std*100:.1f}, {maxv*100:.1f})"
            except KeyError:
                cell = "-"
            row.append(cell)
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    
    return "\n".join(latex)


# ============================================================================
# Experiment 4: Scalability of Interpolation
# ============================================================================

def run_scalability_interpolation(cfg: Config, n_rep: int = 50):
    """Run interpolation scalability experiments."""
    print("\n" + "=" * 60)
    print("Experiment 4: Scalability of Interpolation")
    print("=" * 60)
    
    np.random.seed(cfg.seed)
    
    # 4.1 Scalability vs sample size
    print("\n4.1 Interpolation scalability (sample size)...")
    df_sample = run_scalability_sample(cfg, n_rep=n_rep)

    
    return df_sample


# ============================================================================
# Experiment 5: Scalability of Robust Optimization
# ============================================================================

def run_scalability_robust_optimization(cfg: Config, n_rep: int = 10):
    """Run robust optimization scalability experiments."""
    print("\n" + "=" * 60)
    print("Experiment 5: Scalability of Robust Optimization")
    print("=" * 60)
    
    np.random.seed(cfg.seed)
    
    df_ro = run_scalability_ro(cfg, n_rep=n_rep)
    
    return df_ro


def run_scalability_sample(cfg: Config, n_rep: int = 50):
    """Scalability test: interpolation vs sample size."""
    n_samples_list = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    x_new = np.random.uniform(cfg.x_min, cfg.x_max, cfg.n_features)
    
    results = []
    for n_samples in n_samples_list:
        print(f"  n_samples = {n_samples}...")
        t_binary_total = 0.0
        t_milp_total = 0.0
        
        for rep in range(n_rep):
            print(f"    rep {rep + 1}/{n_rep}...")
            X_samples = np.random.uniform(cfg.x_min, cfg.x_max, size=(n_samples, cfg.n_features))
            Y_lower = true_function(X_samples, cfg.alpha, cfg.cost)
            idx_order = list(np.argsort(Y_lower)[::-1])
            y_sorted = [Y_lower[i] for i in idx_order]
            
            t_binary_total += runtime(worst_case_interpolation, x_new, X_samples,
                                      y_sorted, idx_order, cfg.L)
            t_milp_total += runtime(milp_interpolation_problem, x_new, X_samples,
                                    y_sorted, idx_order, cfg.L)
        
        results.append({
            'n_samples': n_samples,
            'time_binary': t_binary_total / n_rep,
            'time_milp': t_milp_total / n_rep,
        })
    
    df = pd.DataFrame(results)
    print("\nInterpolation Scalability (sample size):")
    print(df.to_string(index=False))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'df_scalability_sample_{timestamp}.csv', index=False)
    
    return df


def run_scalability_ro(cfg: Config, n_rep: int = 10):
    """Scalability test: robust optimization vs sample size."""
    n_samples_list = [32, 64, 128, 256, 512, 1024]
    
    results = []
    for n_samples in n_samples_list:
        print(f"  n_samples = {n_samples}...")
        t_binary_total = 0.0
        t_level_total = 0.0
        
        for _ in range(n_rep):
            X_samples = np.random.uniform(cfg.x_min, cfg.x_max, size=(n_samples, cfg.n_features))
            Y_lower = true_function(X_samples, cfg.alpha, cfg.cost)
            idx_order = list(np.argsort(Y_lower)[::-1])
            y_sorted = [Y_lower[i] for i in idx_order]
            
            t_binary_total += runtime(robust_optimization, X_samples, y_sorted,
                                      idx_order, cfg.L, cfg.A, cfg.b, cfg.x_min, cfg.x_max)
            t_level_total += runtime(level_function_method_optimization, X_samples,
                                     y_sorted, idx_order, cfg.L, cfg.A, cfg.b,
                                     cfg.x_min, cfg.x_max)
        
        results.append({
            'n_samples': n_samples,
            'time_binary': t_binary_total / n_rep,
            'time_level': t_level_total / n_rep,
        })
    
    df = pd.DataFrame(results)
    print("\nRO Scalability (sample size):")
    print(df.to_string(index=False))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'df_scalability_ro_{timestamp}.csv', index=False)
    
    return df


def run_scalability_dimension(cfg: Config, n_rep: int = 20):
    """Scalability test: interpolation vs dimension."""
    n_feature_list = [2, 8, 64]
    n_samples = 200
    
    results = []
    for n_features in n_feature_list:
        print(f"  n_features = {n_features}...")
        t_binary_total = 0.0
        t_milp_total = 0.0
        
        for _ in range(n_rep):
            X_samples = np.random.uniform(cfg.x_min, cfg.x_max, size=(n_samples, n_features))
            alpha = np.random.rand(n_features + 1)
            alpha[1:] = alpha[1:] / alpha[1:].sum()
            cost = np.ones(n_features + 1)
            
            Y_lower = true_function(X_samples, alpha=alpha, cost_param=cost)
            idx_order = list(np.argsort(Y_lower)[::-1])
            y_sorted = [Y_lower[i] for i in idx_order]
            x_new = np.random.uniform(cfg.x_min, cfg.x_max, size=n_features)
            
            t_binary_total += runtime(worst_case_interpolation, x_new, X_samples,
                                      y_sorted, idx_order, cfg.L)
            t_milp_total += runtime(milp_interpolation_problem, x_new, X_samples,
                                    y_sorted, idx_order, cfg.L)
        
        results.append({
            'n_features': n_features,
            'time_binary': t_binary_total / n_rep,
            'time_milp': t_milp_total / n_rep,
        })
    
    df = pd.DataFrame(results)
    print("\nDimension Scalability:")
    print(df.to_string(index=False))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'df_scalability_dim_{timestamp}.csv', index=False)
    
    return df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Robust Quasi-Concave Optimization experiments')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'visualize', 'l1_error', 'optimality_gap', 
                                 'scalability_interp', 'scalability_ro'],
                        help='Which experiment to run')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    parser.add_argument('--n-rep', type=int, default=20,
                        help='Number of repetitions for statistical experiments')
    
    args = parser.parse_args()
    
    cfg = Config()
    show_plots = not args.no_plots
    
    print("=" * 60)
    print("Robust Quasi-Concave Optimization Experiments")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  x_min={cfg.x_min}, x_max={cfg.x_max}")
    print(f"  n_features={cfg.n_features}, L={cfg.L}")
    print(f"  alpha={cfg.alpha}")
    print(f"  cost={cfg.cost}")
    
    if args.experiment in ['all', 'visualize']:
        run_visualization(cfg, show_plots=show_plots)
    
    if args.experiment in ['all', 'l1_error']:
        run_l1_error_experiment(cfg, show_plots=show_plots)
    
    if args.experiment in ['all', 'optimality_gap']:
        run_optimality_gap_experiment(cfg, n_rep=args.n_rep)
    
    if args.experiment in ['all', 'scalability_interp']:
        run_scalability_interpolation(cfg, n_rep=args.n_rep)
    
    if args.experiment in ['all', 'scalability_ro']:
        run_scalability_robust_optimization(cfg, n_rep=min(10, args.n_rep))
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
