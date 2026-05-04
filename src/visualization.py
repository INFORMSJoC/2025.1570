"""
Visualization utilities for Robust Quasi-Concave Optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

from .core import true_function, create_mesh_grid


def plot_true_function_3d(x_min: float, x_max: float, alpha: np.ndarray,
                          cost_param: np.ndarray, n_grid: int = 50,
                          save_path: str = None):
    """Plot 3D surface of the true function."""
    X1, X2, X_grid = create_mesh_grid(x_min, x_max, n_grid)
    Z = true_function(X_grid, alpha=alpha, cost_param=cost_param).reshape(X1.shape)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_true_function_contour(x_min: float, x_max: float, alpha: np.ndarray,
                               cost_param: np.ndarray, n_grid: int = 50,
                               save_path: str = None):
    """Plot contour of the true function."""
    X1, X2, X_grid = create_mesh_grid(x_min, x_max, n_grid)
    Z = true_function(X_grid, alpha=alpha, cost_param=cost_param).reshape(X1.shape)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X1, X2, Z, levels=30, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='True function value')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_interpolation_3d(xx1: np.ndarray, xx2: np.ndarray, Z: np.ndarray,
                          X_samples: np.ndarray = None, Y_samples: np.ndarray = None,
                          title: str = None, save_path: str = None):
    """Plot 3D surface of interpolation with optional sample points."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx1, xx2, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    
    if X_samples is not None and Y_samples is not None:
        ax.scatter(X_samples[:, 0], X_samples[:, 1], Y_samples, 
                   c='red', marker='.', label='Samples')
    
    if title:
        ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_contour_comparison(xx1: np.ndarray, xx2: np.ndarray, Z_interp: np.ndarray,
                            alpha: np.ndarray, cost_param: np.ndarray,
                            save_path: str = None):
    """
    Plot contour comparison between interpolated and true function.
    """
    x_grid = np.column_stack([xx1.ravel(), xx2.ravel()])
    Z_true = true_function(x_grid, alpha=alpha, cost_param=cost_param).reshape(xx1.shape)
    
    z_min = min(Z_interp.min(), Z_true.min())
    z_max = max(Z_interp.max(), Z_true.max())
    levels = np.linspace(z_min, z_max, 10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filled contour of true function
    cf = ax.contourf(xx1, xx2, Z_true, levels=levels, cmap='viridis', alpha=0.9)
    
    # Contour of interpolation (red solid)
    contour1 = ax.contour(xx1, xx2, Z_interp, levels=levels,
                          colors='red', linewidths=1, alpha=0.95, linestyles='solid')
    ax.clabel(contour1, inline=True, fontsize=9, fmt="%.2f", colors='red')
    
    # Contour of true function (white dashed)
    contour2 = ax.contour(xx1, xx2, Z_true, levels=levels,
                          colors='white', linewidths=1, linestyles='dashed', alpha=0.95)
    ax.clabel(contour2, inline=True, fontsize=9, fmt="%.2f", colors='white')
    
    plt.colorbar(cf, ax=ax, shrink=0.8, aspect=20, label='True function value')
    
    legend_handles = [
        Line2D([0], [0], color='red', lw=1, label='Interpolation'),
        Line2D([0], [0], color='white', lw=1, linestyle='dashed', label='True function')
    ]
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, facecolor='white')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_l1_error(df_l1, save_path: str = None):
    """Plot L1 error vs sample size."""
    plt.figure(figsize=(7, 5))
    plt.plot(df_l1['n_samples'], df_l1['L1_PC'], marker='o', label='Piecewise Constant')
    plt.plot(df_l1['n_samples'], df_l1['L1_CR'], marker='s', label='Concave Regression')
    plt.plot(df_l1['n_samples'], df_l1['L1_QC'], marker='d', label='Quasiconcave')
    plt.xlabel('Sample size')
    plt.ylabel('L1 error')
    plt.yscale('log')
    plt.title('L1 error vs. sample size')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
