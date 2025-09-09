"""
Optimization Engine for Dynamic Factor Timing Strategy

Implements L2 regularized mean-variance optimization following the paper's methodology:
- Estimates mean vector μ and covariance matrix Σ from factor interactions
- Solves regularized optimization: w = (Σ + λI)^-1 (μ + λw₀) [Equation 3]
- Applies factor rotation to convert interaction weights to factor weights [Equation 4]
- Optimizes lambda parameter through validation periods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from scipy import linalg
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
import logging

logger = logging.getLogger(__name__)


class RegularizedOptimizer:
    """
    L2 regularized optimizer for dynamic factor timing.
    
    Implements the core optimization methodology from Equations 3-4 in the paper.
    """
    
    def __init__(self, lambda_grid: Optional[List[float]] = None, 
                 covariance_estimator: str = 'sample',
                 numerical_tolerance: float = 1e-8):
        """
        Initialize the regularized optimizer.
        
        Parameters:
        -----------
        lambda_grid : List[float], optional
            Grid of lambda values for validation. Default creates log-spaced grid.
        covariance_estimator : str
            Method for covariance estimation ('sample', 'ledoit_wolf')
        numerical_tolerance : float
            Tolerance for numerical operations
        """
        if lambda_grid is None:
            # Default lambda grid from 0 to 100,000 (as in paper)
            self.lambda_grid = [0.0] + list(np.logspace(-2, 5, 50))  # 51 values total
        else:
            self.lambda_grid = sorted(lambda_grid)
        
        self.covariance_estimator = covariance_estimator
        self.numerical_tolerance = numerical_tolerance
        
        # Storage for optimization results
        self.optimization_results = {}
        
        logger.info(f"RegularizedOptimizer initialized with {len(self.lambda_grid)} lambda values")
    
    def estimate_moments(self, interactions_data: pd.DataFrame, 
                        training_mask: Optional[pd.Series] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean vector μ and covariance matrix Σ from factor interactions.
        
        Parameters:
        -----------
        interactions_data : pd.DataFrame
            Factor interaction timeseries (T × M)
        training_mask : pd.Series, optional
            Boolean mask for training period. If None, uses all data.
            
        Returns:
        --------
        Tuple of (mean_vector, covariance_matrix)
        """
        # Apply training mask if provided
        if training_mask is not None:
            if len(training_mask) != len(interactions_data):
                raise ValueError("Training mask length must match interactions data")
            training_data = interactions_data[training_mask]
        else:
            training_data = interactions_data
        
        if len(training_data) == 0:
            raise ValueError("No training data available")
        
        # Remove any remaining NaN values
        training_data = training_data.dropna()
        
        if len(training_data) < 2:
            raise ValueError("Insufficient training data after removing NaNs")
        
        # Estimate mean vector μ (M × 1)
        mean_vector = training_data.mean().values
        
        # Estimate covariance matrix Σ (M × M)  
        if self.covariance_estimator == 'ledoit_wolf':
            # Use Ledoit-Wolf shrinkage estimator
            lw = LedoitWolf()
            covariance_matrix = lw.fit(training_data.values).covariance_
        elif self.covariance_estimator == 'sample':
            # Use sample covariance
            covariance_matrix = np.cov(training_data.values, rowvar=False, ddof=1)
        else:
            raise ValueError(f"Unknown covariance estimator: {self.covariance_estimator}")
        
        # Check for numerical issues
        if np.any(np.isnan(mean_vector)) or np.any(np.isnan(covariance_matrix)):
            raise ValueError("NaN values in estimated moments")
        
        # Check covariance matrix properties
        if not self._is_positive_semidefinite(covariance_matrix):
            warnings.warn("Covariance matrix is not positive semidefinite, adding regularization")
            covariance_matrix = self._regularize_covariance(covariance_matrix)
        
        logger.debug(f"Estimated moments: mean shape {mean_vector.shape}, cov shape {covariance_matrix.shape}")
        
        return mean_vector, covariance_matrix
    
    def solve_regularized_weights(self, mu: np.ndarray, sigma: np.ndarray, 
                                lambda_param: float, w0: np.ndarray) -> np.ndarray:
        """
        Solve regularized optimization problem: w = (Σ + λI)^-1 (μ + λw₀)
        
        This implements Equation 3 from the paper.
        
        Parameters:
        -----------
        mu : np.ndarray
            Mean vector (M × 1)
        sigma : np.ndarray  
            Covariance matrix (M × M)
        lambda_param : float
            Regularization parameter λ
        w0 : np.ndarray
            Default weight vector (M × 1)
            
        Returns:
        --------
        Optimal interaction weights (M × 1)
        """
        M = len(mu)
        
        # Validate inputs
        if sigma.shape != (M, M):
            raise ValueError(f"Covariance matrix shape {sigma.shape} != expected ({M}, {M})")
        if len(w0) != M:
            raise ValueError(f"Default weights length {len(w0)} != expected {M}")
        if lambda_param < 0:
            raise ValueError("Lambda parameter must be non-negative")
        
        # Create regularized system: (Σ + λI)
        I = np.eye(M)
        regularized_cov = sigma + lambda_param * I
        
        # Create right-hand side: (μ + λw₀)
        rhs = mu + lambda_param * w0
        
        # Solve linear system: w = (Σ + λI)^-1 (μ + λw₀)
        try:
            # Use Cholesky decomposition for efficiency if positive definite
            if lambda_param > self.numerical_tolerance:
                weights = linalg.solve(regularized_cov, rhs, assume_a='pos')
            else:
                # Use LU decomposition for λ=0 case
                weights = linalg.solve(regularized_cov, rhs)
        except linalg.LinAlgError as e:
            # Fallback to pseudo-inverse if system is singular
            warnings.warn(f"Linear system solving failed: {e}, using pseudo-inverse")
            weights = np.linalg.pinv(regularized_cov) @ rhs
        
        # Validate output
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            raise ValueError("Invalid weights computed (NaN or Inf)")
        
        return weights
    
    def apply_factor_rotation(self, interaction_weights: np.ndarray, 
                            rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Apply factor rotation to convert interaction weights to factor weights.
        
        Implements Equation 4: w_f = (w^T R)^T
        
        Parameters:
        -----------
        interaction_weights : np.ndarray
            Interaction weights (M × 1)
        rotation_matrix : np.ndarray
            Rotation matrix (M × N)
            
        Returns:
        --------
        Factor weights (N × 1)
        """
        M, N = rotation_matrix.shape
        
        if len(interaction_weights) != M:
            raise ValueError(f"Interaction weights length {len(interaction_weights)} != matrix rows {M}")
        
        # Apply rotation: w_f = R^T w
        factor_weights = rotation_matrix.T @ interaction_weights
        
        return factor_weights
    
    def rescale_factor_weights(self, factor_weights: np.ndarray) -> np.ndarray:
        """
        Rescale factor weights: -1 ≤ w_f ≤ 1 and sum(abs(w_f)) = 1
        
        Parameters:
        -----------
        factor_weights : np.ndarray
            Raw factor weights (N × 1)
            
        Returns:
        --------
        Rescaled factor weights (N × 1)
        """
        # Calculate sum of absolute weights
        abs_sum = np.sum(np.abs(factor_weights))
        
        if abs_sum <= self.numerical_tolerance:
            # Handle zero weights case - return equal weights
            N = len(factor_weights)
            return np.ones(N) / N
        
        # Rescale to sum(abs(w)) = 1
        rescaled_weights = factor_weights / abs_sum
        
        # Ensure bounds: -1 ≤ w_f ≤ 1 (should already be satisfied after rescaling)
        rescaled_weights = np.clip(rescaled_weights, -1.0, 1.0)
        
        return rescaled_weights
    
    def optimize_lambda_validation(self, interactions_data: pd.DataFrame,
                                 factor_list: List[str], rotation_matrices: Dict[str, np.ndarray],
                                 validation_periods: List[Tuple[str, str]]) -> float:
        """
        Find optimal λ that maximizes Sharpe ratio across validation periods.
        
        Parameters:
        -----------
        interactions_data : pd.DataFrame
            Factor interaction timeseries
        factor_list : List[str]
            List of factor names
        rotation_matrices : Dict[str, np.ndarray]
            Rotation matrices by date
        validation_periods : List[Tuple[str, str]]
            List of (start_date, end_date) for validation periods
            
        Returns:
        --------
        Optimal lambda value
        """
        if not validation_periods:
            raise ValueError("No validation periods provided")
        
        lambda_scores = {}
        
        for lambda_param in self.lambda_grid:
            total_sharpe = 0.0
            valid_periods = 0
            
            for period_start, period_end in validation_periods:
                try:
                    # Calculate Sharpe ratio for this lambda and period
                    sharpe = self._calculate_validation_sharpe(
                        interactions_data, factor_list, rotation_matrices,
                        lambda_param, period_start, period_end
                    )
                    
                    if not np.isnan(sharpe) and np.isfinite(sharpe):
                        total_sharpe += sharpe
                        valid_periods += 1
                
                except Exception as e:
                    logger.warning(f"Failed to calculate Sharpe for λ={lambda_param}, period {period_start}-{period_end}: {e}")
                    continue
            
            # Average Sharpe ratio across validation periods
            if valid_periods > 0:
                lambda_scores[lambda_param] = total_sharpe / valid_periods
            else:
                lambda_scores[lambda_param] = -np.inf
        
        if not lambda_scores or all(score == -np.inf for score in lambda_scores.values()):
            warnings.warn("No valid lambda scores computed, using λ=0")
            return 0.0
        
        # Find lambda with highest average Sharpe ratio
        optimal_lambda = max(lambda_scores, key=lambda_scores.get)
        
        logger.info(f"Optimal lambda: {optimal_lambda:.6f} with Sharpe: {lambda_scores[optimal_lambda]:.4f}")
        
        return optimal_lambda
    
    def _calculate_validation_sharpe(self, interactions_data: pd.DataFrame,
                                   factor_list: List[str], rotation_matrices: Dict[str, np.ndarray],
                                   lambda_param: float, start_date: str, end_date: str) -> float:
        """Calculate Sharpe ratio for validation period with given lambda."""
        
        # Get data for validation period
        period_mask = (interactions_data.index >= start_date) & (interactions_data.index <= end_date)
        period_data = interactions_data[period_mask]
        
        if len(period_data) < 2:
            return np.nan
        
        # Use training data up to validation start for moment estimation
        training_mask = interactions_data.index < start_date
        training_data = interactions_data[training_mask]
        
        if len(training_data) < 12:  # Minimum 12 months training
            return np.nan
        
        # Estimate moments from training data
        mu, sigma = self.estimate_moments(interactions_data, training_mask)
        
        # Create default weights
        w0 = self._create_default_weights(len(factor_list), len(interactions_data.columns))
        
        # Solve for optimal interaction weights
        interaction_weights = self.solve_regularized_weights(mu, sigma, lambda_param, w0)
        
        # Apply factor rotation (use middle date of validation period for rotation matrix)
        middle_date = period_data.index[len(period_data) // 2]
        if middle_date.strftime('%Y-%m-%d') in rotation_matrices:
            R = rotation_matrices[middle_date.strftime('%Y-%m-%d')]
        else:
            # Use closest available rotation matrix
            available_dates = list(rotation_matrices.keys())
            closest_date = min(available_dates, key=lambda x: abs(pd.to_datetime(x) - middle_date))
            R = rotation_matrices[closest_date]
        
        factor_weights = self.apply_factor_rotation(interaction_weights, R)
        factor_weights = self.rescale_factor_weights(factor_weights)
        
        # Calculate portfolio returns (simplified - assume factor returns available)
        # This would need factor returns for the validation period
        # For now, use a proxy calculation
        returns = period_data.iloc[:, :len(factor_list)].values @ factor_weights
        returns_series = pd.Series(returns, index=period_data.index)
        
        # Calculate Sharpe ratio
        if len(returns_series) < 2 or returns_series.std() <= self.numerical_tolerance:
            return np.nan
        
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(12)  # Annualized
        
        return sharpe_ratio
    
    def _create_default_weights(self, num_factors: int, total_interactions: int) -> np.ndarray:
        """
        Create default weight vector w₀.
        
        From paper: 1/N for pure factor returns, 0 for interactions.
        """
        w0 = np.zeros(total_interactions)
        
        # Set 1/N for pure factor components (first N elements)
        for i in range(num_factors):
            factor_start_idx = i * (total_interactions // num_factors)
            w0[factor_start_idx] = 1.0 / num_factors
        
        return w0
    
    def _is_positive_semidefinite(self, matrix: np.ndarray) -> bool:
        """Check if matrix is positive semidefinite."""
        try:
            eigenvals = np.linalg.eigvals(matrix)
            return np.all(eigenvals >= -self.numerical_tolerance)
        except np.linalg.LinAlgError:
            return False
    
    def _regularize_covariance(self, cov_matrix: np.ndarray, reg_param: float = 1e-6) -> np.ndarray:
        """Add small regularization to covariance matrix."""
        return cov_matrix + reg_param * np.eye(cov_matrix.shape[0])
    
    def calculate_portfolio_return(self, factor_weights: np.ndarray, 
                                 factor_returns: pd.Series) -> float:
        """
        Calculate portfolio return given factor weights and returns.
        
        Parameters:
        -----------
        factor_weights : np.ndarray
            Factor weights (N × 1)
        factor_returns : pd.Series
            Factor returns for the period (N × 1)
            
        Returns:
        --------
        Portfolio return
        """
        if len(factor_weights) != len(factor_returns):
            raise ValueError("Factor weights and returns dimensions must match")
        
        return np.dot(factor_weights, factor_returns.values)
    
    def get_optimization_summary(self) -> Dict[str, any]:
        """Get summary of optimization settings and results."""
        return {
            'lambda_grid_size': len(self.lambda_grid),
            'lambda_range': (min(self.lambda_grid), max(self.lambda_grid)),
            'covariance_estimator': self.covariance_estimator,
            'numerical_tolerance': self.numerical_tolerance,
            'optimization_runs': len(self.optimization_results)
        }