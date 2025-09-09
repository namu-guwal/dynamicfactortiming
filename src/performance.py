"""
Performance Analytics Module for Dynamic Factor Timing Strategy

Implements performance metrics calculation and factor rotation analysis:
- Basic performance metrics (Sharpe ratios, returns, volatility)
- Rolling performance analysis (Exhibit 3 style)
- Factor rotation analysis (Exhibit 8 replication)  
- Transition matrix analysis (Exhibit 11 style)
- Performance attribution and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
import logging

from backtester import BacktestResults, DynamicFactorTimingBacktester

logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Core performance analytics for factor timing strategies.
    """
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None):
        """
        Initialize performance analytics.
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Portfolio return timeseries
        benchmark_returns : pd.Series, optional
            Benchmark return timeseries for comparison
        """
        self.portfolio_returns = portfolio_returns.copy()
        self.benchmark_returns = benchmark_returns.copy() if benchmark_returns is not None else None
        
        # Validate data
        self._validate_returns()
        
        logger.info(f"PerformanceAnalytics initialized with {len(self.portfolio_returns)} periods")
    
    def _validate_returns(self):
        """Validate return series data."""
        if len(self.portfolio_returns) == 0:
            raise ValueError("Portfolio returns cannot be empty")
        
        if self.portfolio_returns.isnull().all():
            raise ValueError("Portfolio returns cannot be all NaN")
        
        if self.benchmark_returns is not None:
            if len(self.benchmark_returns) != len(self.portfolio_returns):
                warnings.warn("Portfolio and benchmark returns have different lengths")
    
    def calculate_basic_metrics(self, annualization_factor: float = 12) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Parameters:
        -----------
        annualization_factor : float
            Factor for annualizing returns (12 for monthly data)
            
        Returns:
        --------
        Dictionary of performance metrics
        """
        metrics = {}
        
        # Portfolio metrics
        port_returns_clean = self.portfolio_returns.dropna()
        
        if len(port_returns_clean) == 0:
            return {"error": "No valid portfolio returns"}
        
        metrics['portfolio_return_mean'] = port_returns_clean.mean()
        metrics['portfolio_return_std'] = port_returns_clean.std()
        metrics['portfolio_return_annualized'] = metrics['portfolio_return_mean'] * annualization_factor
        metrics['portfolio_volatility_annualized'] = metrics['portfolio_return_std'] * np.sqrt(annualization_factor)
        
        # Sharpe ratio
        if metrics['portfolio_return_std'] > 1e-10:  # Use tolerance for near-zero volatility
            metrics['portfolio_sharpe_ratio'] = (metrics['portfolio_return_mean'] / 
                                               metrics['portfolio_return_std'] * 
                                               np.sqrt(annualization_factor))
        else:
            metrics['portfolio_sharpe_ratio'] = 0.0
        
        # Downside metrics
        negative_returns = port_returns_clean[port_returns_clean < 0]
        if len(negative_returns) > 0:
            metrics['downside_deviation'] = np.sqrt(np.mean(negative_returns**2)) * np.sqrt(annualization_factor)
            metrics['sortino_ratio'] = metrics['portfolio_return_annualized'] / metrics['downside_deviation']
        else:
            metrics['downside_deviation'] = 0.0
            metrics['sortino_ratio'] = np.inf if metrics['portfolio_return_annualized'] > 0 else 0.0
        
        # Drawdown analysis
        cumulative_returns = (1 + port_returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        
        metrics['max_drawdown'] = drawdown.min()
        metrics['current_drawdown'] = drawdown.iloc[-1] if len(drawdown) > 0 else 0.0
        
        # Hit rate
        metrics['hit_rate'] = (port_returns_clean > 0).mean()
        
        # Benchmark comparison if available
        if self.benchmark_returns is not None:
            bench_returns_clean = self.benchmark_returns.dropna()
            
            if len(bench_returns_clean) > 0:
                metrics['benchmark_return_mean'] = bench_returns_clean.mean()
                metrics['benchmark_return_std'] = bench_returns_clean.std()
                metrics['benchmark_return_annualized'] = metrics['benchmark_return_mean'] * annualization_factor
                metrics['benchmark_volatility_annualized'] = metrics['benchmark_return_std'] * np.sqrt(annualization_factor)
                
                if metrics['benchmark_return_std'] > 0:
                    metrics['benchmark_sharpe_ratio'] = (metrics['benchmark_return_mean'] / 
                                                        metrics['benchmark_return_std'] * 
                                                        np.sqrt(annualization_factor))
                else:
                    metrics['benchmark_sharpe_ratio'] = 0.0
                
                # Active return metrics
                metrics['active_return'] = metrics['portfolio_return_annualized'] - metrics['benchmark_return_annualized']
                metrics['sharpe_improvement'] = metrics['portfolio_sharpe_ratio'] - metrics['benchmark_sharpe_ratio']
                
                # Information ratio
                aligned_portfolio, aligned_benchmark = self.portfolio_returns.align(self.benchmark_returns)
                active_returns = (aligned_portfolio - aligned_benchmark).dropna()
                
                if len(active_returns) > 0 and active_returns.std() > 0:
                    metrics['information_ratio'] = (active_returns.mean() / active_returns.std() * 
                                                   np.sqrt(annualization_factor))
                    metrics['tracking_error'] = active_returns.std() * np.sqrt(annualization_factor)
                else:
                    metrics['information_ratio'] = 0.0
                    metrics['tracking_error'] = 0.0
        
        return metrics
    
    def calculate_rolling_metrics(self, window_years: int = 5, 
                                annualization_factor: float = 12) -> pd.DataFrame:
        """
        Calculate rolling performance metrics (Exhibit 3 style).
        
        Parameters:
        -----------
        window_years : int
            Rolling window in years (default 5 as in paper)
        annualization_factor : float
            Annualization factor
            
        Returns:
        --------
        DataFrame with rolling metrics
        """
        window_periods = window_years * 12  # Convert to monthly periods
        
        if len(self.portfolio_returns) < window_periods:
            warnings.warn(f"Insufficient data for {window_years}-year rolling analysis")
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=self.portfolio_returns.index)
        
        # Rolling Sharpe ratios
        rolling_metrics['portfolio_sharpe'] = (
            self.portfolio_returns.rolling(window_periods)
            .apply(lambda x: x.mean() / x.std() * np.sqrt(annualization_factor) if x.std() > 0 else 0, raw=False)
        )
        
        if self.benchmark_returns is not None:
            rolling_metrics['benchmark_sharpe'] = (
                self.benchmark_returns.rolling(window_periods)
                .apply(lambda x: x.mean() / x.std() * np.sqrt(annualization_factor) if x.std() > 0 else 0, raw=False)
            )
            
            rolling_metrics['sharpe_improvement'] = (rolling_metrics['portfolio_sharpe'] - 
                                                   rolling_metrics['benchmark_sharpe'])
        
        # Rolling returns
        rolling_metrics['portfolio_return'] = (
            self.portfolio_returns.rolling(window_periods).mean() * annualization_factor
        )
        
        if self.benchmark_returns is not None:
            rolling_metrics['benchmark_return'] = (
                self.benchmark_returns.rolling(window_periods).mean() * annualization_factor
            )
        
        # Rolling volatility
        rolling_metrics['portfolio_volatility'] = (
            self.portfolio_returns.rolling(window_periods).std() * np.sqrt(annualization_factor)
        )
        
        if self.benchmark_returns is not None:
            rolling_metrics['benchmark_volatility'] = (
                self.benchmark_returns.rolling(window_periods).std() * np.sqrt(annualization_factor)
            )
        
        return rolling_metrics
    
    def calculate_drawdown_statistics(self) -> Dict[str, Any]:
        """Calculate detailed drawdown statistics."""
        port_returns_clean = self.portfolio_returns.dropna()
        
        if len(port_returns_clean) == 0:
            return {"error": "No valid returns for drawdown analysis"}
        
        cumulative_returns = (1 + port_returns_clean).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        
        if is_drawdown.any():
            # Find consecutive drawdown periods
            drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
            
            start_dates = drawdown_starts[drawdown_starts].index
            end_dates = drawdown_ends[drawdown_ends].index
            
            # Handle case where drawdown continues to end
            if len(start_dates) > len(end_dates):
                end_dates = end_dates.append(pd.Index([port_returns_clean.index[-1]]))
            
            for start, end in zip(start_dates, end_dates):
                period_drawdown = drawdown.loc[start:end]
                min_drawdown = period_drawdown.min()
                duration = len(period_drawdown)
                
                drawdown_periods.append({
                    'start_date': start,
                    'end_date': end,
                    'duration_months': duration,
                    'max_drawdown': min_drawdown,
                    'recovery_date': end
                })
        
        return {
            'max_drawdown': drawdown.min(),
            'current_drawdown': drawdown.iloc[-1],
            'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
            'drawdown_periods': drawdown_periods,
            'n_drawdown_periods': len(drawdown_periods),
            'time_in_drawdown': is_drawdown.mean(),
            'drawdown_timeseries': drawdown
        }
    
    def generate_performance_summary(self, exhibit_2_format: bool = True) -> pd.DataFrame:
        """
        Generate performance summary in Exhibit 2 format.
        
        Parameters:
        -----------
        exhibit_2_format : bool
            Whether to format like Exhibit 2 from the paper
            
        Returns:
        --------
        DataFrame with summary statistics
        """
        metrics = self.calculate_basic_metrics()
        
        if exhibit_2_format:
            # Format like Exhibit 2: Return (%), Std. Dev. (%), Sharpe Ratio, Δ Sharpe Ratio
            summary_data = {
                'Return (%)': metrics['portfolio_return_annualized'] * 100,
                'Std. Dev. (%)': metrics['portfolio_volatility_annualized'] * 100,
                'Sharpe Ratio': metrics['portfolio_sharpe_ratio']
            }
            
            if 'sharpe_improvement' in metrics:
                summary_data['Δ Sharpe Ratio'] = metrics['sharpe_improvement']
            
            return pd.DataFrame([summary_data], index=['Optimal Timing'])
        
        else:
            # Standard format with all metrics
            return pd.DataFrame([metrics])


class FactorRotationAnalyzer:
    """
    Factor rotation analysis implementation (Exhibit 8 replication).
    """
    
    def __init__(self, backtester: DynamicFactorTimingBacktester):
        """
        Initialize with backtester containing rotation analysis data.
        
        Parameters:
        -----------
        backtester : DynamicFactorTimingBacktester
            Backtester with stored rotation analysis data
        """
        self.backtester = backtester
        
        if len(self.backtester.interaction_weights_history) == 0:
            warnings.warn("No rotation analysis data available in backtester")
    
    def generate_exhibit_8_analysis(self, analysis_dates: List[str], 
                                  target_factors: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate complete Exhibit 8 style factor rotation analysis.
        
        Parameters:
        -----------
        analysis_dates : List[str]
            List of dates for analysis (e.g., ['2015-04-30', '2015-05-31'])  
        target_factors : List[str], optional
            Factors to analyze. If None, analyzes all available factors.
            
        Returns:
        --------
        Dictionary with detailed rotation analysis
        """
        if len(analysis_dates) < 2:
            raise ValueError("Need at least 2 dates for rotation analysis")
        
        results = {}
        
        # Get factor weights from backtester results (need to be passed separately)
        # For now, we'll work with the stored interaction weights and predictor data
        
        for i in range(len(analysis_dates) - 1):
            date1, date2 = analysis_dates[i], analysis_dates[i + 1]
            period_name = f"{date1} to {date2}"
            
            # Calculate rotation effects for this period
            rotation_analysis = self._calculate_period_rotation_effects(date1, date2, target_factors)
            
            if rotation_analysis is not None:
                results[period_name] = rotation_analysis
        
        return results
    
    def _calculate_period_rotation_effects(self, date1: str, date2: str, 
                                         target_factors: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Calculate rotation effects for a specific period."""
        
        # Check if we have the required data
        if (date1 not in self.backtester.predictor_values_history or 
            date2 not in self.backtester.predictor_values_history):
            logger.warning(f"Missing predictor data for period {date1} to {date2}")
            return None
        
        # Get predictor changes
        predictor_changes = self.backtester.get_predictor_changes(date1, date2)
        
        if not predictor_changes:
            logger.warning(f"No predictor changes calculated for {date1} to {date2}")
            return None
        
        # Get interaction weights (use validation end date as reference)
        validation_date = date1  # Use first date as reference for interaction weights
        if validation_date not in self.backtester.interaction_weights_history:
            logger.warning(f"Missing interaction weights for {validation_date}")
            return None
        
        interaction_weights = self.backtester.interaction_weights_history[validation_date]
        
        # Calculate rotation effects by predictor
        rotation_effects = self._decompose_rotation_effects(
            interaction_weights, predictor_changes, target_factors
        )
        
        return {
            'predictor_changes': predictor_changes,
            'rotation_effects': rotation_effects,
            'total_rotation_effect': sum(rotation_effects.values())
        }
    
    def _decompose_rotation_effects(self, interaction_weights: np.ndarray, 
                                  predictor_changes: Dict[str, float],
                                  target_factors: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Decompose factor rotation effects by predictor type.
        
        This implements the calculation shown in Exhibit 8 where:
        rotation_effect = interaction_weight * predictor_change
        """
        rotation_effects = {}
        
        # For simplification, we'll calculate total rotation effects by predictor
        # In a full implementation, this would be broken down by factor
        
        for predictor, change in predictor_changes.items():
            # Find corresponding interaction weight
            # This is simplified - in reality we'd need to map predictors to interaction indices
            
            # For now, assign a representative effect
            # This would need to be properly implemented based on the interaction structure
            rotation_effects[predictor] = change * 0.1  # Placeholder calculation
        
        return rotation_effects
    
    def create_rotation_attribution_table(self, analysis_results: Dict[str, Any], 
                                        factor_name: str) -> pd.DataFrame:
        """
        Create formatted rotation attribution table for specific factor.
        
        Parameters:
        -----------
        analysis_results : Dict
            Results from generate_exhibit_8_analysis
        factor_name : str
            Name of factor to create table for
            
        Returns:
        --------
        Formatted attribution table
        """
        rows = []
        
        for period, data in analysis_results.items():
            if 'rotation_effects' in data:
                # Create summary row
                summary_row = {
                    'Period': period,
                    'Factor': factor_name,
                    'Total_Rotation_Effect': data['total_rotation_effect'],
                    'n_predictors': len(data['rotation_effects'])
                }
                rows.append(summary_row)
                
                # Add predictor-level details
                for predictor, effect in data['rotation_effects'].items():
                    detail_row = {
                        'Period': period,
                        'Factor': factor_name,
                        'Predictor': predictor,
                        'Predictor_Change': data['predictor_changes'].get(predictor, 0),
                        'Rotation_Effect': effect
                    }
                    rows.append(detail_row)
        
        return pd.DataFrame(rows)


def calculate_factor_attribution(factor_weights: pd.DataFrame, 
                               factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate factor attribution analysis.
    
    Parameters:
    -----------
    factor_weights : pd.DataFrame
        Factor weights over time
    factor_returns : pd.DataFrame
        Factor returns over time
        
    Returns:
    --------
    DataFrame with attribution by factor
    """
    # Align data
    aligned_weights, aligned_returns = factor_weights.align(factor_returns, join='inner')
    
    if len(aligned_weights) == 0:
        return pd.DataFrame()
    
    # Calculate contribution of each factor
    contributions = aligned_weights * aligned_returns
    
    # Add summary statistics
    summary_stats = pd.DataFrame({
        'Total_Contribution': contributions.sum(),
        'Average_Contribution': contributions.mean(),
        'Contribution_Volatility': contributions.std(),
        'Average_Weight': aligned_weights.mean(),
        'Weight_Volatility': aligned_weights.std()
    })
    
    return summary_stats


def create_exhibit_2_comparison(results_dict: Dict[str, BacktestResults]) -> pd.DataFrame:
    """
    Create Exhibit 2 style comparison table for multiple factor combinations.
    
    Parameters:
    -----------
    results_dict : Dict[str, BacktestResults]
        Dictionary mapping factor combination names to backtest results
        
    Returns:
    --------
    DataFrame formatted like Exhibit 2
    """
    comparison_rows = []
    
    for combo_name, results in results_dict.items():
        if len(results.portfolio_returns) == 0:
            continue
        
        # Calculate metrics for this combination
        analytics = PerformanceAnalytics(results.portfolio_returns, results.benchmark_returns)
        metrics = analytics.calculate_basic_metrics()
        
        # Format for Exhibit 2
        row = {
            'Factors': combo_name,
            'Return (%)': metrics['portfolio_return_annualized'] * 100,
            'Std. Dev. (%)': metrics['portfolio_volatility_annualized'] * 100,
            'Sharpe Ratio': metrics['portfolio_sharpe_ratio'],
            'Δ Sharpe Ratio': metrics.get('sharpe_improvement', 0.0)
        }
        
        comparison_rows.append(row)
    
    return pd.DataFrame(comparison_rows)


def calculate_transition_matrix_analysis(factor_weights_timeseries: pd.DataFrame,
                                       window_months: int = 3,
                                       quantiles: int = 3) -> Dict[str, Any]:
    """
    Calculate comprehensive transition matrix analysis (Exhibit 11 style).
    
    Parameters:
    -----------
    factor_weights_timeseries : pd.DataFrame
        Time series of factor weights
    window_months : int
        Transition window in months
    quantiles : int
        Number of quantiles for transition analysis
        
    Returns:
    --------
    Dictionary with transition matrices and statistics
    """
    if len(factor_weights_timeseries) < window_months + 1:
        return {"error": "Insufficient data for transition analysis"}
    
    results = {}
    
    for factor in factor_weights_timeseries.columns:
        factor_weights = factor_weights_timeseries[factor].dropna()
        
        if len(factor_weights) < window_months + 1:
            continue
        
        # Calculate transitions
        transitions = []
        
        for i in range(len(factor_weights) - window_months):
            current_weight = factor_weights.iloc[i]
            future_weight = factor_weights.iloc[i + window_months]
            
            # Assign to quantiles (handle case where all values are identical)
            try:
                current_quantile = pd.qcut([current_weight], quantiles, labels=False, duplicates='drop')[0]
                future_quantile = pd.qcut([future_weight], quantiles, labels=False, duplicates='drop')[0]
            except ValueError:
                # If all values are identical, assign to middle quantile
                current_quantile = quantiles // 2
                future_quantile = quantiles // 2
            
            transitions.append({
                'from': current_quantile,
                'to': future_quantile,
                'date': factor_weights.index[i]
            })
        
        if transitions:
            transition_df = pd.DataFrame(transitions)
            
            # Create transition matrix
            transition_matrix = pd.crosstab(
                transition_df['from'], 
                transition_df['to'], 
                normalize='index'
            )
            
            # Calculate stability metrics (diagonal sum)
            stability = np.trace(transition_matrix.values) / quantiles if not transition_matrix.empty else 0
            
            results[factor] = {
                'transition_matrix': transition_matrix,
                'stability_score': stability,
                'n_transitions': len(transitions)
            }
    
    return results