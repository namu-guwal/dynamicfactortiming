"""
Backtesting Framework for Dynamic Factor Timing Strategy

Implements the rolling window backtesting methodology from Exhibit 1:
- Expanding training windows (minimum 240 months)
- Annual validation periods for lambda optimization
- Out-of-sample testing periods
- Factor weight time series generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from datetime import datetime, timedelta
import logging

from data_processor import DataProcessor
from factor_interactions import FactorInteractionEngine
from optimizer import RegularizedOptimizer

logger = logging.getLogger(__name__)


class BacktestResults:
    """Container for backtesting results."""
    
    def __init__(self):
        self.factor_weights_timeseries = pd.DataFrame()
        self.portfolio_returns = pd.Series()
        self.benchmark_returns = pd.Series()
        self.lambda_timeseries = pd.Series()
        self.rotation_analysis_data = {}
        self.performance_metrics = {}
        self.backtest_periods = []
        
    def add_period_results(self, date: pd.Timestamp, factor_weights: np.ndarray, 
                          factor_names: List[str], portfolio_return: float,
                          benchmark_return: float, optimal_lambda: float):
        """Add results for a single backtesting period."""
        # Add factor weights
        weights_dict = {name: weight for name, weight in zip(factor_names, factor_weights)}
        if self.factor_weights_timeseries.empty:
            self.factor_weights_timeseries = pd.DataFrame([weights_dict], index=[date])
        else:
            new_row = pd.DataFrame([weights_dict], index=[date])
            self.factor_weights_timeseries = pd.concat([self.factor_weights_timeseries, new_row])
        
        # Add returns
        self.portfolio_returns.loc[date] = portfolio_return
        self.benchmark_returns.loc[date] = benchmark_return
        
        # Add lambda
        self.lambda_timeseries.loc[date] = optimal_lambda
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of backtest results."""
        if len(self.portfolio_returns) == 0:
            return {"error": "No backtest results available"}
        
        # Calculate basic performance metrics
        portfolio_mean = self.portfolio_returns.mean()
        portfolio_std = self.portfolio_returns.std()
        portfolio_sharpe = portfolio_mean / portfolio_std * np.sqrt(12) if portfolio_std > 0 else 0
        
        benchmark_mean = self.benchmark_returns.mean()
        benchmark_std = self.benchmark_returns.std()
        benchmark_sharpe = benchmark_mean / benchmark_std * np.sqrt(12) if benchmark_std > 0 else 0
        
        return {
            'n_periods': len(self.portfolio_returns),
            'date_range': (self.portfolio_returns.index.min(), self.portfolio_returns.index.max()),
            'portfolio_return_annualized': portfolio_mean * 12,
            'portfolio_volatility_annualized': portfolio_std * np.sqrt(12),
            'portfolio_sharpe_ratio': portfolio_sharpe,
            'benchmark_return_annualized': benchmark_mean * 12,
            'benchmark_volatility_annualized': benchmark_std * np.sqrt(12),
            'benchmark_sharpe_ratio': benchmark_sharpe,
            'sharpe_ratio_improvement': portfolio_sharpe - benchmark_sharpe,
            'average_lambda': self.lambda_timeseries.mean(),
            'lambda_range': (self.lambda_timeseries.min(), self.lambda_timeseries.max())
        }


class DynamicFactorTimingBacktester:
    """
    Main backtesting framework implementing the methodology from Exhibit 1.
    """
    
    def __init__(self, data_processor: DataProcessor, 
                 interaction_engine: FactorInteractionEngine,
                 optimizer: RegularizedOptimizer):
        """
        Initialize the backtester with core components.
        
        Parameters:
        -----------
        data_processor : DataProcessor
            Data processing component
        interaction_engine : FactorInteractionEngine
            Factor interaction generation component  
        optimizer : RegularizedOptimizer
            Regularized optimization component
        """
        self.data_processor = data_processor
        self.interaction_engine = interaction_engine
        self.optimizer = optimizer
        
        # Storage for detailed backtest data
        self.interaction_weights_history = {}
        self.predictor_values_history = {}
        self.rotation_matrices_history = {}
        self.standardized_predictors_history = {}
        
        logger.info("DynamicFactorTimingBacktester initialized")
    
    def run_full_backtest(self, factor_combination: List[str], 
                         start_date: str = '2005-12-31', 
                         end_date: str = '2023-12-31',
                         min_training_months: int = 240,
                         validation_months: int = 12,
                         testing_months: int = 12) -> BacktestResults:
        """
        Execute complete backtest following Exhibit 1 methodology.
        
        Parameters:
        -----------
        factor_combination : List[str]
            List of factor names to include
        start_date : str
            Start date for backtesting  
        end_date : str
            End date for backtesting
        min_training_months : int
            Minimum training period (default 240 months as in paper)
        validation_months : int
            Validation period length (default 12 months)
        testing_months : int
            Testing period length (default 12 months)
            
        Returns:
        --------
        BacktestResults object with complete results
        """
        # Validate factor combination
        self.data_processor.validate_factor_list(factor_combination)
        
        # Get aligned monthly data
        aligned_data = self.data_processor.align_data_for_period(start_date, end_date)
        
        # Generate backtest periods
        backtest_periods = self._generate_backtest_periods(
            aligned_data['factor_returns'].index,
            min_training_months, validation_months, testing_months
        )
        
        if len(backtest_periods) == 0:
            raise ValueError("No valid backtest periods generated")
        
        logger.info(f"Generated {len(backtest_periods)} backtest periods")
        
        # Initialize results container
        results = BacktestResults()
        results.backtest_periods = backtest_periods
        
        # Run backtest for each period
        for i, period in enumerate(backtest_periods):
            try:
                logger.debug(f"Processing period {i+1}/{len(backtest_periods)}: {period}")
                
                period_results = self._single_period_backtest(
                    period, factor_combination, aligned_data
                )
                
                if period_results is not None:
                    results.add_period_results(**period_results)
                    
            except Exception as e:
                logger.warning(f"Failed to process period {period}: {e}")
                continue
        
        logger.info(f"Backtest completed with {len(results.portfolio_returns)} periods")
        
        return results
    
    def _generate_backtest_periods(self, date_index: pd.DatetimeIndex,
                                 min_training_months: int,
                                 validation_months: int, 
                                 testing_months: int) -> List[Dict[str, Any]]:
        """
        Generate overlapping train/validate/test periods following Exhibit 1.
        """
        periods = []
        
        # Start from minimum training period
        if len(date_index) < min_training_months + validation_months + testing_months:
            warnings.warn("Insufficient data for backtesting")
            return periods
        
        # Create periods with expanding training windows
        for i in range(min_training_months + validation_months, 
                      len(date_index) - testing_months + 1, testing_months):
            
            # Training period (expanding window)
            train_start = date_index[0]
            train_end = date_index[i - validation_months - testing_months]
            
            # Validation periods (all previous validation periods)
            validation_periods = []
            for j in range(min_training_months, i - testing_months, testing_months):
                val_start = date_index[j]
                val_end = date_index[min(j + validation_months - 1, i - testing_months - 1)]
                validation_periods.append((val_start.strftime('%Y-%m-%d'), 
                                         val_end.strftime('%Y-%m-%d')))
            
            # Testing period  
            test_start = date_index[i - testing_months]
            test_end = date_index[i - 1]
            
            period = {
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'validation_periods': validation_periods,
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d'),
                'test_date': test_end  # Use end date for results
            }
            
            periods.append(period)
        
        return periods
    
    def _single_period_backtest(self, period: Dict[str, Any], 
                               factor_combination: List[str],
                               aligned_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Execute single iteration of train/validate/test cycle."""
        
        # Extract period dates
        train_start = period['train_start'] 
        train_end = period['train_end']
        validation_periods = period['validation_periods']
        test_start = period['test_start']
        test_end = period['test_end']
        test_date = period['test_date']
        
        try:
            # Step 1: Generate factor interactions for full period
            interactions_data = self._generate_interactions_for_period(
                factor_combination, aligned_data, train_start, test_end
            )
            
            # Step 2: Estimate moments using training data
            training_mask = ((interactions_data.index >= train_start) & 
                           (interactions_data.index <= train_end))
            
            if training_mask.sum() < 12:  # Minimum training data
                logger.warning(f"Insufficient training data for period ending {test_date}")
                return None
                
            mu, sigma = self.optimizer.estimate_moments(interactions_data, training_mask)
            
            # Step 3: Standardize predictors using training period
            standardized_predictors = self._standardize_predictors_for_period(
                aligned_data, training_mask
            )
            
            # Step 4: Build rotation matrices for validation and testing
            rotation_matrices = self._build_rotation_matrices_for_period(
                factor_combination, standardized_predictors, validation_periods, test_start, test_end
            )
            
            # Step 5: Optimize lambda using validation periods
            if len(validation_periods) > 0:
                optimal_lambda = self.optimizer.optimize_lambda_validation(
                    interactions_data, factor_combination, rotation_matrices, validation_periods
                )
            else:
                optimal_lambda = 0.0  # Default for first period
            
            # Step 6: Generate optimal weights for testing period
            w0 = self.optimizer._create_default_weights(
                len(factor_combination), len(interactions_data.columns)
            )
            
            interaction_weights = self.optimizer.solve_regularized_weights(
                mu, sigma, optimal_lambda, w0
            )
            
            # Step 7: Apply factor rotation for test date
            test_date_str = test_date.strftime('%Y-%m-%d')
            if test_date_str in rotation_matrices:
                R = rotation_matrices[test_date_str]
            else:
                # Use closest available rotation matrix
                available_dates = list(rotation_matrices.keys())
                if available_dates:
                    closest_date = min(available_dates, 
                                     key=lambda x: abs(pd.to_datetime(x) - test_date))
                    R = rotation_matrices[closest_date]
                else:
                    logger.warning(f"No rotation matrix available for {test_date}")
                    return None
            
            factor_weights = self.optimizer.apply_factor_rotation(interaction_weights, R)
            final_weights = self.optimizer.rescale_factor_weights(factor_weights)
            
            # Step 8: Calculate out-of-sample returns
            portfolio_return, benchmark_return = self._calculate_oos_returns(
                factor_combination, final_weights, aligned_data, test_start, test_end
            )
            
            # Step 9: Store detailed data for rotation analysis
            self._store_rotation_analysis_data(
                test_date_str, interaction_weights, standardized_predictors, R
            )
            
            return {
                'date': test_date,
                'factor_weights': final_weights,
                'factor_names': factor_combination,
                'portfolio_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'optimal_lambda': optimal_lambda
            }
            
        except Exception as e:
            logger.error(f"Error in single period backtest for {test_date}: {e}")
            return None
    
    def _generate_interactions_for_period(self, factor_combination: List[str],
                                        aligned_data: Dict[str, pd.DataFrame],
                                        start_date: str, end_date: str) -> pd.DataFrame:
        """Generate factor interactions for specified period."""
        
        # Filter data for period
        period_mask = ((aligned_data['factor_returns'].index >= start_date) & 
                      (aligned_data['factor_returns'].index <= end_date))
        
        period_factor_returns = aligned_data['factor_returns'][period_mask][factor_combination]
        period_characteristics = aligned_data['characteristics'][period_mask]
        period_macro = aligned_data['macro'][period_mask]
        
        # Create interaction engine for this period
        period_engine = FactorInteractionEngine(
            period_factor_returns, period_characteristics, period_macro
        )
        
        # Generate all interactions
        interactions = period_engine.create_all_interactions(factor_combination)
        
        return interactions
    
    def _standardize_predictors_for_period(self, aligned_data: Dict[str, pd.DataFrame],
                                         training_mask: Union[pd.Series, np.ndarray]) -> Dict[str, pd.DataFrame]:
        """Standardize predictors using training period statistics."""
        
        standardized = {}
        
        # Convert training mask to pandas Series if needed
        if isinstance(training_mask, np.ndarray):
            training_mask = pd.Series(training_mask, index=aligned_data['characteristics'].index)
        
        # Standardize characteristics
        standardized['characteristics'] = self.data_processor.standardize_predictors(
            aligned_data['characteristics'], training_mask
        )
        
        # Standardize macro predictors
        standardized['macro'] = self.data_processor.standardize_predictors(
            aligned_data['macro'], training_mask  
        )
        
        return standardized
    
    def _build_rotation_matrices_for_period(self, factor_combination: List[str],
                                          standardized_predictors: Dict[str, pd.DataFrame],
                                          validation_periods: List[Tuple[str, str]],
                                          test_start: str, test_end: str) -> Dict[str, np.ndarray]:
        """Build rotation matrices for validation and testing periods."""
        
        rotation_matrices = {}
        
        # Build matrices for validation period end dates
        for val_start, val_end in validation_periods:
            val_end_date = pd.to_datetime(val_end)
            if val_end_date in standardized_predictors['characteristics'].index:
                R = self.interaction_engine.build_rotation_matrix(
                    val_end_date, factor_combination, standardized_predictors
                )
                rotation_matrices[val_end] = R
        
        # Build matrix for test period end date
        test_end_date = pd.to_datetime(test_end)
        if test_end_date in standardized_predictors['characteristics'].index:
            R = self.interaction_engine.build_rotation_matrix(
                test_end_date, factor_combination, standardized_predictors
            )
            rotation_matrices[test_end] = R
        
        return rotation_matrices
    
    def _calculate_oos_returns(self, factor_combination: List[str], 
                              factor_weights: np.ndarray,
                              aligned_data: Dict[str, pd.DataFrame],
                              test_start: str, test_end: str) -> Tuple[float, float]:
        """Calculate out-of-sample portfolio and benchmark returns."""
        
        # Get factor returns for test period
        test_mask = ((aligned_data['factor_returns'].index >= test_start) & 
                    (aligned_data['factor_returns'].index <= test_end))
        
        test_factor_returns = aligned_data['factor_returns'][test_mask][factor_combination]
        
        if len(test_factor_returns) == 0:
            return 0.0, 0.0
        
        # Calculate portfolio returns
        portfolio_returns = []
        for _, period_returns in test_factor_returns.iterrows():
            period_portfolio_return = np.dot(factor_weights, period_returns.values)
            portfolio_returns.append(period_portfolio_return)
        
        # Calculate benchmark returns (equal-weighted)
        equal_weights = np.ones(len(factor_combination)) / len(factor_combination)
        benchmark_returns = []
        for _, period_returns in test_factor_returns.iterrows():
            period_benchmark_return = np.dot(equal_weights, period_returns.values)
            benchmark_returns.append(period_benchmark_return)
        
        # Return average returns for the test period
        portfolio_return = np.mean(portfolio_returns)
        benchmark_return = np.mean(benchmark_returns)
        
        return portfolio_return, benchmark_return
    
    def _store_rotation_analysis_data(self, date: str, interaction_weights: np.ndarray,
                                    standardized_predictors: Dict[str, pd.DataFrame],
                                    rotation_matrix: np.ndarray):
        """Store data needed for factor rotation analysis (Exhibit 8)."""
        
        self.interaction_weights_history[date] = interaction_weights.copy()
        
        date_ts = pd.to_datetime(date)
        if date_ts in standardized_predictors['characteristics'].index:
            predictor_values = {}
            
            # Store characteristic values
            for col in standardized_predictors['characteristics'].columns:
                predictor_values[col] = standardized_predictors['characteristics'].loc[date_ts, col]
            
            # Store macro values
            for col in standardized_predictors['macro'].columns:
                predictor_values[col] = standardized_predictors['macro'].loc[date_ts, col]
            
            self.predictor_values_history[date] = predictor_values
        
        self.rotation_matrices_history[date] = rotation_matrix.copy()
        self.standardized_predictors_history[date] = {
            'characteristics': standardized_predictors['characteristics'],
            'macro': standardized_predictors['macro']
        }
    
    def get_factor_weights_history(self, results: BacktestResults) -> pd.DataFrame:
        """Get time series of optimal factor weights."""
        return results.factor_weights_timeseries
    
    def get_lambda_history(self, results: BacktestResults) -> pd.Series:
        """Get time series of optimal lambda values."""
        return results.lambda_timeseries
    
    def get_predictor_changes(self, date1: str, date2: str) -> Dict[str, float]:
        """Calculate predictor changes between two dates (for rotation analysis)."""
        if date1 not in self.predictor_values_history or date2 not in self.predictor_values_history:
            return {}
        
        changes = {}
        predictors_1 = self.predictor_values_history[date1]
        predictors_2 = self.predictor_values_history[date2]
        
        for predictor in predictors_1:
            if predictor in predictors_2:
                changes[predictor] = predictors_2[predictor] - predictors_1[predictor]
            else:
                changes[predictor] = 0.0
        
        return changes
    
    def calculate_transition_matrix(self, results: BacktestResults, 
                                  window_months: int = 3) -> pd.DataFrame:
        """
        Calculate transition matrix for factor weights (Exhibit 11 style).
        
        Parameters:
        -----------
        results : BacktestResults
            Backtest results containing factor weights
        window_months : int
            Window for calculating transitions (default 3 months)
            
        Returns:
        --------
        Transition matrix showing weight stability
        """
        weights_ts = results.factor_weights_timeseries
        
        if len(weights_ts) < 2:
            return pd.DataFrame()
        
        # Calculate transitions between periods
        transitions = []
        
        for i in range(len(weights_ts) - window_months):
            current_weights = weights_ts.iloc[i]
            future_weights = weights_ts.iloc[i + window_months]
            
            # Classify weights into quantiles (top/middle/bottom)
            current_quantiles = pd.qcut(current_weights, 3, labels=['Bottom', 'Middle', 'Top'])
            future_quantiles = pd.qcut(future_weights, 3, labels=['Bottom', 'Middle', 'Top'])
            
            # Count transitions
            for factor in current_weights.index:
                transitions.append({
                    'from': current_quantiles[factor],
                    'to': future_quantiles[factor]
                })
        
        # Create transition matrix
        transition_df = pd.DataFrame(transitions)
        if len(transition_df) == 0:
            return pd.DataFrame()
        
        transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'], normalize='index')
        
        return transition_matrix
    
    def get_backtest_summary(self, results: BacktestResults) -> Dict[str, Any]:
        """Get comprehensive backtest summary."""
        
        summary = results.get_summary()
        
        # Add additional backtester-specific metrics
        summary.update({
            'n_factors': len(results.factor_weights_timeseries.columns) if not results.factor_weights_timeseries.empty else 0,
            'factor_names': list(results.factor_weights_timeseries.columns) if not results.factor_weights_timeseries.empty else [],
            'rotation_analysis_periods': len(self.interaction_weights_history),
            'predictor_history_periods': len(self.predictor_values_history)
        })
        
        return summary