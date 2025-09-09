"""
Unit tests for backtesting framework components.

Tests BacktestResults container and DynamicFactorTimingBacktester functionality
including period generation, single period backtesting, and result aggregation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtester import BacktestResults, DynamicFactorTimingBacktester
from data_processor import DataProcessor
from factor_interactions import FactorInteractionEngine  
from optimizer import RegularizedOptimizer


class TestBacktestResults(unittest.TestCase):
    """Test cases for BacktestResults container."""
    
    def setUp(self):
        """Set up test data."""
        self.results = BacktestResults()
        self.dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        self.factor_names = ['value', 'momentum', 'quality']
    
    def test_initialization(self):
        """Test BacktestResults initialization."""
        self.assertIsInstance(self.results.factor_weights_timeseries, pd.DataFrame)
        self.assertIsInstance(self.results.portfolio_returns, pd.Series)
        self.assertIsInstance(self.results.benchmark_returns, pd.Series)
        self.assertIsInstance(self.results.lambda_timeseries, pd.Series)
        self.assertIsInstance(self.results.rotation_analysis_data, dict)
        self.assertIsInstance(self.results.performance_metrics, dict)
        self.assertIsInstance(self.results.backtest_periods, list)
    
    def test_add_period_results(self):
        """Test adding period results."""
        date = self.dates[0]
        factor_weights = np.array([0.4, -0.3, 0.9])
        portfolio_return = 0.015
        benchmark_return = 0.012
        optimal_lambda = 10.5
        
        self.results.add_period_results(
            date, factor_weights, self.factor_names, 
            portfolio_return, benchmark_return, optimal_lambda
        )
        
        # Check factor weights
        self.assertEqual(len(self.results.factor_weights_timeseries), 1)
        self.assertEqual(self.results.factor_weights_timeseries.iloc[0]['value'], 0.4)
        self.assertEqual(self.results.factor_weights_timeseries.iloc[0]['momentum'], -0.3)
        self.assertEqual(self.results.factor_weights_timeseries.iloc[0]['quality'], 0.9)
        
        # Check returns and lambda
        self.assertEqual(self.results.portfolio_returns.iloc[0], portfolio_return)
        self.assertEqual(self.results.benchmark_returns.iloc[0], benchmark_return)
        self.assertEqual(self.results.lambda_timeseries.iloc[0], optimal_lambda)
    
    def test_add_multiple_periods(self):
        """Test adding multiple period results."""
        for i, date in enumerate(self.dates[:3]):
            factor_weights = np.random.uniform(-1, 1, 3)
            factor_weights = factor_weights / np.sum(np.abs(factor_weights))  # Normalize
            
            self.results.add_period_results(
                date, factor_weights, self.factor_names,
                np.random.normal(0.01, 0.05), np.random.normal(0.008, 0.04), 
                np.random.uniform(0, 100)
            )
        
        # Should have 3 periods
        self.assertEqual(len(self.results.factor_weights_timeseries), 3)
        self.assertEqual(len(self.results.portfolio_returns), 3)
        self.assertEqual(len(self.results.benchmark_returns), 3)
        self.assertEqual(len(self.results.lambda_timeseries), 3)
        
        # Check index consistency
        self.assertTrue(self.results.factor_weights_timeseries.index.equals(self.dates[:3]))
        self.assertTrue(self.results.portfolio_returns.index.equals(self.dates[:3]))
    
    def test_get_summary_empty(self):
        """Test summary with empty results."""
        summary = self.results.get_summary()
        self.assertIn('error', summary)
    
    def test_get_summary_with_data(self):
        """Test summary with actual data."""
        # Add some sample data
        np.random.seed(42)
        for i, date in enumerate(self.dates):
            factor_weights = np.random.uniform(-1, 1, 3)
            factor_weights = factor_weights / np.sum(np.abs(factor_weights))
            
            self.results.add_period_results(
                date, factor_weights, self.factor_names,
                np.random.normal(0.01, 0.05), np.random.normal(0.008, 0.04),
                np.random.uniform(0, 100)
            )
        
        summary = self.results.get_summary()
        
        # Check required keys
        required_keys = ['n_periods', 'date_range', 'portfolio_return_annualized',
                        'portfolio_volatility_annualized', 'portfolio_sharpe_ratio',
                        'benchmark_sharpe_ratio', 'sharpe_ratio_improvement']
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values make sense
        self.assertEqual(summary['n_periods'], 12)
        self.assertIsInstance(summary['portfolio_sharpe_ratio'], (int, float))
        self.assertIsInstance(summary['benchmark_sharpe_ratio'], (int, float))


class TestDynamicFactorTimingBacktester(unittest.TestCase):
    """Test cases for main backtesting framework."""
    
    def setUp(self):
        """Set up test components and data."""
        # Create test data - longer time series for backtesting
        self.dates = pd.date_range('2000-01-31', '2023-12-31', freq='ME')
        self.n_periods = len(self.dates)
        
        np.random.seed(123)  # For reproducible tests
        
        # Sample macro predictors
        self.df_macro = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, self.n_periods),
            'yield_slope': np.random.normal(0.01, 0.005, self.n_periods),
            'yield_change': np.random.normal(0, 0.002, self.n_periods),
            'market_excess_3m': np.random.normal(0.005, 0.08, self.n_periods),
            'credit_spread': np.random.normal(0.015, 0.005, self.n_periods)
        }, index=self.dates)
        
        # Sample factor characteristics  
        factors = ['value', 'momentum', 'quality']
        char_data = {}
        
        for factor in factors:
            char_data[f'{factor}_3m_ret'] = np.random.normal(0.02, 0.1, self.n_periods)
            char_data[f'{factor}_12m_ret'] = np.random.normal(0.08, 0.3, self.n_periods)
            char_data[f'{factor}_3m_vol'] = np.random.uniform(0.15, 0.4, self.n_periods)
        
        char_data['value_spread'] = np.random.normal(0.4, 0.2, self.n_periods)
        char_data['prof_spread'] = np.random.normal(0.2, 0.15, self.n_periods)
        char_data['inv_spread'] = np.random.normal(-0.05, 0.1, self.n_periods)
        
        self.df_characteristics = pd.DataFrame(char_data, index=self.dates)
        
        # Sample factor returns with some persistence
        factor_returns_data = {}
        for factor in factors:
            # Add some serial correlation to make it more realistic
            returns = np.random.normal(0.005, 0.05, self.n_periods)
            for i in range(1, self.n_periods):
                returns[i] += 0.1 * returns[i-1]  # Small autocorrelation
            factor_returns_data[factor] = returns
        
        self.df_factor_returns = pd.DataFrame(factor_returns_data, index=self.dates)
        
        # Initialize components
        self.data_processor = DataProcessor(
            self.df_macro, self.df_characteristics, self.df_factor_returns
        )
        
        # Get monthly data
        monthly_data = self.data_processor.convert_to_monthly()
        
        self.interaction_engine = FactorInteractionEngine(
            monthly_data['factor_returns'],
            monthly_data['characteristics'],
            monthly_data['macro']
        )
        
        self.optimizer = RegularizedOptimizer(
            lambda_grid=[0.0, 1.0, 10.0, 100.0]  # Small grid for testing
        )
        
        # Initialize backtester
        self.backtester = DynamicFactorTimingBacktester(
            self.data_processor, self.interaction_engine, self.optimizer
        )
    
    def test_initialization(self):
        """Test backtester initialization."""
        self.assertIsInstance(self.backtester, DynamicFactorTimingBacktester)
        self.assertIsInstance(self.backtester.data_processor, DataProcessor)
        self.assertIsInstance(self.backtester.interaction_engine, FactorInteractionEngine)
        self.assertIsInstance(self.backtester.optimizer, RegularizedOptimizer)
        
        # Check storage dictionaries
        self.assertIsInstance(self.backtester.interaction_weights_history, dict)
        self.assertIsInstance(self.backtester.predictor_values_history, dict)
        self.assertIsInstance(self.backtester.rotation_matrices_history, dict)
    
    def test_generate_backtest_periods(self):
        """Test backtest period generation."""
        # Use shorter parameters for testing
        date_index = self.dates[::12]  # Annual dates only for testing
        periods = self.backtester._generate_backtest_periods(
            date_index, min_training_months=5, validation_months=2, testing_months=1
        )
        
        self.assertIsInstance(periods, list)
        if len(periods) > 0:
            # Check period structure
            period = periods[0]
            required_keys = ['train_start', 'train_end', 'validation_periods', 
                           'test_start', 'test_end', 'test_date']
            for key in required_keys:
                self.assertIn(key, period)
            
            # Check date ordering
            self.assertLessEqual(pd.to_datetime(period['train_start']), 
                               pd.to_datetime(period['train_end']))
            self.assertLessEqual(pd.to_datetime(period['train_end']), 
                               pd.to_datetime(period['test_start']))
    
    def test_generate_backtest_periods_insufficient_data(self):
        """Test period generation with insufficient data."""
        short_index = pd.date_range('2020-01-31', '2020-06-30', freq='ME')  # Only 6 months
        
        periods = self.backtester._generate_backtest_periods(
            short_index, min_training_months=12, validation_months=6, testing_months=6
        )
        
        # Should return empty list and warn
        with warnings.catch_warnings(record=True) as w:
            periods = self.backtester._generate_backtest_periods(
                short_index, min_training_months=12, validation_months=6, testing_months=6
            )
            self.assertEqual(len(periods), 0)
            self.assertGreater(len(w), 0)
    
    def test_generate_interactions_for_period(self):
        """Test interaction generation for specific period."""
        factor_combination = ['value', 'momentum']
        aligned_data = self.data_processor.align_data_for_period('2020-01-01', '2020-12-31')
        
        interactions = self.backtester._generate_interactions_for_period(
            factor_combination, aligned_data, '2020-01-31', '2020-12-31'
        )
        
        self.assertIsInstance(interactions, pd.DataFrame)
        self.assertGreater(len(interactions.columns), len(factor_combination))  # Should have interactions
        
        # Should have 2 * (1 + 6 + 5) = 24 interactions
        expected_interactions = len(factor_combination) * (1 + 6 + 5)
        self.assertEqual(len(interactions.columns), expected_interactions)
    
    def test_standardize_predictors_for_period(self):
        """Test predictor standardization."""
        aligned_data = self.data_processor.align_data_for_period('2020-01-01', '2021-12-31')
        
        # Create training mask (first year)
        training_mask = aligned_data['macro'].index <= '2020-12-31'
        
        standardized = self.backtester._standardize_predictors_for_period(
            aligned_data, training_mask
        )
        
        self.assertIn('characteristics', standardized)
        self.assertIn('macro', standardized)
        
        # Training period should have mean ~0, std ~1
        training_chars = standardized['characteristics'][training_mask]
        training_macro = standardized['macro'][training_mask]
        
        # Check means are close to 0
        np.testing.assert_allclose(training_chars.mean(), 0, atol=1e-10)
        np.testing.assert_allclose(training_macro.mean(), 0, atol=1e-10)
    
    def test_build_rotation_matrices_for_period(self):
        """Test rotation matrix building."""
        factor_combination = ['value', 'momentum']
        aligned_data = self.data_processor.align_data_for_period('2020-01-01', '2021-12-31')
        training_mask = aligned_data['macro'].index <= '2020-12-31'
        
        standardized = self.backtester._standardize_predictors_for_period(
            aligned_data, training_mask
        )
        
        validation_periods = [('2021-01-31', '2021-06-30')]
        
        rotation_matrices = self.backtester._build_rotation_matrices_for_period(
            factor_combination, standardized, validation_periods, '2021-07-31', '2021-12-31'
        )
        
        self.assertIsInstance(rotation_matrices, dict)
        
        # Should have matrices for validation and test periods
        if len(rotation_matrices) > 0:
            # Check matrix dimensions
            for date, R in rotation_matrices.items():
                expected_rows = len(factor_combination) * (1 + 6 + 5)  # M interactions
                expected_cols = len(factor_combination)  # N factors
                self.assertEqual(R.shape, (expected_rows, expected_cols))
    
    def test_calculate_oos_returns(self):
        """Test out-of-sample return calculation."""
        factor_combination = ['value', 'momentum'] 
        factor_weights = np.array([0.6, -0.4])
        aligned_data = self.data_processor.align_data_for_period('2020-01-01', '2020-12-31')
        
        portfolio_return, benchmark_return = self.backtester._calculate_oos_returns(
            factor_combination, factor_weights, aligned_data, '2020-06-30', '2020-12-31'
        )
        
        self.assertIsInstance(portfolio_return, (int, float))
        self.assertIsInstance(benchmark_return, (int, float))
        
        # Returns should be finite
        self.assertTrue(np.isfinite(portfolio_return))
        self.assertTrue(np.isfinite(benchmark_return))
    
    def test_store_rotation_analysis_data(self):
        """Test rotation analysis data storage."""
        date = '2020-06-30'
        interaction_weights = np.random.normal(0, 0.1, 24)  # 2 factors * 12 interactions each
        rotation_matrix = np.random.normal(0, 0.1, (24, 2))
        
        aligned_data = self.data_processor.align_data_for_period('2020-01-01', '2020-12-31')
        training_mask = aligned_data['macro'].index <= '2020-05-31'
        standardized_predictors = self.backtester._standardize_predictors_for_period(
            aligned_data, training_mask
        )
        
        self.backtester._store_rotation_analysis_data(
            date, interaction_weights, standardized_predictors, rotation_matrix
        )
        
        # Check data was stored
        self.assertIn(date, self.backtester.interaction_weights_history)
        self.assertIn(date, self.backtester.rotation_matrices_history)
        
        # Check stored data
        stored_weights = self.backtester.interaction_weights_history[date]
        np.testing.assert_array_equal(stored_weights, interaction_weights)
        
        stored_matrix = self.backtester.rotation_matrices_history[date]
        np.testing.assert_array_equal(stored_matrix, rotation_matrix)
    
    def test_get_predictor_changes(self):
        """Test predictor change calculation."""
        # Store some test data
        date1 = '2020-01-31'
        date2 = '2020-02-29'
        
        predictors_1 = {'real_yield_1y': 0.5, 'yield_slope': -0.3, 'value_3m_ret': 1.2}
        predictors_2 = {'real_yield_1y': 0.8, 'yield_slope': -0.1, 'value_3m_ret': 0.9}
        
        self.backtester.predictor_values_history[date1] = predictors_1
        self.backtester.predictor_values_history[date2] = predictors_2
        
        changes = self.backtester.get_predictor_changes(date1, date2)
        
        # Check calculated changes
        self.assertAlmostEqual(changes['real_yield_1y'], 0.3)  # 0.8 - 0.5
        self.assertAlmostEqual(changes['yield_slope'], 0.2)    # -0.1 - (-0.3)
        self.assertAlmostEqual(changes['value_3m_ret'], -0.3)  # 0.9 - 1.2
    
    def test_get_predictor_changes_missing_dates(self):
        """Test predictor changes with missing dates."""
        changes = self.backtester.get_predictor_changes('2020-01-31', '2020-02-29')
        self.assertEqual(len(changes), 0)
    
    def test_calculate_transition_matrix(self):
        """Test transition matrix calculation."""
        # Create sample results
        results = BacktestResults()
        
        # Add some periods with varying factor weights
        dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        factor_names = ['value', 'momentum', 'quality']
        
        for i, date in enumerate(dates):
            # Create weights that change over time
            weights = np.array([0.5 + 0.1 * np.sin(i), -0.3 + 0.2 * np.cos(i), 0.2 + 0.1 * i])
            weights = weights / np.sum(np.abs(weights))  # Normalize
            
            results.add_period_results(
                date, weights, factor_names, 0.01, 0.008, 10.0
            )
        
        transition_matrix = self.backtester.calculate_transition_matrix(results, window_months=3)
        
        if not transition_matrix.empty:
            # Should be 3x3 matrix (Bottom/Middle/Top)
            self.assertEqual(transition_matrix.shape[0], 3)
            self.assertEqual(transition_matrix.shape[1], 3)
            
            # Each row should sum to 1 (probabilities)
            for i in range(3):
                self.assertAlmostEqual(transition_matrix.iloc[i].sum(), 1.0, places=5)
    
    def test_run_full_backtest_short_period(self):
        """Test full backtest with short time period."""
        factor_combination = ['value', 'momentum']
        
        # Use recent period with sufficient data
        results = self.backtester.run_full_backtest(
            factor_combination, 
            start_date='2020-01-01', 
            end_date='2023-12-31',
            min_training_months=24,  # Shorter for testing
            validation_months=6,
            testing_months=6
        )
        
        self.assertIsInstance(results, BacktestResults)
        
        # Should have some results
        if len(results.portfolio_returns) > 0:
            # Check basic properties
            self.assertGreater(len(results.factor_weights_timeseries.columns), 0)
            self.assertEqual(list(results.factor_weights_timeseries.columns), factor_combination)
            
            # Check return data consistency
            self.assertEqual(len(results.portfolio_returns), len(results.benchmark_returns))
            self.assertEqual(len(results.portfolio_returns), len(results.lambda_timeseries))
            
            # Check weight normalization
            for _, weights in results.factor_weights_timeseries.iterrows():
                abs_sum = np.sum(np.abs(weights.values))
                self.assertAlmostEqual(abs_sum, 1.0, places=5)
    
    def test_run_full_backtest_invalid_factors(self):
        """Test full backtest with invalid factor combination."""
        invalid_factors = ['nonexistent_factor']
        
        with self.assertRaises(ValueError):
            self.backtester.run_full_backtest(invalid_factors)
    
    def test_get_backtest_summary(self):
        """Test backtest summary generation."""
        # Create minimal results
        results = BacktestResults()
        
        # Add one period
        results.add_period_results(
            pd.Timestamp('2020-12-31'),
            np.array([0.6, -0.4]),
            ['value', 'momentum'],
            0.015, 0.012, 25.0
        )
        
        summary = self.backtester.get_backtest_summary(results)
        
        # Should include basic summary plus backtester-specific metrics
        self.assertIn('n_factors', summary)
        self.assertIn('factor_names', summary)
        self.assertIn('rotation_analysis_periods', summary)
        
        self.assertEqual(summary['n_factors'], 2)
        self.assertEqual(summary['factor_names'], ['value', 'momentum'])


class TestBacktesterIntegration(unittest.TestCase):
    """Integration tests for backtester with realistic scenarios."""
    
    def setUp(self):
        """Set up realistic test scenario."""
        # Create longer, more realistic time series
        self.dates = pd.date_range('2000-01-31', '2020-12-31', freq='ME')  # 20 years
        self.n_periods = len(self.dates)
        
        np.random.seed(42)  # For reproducibility
        
        # More realistic macro data with trends and cycles
        t = np.arange(self.n_periods)
        self.df_macro = pd.DataFrame({
            'real_yield_1y': 0.03 + 0.02 * np.sin(t / 12) + np.random.normal(0, 0.005, self.n_periods),
            'yield_slope': 0.015 + 0.01 * np.cos(t / 24) + np.random.normal(0, 0.003, self.n_periods),
            'yield_change': np.random.normal(0, 0.002, self.n_periods),
            'market_excess_3m': np.random.normal(0.008, 0.06, self.n_periods),
            'credit_spread': 0.02 + 0.01 * (t / self.n_periods) + np.random.normal(0, 0.003, self.n_periods)
        }, index=self.dates)
        
        # Factor characteristics with some realism
        factors = ['value', 'momentum', 'quality', 'size']
        char_data = {}
        
        for factor in factors:
            char_data[f'{factor}_3m_ret'] = np.random.normal(0.02, 0.08, self.n_periods)
            char_data[f'{factor}_12m_ret'] = np.random.normal(0.08, 0.25, self.n_periods) 
            char_data[f'{factor}_3m_vol'] = np.random.uniform(0.12, 0.35, self.n_periods)
        
        # Cross-factor spreads
        char_data['value_spread'] = 0.5 + 0.3 * np.sin(t / 36) + np.random.normal(0, 0.15, self.n_periods)
        char_data['prof_spread'] = 0.25 + 0.2 * np.cos(t / 48) + np.random.normal(0, 0.12, self.n_periods)
        char_data['inv_spread'] = -0.1 + 0.15 * np.sin(t / 60) + np.random.normal(0, 0.08, self.n_periods)
        
        self.df_characteristics = pd.DataFrame(char_data, index=self.dates)
        
        # More realistic factor returns with regime changes
        factor_returns_data = {}
        for i, factor in enumerate(factors):
            # Base returns with different risk/return profiles
            base_returns = [0.008, 0.012, 0.006, 0.004][i]  # Different expected returns
            volatilities = [0.18, 0.22, 0.15, 0.20][i]      # Different volatilities
            
            returns = np.random.normal(base_returns / 12, volatilities / np.sqrt(12), self.n_periods)
            
            # Add some serial correlation and regime changes
            for j in range(1, self.n_periods):
                # Autocorrelation
                returns[j] += 0.05 * returns[j-1]
                
                # Occasional regime changes
                if j % 60 == 0:  # Every 5 years
                    returns[j] += np.random.normal(0, 0.02)
            
            factor_returns_data[factor] = returns
        
        self.df_factor_returns = pd.DataFrame(factor_returns_data, index=self.dates)
        
        # Initialize components with smaller lambda grid for faster testing
        self.data_processor = DataProcessor(
            self.df_macro, self.df_characteristics, self.df_factor_returns
        )
        
        monthly_data = self.data_processor.convert_to_monthly()
        
        self.interaction_engine = FactorInteractionEngine(
            monthly_data['factor_returns'],
            monthly_data['characteristics'], 
            monthly_data['macro']
        )
        
        self.optimizer = RegularizedOptimizer(
            lambda_grid=[0.0, 1.0, 10.0, 100.0, 1000.0]  # Smaller grid for testing
        )
        
        self.backtester = DynamicFactorTimingBacktester(
            self.data_processor, self.interaction_engine, self.optimizer
        )
    
    def test_realistic_backtest_two_factors(self):
        """Test realistic backtest with two factors."""
        factor_combination = ['value', 'momentum']
        
        results = self.backtester.run_full_backtest(
            factor_combination,
            start_date='2010-01-01', 
            end_date='2020-12-31',
            min_training_months=60,  # 5 years minimum
            validation_months=12,
            testing_months=12
        )
        
        self.assertIsInstance(results, BacktestResults)
        
        if len(results.portfolio_returns) > 0:
            summary = results.get_summary()
            
            # Check reasonable values
            self.assertGreater(summary['n_periods'], 0)
            self.assertTrue(np.isfinite(summary['portfolio_sharpe_ratio']))
            self.assertTrue(np.isfinite(summary['benchmark_sharpe_ratio']))
            
            # Factor weights should be normalized
            for _, weights in results.factor_weights_timeseries.iterrows():
                abs_sum = np.sum(np.abs(weights.values))
                self.assertAlmostEqual(abs_sum, 1.0, places=4)
                
                # Weights should be within bounds
                self.assertTrue(np.all(np.abs(weights.values) <= 1.0))
    
    def test_realistic_backtest_four_factors(self):
        """Test realistic backtest with four factors."""
        factor_combination = ['value', 'momentum', 'quality', 'size']
        
        # Use shorter period for 4-factor test (more complex)
        results = self.backtester.run_full_backtest(
            factor_combination,
            start_date='2015-01-01',
            end_date='2020-12-31', 
            min_training_months=36,  # 3 years minimum
            validation_months=6,
            testing_months=6
        )
        
        if len(results.portfolio_returns) > 0:
            # Check that all factors are represented
            self.assertEqual(list(results.factor_weights_timeseries.columns), factor_combination)
            
            # Check lambda optimization worked
            self.assertGreater(len(results.lambda_timeseries), 0)
            self.assertTrue(np.all(results.lambda_timeseries >= 0))
            
            # Check rotation analysis data was stored
            self.assertGreater(len(self.backtester.interaction_weights_history), 0)
            self.assertGreater(len(self.backtester.predictor_values_history), 0)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)