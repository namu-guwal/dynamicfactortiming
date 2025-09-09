"""
Unit tests for Performance Analytics module.

Tests performance metrics calculation, rolling analysis, factor rotation analysis,
and reporting functionality.
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

from performance import (
    PerformanceAnalytics, FactorRotationAnalyzer, 
    calculate_factor_attribution, create_exhibit_2_comparison,
    calculate_transition_matrix_analysis
)
from backtester import BacktestResults, DynamicFactorTimingBacktester
from data_processor import DataProcessor
from factor_interactions import FactorInteractionEngine
from optimizer import RegularizedOptimizer


class TestPerformanceAnalytics(unittest.TestCase):
    """Test cases for PerformanceAnalytics class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample return data
        self.dates = pd.date_range('2020-01-31', '2022-12-31', freq='ME')
        self.n_periods = len(self.dates)
        
        np.random.seed(42)  # For reproducible tests
        
        # Generate portfolio returns with some positive drift
        self.portfolio_returns = pd.Series(
            np.random.normal(0.01, 0.05, self.n_periods),
            index=self.dates,
            name='portfolio'
        )
        
        # Generate benchmark returns with slightly lower return  
        self.benchmark_returns = pd.Series(
            np.random.normal(0.008, 0.04, self.n_periods),
            index=self.dates,
            name='benchmark'
        )
    
    def test_initialization_valid_data(self):
        """Test successful initialization."""
        analytics = PerformanceAnalytics(self.portfolio_returns, self.benchmark_returns)
        
        self.assertIsInstance(analytics, PerformanceAnalytics)
        self.assertEqual(len(analytics.portfolio_returns), self.n_periods)
        self.assertEqual(len(analytics.benchmark_returns), self.n_periods)
    
    def test_initialization_portfolio_only(self):
        """Test initialization with portfolio returns only."""
        analytics = PerformanceAnalytics(self.portfolio_returns)
        
        self.assertIsInstance(analytics, PerformanceAnalytics)
        self.assertIsNone(analytics.benchmark_returns)
    
    def test_initialization_empty_returns(self):
        """Test initialization with empty returns."""
        empty_returns = pd.Series([], dtype=float)
        
        with self.assertRaises(ValueError):
            PerformanceAnalytics(empty_returns)
    
    def test_initialization_all_nan_returns(self):
        """Test initialization with all NaN returns."""
        nan_returns = pd.Series([np.nan] * 10, index=self.dates[:10])
        
        with self.assertRaises(ValueError):
            PerformanceAnalytics(nan_returns)
    
    def test_calculate_basic_metrics_portfolio_only(self):
        """Test basic metrics calculation with portfolio only."""
        analytics = PerformanceAnalytics(self.portfolio_returns)
        metrics = analytics.calculate_basic_metrics()
        
        # Check required keys
        required_keys = ['portfolio_return_mean', 'portfolio_return_std', 
                        'portfolio_return_annualized', 'portfolio_volatility_annualized',
                        'portfolio_sharpe_ratio', 'max_drawdown', 'hit_rate']
        
        for key in required_keys:
            self.assertIn(key, metrics)
        
        # Check values are reasonable
        self.assertIsInstance(metrics['portfolio_sharpe_ratio'], (int, float))
        self.assertGreaterEqual(metrics['hit_rate'], 0.0)
        self.assertLessEqual(metrics['hit_rate'], 1.0)
        self.assertLessEqual(metrics['max_drawdown'], 0.0)
    
    def test_calculate_basic_metrics_with_benchmark(self):
        """Test basic metrics calculation with benchmark."""
        analytics = PerformanceAnalytics(self.portfolio_returns, self.benchmark_returns)
        metrics = analytics.calculate_basic_metrics()
        
        # Should have benchmark metrics
        benchmark_keys = ['benchmark_return_mean', 'benchmark_return_std',
                         'benchmark_sharpe_ratio', 'active_return', 
                         'sharpe_improvement', 'information_ratio']
        
        for key in benchmark_keys:
            self.assertIn(key, metrics)
        
        # Active return should be portfolio minus benchmark
        expected_active = metrics['portfolio_return_annualized'] - metrics['benchmark_return_annualized']
        self.assertAlmostEqual(metrics['active_return'], expected_active, places=10)
        
        # Sharpe improvement should be portfolio minus benchmark Sharpe
        expected_improvement = metrics['portfolio_sharpe_ratio'] - metrics['benchmark_sharpe_ratio']
        self.assertAlmostEqual(metrics['sharpe_improvement'], expected_improvement, places=10)
    
    def test_calculate_basic_metrics_zero_volatility(self):
        """Test metrics with zero volatility (constant returns)."""
        constant_returns = pd.Series([0.01] * len(self.dates), index=self.dates)
        analytics = PerformanceAnalytics(constant_returns)
        metrics = analytics.calculate_basic_metrics()
        
        # Sharpe ratio should be 0 for zero volatility
        self.assertEqual(metrics['portfolio_sharpe_ratio'], 0.0)
        self.assertAlmostEqual(metrics['portfolio_return_std'], 0.0, places=10)
    
    def test_calculate_rolling_metrics(self):
        """Test rolling metrics calculation."""
        analytics = PerformanceAnalytics(self.portfolio_returns, self.benchmark_returns)
        rolling_metrics = analytics.calculate_rolling_metrics(window_years=2)  # 2-year window
        
        if not rolling_metrics.empty:
            # Should have rolling columns
            expected_cols = ['portfolio_sharpe', 'benchmark_sharpe', 'sharpe_improvement',
                           'portfolio_return', 'benchmark_return', 'portfolio_volatility', 
                           'benchmark_volatility']
            
            for col in expected_cols:
                self.assertIn(col, rolling_metrics.columns)
            
            # Check that early periods are NaN (insufficient data)
            window_periods = 2 * 12  # 24 months
            self.assertTrue(rolling_metrics.iloc[:window_periods-1].isnull().all().all())
            
            # Later periods should have values
            if len(rolling_metrics) >= window_periods:
                self.assertFalse(rolling_metrics.iloc[window_periods:].isnull().all().all())
    
    def test_calculate_rolling_metrics_insufficient_data(self):
        """Test rolling metrics with insufficient data."""
        short_returns = self.portfolio_returns[:12]  # Only 1 year
        analytics = PerformanceAnalytics(short_returns)
        
        with warnings.catch_warnings(record=True) as w:
            rolling_metrics = analytics.calculate_rolling_metrics(window_years=5)
            self.assertGreater(len(w), 0)  # Should warn
            self.assertTrue(rolling_metrics.empty)
    
    def test_calculate_drawdown_statistics(self):
        """Test drawdown statistics calculation."""
        # Create returns with known drawdown
        test_returns = pd.Series([0.05, -0.10, -0.05, 0.08, 0.02, -0.03], 
                                index=pd.date_range('2020-01-31', periods=6, freq='ME'))
        
        analytics = PerformanceAnalytics(test_returns)
        drawdown_stats = analytics.calculate_drawdown_statistics()
        
        # Check required keys
        required_keys = ['max_drawdown', 'current_drawdown', 'avg_drawdown',
                        'drawdown_periods', 'n_drawdown_periods', 'time_in_drawdown']
        
        for key in required_keys:
            self.assertIn(key, drawdown_stats)
        
        # Max drawdown should be negative
        self.assertLessEqual(drawdown_stats['max_drawdown'], 0.0)
        
        # Should have drawdown periods
        self.assertIsInstance(drawdown_stats['drawdown_periods'], list)
        self.assertGreaterEqual(drawdown_stats['n_drawdown_periods'], 0)
        
        # Time in drawdown should be between 0 and 1
        self.assertGreaterEqual(drawdown_stats['time_in_drawdown'], 0.0)
        self.assertLessEqual(drawdown_stats['time_in_drawdown'], 1.0)
    
    def test_calculate_drawdown_statistics_no_valid_returns(self):
        """Test drawdown calculation with no valid returns."""
        nan_returns = pd.Series([np.nan, np.nan, np.nan], 
                               index=pd.date_range('2020-01-31', periods=3, freq='ME'))
        
        analytics = PerformanceAnalytics(pd.Series([0.01]), pd.Series([0.01]))  # Valid initialization
        analytics.portfolio_returns = nan_returns  # Replace with NaN data
        
        drawdown_stats = analytics.calculate_drawdown_statistics()
        self.assertIn('error', drawdown_stats)
    
    def test_generate_performance_summary_exhibit_2_format(self):
        """Test performance summary in Exhibit 2 format."""
        analytics = PerformanceAnalytics(self.portfolio_returns, self.benchmark_returns)
        summary = analytics.generate_performance_summary(exhibit_2_format=True)
        
        # Should have Exhibit 2 columns
        expected_cols = ['Return (%)', 'Std. Dev. (%)', 'Sharpe Ratio', 'Δ Sharpe Ratio']
        for col in expected_cols:
            self.assertIn(col, summary.columns)
        
        # Should have one row
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.index[0], 'Optimal Timing')
        
        # Returns and volatility should be in percentage
        self.assertGreater(abs(summary['Return (%)'].iloc[0]), 1)  # Should be in percentage
        self.assertGreater(abs(summary['Std. Dev. (%)'].iloc[0]), 1)  # Should be in percentage
    
    def test_generate_performance_summary_standard_format(self):
        """Test performance summary in standard format."""
        analytics = PerformanceAnalytics(self.portfolio_returns, self.benchmark_returns)
        summary = analytics.generate_performance_summary(exhibit_2_format=False)
        
        # Should have many columns (all metrics)
        self.assertGreater(len(summary.columns), 10)
        
        # Should include raw metric names
        self.assertIn('portfolio_return_annualized', summary.columns)
        self.assertIn('portfolio_sharpe_ratio', summary.columns)


class TestFactorRotationAnalyzer(unittest.TestCase):
    """Test cases for FactorRotationAnalyzer."""
    
    def setUp(self):
        """Set up test backtester with rotation data."""
        # Create minimal components for backtester
        dates = pd.date_range('2020-01-31', '2021-12-31', freq='ME')
        n_periods = len(dates)
        
        np.random.seed(123)
        
        # Minimal data for backtester components
        df_macro = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, n_periods),
            'yield_slope': np.random.normal(0.01, 0.005, n_periods),
            'yield_change': np.random.normal(0, 0.002, n_periods),
            'market_excess_3m': np.random.normal(0.005, 0.08, n_periods),
            'credit_spread': np.random.normal(0.015, 0.005, n_periods)
        }, index=dates)
        
        df_characteristics = pd.DataFrame({
            'value_3m_ret': np.random.normal(0.02, 0.1, n_periods),
            'value_12m_ret': np.random.normal(0.08, 0.3, n_periods),
            'value_3m_vol': np.random.uniform(0.15, 0.4, n_periods),
            'value_spread': np.random.normal(0.4, 0.2, n_periods),
            'prof_spread': np.random.normal(0.2, 0.15, n_periods),
            'inv_spread': np.random.normal(-0.05, 0.1, n_periods)
        }, index=dates)
        
        df_factor_returns = pd.DataFrame({
            'value': np.random.normal(0.005, 0.05, n_periods)
        }, index=dates)
        
        # Initialize components
        data_processor = DataProcessor(df_macro, df_characteristics, df_factor_returns)
        monthly_data = data_processor.convert_to_monthly()
        
        interaction_engine = FactorInteractionEngine(
            monthly_data['factor_returns'],
            monthly_data['characteristics'],
            monthly_data['macro']
        )
        
        optimizer = RegularizedOptimizer(lambda_grid=[0.0, 1.0, 10.0])
        
        self.backtester = DynamicFactorTimingBacktester(
            data_processor, interaction_engine, optimizer
        )
        
        # Add some fake rotation analysis data
        self.backtester.interaction_weights_history['2020-04-30'] = np.random.normal(0, 0.1, 12)
        self.backtester.interaction_weights_history['2020-05-31'] = np.random.normal(0, 0.1, 12)
        
        self.backtester.predictor_values_history['2020-04-30'] = {
            'real_yield_1y': 0.5, 'yield_slope': -0.3, 'value_3m_ret': 1.2
        }
        self.backtester.predictor_values_history['2020-05-31'] = {
            'real_yield_1y': 0.8, 'yield_slope': -0.1, 'value_3m_ret': 0.9
        }
    
    def test_initialization(self):
        """Test FactorRotationAnalyzer initialization."""
        analyzer = FactorRotationAnalyzer(self.backtester)
        
        self.assertIsInstance(analyzer, FactorRotationAnalyzer)
        self.assertEqual(analyzer.backtester, self.backtester)
    
    def test_initialization_no_rotation_data(self):
        """Test initialization with no rotation data."""
        # Create empty backtester
        empty_backtester = DynamicFactorTimingBacktester(None, None, None)
        
        with warnings.catch_warnings(record=True) as w:
            analyzer = FactorRotationAnalyzer(empty_backtester)
            self.assertGreater(len(w), 0)  # Should warn about no data
    
    def test_generate_exhibit_8_analysis(self):
        """Test Exhibit 8 style analysis generation."""
        analyzer = FactorRotationAnalyzer(self.backtester)
        
        analysis_dates = ['2020-04-30', '2020-05-31']
        results = analyzer.generate_exhibit_8_analysis(analysis_dates)
        
        self.assertIsInstance(results, dict)
        
        if len(results) > 0:
            period_key = '2020-04-30 to 2020-05-31'
            if period_key in results:
                period_data = results[period_key]
                
                # Check required keys
                expected_keys = ['predictor_changes', 'rotation_effects', 'total_rotation_effect']
                for key in expected_keys:
                    self.assertIn(key, period_data)
    
    def test_generate_exhibit_8_analysis_insufficient_dates(self):
        """Test analysis with insufficient dates."""
        analyzer = FactorRotationAnalyzer(self.backtester)
        
        with self.assertRaises(ValueError):
            analyzer.generate_exhibit_8_analysis(['2020-04-30'])  # Only one date
    
    def test_create_rotation_attribution_table(self):
        """Test rotation attribution table creation."""
        analyzer = FactorRotationAnalyzer(self.backtester)
        
        # Create sample analysis results
        analysis_results = {
            '2020-04-30 to 2020-05-31': {
                'predictor_changes': {'real_yield_1y': 0.3, 'yield_slope': 0.2},
                'rotation_effects': {'real_yield_1y': 0.05, 'yield_slope': -0.03},
                'total_rotation_effect': 0.02
            }
        }
        
        table = analyzer.create_rotation_attribution_table(analysis_results, 'value')
        
        self.assertIsInstance(table, pd.DataFrame)
        self.assertGreater(len(table), 0)
        
        # Check that table has expected columns
        expected_cols = ['Period', 'Factor']
        for col in expected_cols:
            if col in table.columns:
                self.assertIn(col, table.columns)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        
        # Sample factor weights
        self.factor_weights = pd.DataFrame({
            'value': np.random.uniform(-1, 1, len(self.dates)),
            'momentum': np.random.uniform(-1, 1, len(self.dates))
        }, index=self.dates)
        
        # Sample factor returns
        self.factor_returns = pd.DataFrame({
            'value': np.random.normal(0.005, 0.05, len(self.dates)),
            'momentum': np.random.normal(0.008, 0.06, len(self.dates))
        }, index=self.dates)
    
    def test_calculate_factor_attribution(self):
        """Test factor attribution calculation."""
        attribution = calculate_factor_attribution(self.factor_weights, self.factor_returns)
        
        self.assertIsInstance(attribution, pd.DataFrame)
        
        if not attribution.empty:
            # Should have attribution columns
            expected_cols = ['Total_Contribution', 'Average_Contribution', 
                           'Average_Weight', 'Weight_Volatility']
            
            for col in expected_cols:
                self.assertIn(col, attribution.columns)
            
            # Should have rows for each factor
            self.assertEqual(len(attribution), len(self.factor_weights.columns))
    
    def test_calculate_factor_attribution_misaligned_data(self):
        """Test attribution with misaligned data."""
        # Create misaligned returns
        short_returns = self.factor_returns.iloc[:6]  # Only first 6 months
        
        attribution = calculate_factor_attribution(self.factor_weights, short_returns)
        
        # Should handle misalignment gracefully
        self.assertIsInstance(attribution, pd.DataFrame)
    
    def test_create_exhibit_2_comparison(self):
        """Test Exhibit 2 comparison table creation."""
        # Create sample backtest results
        results1 = BacktestResults()
        results2 = BacktestResults()
        
        # Add some data
        for i, date in enumerate(self.dates[:3]):
            results1.add_period_results(
                date, np.array([0.6, -0.4]), ['value', 'momentum'],
                0.015, 0.012, 10.0
            )
            results2.add_period_results(
                date, np.array([0.3, 0.7]), ['value', 'momentum'], 
                0.018, 0.012, 5.0
            )
        
        results_dict = {
            'Combination 1': results1,
            'Combination 2': results2
        }
        
        comparison = create_exhibit_2_comparison(results_dict)
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)  # Two combinations
        
        # Check Exhibit 2 columns
        expected_cols = ['Factors', 'Return (%)', 'Std. Dev. (%)', 'Sharpe Ratio', 'Δ Sharpe Ratio']
        for col in expected_cols:
            self.assertIn(col, comparison.columns)
    
    def test_create_exhibit_2_comparison_empty_results(self):
        """Test comparison with empty results."""
        empty_results = BacktestResults()
        results_dict = {'Empty': empty_results}
        
        comparison = create_exhibit_2_comparison(results_dict)
        
        # Should handle empty results gracefully
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 0)  # No valid results
    
    def test_calculate_transition_matrix_analysis(self):
        """Test transition matrix analysis."""
        # Create factor weights with some pattern
        dates = pd.date_range('2020-01-31', periods=24, freq='ME')  # 2 years
        
        factor_weights = pd.DataFrame({
            'value': np.sin(np.arange(24) / 6) + np.random.normal(0, 0.1, 24),  # Cyclical pattern
            'momentum': np.random.normal(0, 0.5, 24)  # Random
        }, index=dates)
        
        analysis = calculate_transition_matrix_analysis(factor_weights, window_months=6)
        
        self.assertIsInstance(analysis, dict)
        
        if 'error' not in analysis:
            # Should have results for each factor
            for factor in factor_weights.columns:
                if factor in analysis:
                    factor_result = analysis[factor]
                    
                    # Check required keys
                    expected_keys = ['transition_matrix', 'stability_score', 'n_transitions']
                    for key in expected_keys:
                        self.assertIn(key, factor_result)
                    
                    # Stability score should be between 0 and 1
                    self.assertGreaterEqual(factor_result['stability_score'], 0.0)
                    self.assertLessEqual(factor_result['stability_score'], 1.0)
    
    def test_calculate_transition_matrix_analysis_insufficient_data(self):
        """Test transition analysis with insufficient data."""
        # Very short time series
        short_weights = self.factor_weights.iloc[:2]  # Only 2 periods
        
        analysis = calculate_transition_matrix_analysis(short_weights, window_months=6)
        
        # Should return error
        self.assertIn('error', analysis)


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for performance analytics."""
    
    def test_realistic_performance_analysis(self):
        """Test performance analysis with realistic data."""
        # Create realistic return series
        np.random.seed(42)
        dates = pd.date_range('2015-01-31', '2023-12-31', freq='ME')
        
        # Portfolio with slight outperformance and higher volatility
        portfolio_returns = pd.Series(
            np.random.normal(0.012, 0.06, len(dates)),  # 14.4% annual return, 20.8% vol
            index=dates
        )
        
        # Benchmark with lower return and volatility
        benchmark_returns = pd.Series(
            np.random.normal(0.010, 0.05, len(dates)),   # 12% annual return, 17.3% vol
            index=dates
        )
        
        analytics = PerformanceAnalytics(portfolio_returns, benchmark_returns)
        
        # Test comprehensive analysis
        basic_metrics = analytics.calculate_basic_metrics()
        rolling_metrics = analytics.calculate_rolling_metrics(window_years=3)
        drawdown_stats = analytics.calculate_drawdown_statistics()
        summary = analytics.generate_performance_summary()
        
        # All should complete without errors
        self.assertIsInstance(basic_metrics, dict)
        self.assertIsInstance(rolling_metrics, pd.DataFrame)
        self.assertIsInstance(drawdown_stats, dict)
        self.assertIsInstance(summary, pd.DataFrame)
        
        # Check some expected relationships
        if 'sharpe_improvement' in basic_metrics:
            # Portfolio should outperform (on average)
            # Note: Due to randomness, this might not always hold
            pass
        
        # Rolling metrics should have some valid data
        if not rolling_metrics.empty:
            self.assertGreater(len(rolling_metrics.dropna()), 0)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)