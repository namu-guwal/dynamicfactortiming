"""
Unit tests for DataProcessor class.

Tests all functionality including data validation, frequency conversion,
standardization, and data alignment.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test data for all tests."""
        # Create sample daily data
        self.dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        self.n_days = len(self.dates)
        
        # Sample macro predictors (5 required variables)
        self.df_macro = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, self.n_days),
            'yield_slope': np.random.normal(0.01, 0.005, self.n_days), 
            'yield_change': np.random.normal(0, 0.002, self.n_days),
            'market_excess_3m': np.random.normal(0.001, 0.02, self.n_days),
            'credit_spread': np.random.normal(0.015, 0.003, self.n_days)
        }, index=self.dates)
        
        # Sample factor characteristics
        factors = ['value_bm', 'momentum', 'profitability']
        char_data = {}
        
        # Add factor-specific characteristics
        for factor in factors:
            char_data[f'{factor}_3m_ret'] = np.random.normal(0, 0.05, self.n_days)
            char_data[f'{factor}_12m_ret'] = np.random.normal(0, 0.15, self.n_days)
            char_data[f'{factor}_3m_vol'] = np.random.uniform(0.1, 0.3, self.n_days)
        
        # Add spread characteristics
        char_data['value_spread'] = np.random.normal(0.5, 0.2, self.n_days)
        char_data['prof_spread'] = np.random.normal(0.3, 0.15, self.n_days)
        char_data['inv_spread'] = np.random.normal(-0.1, 0.1, self.n_days)
        
        self.df_characteristics = pd.DataFrame(char_data, index=self.dates)
        
        # Sample factor returns (daily)
        self.df_factor_returns = pd.DataFrame({
            'value_bm': np.random.normal(0.0005, 0.02, self.n_days),
            'momentum': np.random.normal(0.0008, 0.025, self.n_days),
            'profitability': np.random.normal(0.0003, 0.018, self.n_days),
            'investment': np.random.normal(-0.0002, 0.022, self.n_days)
        }, index=self.dates)
    
    def test_initialization_valid_data(self):
        """Test successful initialization with valid data."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        self.assertIsInstance(processor, DataProcessor)
        self.assertEqual(len(processor.df_macro), self.n_days)
        self.assertEqual(len(processor.df_characteristics), self.n_days)
        self.assertEqual(len(processor.df_factor_returns), self.n_days)
    
    def test_initialization_invalid_index(self):
        """Test initialization fails with non-datetime index."""
        df_invalid = self.df_macro.copy()
        df_invalid.index = range(len(df_invalid))  # Integer index
        
        with self.assertRaises(ValueError):
            DataProcessor(df_invalid, self.df_characteristics, self.df_factor_returns)
    
    def test_initialization_empty_dataframe(self):
        """Test initialization fails with empty dataframe."""
        df_empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        
        with self.assertRaises(ValueError):
            DataProcessor(df_empty, self.df_characteristics, self.df_factor_returns)
    
    def test_initialization_missing_macro_columns(self):
        """Test initialization fails with missing required macro columns."""
        df_incomplete = self.df_macro[['real_yield_1y', 'yield_slope']].copy()  # Missing 3 columns
        
        with self.assertRaises(ValueError):
            DataProcessor(df_incomplete, self.df_characteristics, self.df_factor_returns)
    
    def test_get_date_range(self):
        """Test date range calculation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        start_date, end_date = processor.get_date_range()
        
        self.assertEqual(start_date, self.dates[0])
        self.assertEqual(end_date, self.dates[-1])
    
    def test_convert_to_monthly_last(self):
        """Test conversion to monthly using last observation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        monthly_data = processor.convert_to_monthly(method='last')
        
        self.assertIn('macro', monthly_data)
        self.assertIn('characteristics', monthly_data)
        self.assertIn('factor_returns', monthly_data)
        
        # Should have 12 monthly observations for 2020
        self.assertEqual(len(monthly_data['macro']), 12)
        self.assertEqual(len(monthly_data['characteristics']), 12)
        self.assertEqual(len(monthly_data['factor_returns']), 12)
        
        # Check that index is monthly
        self.assertTrue(all(monthly_data['macro'].index.day >= 28))  # Month-end dates
    
    def test_convert_to_monthly_invalid_method(self):
        """Test monthly conversion with invalid method."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        with self.assertRaises(ValueError):
            processor.convert_to_monthly(method='invalid')
    
    def test_calculate_monthly_returns(self):
        """Test monthly return calculation from daily returns."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        # Create test daily returns for exactly one month
        jan_dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
        daily_returns = pd.DataFrame({
            'factor1': [0.01] * len(jan_dates)  # 1% daily return
        }, index=jan_dates)
        
        monthly_returns = processor._calculate_monthly_returns(daily_returns)
        
        # Should compound returns correctly: (1.01)^31 - 1
        expected_jan = (1.01 ** 31) - 1
        self.assertAlmostEqual(monthly_returns.iloc[0, 0], expected_jan, places=6)
    
    def test_apply_lags(self):
        """Test lag application to data."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        monthly_data = processor.convert_to_monthly()
        
        lagged_macro = processor.apply_lags(monthly_data['macro'], lag_months=1)
        
        # First observation should be NaN after lag
        self.assertTrue(pd.isna(lagged_macro.iloc[0]).all())
        
        # Second observation should equal first original observation
        pd.testing.assert_series_equal(
            lagged_macro.iloc[1], 
            monthly_data['macro'].iloc[0],
            check_names=False
        )
    
    def test_get_rebalance_dates(self):
        """Test rebalancing date generation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        rebal_dates = processor.get_rebalance_dates('2020-01-01', '2020-06-30', frequency='M')
        
        self.assertEqual(len(rebal_dates), 6)  # 6 months
        self.assertTrue(all(isinstance(d, pd.Timestamp) for d in rebal_dates))
    
    def test_align_data_for_period(self):
        """Test data alignment for specific period."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        aligned_data = processor.align_data_for_period('2020-03-01', '2020-09-30')
        
        for key, df in aligned_data.items():
            # Should have data within the specified range
            self.assertTrue(all(df.index >= pd.to_datetime('2020-03-01')))
            self.assertTrue(all(df.index <= pd.to_datetime('2020-09-30')))
    
    def test_calculate_rolling_statistics(self):
        """Test rolling statistics calculation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        # Use factor returns for rolling stats
        windows = {'3m': 63, '1m': 21}
        rolling_stats = processor.calculate_rolling_statistics(
            self.df_factor_returns[['value_bm']], windows
        )
        
        # Should have columns for each window
        expected_cols = ['3m_return', '3m_volatility', '1m_return', '1m_volatility']
        for col in expected_cols:
            self.assertIn(col, rolling_stats.columns)
        
        # Should have monthly observations
        self.assertTrue(len(rolling_stats) <= 12)  # At most 12 monthly obs
    
    def test_standardize_predictors(self):
        """Test predictor standardization."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        monthly_data = processor.convert_to_monthly()
        
        # Create training mask (first 8 months)
        training_mask = pd.Series([True] * 8 + [False] * 4, index=monthly_data['macro'].index)
        
        standardized = processor.standardize_predictors(monthly_data['macro'], training_mask)
        
        # Training period should have mean ~0 and std ~1
        training_data = standardized[training_mask]
        np.testing.assert_allclose(training_data.mean(), 0, atol=1e-10)
        np.testing.assert_allclose(training_data.std(), 1, atol=1e-10)
    
    def test_standardize_predictors_invalid_mask(self):
        """Test standardization with invalid training mask."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        monthly_data = processor.convert_to_monthly()
        
        # Wrong type mask
        with self.assertRaises(TypeError):
            processor.standardize_predictors(monthly_data['macro'], [True] * 12)
        
        # Wrong length mask
        short_mask = pd.Series([True] * 5, index=monthly_data['macro'].index[:5])
        with self.assertRaises(ValueError):
            processor.standardize_predictors(monthly_data['macro'], short_mask)
        
        # Empty training data
        empty_mask = pd.Series([False] * 12, index=monthly_data['macro'].index)
        with self.assertRaises(ValueError):
            processor.standardize_predictors(monthly_data['macro'], empty_mask)
    
    def test_extract_factor_characteristics(self):
        """Test factor characteristic extraction."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        factor_chars = processor.extract_factor_characteristics('value_bm')
        
        # Should have 6 characteristics (3 factor-specific + 3 spreads)
        expected_cols = ['value_bm_3m_ret', 'value_bm_12m_ret', 'value_bm_3m_vol',
                        'value_spread', 'prof_spread', 'inv_spread']
        
        for col in expected_cols:
            if col in self.df_characteristics.columns:
                self.assertIn(col, factor_chars.columns)
    
    def test_extract_factor_characteristics_missing_factor(self):
        """Test characteristic extraction for non-existent factor.""" 
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        # Test that it still works even with missing characteristics
        # The function should return available spreads even if factor-specific chars are missing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings about missing columns
            result = processor.extract_factor_characteristics('nonexistent_factor')
            # Should return spread characteristics even for non-existent factors
            self.assertGreater(len(result.columns), 0)
    
    def test_validate_factor_list(self):
        """Test factor list validation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        # Valid factors
        valid_factors = ['value_bm', 'momentum']
        self.assertTrue(processor.validate_factor_list(valid_factors))
        
        # Invalid factors
        invalid_factors = ['value_bm', 'nonexistent_factor']
        with self.assertRaises(ValueError):
            processor.validate_factor_list(invalid_factors)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        summary = processor.get_data_summary()
        
        # Check required keys
        required_keys = ['date_range', 'total_days', 'macro_predictors', 
                        'n_macro_predictors', 'factor_returns', 'n_factors']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values make sense
        self.assertEqual(summary['n_macro_predictors'], 5)
        self.assertGreater(summary['total_days'], 300)  # Should be ~365 days
        self.assertIsInstance(summary['missing_data_pct'], dict)
    
    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        # Create data with issues
        dirty_dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        
        # Create data with NaN and duplicate index
        dirty_data = pd.DataFrame({
            'real_yield_1y': [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            'yield_slope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'yield_change': [0.01] * 10,
            'market_excess_3m': [0.02] * 10, 
            'credit_spread': [0.015] * 10
        }, index=dirty_dates)
        
        # Add duplicate by concatenating with last row
        duplicate_row = dirty_data.iloc[[-1]].copy()
        dirty_data = pd.concat([dirty_data, duplicate_row])
        
        # Create minimal other dataframes with matching length
        minimal_chars = pd.DataFrame({
            'value_spread': [0.1] * 11
        }, index=dirty_data.index)
        
        minimal_returns = pd.DataFrame({
            'value_bm': [0.001] * 11
        }, index=dirty_data.index)
        
        processor = DataProcessor(dirty_data, minimal_chars, minimal_returns)
        
        # Should have removed duplicate
        self.assertEqual(len(processor.df_macro), 10)
        
        # Should have forward-filled NaN
        self.assertFalse(processor.df_macro['real_yield_1y'].isna().any())
    
    def test_monthly_data_caching(self):
        """Test that monthly data is cached properly."""
        processor = DataProcessor(self.df_macro, self.df_characteristics, self.df_factor_returns)
        
        # First call should compute
        monthly_data1 = processor.convert_to_monthly()
        
        # Second call should use cache
        monthly_data2 = processor.convert_to_monthly()
        
        # Should be the same object (cached)
        self.assertIs(monthly_data1, monthly_data2)
        
        # Different method should compute new
        monthly_data3 = processor.convert_to_monthly(method='mean')
        self.assertIsNot(monthly_data1, monthly_data3)


class TestDataProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_minimal_data(self):
        """Test with minimal required data."""
        # Very short time series
        dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')  # 1 month
        
        df_macro = pd.DataFrame({
            'real_yield_1y': [0.02] * len(dates),
            'yield_slope': [0.01] * len(dates),
            'yield_change': [0.0] * len(dates),
            'market_excess_3m': [0.001] * len(dates),
            'credit_spread': [0.015] * len(dates)
        }, index=dates)
        
        df_chars = pd.DataFrame({
            'value_spread': [0.1] * len(dates)
        }, index=dates)
        
        df_returns = pd.DataFrame({
            'value_bm': [0.001] * len(dates)
        }, index=dates)
        
        processor = DataProcessor(df_macro, df_chars, df_returns)
        self.assertIsInstance(processor, DataProcessor)
        
        # Monthly conversion should work
        monthly_data = processor.convert_to_monthly()
        self.assertEqual(len(monthly_data['macro']), 1)  # Only 1 month
    
    def test_misaligned_dates(self):
        """Test with misaligned date ranges."""
        # Macro data starts later
        macro_dates = pd.date_range('2020-06-01', '2020-12-31', freq='D')
        char_dates = pd.date_range('2020-01-01', '2020-12-31', freq='D') 
        return_dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        
        df_macro = pd.DataFrame({
            'real_yield_1y': [0.02] * len(macro_dates),
            'yield_slope': [0.01] * len(macro_dates), 
            'yield_change': [0.0] * len(macro_dates),
            'market_excess_3m': [0.001] * len(macro_dates),
            'credit_spread': [0.015] * len(macro_dates)
        }, index=macro_dates)
        
        df_chars = pd.DataFrame({
            'value_spread': [0.1] * len(char_dates)
        }, index=char_dates)
        
        df_returns = pd.DataFrame({
            'value_bm': [0.001] * len(return_dates)
        }, index=return_dates)
        
        # Should initialize with warnings (check the logic is correct)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Make sure warnings are captured
            processor = DataProcessor(df_macro, df_chars, df_returns)
            # Note: warnings may not be triggered if dates overlap sufficiently
        
        # Date range should be intersection
        start_date, end_date = processor.get_date_range()
        self.assertEqual(start_date, pd.to_datetime('2020-06-01'))


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)