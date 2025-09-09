"""
Unit tests for FactorInteractionEngine class.

Tests all functionality including interaction generation, rotation matrices,
and predictor change calculations.
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

from factor_interactions import FactorInteractionEngine


class TestFactorInteractionEngine(unittest.TestCase):
    """Test cases for FactorInteractionEngine class."""
    
    def setUp(self):
        """Set up test data for all tests."""
        # Create monthly test data
        self.dates = pd.date_range('2020-01-31', '2020-12-31', freq='ME')
        self.n_months = len(self.dates)
        
        # Sample factor returns (N=3 factors)
        self.factor_returns = pd.DataFrame({
            'value_bm': np.random.normal(0.005, 0.05, self.n_months),
            'momentum': np.random.normal(0.008, 0.06, self.n_months), 
            'profitability': np.random.normal(0.003, 0.04, self.n_months)
        }, index=self.dates)
        
        # Sample factor characteristics (6 per factor)
        char_data = {}
        for factor in self.factor_returns.columns:
            char_data[f'{factor}_3m_ret'] = np.random.normal(0.02, 0.1, self.n_months)
            char_data[f'{factor}_12m_ret'] = np.random.normal(0.08, 0.3, self.n_months)
            char_data[f'{factor}_3m_vol'] = np.random.uniform(0.15, 0.4, self.n_months)
        
        # Add spread characteristics
        char_data['value_spread'] = np.random.normal(0.4, 0.2, self.n_months)
        char_data['prof_spread'] = np.random.normal(0.2, 0.15, self.n_months)
        char_data['inv_spread'] = np.random.normal(-0.05, 0.1, self.n_months)
        
        self.factor_characteristics = pd.DataFrame(char_data, index=self.dates)
        
        # Sample macro predictors (J=5 variables)
        self.macro_predictors = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, self.n_months),
            'yield_slope': np.random.normal(0.01, 0.005, self.n_months),
            'yield_change': np.random.normal(0, 0.002, self.n_months),
            'market_excess_3m': np.random.normal(0.005, 0.08, self.n_months),
            'credit_spread': np.random.normal(0.015, 0.005, self.n_months)
        }, index=self.dates)
    
    def test_initialization_valid_data(self):
        """Test successful initialization with valid data."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        self.assertIsInstance(engine, FactorInteractionEngine)
        self.assertEqual(engine.N, 3)  # 3 factors
        self.assertEqual(engine.K, 6)  # 6 characteristics per factor
        self.assertEqual(engine.J, 5)  # 5 macro predictors
        self.assertEqual(engine.M, 3 * (1 + 6 + 5))  # 3 * 12 = 36 total interactions
    
    def test_initialization_misaligned_dates(self):
        """Test initialization with misaligned date indices."""
        # Create macro data with different dates
        misaligned_dates = pd.date_range('2020-02-29', '2020-11-30', freq='ME')
        misaligned_macro = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, len(misaligned_dates))
        }, index=misaligned_dates)
        
        # Should initialize with warnings
        with warnings.catch_warnings(record=True) as w:
            engine = FactorInteractionEngine(
                self.factor_returns, self.factor_characteristics, misaligned_macro
            )
            self.assertGreater(len(w), 0)  # Should have warnings
        
        self.assertIsInstance(engine, FactorInteractionEngine)
    
    def test_initialization_invalid_data(self):
        """Test initialization with invalid data."""
        # Empty factor returns
        empty_returns = pd.DataFrame(index=self.dates)
        with self.assertRaises(ValueError):
            FactorInteractionEngine(empty_returns, self.factor_characteristics, self.macro_predictors)
        
        # Empty macro predictors
        empty_macro = pd.DataFrame(index=self.dates)
        with self.assertRaises(ValueError):
            FactorInteractionEngine(self.factor_returns, self.factor_characteristics, empty_macro)
    
    def test_create_all_interactions(self):
        """Test creation of all interaction timeseries."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'momentum']
        interactions = engine.create_all_interactions(factor_list)
        
        # Should have N*(1+K+J) = 2*(1+6+5) = 24 interactions
        expected_interactions = 2 * (1 + 6 + 5)
        self.assertEqual(len(interactions.columns), expected_interactions)
        
        # Should have same number of periods as input data
        self.assertEqual(len(interactions), self.n_months)
        
        # Check for pure factor return columns
        self.assertIn('value_bm', interactions.columns)
        self.assertIn('momentum', interactions.columns)
        
        # Check for some interaction columns
        self.assertIn('value_bm_x_value_bm_3m_ret', interactions.columns)
        self.assertIn('momentum_x_real_yield_1y', interactions.columns)
    
    def test_create_all_interactions_missing_factors(self):
        """Test interaction creation with missing factors."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        # Request non-existent factor
        with self.assertRaises(ValueError):
            engine.create_all_interactions(['value_bm', 'nonexistent_factor'])
    
    def test_factor_characteristic_interactions(self):
        """Test factor-characteristic interaction generation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm']
        fxc_interactions, fxc_names = engine._create_factor_characteristic_interactions(
            self.factor_returns[factor_list], factor_list
        )
        
        # Should have 6 interactions per factor
        self.assertEqual(len(fxc_interactions), 6)
        self.assertEqual(len(fxc_names), 6)
        
        # Check that interactions use lagged characteristics
        # First period should be NaN due to lag
        self.assertTrue(pd.isna(fxc_interactions[0].iloc[0]))
        
        # Second period should be factor_return[1] * characteristic[0]
        expected_second = (self.factor_returns['value_bm'].iloc[1] * 
                          self.factor_characteristics['value_bm_3m_ret'].iloc[0])
        self.assertAlmostEqual(fxc_interactions[0].iloc[1], expected_second, places=10)
    
    def test_factor_macro_interactions(self):
        """Test factor-macro interaction generation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['momentum']
        fxm_interactions, fxm_names = engine._create_factor_macro_interactions(
            self.factor_returns[factor_list], factor_list
        )
        
        # Should have J interactions per factor
        self.assertEqual(len(fxm_interactions), engine.J)
        self.assertEqual(len(fxm_names), engine.J)
        
        # Check that interactions use lagged macro predictors
        # First period should be NaN due to lag
        self.assertTrue(pd.isna(fxm_interactions[0].iloc[0]))
        
        # Second period should be factor_return[1] * macro_predictor[0]
        expected_second = (self.factor_returns['momentum'].iloc[1] * 
                          self.macro_predictors['real_yield_1y'].iloc[0])
        self.assertAlmostEqual(fxm_interactions[0].iloc[1], expected_second, places=10)
    
    def test_get_factor_characteristics(self):
        """Test factor characteristic extraction."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_chars = engine._get_factor_characteristics('value_bm')
        
        # Should have 6 characteristics
        self.assertEqual(len(factor_chars.columns), 6)
        
        # Should include factor-specific characteristics
        expected_factor_chars = ['value_bm_3m_ret', 'value_bm_12m_ret', 'value_bm_3m_vol']
        for char in expected_factor_chars:
            self.assertIn(char, factor_chars.columns)
        
        # Should include spread characteristics
        expected_spreads = ['value_spread', 'prof_spread', 'inv_spread']
        for spread in expected_spreads:
            self.assertIn(spread, factor_chars.columns)
    
    def test_get_factor_characteristics_missing(self):
        """Test factor characteristic extraction with missing characteristics."""
        # Remove some characteristics from the data
        incomplete_chars = self.factor_characteristics[['value_spread', 'prof_spread']].copy()
        
        engine = FactorInteractionEngine(
            self.factor_returns, incomplete_chars, self.macro_predictors
        )
        
        # Should still return 6 characteristics, filling missing ones with zeros
        with warnings.catch_warnings(record=True) as w:
            factor_chars = engine._get_factor_characteristics('value_bm')
            self.assertGreater(len(w), 0)  # Should have warnings about missing characteristics
        
        self.assertEqual(len(factor_chars.columns), 6)
        
        # Missing characteristics should be filled with zeros
        self.assertTrue((factor_chars['value_bm_3m_ret'] == 0.0).all())
    
    def test_build_rotation_matrix(self):
        """Test rotation matrix construction."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'momentum']
        test_date = self.dates[5]  # Use middle date
        
        R = engine.build_rotation_matrix(test_date, factor_list)
        
        # Should have dimensions M×N
        N = len(factor_list)
        M = N * (1 + engine.K + engine.J)
        self.assertEqual(R.shape, (M, N))
        
        # Each column should have (1 + K + J) non-zero elements  
        for n in range(N):
            col = R[:, n]
            # First element in each factor block should be 1.0 (pure factor)
            factor_start_idx = n * (1 + engine.K + engine.J)
            self.assertEqual(col[factor_start_idx], 1.0)
    
    def test_build_rotation_matrix_with_standardized_data(self):
        """Test rotation matrix with standardized predictors."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm']
        test_date = self.dates[3]
        
        # Create standardized predictor data
        standardized_predictors = {
            'characteristics': self.factor_characteristics * 2,  # Scale up for test
            'macro': self.macro_predictors * 3  # Scale up for test
        }
        
        R = engine.build_rotation_matrix(test_date, factor_list, standardized_predictors)
        R_raw = engine.build_rotation_matrix(test_date, factor_list, None)
        
        # Matrices should be different when using standardized vs raw data
        self.assertFalse(np.array_equal(R, R_raw))
        
        # Pure factor component should still be 1.0
        self.assertEqual(R[0, 0], 1.0)
    
    def test_get_interaction_names(self):
        """Test interaction name generation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'profitability']
        names = engine.get_interaction_names(factor_list)
        
        # Should have N*(1+K+J) names
        expected_count = 2 * (1 + 6 + 5)
        self.assertEqual(len(names), expected_count)
        
        # Check some specific names
        self.assertIn('value_bm', names)
        self.assertIn('profitability', names)
        self.assertIn('value_bm_x_value_bm_3m_ret', names)
        self.assertIn('profitability_x_real_yield_1y', names)
    
    def test_validate_rotation_matrix(self):
        """Test rotation matrix validation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'momentum']
        test_date = self.dates[2]
        
        # Valid matrix
        R = engine.build_rotation_matrix(test_date, factor_list)
        self.assertTrue(engine.validate_rotation_matrix(R, factor_list))
        
        # Invalid matrix dimensions
        R_wrong_shape = np.zeros((10, 2))
        with self.assertRaises(ValueError):
            engine.validate_rotation_matrix(R_wrong_shape, factor_list)
    
    def test_calculate_predictor_changes(self):
        """Test predictor change calculation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        date1 = self.dates[3]
        date2 = self.dates[4] 
        
        # Create standardized data (z-scores)
        standardized_data = {
            'characteristics': (self.factor_characteristics - self.factor_characteristics.mean()) / 
                             self.factor_characteristics.std(),
            'macro': (self.macro_predictors - self.macro_predictors.mean()) / 
                    self.macro_predictors.std()
        }
        
        changes = engine.calculate_predictor_changes(date1, date2, standardized_data)
        
        # Should have changes for all predictors
        all_predictors = (list(self.factor_characteristics.columns) + 
                         list(self.macro_predictors.columns))
        for predictor in all_predictors:
            self.assertIn(predictor, changes)
        
        # Check calculation is correct for one predictor
        expected_change = (standardized_data['macro'].loc[date2, 'real_yield_1y'] - 
                          standardized_data['macro'].loc[date1, 'real_yield_1y'])
        self.assertAlmostEqual(changes['real_yield_1y'], expected_change, places=10)
    
    def test_calculate_predictor_changes_missing_dates(self):
        """Test predictor change calculation with missing dates."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        # Use dates not in the data
        bad_date1 = pd.Timestamp('2019-12-31')
        bad_date2 = pd.Timestamp('2021-01-31')
        
        standardized_data = {
            'characteristics': self.factor_characteristics,
            'macro': self.macro_predictors
        }
        
        changes = engine.calculate_predictor_changes(bad_date1, bad_date2, standardized_data)
        
        # Should return zeros for missing dates
        for predictor, change in changes.items():
            self.assertEqual(change, 0.0)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        summary = engine.get_data_summary()
        
        # Check required keys
        required_keys = ['n_factors', 'n_characteristics_per_factor', 'n_macro_predictors',
                        'total_interactions', 'factor_names', 'macro_predictor_names']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check values
        self.assertEqual(summary['n_factors'], 3)
        self.assertEqual(summary['n_characteristics_per_factor'], 6)
        self.assertEqual(summary['n_macro_predictors'], 5)
        self.assertEqual(summary['total_interactions'], 36)
        self.assertEqual(len(summary['factor_names']), 3)


class TestFactorInteractionEngineIntegration(unittest.TestCase):
    """Integration tests for complex scenarios."""
    
    def setUp(self):
        """Set up more complex test data."""
        self.dates = pd.date_range('2018-01-31', '2023-12-31', freq='ME')
        self.n_months = len(self.dates)
        
        # More realistic factor returns with some serial correlation
        np.random.seed(42)  # For reproducible tests
        
        self.factor_returns = pd.DataFrame({
            'value_bm': np.random.normal(0.005, 0.05, self.n_months),
            'momentum': np.random.normal(0.008, 0.06, self.n_months), 
            'profitability': np.random.normal(0.003, 0.04, self.n_months),
            'quality': np.random.normal(0.004, 0.045, self.n_months),
            'size': np.random.normal(0.001, 0.055, self.n_months),
            'low_vol': np.random.normal(0.002, 0.035, self.n_months)
        }, index=self.dates)
        
        # Full factor characteristics
        char_data = {}
        for factor in self.factor_returns.columns:
            char_data[f'{factor}_3m_ret'] = np.random.normal(0.02, 0.1, self.n_months)
            char_data[f'{factor}_12m_ret'] = np.random.normal(0.08, 0.3, self.n_months)
            char_data[f'{factor}_3m_vol'] = np.random.uniform(0.15, 0.4, self.n_months)
        
        char_data['value_spread'] = np.random.normal(0.4, 0.2, self.n_months)
        char_data['prof_spread'] = np.random.normal(0.2, 0.15, self.n_months)
        char_data['inv_spread'] = np.random.normal(-0.05, 0.1, self.n_months)
        
        self.factor_characteristics = pd.DataFrame(char_data, index=self.dates)
        
        # Full macro predictors
        self.macro_predictors = pd.DataFrame({
            'real_yield_1y': np.random.normal(0.02, 0.01, self.n_months),
            'yield_slope': np.random.normal(0.01, 0.005, self.n_months),
            'yield_change': np.random.normal(0, 0.002, self.n_months),
            'market_excess_3m': np.random.normal(0.005, 0.08, self.n_months),
            'credit_spread': np.random.normal(0.015, 0.005, self.n_months)
        }, index=self.dates)
    
    def test_full_interaction_generation(self):
        """Test full interaction generation with all factors."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        all_factors = list(self.factor_returns.columns)
        interactions = engine.create_all_interactions(all_factors)
        
        # Should have N*(1+K+J) = 6*(1+6+5) = 72 interactions
        expected_interactions = 6 * (1 + 6 + 5)
        self.assertEqual(len(interactions.columns), expected_interactions)
        
        # Should have same time series length
        self.assertEqual(len(interactions), self.n_months)
        
        # Check that there are no completely empty columns
        for col in interactions.columns:
            self.assertFalse(interactions[col].isna().all(), f"Column {col} is all NaN")
    
    def test_rotation_matrix_consistency(self):
        """Test that rotation matrices are consistent across dates."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'momentum', 'profitability']
        
        # Build rotation matrices for different dates
        dates_to_test = [self.dates[12], self.dates[36], self.dates[60]]
        matrices = []
        
        for date in dates_to_test:
            R = engine.build_rotation_matrix(date, factor_list)
            matrices.append(R)
            
            # Validate each matrix
            self.assertTrue(engine.validate_rotation_matrix(R, factor_list))
        
        # All matrices should have the same shape
        for R in matrices[1:]:
            self.assertEqual(R.shape, matrices[0].shape)
        
        # Pure factor components should always be 1.0
        N = len(factor_list)
        for R in matrices:
            for n in range(N):
                factor_start_idx = n * (1 + engine.K + engine.J)
                self.assertEqual(R[factor_start_idx, n], 1.0)
    
    def test_interaction_names_uniqueness(self):
        """Test that all interaction names are unique."""
        engine = FactorInteractionEngine(
            self.factor_returns, self.factor_characteristics, self.macro_predictors
        )
        
        factor_list = ['value_bm', 'momentum', 'profitability', 'quality']
        names = engine.get_interaction_names(factor_list)
        
        # All names should be unique
        self.assertEqual(len(names), len(set(names)))
        
        # Should have correct total count
        expected_count = len(factor_list) * (1 + engine.K + engine.J)
        self.assertEqual(len(names), expected_count)


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)