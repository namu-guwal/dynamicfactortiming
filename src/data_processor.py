"""
Data Processing Module for Dynamic Factor Timing Strategy

Handles data ingestion, validation, frequency conversion, and preprocessing
of macro predictors, factor characteristics, and factor returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Core data processing class for Dynamic Factor Timing strategy.
    
    Handles three daily dataframes:
    1. Macro predictors (5 variables)
    2. Factor characteristics (6 per factor) 
    3. Factor returns (long/short factor portfolios)
    """
    
    def __init__(self, df_macro: pd.DataFrame, df_characteristics: pd.DataFrame, 
                 df_factor_returns: pd.DataFrame):
        """
        Initialize DataProcessor with daily dataframes.
        
        Parameters:
        -----------
        df_macro : pd.DataFrame
            Daily macro/market predictors with date index
        df_characteristics : pd.DataFrame  
            Daily factor characteristics with date index
        df_factor_returns : pd.DataFrame
            Daily long/short factor returns with date index
        """
        self.df_macro_raw = df_macro.copy()
        self.df_characteristics_raw = df_characteristics.copy()
        self.df_factor_returns_raw = df_factor_returns.copy()
        
        # Validate input data
        self._validate_input_data()
        
        # Clean and prepare data
        self.df_macro = self._clean_dataframe(self.df_macro_raw)
        self.df_characteristics = self._clean_dataframe(self.df_characteristics_raw)  
        self.df_factor_returns = self._clean_dataframe(self.df_factor_returns_raw)
        
        # Store processed monthly data
        self._monthly_data_cache = {}
        
        logger.info(f"DataProcessor initialized successfully")
        logger.info(f"Data period: {self.get_date_range()}")
    
    def _validate_input_data(self) -> None:
        """Validate input dataframes for consistency and requirements."""
        
        # Check that all dataframes have date indices
        for name, df in [('macro', self.df_macro_raw), 
                        ('characteristics', self.df_characteristics_raw),
                        ('factor_returns', self.df_factor_returns_raw)]:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(f"{name} dataframe must have DatetimeIndex")
            if df.empty:
                raise ValueError(f"{name} dataframe cannot be empty")
        
        # Check for required macro predictor columns
        required_macro_cols = ['real_yield_1y', 'yield_slope', 'yield_change', 
                              'market_excess_3m', 'credit_spread']
        missing_macro = set(required_macro_cols) - set(self.df_macro_raw.columns)
        if missing_macro:
            raise ValueError(f"Missing required macro columns: {missing_macro}")
        
        # Validate date ranges overlap
        macro_range = (self.df_macro_raw.index.min(), self.df_macro_raw.index.max())
        char_range = (self.df_characteristics_raw.index.min(), self.df_characteristics_raw.index.max())
        factor_range = (self.df_factor_returns_raw.index.min(), self.df_factor_returns_raw.index.max())
        
        if not (macro_range[0] <= char_range[1] and char_range[0] <= macro_range[1]):
            warnings.warn("Limited overlap between macro and characteristics data")
        if not (macro_range[0] <= factor_range[1] and factor_range[0] <= macro_range[1]):
            warnings.warn("Limited overlap between macro and factor returns data")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe for processing."""
        df_clean = df.copy()
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
        
        # Handle missing values with forward fill (limited)
        df_clean = df_clean.ffill(limit=5)
        
        return df_clean
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the common date range across all datasets."""
        start_dates = [
            self.df_macro.index.min(),
            self.df_characteristics.index.min(), 
            self.df_factor_returns.index.min()
        ]
        end_dates = [
            self.df_macro.index.max(),
            self.df_characteristics.index.max(),
            self.df_factor_returns.index.max()
        ]
        
        return max(start_dates), min(end_dates)
    
    def convert_to_monthly(self, method: str = 'last') -> Dict[str, pd.DataFrame]:
        """
        Convert daily data to monthly observations.
        
        Parameters:
        -----------
        method : str
            Method for conversion ('last' for month-end, 'mean' for monthly average)
            
        Returns:
        --------
        Dict with keys 'macro', 'characteristics', 'factor_returns'
        """
        if method not in ['last', 'mean']:
            raise ValueError("Method must be 'last' or 'mean'")
        
        if method in self._monthly_data_cache:
            return self._monthly_data_cache[method]
        
        if method == 'last':
            # Use month-end values
            monthly_macro = self.df_macro.resample('ME').last()
            monthly_characteristics = self.df_characteristics.resample('ME').last()
            # Calculate monthly returns from daily returns
            monthly_factor_returns = self._calculate_monthly_returns(self.df_factor_returns)
        else:
            # Use monthly averages
            monthly_macro = self.df_macro.resample('ME').mean()
            monthly_characteristics = self.df_characteristics.resample('ME').mean()
            monthly_factor_returns = self._calculate_monthly_returns(self.df_factor_returns)
        
        monthly_data = {
            'macro': monthly_macro,
            'characteristics': monthly_characteristics,
            'factor_returns': monthly_factor_returns
        }
        
        self._monthly_data_cache[method] = monthly_data
        return monthly_data
    
    def _calculate_monthly_returns(self, daily_returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate monthly returns from daily returns."""
        # Convert to monthly compound returns: (1 + daily_ret).prod() - 1
        monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
        return monthly_returns
    
    def apply_lags(self, data: pd.DataFrame, lag_months: int = 1) -> pd.DataFrame:
        """
        Apply lag to predictor data (T-1 convention).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to lag
        lag_months : int
            Number of months to lag
            
        Returns:
        --------
        Lagged dataframe
        """
        return data.shift(lag_months)
    
    def get_rebalance_dates(self, start_date: str, end_date: str, 
                           frequency: str = 'M') -> pd.DatetimeIndex:
        """
        Generate rebalancing dates for backtesting.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str  
            End date in YYYY-MM-DD format
        frequency : str
            Rebalancing frequency ('M' for monthly)
            
        Returns:
        --------
        DatetimeIndex of rebalancing dates
        """
        freq_map = {'M': 'ME'}  # Handle deprecated frequency codes
        freq = freq_map.get(frequency, frequency)
        return pd.date_range(start=start_date, end=end_date, freq=freq)
    
    def align_data_for_period(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Align all datasets for specific backtesting period.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        Dictionary of aligned monthly datasets
        """
        monthly_data = self.convert_to_monthly()
        
        # Convert dates to pandas timestamps
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        aligned_data = {}
        for key, df in monthly_data.items():
            mask = (df.index >= start_ts) & (df.index <= end_ts)
            aligned_data[key] = df[mask]
        
        # Check for sufficient data
        min_periods = 12  # At least 12 months
        for key, df in aligned_data.items():
            if len(df) < min_periods:
                warnings.warn(f"Limited data for {key}: {len(df)} periods")
        
        return aligned_data
    
    def calculate_rolling_statistics(self, data: pd.DataFrame, 
                                   windows: Dict[str, int]) -> pd.DataFrame:
        """
        Calculate rolling statistics and convert to monthly observations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Daily data for rolling calculations
        windows : Dict[str, int]
            Rolling windows in trading days, e.g., {'3m': 63, '12m': 252}
            
        Returns:
        --------
        DataFrame with rolling statistics
        """
        rolling_stats = pd.DataFrame(index=data.index)
        
        for name, window in windows.items():
            # Rolling returns
            rolling_stats[f'{name}_return'] = data.rolling(window).apply(
                lambda x: (1 + x).prod() - 1, raw=False
            )
            
            # Rolling volatility  
            rolling_stats[f'{name}_volatility'] = data.rolling(window).std()
        
        # Convert to monthly observations
        return rolling_stats.resample('ME').last()
    
    def standardize_predictors(self, predictors: pd.DataFrame, 
                             training_mask: pd.Series) -> pd.DataFrame:
        """
        Z-score standardization using training period statistics only.
        
        Parameters:
        -----------
        predictors : pd.DataFrame
            Predictor data to standardize
        training_mask : pd.Series
            Boolean mask indicating training period
            
        Returns:
        --------
        Standardized predictors dataframe
        """
        if not isinstance(training_mask, pd.Series):
            raise TypeError("training_mask must be pandas Series")
        if len(training_mask) != len(predictors):
            raise ValueError("training_mask length must match predictors")
        
        training_data = predictors[training_mask]
        
        if len(training_data) == 0:
            raise ValueError("No training data available")
        
        # Calculate training period statistics
        means = training_data.mean()
        stds = training_data.std()
        
        # Handle zero standard deviations
        stds = stds.replace(0, 1)  # Avoid division by zero
        
        # Standardize entire series using training statistics
        standardized = (predictors - means) / stds
        
        return standardized
    
    def extract_factor_characteristics(self, factor_name: str) -> pd.DataFrame:
        """
        Extract characteristics for a specific factor.
        
        Parameters:
        -----------
        factor_name : str
            Name of the factor (e.g., 'value_bm', 'momentum')
            
        Returns:
        --------
        DataFrame with 6 characteristics for the factor
        """
        # Expected characteristic columns for each factor
        char_suffixes = ['_3m_ret', '_12m_ret', '_3m_vol']
        spread_chars = ['value_spread', 'prof_spread', 'inv_spread']
        
        factor_chars = []
        
        # Factor-specific characteristics
        for suffix in char_suffixes:
            col_name = f'{factor_name}{suffix}'
            if col_name in self.df_characteristics.columns:
                factor_chars.append(col_name)
            else:
                warnings.warn(f"Missing factor characteristic: {col_name}")
        
        # Cross-factor spreads (same for all factors)
        for spread in spread_chars:
            if spread in self.df_characteristics.columns:
                factor_chars.append(spread)
            else:
                warnings.warn(f"Missing spread characteristic: {spread}")
        
        if not factor_chars:
            raise ValueError(f"No characteristics found for factor: {factor_name}")
        
        return self.df_characteristics[factor_chars]
    
    def validate_factor_list(self, factor_list: List[str]) -> bool:
        """
        Validate that all factors in list are available in factor returns.
        
        Parameters:
        -----------
        factor_list : List[str]
            List of factor names to validate
            
        Returns:
        --------
        True if all factors are available
        """
        available_factors = set(self.df_factor_returns.columns)
        requested_factors = set(factor_list)
        
        missing_factors = requested_factors - available_factors
        if missing_factors:
            raise ValueError(f"Missing factors in returns data: {missing_factors}")
        
        return True
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary statistics of the loaded data."""
        start_date, end_date = self.get_date_range()
        
        summary = {
            'date_range': (start_date, end_date),
            'total_days': (end_date - start_date).days,
            'macro_predictors': list(self.df_macro.columns),
            'n_macro_predictors': len(self.df_macro.columns),
            'factor_characteristics': list(self.df_characteristics.columns),
            'n_factor_characteristics': len(self.df_characteristics.columns),
            'factor_returns': list(self.df_factor_returns.columns),
            'n_factors': len(self.df_factor_returns.columns),
            'missing_data_pct': {
                'macro': self.df_macro.isnull().mean().mean(),
                'characteristics': self.df_characteristics.isnull().mean().mean(),
                'factor_returns': self.df_factor_returns.isnull().mean().mean()
            }
        }
        
        return summary