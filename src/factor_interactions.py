"""
Factor Interactions Module for Dynamic Factor Timing Strategy

Generates factor interaction timeseries following the methodology in the paper:
- FXC: Factor × Characteristic interactions (Equation 1)
- FXM: Factor × Macro interactions (Equation 2)
- Creates rotation matrices for converting interaction weights to factor weights (Equation 4)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
import logging

logger = logging.getLogger(__name__)


class FactorInteractionEngine:
    """
    Engine for creating factor interaction timeseries and rotation matrices.
    
    Implements the methodology from the paper:
    - Pure factor returns (N timeseries)
    - Factor-characteristic interactions (N×K timeseries) 
    - Factor-macro interactions (N×J timeseries)
    - Total M = N×(1+K+J) interaction timeseries
    """
    
    def __init__(self, factor_returns: pd.DataFrame, factor_characteristics: pd.DataFrame, 
                 macro_predictors: pd.DataFrame):
        """
        Initialize the Factor Interaction Engine.
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Monthly factor returns (N columns)
        factor_characteristics : pd.DataFrame  
            Monthly factor characteristics (6×N columns)
        macro_predictors : pd.DataFrame
            Monthly macro/market predictors (J columns)
        """
        self.factor_returns = factor_returns.copy()
        self.factor_characteristics = factor_characteristics.copy()
        self.macro_predictors = macro_predictors.copy()
        
        # Validate input data
        self._validate_inputs()
        
        # Get dimensions
        self.N = len(self.factor_returns.columns)  # Number of factors
        self.K = 6  # Number of characteristics per factor (fixed by methodology)
        self.J = len(self.macro_predictors.columns)  # Number of macro predictors
        self.M = self.N * (1 + self.K + self.J)  # Total interactions
        
        logger.info(f"FactorInteractionEngine initialized: N={self.N}, K={self.K}, J={self.J}, M={self.M}")
    
    def _validate_inputs(self) -> None:
        """Validate input dataframes."""
        # Check for matching date indices
        if not self.factor_returns.index.equals(self.factor_characteristics.index):
            warnings.warn("Factor returns and characteristics have different date indices")
        if not self.factor_returns.index.equals(self.macro_predictors.index):
            warnings.warn("Factor returns and macro predictors have different date indices")
        
        # Check for minimum data requirements
        if len(self.factor_returns) < 12:
            warnings.warn("Less than 12 months of data available")
        if len(self.factor_returns.columns) == 0:
            raise ValueError("No factors provided in factor_returns")
        if len(self.macro_predictors.columns) == 0:
            raise ValueError("No macro predictors provided")
    
    def create_all_interactions(self, factor_list: List[str]) -> pd.DataFrame:
        """
        Generate all M factor interaction timeseries.
        
        Parameters:
        -----------
        factor_list : List[str]
            List of factor names to include
            
        Returns:
        --------
        DataFrame with M interaction columns and T time observations
        """
        # Validate factor list
        missing_factors = set(factor_list) - set(self.factor_returns.columns)
        if missing_factors:
            raise ValueError(f"Missing factors in returns data: {missing_factors}")
        
        # Subset data to requested factors
        factor_returns_subset = self.factor_returns[factor_list]
        
        # Generate all interaction components
        interactions_list = []
        interaction_names = []
        
        # 1. Pure factor returns (N timeseries)
        for factor in factor_list:
            interactions_list.append(factor_returns_subset[factor])
            interaction_names.append(f"{factor}")
        
        # 2. Factor-characteristic interactions (N×K timeseries)  
        fxc_interactions, fxc_names = self._create_factor_characteristic_interactions(
            factor_returns_subset, factor_list
        )
        interactions_list.extend(fxc_interactions)
        interaction_names.extend(fxc_names)
        
        # 3. Factor-macro interactions (N×J timeseries)
        fxm_interactions, fxm_names = self._create_factor_macro_interactions(
            factor_returns_subset, factor_list
        )
        interactions_list.extend(fxm_interactions)
        interaction_names.extend(fxm_names)
        
        # Combine all interactions
        interactions_df = pd.concat(interactions_list, axis=1, keys=interaction_names)
        interactions_df.columns = interaction_names
        
        logger.info(f"Created {len(interactions_df.columns)} interaction timeseries")
        
        return interactions_df
    
    def _create_factor_characteristic_interactions(self, factor_returns: pd.DataFrame, 
                                                 factor_list: List[str]) -> Tuple[List[pd.Series], List[str]]:
        """
        Create factor-characteristic interactions: FXC[n,k,t] = F[n,t] * XC[n,k,t-1]
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns subset
        factor_list : List[str] 
            List of factor names
            
        Returns:
        --------
        Tuple of (interaction_series_list, interaction_names)
        """
        interactions = []
        names = []
        
        for factor in factor_list:
            # Get characteristics for this factor (6 per factor)
            factor_chars = self._get_factor_characteristics(factor)
            
            for char_name, char_series in factor_chars.items():
                # Apply T-1 lag to characteristics
                lagged_char = char_series.shift(1)
                
                # Create interaction: F[n,t] * XC[n,k,t-1]
                interaction = factor_returns[factor] * lagged_char
                
                interactions.append(interaction)
                names.append(f"{factor}_x_{char_name}")
        
        return interactions, names
    
    def _create_factor_macro_interactions(self, factor_returns: pd.DataFrame,
                                        factor_list: List[str]) -> Tuple[List[pd.Series], List[str]]:
        """
        Create factor-macro interactions: FXM[n,j,t] = F[n,t] * XM[j,t-1]
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns subset  
        factor_list : List[str]
            List of factor names
            
        Returns:
        --------
        Tuple of (interaction_series_list, interaction_names)
        """
        interactions = []
        names = []
        
        for factor in factor_list:
            for macro_var in self.macro_predictors.columns:
                # Apply T-1 lag to macro predictors
                lagged_macro = self.macro_predictors[macro_var].shift(1)
                
                # Create interaction: F[n,t] * XM[j,t-1] 
                interaction = factor_returns[factor] * lagged_macro
                
                interactions.append(interaction)
                names.append(f"{factor}_x_{macro_var}")
        
        return interactions, names
    
    def _get_factor_characteristics(self, factor_name: str) -> pd.DataFrame:
        """
        Get the 6 characteristics for a specific factor.
        
        Parameters:
        -----------
        factor_name : str
            Name of the factor
            
        Returns:
        --------
        DataFrame with 6 characteristics for the factor
        """
        # Expected characteristic patterns
        factor_specific_suffixes = ['_3m_ret', '_12m_ret', '_3m_vol']
        spread_characteristics = ['value_spread', 'prof_spread', 'inv_spread']
        
        characteristics = {}
        
        # Factor-specific characteristics
        for suffix in factor_specific_suffixes:
            col_name = f"{factor_name}{suffix}"
            if col_name in self.factor_characteristics.columns:
                characteristics[col_name] = self.factor_characteristics[col_name]
            else:
                # Create dummy characteristic if missing (filled with zeros)
                characteristics[col_name] = pd.Series(0.0, index=self.factor_characteristics.index)
                warnings.warn(f"Missing characteristic {col_name}, using zeros")
        
        # Cross-factor spread characteristics (same for all factors)
        for spread in spread_characteristics:
            if spread in self.factor_characteristics.columns:
                characteristics[spread] = self.factor_characteristics[spread]
            else:
                # Create dummy spread if missing
                characteristics[spread] = pd.Series(0.0, index=self.factor_characteristics.index)
                warnings.warn(f"Missing spread characteristic {spread}, using zeros")
        
        return pd.DataFrame(characteristics)
    
    def build_rotation_matrix(self, date: pd.Timestamp, factor_list: List[str],
                            standardized_predictors: Optional[Dict[str, pd.DataFrame]] = None) -> np.ndarray:
        """
        Build rotation matrix R for converting interaction weights to factor weights.
        
        From Equation 4: w_f = (w^T R)^T
        R is M×N matrix where each column has (1+K+J) non-zero elements for each factor.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Date for which to build rotation matrix (end of training period)
        factor_list : List[str]
            List of factor names
        standardized_predictors : Dict[str, pd.DataFrame], optional
            Pre-standardized predictors. If None, uses raw values.
            
        Returns:
        --------
        M×N rotation matrix
        """
        N = len(factor_list)
        M = N * (1 + self.K + self.J)
        
        R = np.zeros((M, N))
        
        # Fill rotation matrix column by column (one column per factor)
        for n, factor in enumerate(factor_list):
            row_idx = 0
            
            # 1. Pure factor return component (always 1.0)
            factor_start_idx = n * (1 + self.K + self.J)
            R[factor_start_idx + row_idx, n] = 1.0
            row_idx += 1
            
            # 2. Factor characteristic components
            factor_chars = self._get_factor_characteristics(factor)
            for char_name in factor_chars.columns:
                # Use standardized predictor value if available, else raw value
                if standardized_predictors and 'characteristics' in standardized_predictors:
                    if char_name in standardized_predictors['characteristics'].columns:
                        predictor_value = standardized_predictors['characteristics'].loc[date, char_name]
                    else:
                        predictor_value = 0.0  # Missing characteristic
                else:
                    if char_name in self.factor_characteristics.columns:
                        predictor_value = self.factor_characteristics.loc[date, char_name]
                    else:
                        predictor_value = 0.0
                
                R[factor_start_idx + row_idx, n] = predictor_value
                row_idx += 1
            
            # 3. Macro predictor components  
            for macro_var in self.macro_predictors.columns:
                # Use standardized predictor value if available, else raw value
                if standardized_predictors and 'macro' in standardized_predictors:
                    if macro_var in standardized_predictors['macro'].columns:
                        predictor_value = standardized_predictors['macro'].loc[date, macro_var]
                    else:
                        predictor_value = 0.0  # Missing macro predictor
                else:
                    if macro_var in self.macro_predictors.columns:
                        predictor_value = self.macro_predictors.loc[date, macro_var]
                    else:
                        predictor_value = 0.0
                
                R[factor_start_idx + row_idx, n] = predictor_value
                row_idx += 1
        
        return R
    
    def get_interaction_names(self, factor_list: List[str]) -> List[str]:
        """
        Generate descriptive names for all interaction timeseries.
        
        Parameters:
        -----------
        factor_list : List[str]
            List of factor names
            
        Returns:
        --------
        List of interaction names in order
        """
        names = []
        
        # Pure factor names
        for factor in factor_list:
            names.append(f"{factor}")
        
        # Factor-characteristic interaction names
        for factor in factor_list:
            factor_chars = self._get_factor_characteristics(factor)
            for char_name in factor_chars.columns:
                names.append(f"{factor}_x_{char_name}")
        
        # Factor-macro interaction names
        for factor in factor_list:
            for macro_var in self.macro_predictors.columns:
                names.append(f"{factor}_x_{macro_var}")
        
        return names
    
    def validate_rotation_matrix(self, R: np.ndarray, factor_list: List[str]) -> bool:
        """
        Validate rotation matrix dimensions and structure.
        
        Parameters:
        -----------
        R : np.ndarray
            Rotation matrix to validate
        factor_list : List[str]
            List of factor names
            
        Returns:
        --------
        True if valid, raises ValueError if invalid
        """
        N = len(factor_list)
        M = N * (1 + self.K + self.J)
        
        if R.shape != (M, N):
            raise ValueError(f"Rotation matrix shape {R.shape} != expected {(M, N)}")
        
        # Check that each column has exactly (1 + K + J) non-zero elements
        for n in range(N):
            col = R[:, n]
            non_zero_count = np.count_nonzero(col)
            expected_nonzero = 1 + self.K + self.J
            
            if non_zero_count < expected_nonzero:
                warnings.warn(f"Column {n} has {non_zero_count} non-zero elements, expected {expected_nonzero}")
        
        return True
    
    def calculate_predictor_changes(self, date1: pd.Timestamp, date2: pd.Timestamp,
                                  standardized_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate changes in standardized predictors between two dates.
        
        Used for factor rotation analysis (Exhibit 8).
        
        Parameters:
        -----------
        date1 : pd.Timestamp
            Start date
        date2 : pd.Timestamp
            End date
        standardized_data : Dict[str, pd.DataFrame]
            Standardized predictor data
            
        Returns:
        --------
        Dictionary of predictor changes (z-score changes)
        """
        changes = {}
        
        # Factor characteristic changes
        if 'characteristics' in standardized_data:
            chars = standardized_data['characteristics']
            for col in chars.columns:
                if date1 in chars.index and date2 in chars.index:
                    changes[col] = chars.loc[date2, col] - chars.loc[date1, col]
                else:
                    changes[col] = 0.0
        
        # Macro predictor changes
        if 'macro' in standardized_data:
            macro = standardized_data['macro']
            for col in macro.columns:
                if date1 in macro.index and date2 in macro.index:
                    changes[col] = macro.loc[date2, col] - macro.loc[date1, col]
                else:
                    changes[col] = 0.0
        
        return changes
    
    def get_data_summary(self) -> Dict[str, any]:
        """Get summary of the interaction engine data."""
        return {
            'n_factors': self.N,
            'n_characteristics_per_factor': self.K,
            'n_macro_predictors': self.J,
            'total_interactions': self.M,
            'factor_names': list(self.factor_returns.columns),
            'macro_predictor_names': list(self.macro_predictors.columns),
            'date_range': (self.factor_returns.index.min(), self.factor_returns.index.max()),
            'n_periods': len(self.factor_returns)
        }