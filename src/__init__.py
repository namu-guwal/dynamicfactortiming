"""
Dynamic Factor Timing Strategy Implementation

Based on the Northern Trust paper "Dynamic Factor Timing" by Lehnherr, Mehta, and Nagel (2024).
Implements a regularization approach for factor timing using machine learning techniques.
"""

__version__ = "0.1.0"
__author__ = "Implementation based on Northern Trust research"

from .data_processor import DataProcessor
from .factor_interactions import FactorInteractionEngine
from .optimizer import RegularizedOptimizer
from .backtester import DynamicFactorTimingBacktester
from .performance import PerformanceAnalytics, FactorRotationAnalyzer
from .long_only import LongOnlyFactorTiming

__all__ = [
    'DataProcessor',
    'FactorInteractionEngine', 
    'RegularizedOptimizer',
    'DynamicFactorTimingBacktester',
    'PerformanceAnalytics',
    'FactorRotationAnalyzer',
    'LongOnlyFactorTiming'
]