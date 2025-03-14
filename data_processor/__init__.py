# data_processor/__init__.py
"""
Data Processing Framework
A comprehensive toolkit for data preprocessing, normalization, transformation and validation.
"""

from data_processor.core.processor import DataProcessor
from data_processor.core.modules.base import BaseModule
from data_processor.core.modules.cleaner import DataCleaner
from data_processor.core.modules.normalizer import DataNormalizer
from data_processor.core.modules.transformer import DataTransformer
from data_processor.core.modules.validator import DataValidator

__version__ = "1.0.0"
__all__ = [
    'DataProcessor',
    'BaseModule',
    'DataCleaner',
    'DataNormalizer',
    'DataTransformer',
    'DataValidator'
]