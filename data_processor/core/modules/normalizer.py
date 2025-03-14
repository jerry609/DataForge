# data_processor/core/modules/normalizer.py
"""Data normalization module for standardizing and transforming variables"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from data_processor.core.modules.base import BaseModule
from data_processor.utils.logging_utils import get_logger

logger = get_logger("DataNormalizer")

class DataNormalizer(BaseModule):
    """Data normalization module for standardizing and transforming variables"""

    def __init__(self):
        """Initialize the data normalizer"""
        super().__init__()
        self.scalers = {}
        logger.info("Initialized DataNormalizer")

    def z_score_normalize(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Z-Score normalization

        Args:
            df: Input DataFrame
            columns: Columns to normalize, None means all numeric columns

        Returns:
            Normalized DataFrame
        """
        result_df = df.copy()

        # If no columns specified, select all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(columns) > 0:
            # Create and train scaler
            scaler = StandardScaler()
            scaled_values = scaler.fit_transform(df[columns])

            # Put scaled values back into DataFrame
            result_df[columns] = scaled_values

            # Store scaler for future use
            self.scalers['z_score'] = {
                'scaler': scaler,
                'columns': columns
            }

            # Log operation
            self.log_operation({
                'operation': 'normalize',
                'method': 'z_score',
                'columns': columns
            })
            logger.info(f"Applied Z-Score normalization to columns: {columns}")

        return result_df

    def min_max_normalize(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """
        Min-Max normalization

        Args:
            df: Input DataFrame
            columns: Columns to normalize, None means all numeric columns
            feature_range: Target range for normalization

        Returns:
            Normalized DataFrame
        """
        result_df = df.copy()

        # If no columns specified, select all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(columns) > 0:
            # Create and train scaler
            scaler = MinMaxScaler(feature_range=feature_range)
            scaled_values = scaler.fit_transform(df[columns])

            # Put scaled values back into DataFrame
            result_df[columns] = scaled_values

            # Store scaler for future use
            self.scalers['min_max'] = {
                'scaler': scaler,
                'columns': columns
            }

            # Log operation
            self.log_operation({
                'operation': 'normalize',
                'method': 'min_max',
                'columns': columns,
                'feature_range': feature_range
            })
            logger.info(f"Applied Min-Max normalization to columns: {columns}, range={feature_range}")

        return result_df

    def robust_scale(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Robust scaling (based on quartiles, less sensitive to outliers)

        Args:
            df: Input DataFrame
            columns: Columns to scale, None means all numeric columns

        Returns:
            Scaled DataFrame
        """
        result_df = df.copy()

        # If no columns specified, select all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(columns) > 0:
            # Create and train scaler
            scaler = RobustScaler()
            scaled_values = scaler.fit_transform(df[columns])

            # Put scaled values back into DataFrame
            result_df[columns] = scaled_values

            # Store scaler for future use
            self.scalers['robust'] = {
                'scaler': scaler,
                'columns': columns
            }

            # Log operation
            self.log_operation({
                'operation': 'normalize',
                'method': 'robust',
                'columns': columns
            })
            logger.info(f"Applied Robust scaling to columns: {columns}")

        return result_df

    def log_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                      base: float = np.e, offset: float = 1.0) -> pd.DataFrame:
        """
        Logarithmic transformation

        Args:
            df: Input DataFrame
            columns: Columns to transform, None means all numeric columns
            base: Logarithm base
            offset: Offset to add to values to avoid log(0) or log(negative)

        Returns:
            Transformed DataFrame
        """
        result_df = df.copy()

        # If no columns specified, select all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Apply log transform
                if base == np.e:
                    result_df[col] = np.log(df[col] + offset)
                else:
                    result_df[col] = np.log(df[col] + offset) / np.log(base)

                # Log operation
                self.log_operation({
                    'operation': 'transform',
                    'method': 'log',
                    'column': col,
                    'base': 'e' if base == np.e else base,
                    'offset': offset
                })
                logger.info(f"Applied log transform to column '{col}', base={base}, offset={offset}")

        return result_df

    def inverse_transform(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Inverse transform (convert normalized/standardized data back to original scale)

        Args:
            df: Input DataFrame
            method: Method used for original scaling: 'z_score', 'min_max', 'robust'

        Returns:
            Inverse-transformed DataFrame
        """
        result_df = df.copy()

        if method in self.scalers:
            scaler = self.scalers[method]['scaler']
            columns = self.scalers[method]['columns']

            # Apply inverse transform
            original_values = scaler.inverse_transform(df[columns])

            # Put inverse-transformed values back into DataFrame
            result_df[columns] = original_values

            # Log operation
            self.log_operation({
                'operation': 'inverse_transform',
                'method': method,
                'columns': columns
            })
            logger.info(f"Applied {method} inverse transform to columns: {columns}")
        else:
            logger.warning(f"No scaler found for method '{method}', can't perform inverse transform")

        return result_df