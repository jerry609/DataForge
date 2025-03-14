# data_processor/core/modules/transformer.py
"""Data transformation module, handling type conversion, encoding and feature extraction"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

from data_processor.core.modules.base import BaseModule
from data_processor.utils.logging_utils import get_logger

logger = get_logger("DataTransformer")

class DataTransformer(BaseModule):
    """Data transformation module, handling type conversion, encoding and feature extraction"""

    def __init__(self):
        """Initialize the data transformer"""
        super().__init__()
        self.encoding_maps = {}
        logger.info("Initialized DataTransformer")

    def convert_types(self, df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
        """
        Type conversion

        Args:
            df: Input DataFrame
            type_map: Mapping from column names to types, e.g. {'age': 'int', 'price': 'float'}

        Returns:
            DataFrame with converted types
        """
        result_df = df.copy()
        conversion_errors = {}

        for col, dtype in type_map.items():
            if col in df.columns:
                try:
                    if dtype == 'int':
                        result_df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')  # Use Int64 to allow NaN
                    elif dtype == 'float':
                        result_df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'bool':
                        result_df[col] = df[col].astype(bool)
                    elif dtype == 'str' or dtype == 'string':
                        result_df[col] = df[col].astype(str)
                    elif dtype == 'datetime':
                        result_df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif dtype == 'category':
                        result_df[col] = df[col].astype('category')
                    else:
                        result_df[col] = df[col].astype(dtype)

                    # Log operation
                    self.log_operation({
                        'operation': 'convert_type',
                        'column': col,
                        'from_type': str(df[col].dtype),
                        'to_type': dtype
                    })
                    logger.info(f"Converted column '{col}' from {df[col].dtype} to {dtype}")

                except Exception as e:
                    conversion_errors[col] = str(e)
                    logger.error(f"Failed to convert column '{col}' type: {e}")

        if conversion_errors:
            logger.warning(f"Type conversion errors: {conversion_errors}")

        return result_df

    def label_encode(self, df: pd.DataFrame, columns: List[str],
                     mapping: Optional[Dict[str, Dict[Any, int]]] = None) -> pd.DataFrame:
        """
        Label encoding

        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            mapping: Custom encoding mapping, format: {column_name: {original_value: encoded_value}}

        Returns:
            Encoded DataFrame
        """
        result_df = df.copy()

        for col in columns:
            if col in df.columns:
                # Use custom mapping or create new mapping
                if mapping and col in mapping:
                    # Use provided mapping
                    col_map = mapping[col]
                    result_df[col] = df[col].map(col_map)

                    # Handle values not in mapping
                    if result_df[col].isna().any() and not df[col].isna().any():
                        max_val = max(col_map.values()) if col_map else 0
                        # Assign new values to unseen categories
                        for val in df[col].unique():
                            if val not in col_map and not pd.isna(val):
                                max_val += 1
                                col_map[val] = max_val
                                result_df.loc[df[col] == val, col] = max_val
                else:
                    # Create new mapping
                    unique_values = df[col].dropna().unique()
                    col_map = {val: i for i, val in enumerate(unique_values)}
                    result_df[col] = df[col].map(col_map)

                # Ensure NaN values remain NaN
                result_df.loc[df[col].isna(), col] = np.nan

                # Save encoding mapping
                self.encoding_maps[col] = {
                    'type': 'label',
                    'mapping': col_map
                }

                # Log operation
                self.log_operation({
                    'operation': 'label_encode',
                    'column': col,
                    'mapping': str(col_map)
                })
                logger.info(f"Applied label encoding to column '{col}', mapping {len(col_map)} different values")

        return result_df

    def one_hot_encode(self, df: pd.DataFrame, columns: List[str],
                       drop_first: bool = False, max_categories: Optional[int] = None) -> pd.DataFrame:
        """
        One-Hot encoding

        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            drop_first: Whether to drop first category to avoid multicollinearity
            max_categories: Maximum number of categories to retain per column, excess will be grouped

        Returns:
            Encoded DataFrame
        """
        result_df = df.copy()

        for col in columns:
            if col in df.columns:
                # Handle high cardinality categorical variables
                if max_categories and df[col].nunique() > max_categories:
                    # Get most common categories
                    top_categories = df[col].value_counts().nlargest(max_categories).index.tolist()

                    # Replace values not in top categories with "Other"
                    result_df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

                    logger.info(f"Column '{col}' has {df[col].nunique()} different values, limited to top {max_categories} categories")

                # Perform One-Hot encoding
                dummies = pd.get_dummies(result_df[col], prefix=col, drop_first=drop_first)

                # Save category mapping
                categories = result_df[col].unique().tolist()
                self.encoding_maps[col] = {
                    'type': 'one_hot',
                    'categories': categories,
                    'drop_first': drop_first
                }

                # Merge into result DataFrame
                result_df = pd.concat([result_df, dummies], axis=1)
                result_df = result_df.drop(columns=[col])

                # Log operation
                self.log_operation({
                    'operation': 'one_hot_encode',
                    'column': col,
                    'categories': categories,
                    'drop_first': drop_first,
                    'max_categories': max_categories
                })
                logger.info(f"Applied One-Hot encoding to column '{col}', generated {len(dummies.columns)} new columns")

        return result_df

    def extract_datetime_features(self, df: pd.DataFrame, date_columns: List[str],
                                  features: List[str] = ['year', 'month', 'day', 'dayofweek', 'hour']) -> pd.DataFrame:
        """
        Extract features from datetime columns

        Args:
            df: Input DataFrame
            date_columns: Datetime columns
            features: Features to extract, can include 'year', 'month', 'day', 'dayofweek',
                    'weekday_name', 'quarter', 'hour', 'minute', 'is_weekend',
                    'is_month_start', 'is_month_end'

        Returns:
            DataFrame with new features
        """
        result_df = df.copy()

        for col in date_columns:
            if col in df.columns:
                # Ensure column is datetime type
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    try:
                        result_df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Failed to convert column '{col}' to datetime: {e}")
                        continue

                # Extract specified datetime features
                prefix = f"{col}_"

                if 'year' in features:
                    result_df[prefix + 'year'] = result_df[col].dt.year

                if 'month' in features:
                    result_df[prefix + 'month'] = result_df[col].dt.month

                if 'day' in features:
                    result_df[prefix + 'day'] = result_df[col].dt.day

                if 'dayofweek' in features:
                    result_df[prefix + 'dayofweek'] = result_df[col].dt.dayofweek

                if 'weekday_name' in features:
                    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                                  3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                    result_df[prefix + 'weekday_name'] = result_df[col].dt.dayofweek.map(weekday_map)

                if 'quarter' in features:
                    result_df[prefix + 'quarter'] = result_df[col].dt.quarter

                if 'hour' in features:
                    result_df[prefix + 'hour'] = result_df[col].dt.hour

                if 'minute' in features:
                    result_df[prefix + 'minute'] = result_df[col].dt.minute

                if 'is_weekend' in features:
                    result_df[prefix + 'is_weekend'] = result_df[col].dt.dayofweek.isin([5, 6]).astype(int)

                if 'is_month_start' in features:
                    result_df[prefix + 'is_month_start'] = result_df[col].dt.is_month_start.astype(int)

                if 'is_month_end' in features:
                    result_df[prefix + 'is_month_end'] = result_df[col].dt.is_month_end.astype(int)

                # Cyclical features using trigonometric transformations
                if 'cyclical_month' in features:
                    # Use sine and cosine to represent cyclical features, so Jan and Dec are close
                    result_df[prefix + 'month_sin'] = np.sin(2 * np.pi * result_df[col].dt.month / 12)
                    result_df[prefix + 'month_cos'] = np.cos(2 * np.pi * result_df[col].dt.month / 12)

                if 'cyclical_hour' in features:
                    result_df[prefix + 'hour_sin'] = np.sin(2 * np.pi * result_df[col].dt.hour / 24)
                    result_df[prefix + 'hour_cos'] = np.cos(2 * np.pi * result_df[col].dt.hour / 24)

                if 'cyclical_dayofweek' in features:
                    result_df[prefix + 'dayofweek_sin'] = np.sin(2 * np.pi * result_df[col].dt.dayofweek / 7)
                    result_df[prefix + 'dayofweek_cos'] = np.cos(2 * np.pi * result_df[col].dt.dayofweek / 7)

                # Log operation
                self.log_operation({
                    'operation': 'extract_datetime_features',
                    'column': col,
                    'features': features
                })
                logger.info(f"Extracted datetime features from column '{col}': {features}")

        return result_df
