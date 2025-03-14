# data_processor/core/modules/cleaner.py
"""Data cleaning module for handling missing values, outliers, duplicates, and noise"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

from data_processor.core.modules.base import BaseModule
from data_processor.utils.logging_utils import get_logger

logger = get_logger("DataCleaner")

class DataCleaner(BaseModule):
    """Data cleaning module for handling missing values, outliers, duplicates, and noise"""

    def __init__(self, missing_threshold: float = 0.3, extreme_threshold: float = 3.0):
        """
        Initialize the data cleaner

        Args:
            missing_threshold: Threshold for removing features/samples with missing values
            extreme_threshold: Z-score threshold for outlier detection
        """
        super().__init__()
        self.missing_threshold = missing_threshold
        self.extreme_threshold = extreme_threshold
        logger.info(
            f"Initialized DataCleaner: missing_threshold={missing_threshold}, extreme_threshold={extreme_threshold}")

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset

        Args:
            df: Input DataFrame
            strategy: Strategy to handle missing values: 'drop', 'mean', 'median',
                     'mode', 'forward', 'backward', 'knn', 'auto'

        Returns:
            DataFrame with handled missing values
        """
        # Record original missing value stats
        missing_stats = df.isnull().sum()
        missing_percent = missing_stats / len(df)
        logger.info(f"Missing value stats:\n{missing_percent}")

        # Deep copy to avoid modifying original data
        result_df = df.copy()

        # Drop columns with high missing rates
        cols_to_drop = missing_percent[missing_percent > self.missing_threshold].index.tolist()
        if cols_to_drop:
            result_df = result_df.drop(columns=cols_to_drop)
            self.log_operation({
                'operation': 'drop_columns',
                'columns': cols_to_drop,
                'reason': 'high_missing_rate'
            })
            logger.info(f"Dropped high missing rate columns: {cols_to_drop}")

        # Process remaining missing values based on strategy
        if strategy == 'auto':
            # Automatically choose appropriate strategy for each column
            for col in result_df.columns:
                if result_df[col].isnull().sum() == 0:
                    continue

                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Fill numeric columns with median
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'median'
                    })
                elif pd.api.types.is_datetime64_dtype(result_df[col]):
                    # Forward fill datetime columns
                    result_df[col] = result_df[col].fillna(method='ffill')
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'forward_fill'
                    })
                else:
                    # Fill categorical columns with mode
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'mode'
                    })
        elif strategy == 'drop':
            # Drop rows with missing values
            before_rows = len(result_df)
            result_df = result_df.dropna()
            dropped_rows = before_rows - len(result_df)
            self.log_operation({
                'operation': 'drop_rows',
                'count': dropped_rows,
                'reason': 'missing_values'
            })
            logger.info(f"Dropped rows with missing values: {dropped_rows} rows")
        elif strategy == 'mean':
            # Mean imputation (numeric columns only)
            numeric_cols = result_df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].fillna(result_df[col].mean())
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'mean'
                    })
        elif strategy == 'median':
            # Median imputation (numeric columns only)
            numeric_cols = result_df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'median'
                    })
        elif strategy == 'mode':
            # Mode imputation (all columns)
            for col in result_df.columns:
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].fillna(result_df[col].mode()[0])
                    self.log_operation({
                        'operation': 'fill_na',
                        'column': col,
                        'method': 'mode'
                    })
        elif strategy == 'forward':
            # Forward fill
            result_df = result_df.fillna(method='ffill')
            self.log_operation({
                'operation': 'fill_na',
                'method': 'forward_fill'
            })
        elif strategy == 'backward':
            # Backward fill
            result_df = result_df.fillna(method='bfill')
            self.log_operation({
                'operation': 'fill_na',
                'method': 'backward_fill'
            })
        elif strategy == 'knn':
            # KNN imputation (numeric columns only)
            numeric_cols = result_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                numeric_df = result_df[numeric_cols].copy()

                # Use KNN imputer
                imputer = KNNImputer(n_neighbors=5)
                imputed_values = imputer.fit_transform(numeric_df)

                # Insert imputed values back into DataFrame
                imputed_df = pd.DataFrame(imputed_values, columns=numeric_cols, index=result_df.index)
                for col in numeric_cols:
                    result_df[col] = imputed_df[col]

                self.log_operation({
                    'operation': 'fill_na',
                    'columns': numeric_cols.tolist(),
                    'method': 'knn'
                })
                logger.info(f"Used KNN to impute missing values in numeric columns")

        return result_df

    def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers in the dataset

        Args:
            df: Input DataFrame
            method: Detection method: 'zscore', 'iqr', 'isolation_forest'

        Returns:
            Tuple of (processed DataFrame, outliers DataFrame)
        """
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        result_df = df.copy()
        outliers_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

        if method == 'zscore':
            # Detect outliers using Z-score
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_indices = df[col].dropna()[z_scores > self.extreme_threshold].index
                outliers_mask.loc[outlier_indices, col] = True

                # Log operation
                self.log_operation({
                    'operation': 'detect_outliers',
                    'column': col,
                    'method': 'zscore',
                    'threshold': self.extreme_threshold,
                    'outlier_count': len(outlier_indices)
                })
                logger.info(f"Used Z-score to detect outliers in column '{col}': found {len(outlier_indices)}")

        elif method == 'iqr':
            # Detect outliers using IQR
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outliers_mask.loc[outlier_indices, col] = True

                # Log operation
                self.log_operation({
                    'operation': 'detect_outliers',
                    'column': col,
                    'method': 'iqr',
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': len(outlier_indices)
                })
                logger.info(f"Used IQR to detect outliers in column '{col}': found {len(outlier_indices)}")

        elif method == 'isolation_forest':
            # Detect outliers using Isolation Forest
            if len(numeric_cols) > 0:
                # Fill missing values for Isolation Forest algorithm
                temp_df = df[numeric_cols].fillna(df[numeric_cols].median())

                # Train Isolation Forest model
                model = IsolationForest(contamination=0.05, random_state=42)
                preds = model.fit_predict(temp_df)

                # -1 indicates outliers, 1 indicates normal points
                outlier_indices = df.index[preds == -1]

                # Mark all numeric columns as outliers for these rows
                for col in numeric_cols:
                    outliers_mask.loc[outlier_indices, col] = True

                # Log operation
                self.log_operation({
                    'operation': 'detect_outliers',
                    'method': 'isolation_forest',
                    'contamination': 0.05,
                    'outlier_count': len(outlier_indices)
                })
                logger.info(f"Used Isolation Forest to detect outliers: found {len(outlier_indices)}")

        # Create outliers DataFrame
        outliers_df = df.copy()
        outliers_df = outliers_df[outliers_mask.any(axis=1)]

        return result_df, outliers_df

    def handle_outliers(self, df: pd.DataFrame, outliers_df: pd.DataFrame, strategy: str = 'winsorize') -> pd.DataFrame:
        """
        Handle detected outliers

        Args:
            df: Input DataFrame
            outliers_df: DataFrame containing outliers
            strategy: Strategy to handle outliers: 'remove', 'winsorize', 'cap'

        Returns:
            DataFrame with handled outliers
        """
        result_df = df.copy()

        if strategy == 'remove':
            # Remove outlier rows
            outlier_indices = outliers_df.index
            result_df = result_df.drop(index=outlier_indices)

            # Log operation
            self.log_operation({
                'operation': 'handle_outliers',
                'method': 'remove',
                'removed_count': len(outlier_indices)
            })
            logger.info(f"Removed outlier rows: {len(outlier_indices)} rows")

        elif strategy == 'winsorize':
            # Winsorize outliers (set values outside range to boundary values)
            numeric_cols = df.select_dtypes(include=np.number).columns

            for col in numeric_cols:
                if col in outliers_df.columns:
                    # Calculate winsorizing points (using 5% and 95% percentiles)
                    lower_bound = np.percentile(df[col], 5)
                    upper_bound = np.percentile(df[col], 95)

                    # Winsorize outliers
                    result_df[col] = np.where(result_df[col] < lower_bound, lower_bound, result_df[col])
                    result_df[col] = np.where(result_df[col] > upper_bound, upper_bound, result_df[col])

                    # Log operation
                    self.log_operation({
                        'operation': 'handle_outliers',
                        'column': col,
                        'method': 'winsorize',
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
            logger.info(f"Winsorized outliers")

        elif strategy == 'cap':
            # Cap outliers (using IQR method to calculate boundaries)
            numeric_cols = df.select_dtypes(include=np.number).columns

            for col in numeric_cols:
                if col in outliers_df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    # Cap outliers
                    result_df[col] = np.where(result_df[col] < lower_bound, lower_bound, result_df[col])
                    result_df[col] = np.where(result_df[col] > upper_bound, upper_bound, result_df[col])

                    # Log operation
                    self.log_operation({
                        'operation': 'handle_outliers',
                        'column': col,
                        'method': 'cap',
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
            logger.info(f"Capped outliers")

        return result_df

    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate data

        Args:
            df: Input DataFrame
            subset: Subset of columns to consider for duplicates, None means use all columns

        Returns:
            DataFrame with duplicates removed
        """
        # Count duplicate rows
        if subset:
            dup_count = df.duplicated(subset=subset).sum()
        else:
            dup_count = df.duplicated().sum()

        # Remove duplicate rows
        result_df = df.drop_duplicates(subset=subset, keep='first')

        # Log operation
        self.log_operation({
            'operation': 'remove_duplicates',
            'subset': subset,
            'removed_count': dup_count
        })
        logger.info(f"Removed duplicate rows: {dup_count} rows")

        return result_df

    def filter_noise(self, df: pd.DataFrame, columns: List[str], window_size: int = 3) -> pd.DataFrame:
        """
        Filter noise data (using moving average smoothing)

        Args:
            df: Input DataFrame
            columns: Columns to smooth
            window_size: Size of moving window

        Returns:
            Smoothed DataFrame
        """
        result_df = df.copy()

        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Apply moving average smoothing
                result_df[col] = df[col].rolling(window=window_size, center=True).mean()

                # Handle NaN values at window ends
                result_df[col] = result_df[col].fillna(df[col])

                # Log operation
                self.log_operation({
                    'operation': 'filter_noise',
                    'column': col,
                    'method': 'moving_average',
                    'window_size': window_size
                })
                logger.info(f"Applied moving average smoothing to column '{col}' with window size={window_size}")

        return result_df