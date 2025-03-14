import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_processor.log")
    ]
)
logger = logging.getLogger("DataProcessor")


# Base class for all modules
class BaseModule:
    """Base class for all data processing modules"""

    def __init__(self):
        self.operation_log = []

    def log_operation(self, operation_info: Dict):
        """Add timestamp and log an operation"""
        operation_info['timestamp'] = datetime.now().isoformat()
        self.operation_log.append(operation_info)
        return operation_info


##############################################
#    DATA PREPROCESSING MODULE COMPONENTS    #
##############################################

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
                        result_df[col] = pd.to_numeric(df[col], errors='coerce').astype(
                            'Int64')  # Use Int64 to allow NaN
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

                    logger.info(
                        f"Column '{col}' has {df[col].nunique()} different values, limited to top {max_categories} categories")

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
        从日期时间列提取特征

        Args:
            df: 输入数据框
            date_columns: 日期时间列
            features: 要提取的特征列表，可包括 'year', 'month', 'day', 'dayofweek',
                      'weekday_name', 'quarter', 'hour', 'minute', 'is_weekend',
                      'is_month_start', 'is_month_end'

        Returns:
            包含新特征的数据框
        """
        result_df = df.copy()

        for col in date_columns:
            if col in df.columns:
                # 确保列是日期时间类型
                if not pd.api.types.is_datetime64_dtype(df[col]):
                    try:
                        result_df[col] = pd.to_datetime(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"将列'{col}'转换为日期时间失败: {e}")
                        continue

                # 提取指定的日期特征
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
                    weekday_map = {0: '星期一', 1: '星期二', 2: '星期三',
                                   3: '星期四', 4: '星期五', 5: '星期六', 6: '星期日'}
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

                # 对周期性特征进行三角变换
                if 'cyclical_month' in features:
                    # 使用正弦和余弦表示循环特征，避免1月和12月看起来很远
                    result_df[prefix + 'month_sin'] = np.sin(2 * np.pi * result_df[col].dt.month / 12)
                    result_df[prefix + 'month_cos'] = np.cos(2 * np.pi * result_df[col].dt.month / 12)

                if 'cyclical_hour' in features:
                    result_df[prefix + 'hour_sin'] = np.sin(2 * np.pi * result_df[col].dt.hour / 24)
                    result_df[prefix + 'hour_cos'] = np.cos(2 * np.pi * result_df[col].dt.hour / 24)

                if 'cyclical_dayofweek' in features:
                    result_df[prefix + 'dayofweek_sin'] = np.sin(2 * np.pi * result_df[col].dt.dayofweek / 7)
                    result_df[prefix + 'dayofweek_cos'] = np.cos(2 * np.pi * result_df[col].dt.dayofweek / 7)

                # 记录操作
                self.log_operation({
                    'operation': 'extract_datetime_features',
                    'column': col,
                    'features': features
                })
                logger.info(f"从列'{col}'提取日期时间特征: {features}")

        return result_df


##############################################
#       DATA VALIDATION MODULE COMPONENTS    #
##############################################

class DataValidator(BaseModule):
    """数据验证模块，检查数据完整性、格式和业务规则"""

    def __init__(self):
        """初始化数据验证器"""
        super().__init__()
        self.validation_results = {
            'completeness': {},
            'format': {},
            'business_rules': {}
        }
        logger.info("初始化数据验证器")

    def check_required_fields(self, df: pd.DataFrame, required_fields: List[str]) -> Tuple[bool, Dict]:
        """
        检查必要字段是否存在且非空

        Args:
            df: 输入数据框
            required_fields: 必要字段列表

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'missing_fields': [], 'null_fields': {}}

        # 检查字段是否存在
        for field in required_fields:
            if field not in df.columns:
                result['missing_fields'].append(field)

        # 检查必要字段是否有空值
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    result['null_fields'][field] = null_count

        # 验证结果
        passed = (len(result['missing_fields']) == 0 and len(result['null_fields']) == 0)

        # 记录操作
        self.validation_results['completeness']['required_fields'] = result
        self.log_operation({
            'operation': 'check_required_fields',
            'required_fields': required_fields,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("必要字段验证通过")
        else:
            logger.warning(f"必要字段验证失败: {result}")

        return passed, result

    def check_aggregation(self, df: pd.DataFrame,
                          group_col: str, sum_col: str, total_col: str,
                          tolerance: float = 0.01) -> Tuple[bool, Dict]:
        """
        验证数据聚合（如各子项之和等于总和）

        Args:
            df: 输入数据框
            group_col: 分组列（如类别、部门等）
            sum_col: 求和列（如销售额、数量等）
            total_col: 总计列
            tolerance: 允许的误差范围

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'inconsistent_groups': []}

        # 按组计算总和
        group_sums = df.groupby(group_col)[sum_col].sum().reset_index()

        # 与总计列比较
        for group in group_sums[group_col].unique():
            group_sum = group_sums.loc[group_sums[group_col] == group, sum_col].values[0]
            total_value = df.loc[df[group_col] == group, total_col].values[0]

            # 检查总和是否在容差范围内
            if abs(group_sum - total_value) > tolerance:
                result['inconsistent_groups'].append({
                    'group': group,
                    'calculated_sum': group_sum,
                    'reported_total': total_value,
                    'difference': group_sum - total_value
                })

        # 验证结果
        passed = len(result['inconsistent_groups']) == 0

        # 记录操作
        self.validation_results['completeness']['aggregation'] = result
        self.log_operation({
            'operation': 'check_aggregation',
            'group_col': group_col,
            'sum_col': sum_col,
            'total_col': total_col,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("数据聚合验证通过")
        else:
            logger.warning(f"数据聚合验证失败: 发现{len(result['inconsistent_groups'])}个不一致组")

        return passed, result

    def check_record_count(self, df: pd.DataFrame, expected_count: int,
                           tolerance_percent: float = 5.0) -> Tuple[bool, Dict]:
        """
        验证记录数量是否符合预期

        Args:
            df: 输入数据框
            expected_count: 预期记录数
            tolerance_percent: 允许的误差百分比

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        actual_count = len(df)
        abs_diff = abs(actual_count - expected_count)
        percent_diff = (abs_diff / expected_count) * 100 if expected_count > 0 else float('inf')

        result = {
            'expected_count': expected_count,
            'actual_count': actual_count,
            'absolute_difference': abs_diff,
            'percent_difference': percent_diff,
            'tolerance_percent': tolerance_percent
        }

        # 验证结果
        passed = percent_diff <= tolerance_percent

        # 记录操作
        self.validation_results['completeness']['record_count'] = result
        self.log_operation({
            'operation': 'check_record_count',
            'expected_count': expected_count,
            'actual_count': actual_count,
            'tolerance_percent': tolerance_percent,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info(f"记录数量验证通过: 预期{expected_count}, 实际{actual_count}")
        else:
            logger.warning(f"记录数量验证失败: 预期{expected_count}, 实际{actual_count}, 差异{percent_diff:.2f}%")

        return passed, result

    def check_foreign_key_integrity(self, df: pd.DataFrame, fk_col: str,
                                    reference_df: pd.DataFrame, reference_col: str) -> Tuple[bool, Dict]:
        """
        检查外键完整性

        Args:
            df: 输入数据框（包含外键）
            fk_col: 外键列名
            reference_df: 参考数据框（包含主键）
            reference_col: 参考表中的主键列名

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        # 获取参考表中的唯一值
        reference_values = set(reference_df[reference_col].dropna().unique())

        # 检查外键值是否在参考集中
        fk_values = set(df[fk_col].dropna().unique())
        invalid_values = fk_values - reference_values

        result = {
            'fk_column': fk_col,
            'reference_table': type(reference_df).__name__,
            'reference_column': reference_col,
            'invalid_values': list(invalid_values),
            'invalid_count': len(invalid_values),
            'reference_count': len(reference_values)
        }

        # 验证结果
        passed = len(invalid_values) == 0

        # 记录操作
        self.validation_results['completeness']['foreign_key'] = result
        self.log_operation({
            'operation': 'check_foreign_key_integrity',
            'fk_column': fk_col,
            'reference_column': reference_col,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info(f"外键完整性验证通过: 列'{fk_col}'的所有值都在参考列'{reference_col}'中")
        else:
            logger.warning(f"外键完整性验证失败: 发现{len(invalid_values)}个无效值")

        return passed, result

    def check_data_types(self, df: pd.DataFrame, type_specs: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        检查数据类型是否符合预期

        Args:
            df: 输入数据框
            type_specs: 列名到预期类型的映射

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'type_mismatches': []}

        for col, expected_type in type_specs.items():
            if col in df.columns:
                # 获取当前列的类型
                current_type = str(df[col].dtype)

                # 检查类型是否匹配
                type_match = False

                if expected_type == 'int' and ('int' in current_type or 'Int' in current_type):
                    type_match = True
                elif expected_type == 'float' and ('float' in current_type):
                    type_match = True
                elif expected_type == 'str' and ('object' in current_type or 'string' in current_type):
                    type_match = True
                elif expected_type == 'bool' and ('bool' in current_type):
                    type_match = True
                elif expected_type == 'datetime' and ('datetime' in current_type):
                    type_match = True
                elif expected_type == 'category' and ('category' in current_type):
                    type_match = True
                elif expected_type == current_type:
                    type_match = True

                if not type_match:
                    result['type_mismatches'].append({
                        'column': col,
                        'expected_type': expected_type,
                        'actual_type': current_type
                    })
            else:
                result['type_mismatches'].append({
                    'column': col,
                    'error': 'column_not_found'
                })

        # 验证结果
        passed = len(result['type_mismatches']) == 0

        # 记录操作
        self.validation_results['format']['data_types'] = result
        self.log_operation({
            'operation': 'check_data_types',
            'type_specs': type_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("数据类型验证通过")
        else:
            logger.warning(f"数据类型验证失败: 发现{len(result['type_mismatches'])}个类型不匹配")

        return passed, result

    def check_value_ranges(self, df: pd.DataFrame,
                           range_specs: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
        """
        检查值是否在指定范围内

        Args:
            df: 输入数据框
            range_specs: 列名到范围规范的映射，格式为
                        {列名: {'min': 最小值, 'max': 最大值, 'inclusive': True/False}}

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'out_of_range': {}}

        for col, range_spec in range_specs.items():
            if col in df.columns:
                min_val = range_spec.get('min')
                max_val = range_spec.get('max')
                inclusive = range_spec.get('inclusive', True)

                # 检查最小值
                if min_val is not None:
                    if inclusive:
                        out_of_range = df[df[col] < min_val]
                    else:
                        out_of_range = df[df[col] <= min_val]

                    if not out_of_range.empty:
                        result['out_of_range'][f"{col}_below_min"] = {
                            'count': len(out_of_range),
                            'min_allowed': min_val,
                            'inclusive': inclusive
                        }

                # 检查最大值
                if max_val is not None:
                    if inclusive:
                        out_of_range = df[df[col] > max_val]
                    else:
                        out_of_range = df[df[col] >= max_val]

                    if not out_of_range.empty:
                        result['out_of_range'][f"{col}_above_max"] = {
                            'count': len(out_of_range),
                            'max_allowed': max_val,
                            'inclusive': inclusive
                        }

        # 验证结果
        passed = len(result['out_of_range']) == 0

        # 记录操作
        self.validation_results['format']['value_ranges'] = result
        self.log_operation({
            'operation': 'check_value_ranges',
            'range_specs': range_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("值范围验证通过")
        else:
            logger.warning(f"值范围验证失败: 发现{len(result['out_of_range'])}个超出范围项")

        return passed, result

    def check_regex_patterns(self, df: pd.DataFrame,
                             pattern_specs: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        使用正则表达式验证数据格式

        Args:
            df: 输入数据框
            pattern_specs: 列名到正则表达式模式的映射

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'pattern_mismatches': {}}

        for col, pattern in pattern_specs.items():
            if col in df.columns:
                # 应用正则表达式
                mask = df[col].astype(str).str.match(pattern) == False
                mismatches = df[mask]

                if not mismatches.empty:
                    result['pattern_mismatches'][col] = {
                        'count': len(mismatches),
                        'pattern': pattern,
                        'sample_values': mismatches[col].head(5).tolist()
                    }

        # 验证结果
        passed = len(result['pattern_mismatches']) == 0

        # 记录操作
        self.validation_results['format']['regex_patterns'] = result
        self.log_operation({
            'operation': 'check_regex_patterns',
            'pattern_specs': pattern_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("正则表达式模式验证通过")
        else:
            logger.warning(f"正则表达式模式验证失败: 发现{len(result['pattern_mismatches'])}个不匹配列")

        return passed, result

    def check_structural_consistency(self, df: pd.DataFrame,
                                     schema: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
        """
        检查数据结构一致性（列名、类型、约束等）

        Args:
            df: 输入数据框
            schema: 数据架构定义

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }

        # 检查所需的列是否都存在
        schema_columns = set(schema.keys())
        df_columns = set(df.columns)

        result['missing_columns'] = list(schema_columns - df_columns)
        result['extra_columns'] = list(df_columns - schema_columns)

        # 检查列类型
        for col, col_schema in schema.items():
            if col in df.columns:
                expected_type = col_schema.get('type')
                if expected_type:
                    current_type = str(df[col].dtype)

                    # 检查类型是否匹配
                    type_match = False

                    if expected_type == 'int' and ('int' in current_type or 'Int' in current_type):
                        type_match = True
                    elif expected_type == 'float' and ('float' in current_type):
                        type_match = True
                    elif expected_type == 'str' and ('object' in current_type or 'string' in current_type):
                        type_match = True
                    elif expected_type == 'bool' and ('bool' in current_type):
                        type_match = True
                    elif expected_type == 'datetime' and ('datetime' in current_type):
                        type_match = True
                    elif expected_type == 'category' and ('category' in current_type):
                        type_match = True
                    elif expected_type == current_type:
                        type_match = True

                    if not type_match:
                        result['type_mismatches'].append({
                            'column': col,
                            'expected_type': expected_type,
                            'actual_type': current_type
                        })

        # 验证结果
        passed = (len(result['missing_columns']) == 0 and
                  len(result['type_mismatches']) == 0)

        # 记录操作
        self.validation_results['format']['structural_consistency'] = result
        self.log_operation({
            'operation': 'check_structural_consistency',
            'schema': schema,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("结构一致性验证通过")
        else:
            logger.warning(f"结构一致性验证失败: 缺少{len(result['missing_columns'])}列, "
                           f"类型不匹配{len(result['type_mismatches'])}列")

        return passed, result

    def check_domain_rules(self, df: pd.DataFrame,
                           rule_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> Tuple[bool, Dict]:
        """
        检查领域特定规则

        Args:
            df: 输入数据框
            rule_specs: 规则名称到规则函数的映射，每个函数应返回一个布尔Series

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'rule_violations': {}}

        for rule_name, rule_func in rule_specs.items():
            # 应用规则函数
            try:
                violations = ~rule_func(df)
                violation_count = violations.sum()

                if violation_count > 0:
                    result['rule_violations'][rule_name] = {
                        'count': int(violation_count),
                        'first_few_indices': df[violations].index[:5].tolist()
                    }
            except Exception as e:
                result['rule_violations'][rule_name] = {
                    'error': str(e)
                }
                logger.error(f"应用规则'{rule_name}'时发生错误: {e}")

        # 验证结果
        passed = len(result['rule_violations']) == 0

        # 记录操作
        self.validation_results['business_rules']['domain_rules'] = result
        self.log_operation({
            'operation': 'check_domain_rules',
            'rules': list(rule_specs.keys()),
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("领域规则验证通过")
        else:
            logger.warning(f"领域规则验证失败: 发现{len(result['rule_violations'])}个规则违反")

        return passed, result

    def check_cross_field_relations(self, df: pd.DataFrame,
                                    relation_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> Tuple[
        bool, Dict]:
        """
        检查跨字段关系

        Args:
            df: 输入数据框
            relation_specs: 关系名称到验证函数的映射，每个函数应返回一个布尔Series

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'relation_violations': {}}

        for relation_name, relation_func in relation_specs.items():
            # 应用关系函数
            try:
                violations = ~relation_func(df)
                violation_count = violations.sum()

                if violation_count > 0:
                    result['relation_violations'][relation_name] = {
                        'count': int(violation_count),
                        'first_few_indices': df[violations].index[:5].tolist()
                    }
            except Exception as e:
                result['relation_violations'][relation_name] = {
                    'error': str(e)
                }
                logger.error(f"应用关系验证'{relation_name}'时发生错误: {e}")

        # 验证结果
        passed = len(result['relation_violations']) == 0

        # 记录操作
        self.validation_results['business_rules']['cross_field_relations'] = result
        self.log_operation({
            'operation': 'check_cross_field_relations',
            'relations': list(relation_specs.keys()),
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("跨字段关系验证通过")
        else:
            logger.warning(f"跨字段关系验证失败: 发现{len(result['relation_violations'])}个关系违反")

        return passed, result

    def check_time_series_consistency(self, df: pd.DataFrame, time_col: str,
                                      value_col: str, group_col: Optional[str] = None,
                                      max_change_percent: float = 200.0) -> Tuple[bool, Dict]:
        """
        检查时序数据一致性（异常变化率）

        Args:
            df: 输入数据框
            time_col: 时间列
            value_col: 值列
            group_col: 分组列（如不同产品、地区等）
            max_change_percent: 允许的最大变化百分比

        Returns:
            检查结果元组: (通过验证?, 详细结果)
        """
        result = {'change_rate_violations': []}

        # 确保时间列是日期时间类型
        df_sorted = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors='coerce')

        # 按时间排序
        df_sorted = df_sorted.sort_values(time_col)

        if group_col:
            # 按组计算变化率
            for group_name, group_df in df_sorted.groupby(group_col):
                # 计算百分比变化
                pct_change = group_df[value_col].pct_change() * 100

                # 找出超过阈值的变化
                violations = pct_change.abs() > max_change_percent

                if violations.any():
                    violation_times = group_df[violations][time_col]
                    violation_values = group_df[violations][value_col]
                    previous_values = group_df[violations][value_col].shift(1)
                    change_percents = pct_change[violations]

                    for i, (time, value, prev_val, change_pct) in enumerate(zip(
                            violation_times, violation_values, previous_values, change_percents)):
                        result['change_rate_violations'].append({
                            'group': group_name,
                            'time': time,
                            'value': value,
                            'previous_value': prev_val,
                            'change_percent': change_pct
                        })
        else:
            # 计算整体变化率
            pct_change = df_sorted[value_col].pct_change() * 100

            # 找出超过阈值的变化
            violations = pct_change.abs() > max_change_percent

            if violations.any():
                violation_times = df_sorted[violations][time_col]
                violation_values = df_sorted[violations][value_col]
                previous_values = df_sorted[violations][value_col].shift(1)
                change_percents = pct_change[violations]

                for i, (time, value, prev_val, change_pct) in enumerate(zip(
                        violation_times, violation_values, previous_values, change_percents)):
                    result['change_rate_violations'].append({
                        'time': time,
                        'value': value,
                        'previous_value': prev_val,
                        'change_percent': change_pct
                    })

        # 验证结果
        passed = len(result['change_rate_violations']) == 0

        # 记录操作
        self.validation_results['business_rules']['time_series_consistency'] = result
        self.log_operation({
            'operation': 'check_time_series_consistency',
            'time_col': time_col,
            'value_col': value_col,
            'group_col': group_col,
            'max_change_percent': max_change_percent,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("时序数据一致性验证通过")
        else:
            logger.warning(f"时序数据一致性验证失败: 发现{len(result['change_rate_violations'])}个异常变化")

        return passed, result

        def check_workflow_compliance(self, df: pd.DataFrame,
                                      state_col: str, timestamp_col: str, id_col: str,
                                      valid_transitions: Dict[str, List[str]]) -> Tuple[bool, Dict]:
            """
            检查业务流程合规性（状态转换是否有效）

            Args:
                df: 输入数据框
                state_col: 状态列
                timestamp_col: 时间戳列
                id_col: ID列（标识不同实体，如订单、客户等）
                valid_transitions: 有效的状态转换词典 {当前状态: [有效的下一个状态列表]}

            Returns:
                检查结果元组: (通过验证?, 详细结果)
            """
            result = {'invalid_transitions': []}

            # 确保时间列是日期时间类型
            df_sorted = df.copy()
            if not pd.api.types.is_datetime64_dtype(df_sorted[timestamp_col]):
                df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')

            # 按ID和时间排序
            df_sorted = df_sorted.sort_values([id_col, timestamp_col])

            # 检查每个实体的状态转换
            for entity_id, entity_df in df_sorted.groupby(id_col):
                if len(entity_df) <= 1:
                    continue  # 只有一个状态，没有转换

                # 获取状态序列
                states = entity_df[state_col].tolist()
                times = entity_df[timestamp_col].tolist()

                # 检查转换有效性
                for i in range(1, len(states)):
                    prev_state = states[i - 1]
                    curr_state = states[i]

                    if prev_state in valid_transitions:
                        if curr_state not in valid_transitions[prev_state]:
                            result['invalid_transitions'].append({
                                'entity_id': entity_id,
                                'from_state': prev_state,
                                'to_state': curr_state,
                                'from_time': times[i - 1],
                                'to_time': times[i]
                            })
                    else:
                        # 前一个状态不在转换规则中
                        result['invalid_transitions'].append({
                            'entity_id': entity_id,
                            'from_state': prev_state,
                            'to_state': curr_state,
                            'from_time': times[i - 1],
                            'to_time': times[i],
                            'error': 'undefined_origin_state'
                        })

            # 验证结果
            passed = len(result['invalid_transitions']) == 0

            # 记录操作
            self.validation_results['business_rules']['workflow_compliance'] = result
            self.log_operation({
                'operation': 'check_workflow_compliance',
                'state_col': state_col,
                'timestamp_col': timestamp_col,
                'id_col': id_col,
                'valid_transitions': valid_transitions,
                'result': result,
                'passed': passed
            })

            if passed:
                logger.info("业务流程合规性验证通过")
            else:
                logger.warning(f"业务流程合规性验证失败: 发现{len(result['invalid_transitions'])}个无效转换")

            return passed, result

        def get_validation_summary(self) -> Dict:
            """
            获取验证结果摘要

            Returns:
                包含验证结果的字典
            """
            # 计算每个类别的通过率
            summary = {category: {'total': 0, 'passed': 0} for category in self.validation_results.keys()}

            for category, checks in self.validation_results.items():
                for check_name, check_result in checks.items():
                    summary[category]['total'] += 1
                    if isinstance(check_result, dict) and check_result.get('passed', False):
                        summary[category]['passed'] += 1

            # 计算总体通过率
            total_checks = sum(cat['total'] for cat in summary.values())
            passed_checks = sum(cat['passed'] for cat in summary.values())

            summary['overall'] = {
                'total': total_checks,
                'passed': passed_checks,
                'pass_rate': passed_checks / total_checks if total_checks > 0 else 1.0
            }

            return summary

    ##############################################
    #          INTEGRATED DATA PROCESSOR         #
    ##############################################

    class DataProcessor:
        """
        集成的数据处理器，结合清洗、标准化、转换和验证功能
        """

        def __init__(self):
            """初始化数据处理器"""
            self.cleaner = DataCleaner()
            self.normalizer = DataNormalizer()
            self.transformer = DataTransformer()
            self.validator = DataValidator()
            self.processed_data = None
            self.anomalies = None
            self.processing_steps = []
            self.input_stats = None
            logger.info("初始化数据处理器")

        def process(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
            """
            根据配置处理数据

            Args:
                df: 输入数据框
                config: 处理配置

            Returns:
                处理后的数据框
            """
            result_df = df.copy()
            self.processing_steps = []

            # 记录输入数据统计
            self.input_stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().sum()
            }
            logger.info(f"开始处理数据: {self.input_stats['rows']}行, {self.input_stats['columns']}列")

            # 1. 数据清洗
            if 'cleaning' in config:
                cleaning_config = config['cleaning']

                # 处理缺失值
                if 'missing_values' in cleaning_config:
                    missing_strategy = cleaning_config['missing_values'].get('strategy', 'auto')
                    result_df = self.cleaner.handle_missing_values(result_df, strategy=missing_strategy)
                    self.processing_steps.append({'step': 'handle_missing_values', 'strategy': missing_strategy})

                # 检测异常值
                if 'outliers' in cleaning_config:
                    outlier_method = cleaning_config['outliers'].get('detection_method', 'zscore')
                    result_df, outliers_df = self.cleaner.detect_outliers(result_df, method=outlier_method)
                    self.anomalies = outliers_df

                    # 处理异常值
                    if cleaning_config['outliers'].get('handle', True):
                        handle_strategy = cleaning_config['outliers'].get('handle_strategy', 'winsorize')
                        result_df = self.cleaner.handle_outliers(result_df, outliers_df, strategy=handle_strategy)
                        self.processing_steps.append({'step': 'handle_outliers', 'strategy': handle_strategy})

                # 移除重复数据
                if cleaning_config.get('remove_duplicates', True):
                    subset = cleaning_config.get('duplicate_subset')
                    result_df = self.cleaner.remove_duplicates(result_df, subset=subset)
                    self.processing_steps.append({'step': 'remove_duplicates', 'subset': subset})

                # 平滑噪声数据
                if 'noise_filtering' in cleaning_config:
                    columns = cleaning_config['noise_filtering'].get('columns', [])
                    window_size = cleaning_config['noise_filtering'].get('window_size', 3)
                    if columns:
                        result_df = self.cleaner.filter_noise(result_df, columns, window_size=window_size)
                        self.processing_steps.append({'step': 'filter_noise', 'window_size': window_size})

            # 2. 数据标准化/归一化
            if 'normalization' in config:
                norm_config = config['normalization']

                # Z-Score标准化
                if norm_config.get('z_score', False):
                    columns = norm_config.get('z_score_columns')
                    result_df = self.normalizer.z_score_normalize(result_df, columns=columns)
                    self.processing_steps.append({'step': 'z_score_normalize', 'columns': columns})

                # Min-Max归一化
                if norm_config.get('min_max', False):
                    columns = norm_config.get('min_max_columns')
                    feature_range = norm_config.get('min_max_range', (0, 1))
                    result_df = self.normalizer.min_max_normalize(result_df, columns=columns,
                                                                  feature_range=feature_range)
                    self.processing_steps.append({'step': 'min_max_normalize', 'feature_range': feature_range})

                # Robust缩放
                if norm_config.get('robust', False):
                    columns = norm_config.get('robust_columns')
                    result_df = self.normalizer.robust_scale(result_df, columns=columns)
                    self.processing_steps.append({'step': 'robust_scale'})

                # 对数变换
                if norm_config.get('log_transform', False):
                    columns = norm_config.get('log_columns')
                    base = norm_config.get('log_base', np.e)
                    offset = norm_config.get('log_offset', 1.0)
                    result_df = self.normalizer.log_transform(result_df, columns=columns, base=base, offset=offset)
                    self.processing_steps.append({'step': 'log_transform', 'base': base, 'offset': offset})

            # 3. 数据转换
            if 'transformation' in config:
                trans_config = config['transformation']

                # 类型转换
                if 'type_conversion' in trans_config:
                    type_map = trans_config['type_conversion']
                    result_df = self.transformer.convert_types(result_df, type_map)
                    self.processing_steps.append({'step': 'convert_types'})

                # One-Hot编码
                if 'one_hot_encoding' in trans_config:
                    columns = trans_config['one_hot_encoding'].get('columns', [])
                    drop_first = trans_config['one_hot_encoding'].get('drop_first', False)
                    max_categories = trans_config['one_hot_encoding'].get('max_categories')
                    result_df = self.transformer.one_hot_encode(result_df, columns, drop_first=drop_first,
                                                                max_categories=max_categories)
                    self.processing_steps.append({'step': 'one_hot_encode'})

                # 标签编码
                if 'label_encoding' in trans_config:
                    columns = trans_config['label_encoding'].get('columns', [])
                    mapping = trans_config['label_encoding'].get('mapping')
                    result_df = self.transformer.label_encode(result_df, columns, mapping=mapping)
                    self.processing_steps.append({'step': 'label_encode'})

                # 时间特征提取
                if 'datetime_features' in trans_config:
                    date_columns = trans_config['datetime_features'].get('columns', [])
                    features = trans_config['datetime_features'].get('features', [])
                    result_df = self.transformer.extract_datetime_features(result_df, date_columns, features=features)
                    self.processing_steps.append({'step': 'extract_datetime_features'})
            import pandas as pd
            import numpy as np
            import re
            import json
            import logging
            from typing import Dict, List, Union, Tuple, Optional, Any, Callable
            from datetime import datetime
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            from sklearn.impute import KNNImputer
            from sklearn.ensemble import IsolationForest
            from scipy import stats

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler("data_processor.log")
                ]
            )
            logger = logging.getLogger("DataProcessor")

            # Base class for all modules
            class BaseModule:
                """Base class for all data processing modules"""

                def __init__(self):
                    self.operation_log = []

                def log_operation(self, operation_info: Dict):
                    """Add timestamp and log an operation"""
                    operation_info['timestamp'] = datetime.now().isoformat()
                    self.operation_log.append(operation_info)
                    return operation_info

            ##############################################
            #    DATA PREPROCESSING MODULE COMPONENTS    #
            ##############################################

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

                def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore') -> Tuple[
                    pd.DataFrame, pd.DataFrame]:
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
                            logger.info(
                                f"Used Z-score to detect outliers in column '{col}': found {len(outlier_indices)}")

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

                def handle_outliers(self, df: pd.DataFrame, outliers_df: pd.DataFrame,
                                    strategy: str = 'winsorize') -> pd.DataFrame:
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
                            logger.info(
                                f"Applied moving average smoothing to column '{col}' with window size={window_size}")

                    return result_df

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
                                    result_df[col] = pd.to_numeric(df[col], errors='coerce').astype(
                                        'Int64')  # Use Int64 to allow NaN
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
                            logger.info(
                                f"Applied label encoding to column '{col}', mapping {len(col_map)} different values")

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

                                logger.info(
                                    f"Column '{col}' has {df[col].nunique()} different values, limited to top {max_categories} categories")

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
                            logger.info(
                                f"Applied One-Hot encoding to column '{col}', generated {len(dummies.columns)} new columns")

                    return result_df

                def extract_datetime_features(self, df: pd.DataFrame, date_columns: List[str],
                                              features: List[str] = ['year', 'month', 'day', 'dayofweek',
                                                                     'hour']) -> pd.DataFrame:
                    """
                    从日期时间列提取特征

                    Args:
                        df: 输入数据框
                        date_columns: 日期时间列
                        features: 要提取的特征列表，可包括 'year', 'month', 'day', 'dayofweek',
                                  'weekday_name', 'quarter', 'hour', 'minute', 'is_weekend',
                                  'is_month_start', 'is_month_end'

                    Returns:
                        包含新特征的数据框
                    """
                    result_df = df.copy()

                    for col in date_columns:
                        if col in df.columns:
                            # 确保列是日期时间类型
                            if not pd.api.types.is_datetime64_dtype(df[col]):
                                try:
                                    result_df[col] = pd.to_datetime(df[col], errors='coerce')
                                except Exception as e:
                                    logger.error(f"将列'{col}'转换为日期时间失败: {e}")
                                    continue

                            # 提取指定的日期特征
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
                                weekday_map = {0: '星期一', 1: '星期二', 2: '星期三',
                                               3: '星期四', 4: '星期五', 5: '星期六', 6: '星期日'}
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

                            # 对周期性特征进行三角变换
                            if 'cyclical_month' in features:
                                # 使用正弦和余弦表示循环特征，避免1月和12月看起来很远
                                result_df[prefix + 'month_sin'] = np.sin(2 * np.pi * result_df[col].dt.month / 12)
                                result_df[prefix + 'month_cos'] = np.cos(2 * np.pi * result_df[col].dt.month / 12)

                            if 'cyclical_hour' in features:
                                result_df[prefix + 'hour_sin'] = np.sin(2 * np.pi * result_df[col].dt.hour / 24)
                                result_df[prefix + 'hour_cos'] = np.cos(2 * np.pi * result_df[col].dt.hour / 24)

                            if 'cyclical_dayofweek' in features:
                                result_df[prefix + 'dayofweek_sin'] = np.sin(
                                    2 * np.pi * result_df[col].dt.dayofweek / 7)
                                result_df[prefix + 'dayofweek_cos'] = np.cos(
                                    2 * np.pi * result_df[col].dt.dayofweek / 7)

                            # 记录操作
                            self.log_operation({
                                'operation': 'extract_datetime_features',
                                'column': col,
                                'features': features
                            })
                            logger.info(f"从列'{col}'提取日期时间特征: {features}")

                    return result_df

            ##############################################
            #       DATA VALIDATION MODULE COMPONENTS    #
            ##############################################

            class DataValidator(BaseModule):
                """数据验证模块，检查数据完整性、格式和业务规则"""

                def __init__(self):
                    """初始化数据验证器"""
                    super().__init__()
                    self.validation_results = {
                        'completeness': {},
                        'format': {},
                        'business_rules': {}
                    }
                    logger.info("初始化数据验证器")

                def check_required_fields(self, df: pd.DataFrame, required_fields: List[str]) -> Tuple[bool, Dict]:
                    """
                    检查必要字段是否存在且非空

                    Args:
                        df: 输入数据框
                        required_fields: 必要字段列表

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'missing_fields': [], 'null_fields': {}}

                    # 检查字段是否存在
                    for field in required_fields:
                        if field not in df.columns:
                            result['missing_fields'].append(field)

                    # 检查必要字段是否有空值
                    for field in required_fields:
                        if field in df.columns:
                            null_count = df[field].isnull().sum()
                            if null_count > 0:
                                result['null_fields'][field] = null_count

                    # 验证结果
                    passed = (len(result['missing_fields']) == 0 and len(result['null_fields']) == 0)

                    # 记录操作
                    self.validation_results['completeness']['required_fields'] = result
                    self.log_operation({
                        'operation': 'check_required_fields',
                        'required_fields': required_fields,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("必要字段验证通过")
                    else:
                        logger.warning(f"必要字段验证失败: {result}")

                    return passed, result

                def check_aggregation(self, df: pd.DataFrame,
                                      group_col: str, sum_col: str, total_col: str,
                                      tolerance: float = 0.01) -> Tuple[bool, Dict]:
                    """
                    验证数据聚合（如各子项之和等于总和）

                    Args:
                        df: 输入数据框
                        group_col: 分组列（如类别、部门等）
                        sum_col: 求和列（如销售额、数量等）
                        total_col: 总计列
                        tolerance: 允许的误差范围

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'inconsistent_groups': []}

                    # 按组计算总和
                    group_sums = df.groupby(group_col)[sum_col].sum().reset_index()

                    # 与总计列比较
                    for group in group_sums[group_col].unique():
                        group_sum = group_sums.loc[group_sums[group_col] == group, sum_col].values[0]
                        total_value = df.loc[df[group_col] == group, total_col].values[0]

                        # 检查总和是否在容差范围内
                        if abs(group_sum - total_value) > tolerance:
                            result['inconsistent_groups'].append({
                                'group': group,
                                'calculated_sum': group_sum,
                                'reported_total': total_value,
                                'difference': group_sum - total_value
                            })

                    # 验证结果
                    passed = len(result['inconsistent_groups']) == 0

                    # 记录操作
                    self.validation_results['completeness']['aggregation'] = result
                    self.log_operation({
                        'operation': 'check_aggregation',
                        'group_col': group_col,
                        'sum_col': sum_col,
                        'total_col': total_col,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("数据聚合验证通过")
                    else:
                        logger.warning(f"数据聚合验证失败: 发现{len(result['inconsistent_groups'])}个不一致组")

                    return passed, result

                def check_record_count(self, df: pd.DataFrame, expected_count: int,
                                       tolerance_percent: float = 5.0) -> Tuple[bool, Dict]:
                    """
                    验证记录数量是否符合预期

                    Args:
                        df: 输入数据框
                        expected_count: 预期记录数
                        tolerance_percent: 允许的误差百分比

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    actual_count = len(df)
                    abs_diff = abs(actual_count - expected_count)
                    percent_diff = (abs_diff / expected_count) * 100 if expected_count > 0 else float('inf')

                    result = {
                        'expected_count': expected_count,
                        'actual_count': actual_count,
                        'absolute_difference': abs_diff,
                        'percent_difference': percent_diff,
                        'tolerance_percent': tolerance_percent
                    }

                    # 验证结果
                    passed = percent_diff <= tolerance_percent

                    # 记录操作
                    self.validation_results['completeness']['record_count'] = result
                    self.log_operation({
                        'operation': 'check_record_count',
                        'expected_count': expected_count,
                        'actual_count': actual_count,
                        'tolerance_percent': tolerance_percent,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info(f"记录数量验证通过: 预期{expected_count}, 实际{actual_count}")
                    else:
                        logger.warning(
                            f"记录数量验证失败: 预期{expected_count}, 实际{actual_count}, 差异{percent_diff:.2f}%")

                    return passed, result

                def check_foreign_key_integrity(self, df: pd.DataFrame, fk_col: str,
                                                reference_df: pd.DataFrame, reference_col: str) -> Tuple[bool, Dict]:
                    """
                    检查外键完整性

                    Args:
                        df: 输入数据框（包含外键）
                        fk_col: 外键列名
                        reference_df: 参考数据框（包含主键）
                        reference_col: 参考表中的主键列名

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    # 获取参考表中的唯一值
                    reference_values = set(reference_df[reference_col].dropna().unique())

                    # 检查外键值是否在参考集中
                    fk_values = set(df[fk_col].dropna().unique())
                    invalid_values = fk_values - reference_values

                    result = {
                        'fk_column': fk_col,
                        'reference_table': type(reference_df).__name__,
                        'reference_column': reference_col,
                        'invalid_values': list(invalid_values),
                        'invalid_count': len(invalid_values),
                        'reference_count': len(reference_values)
                    }

                    # 验证结果
                    passed = len(invalid_values) == 0

                    # 记录操作
                    self.validation_results['completeness']['foreign_key'] = result
                    self.log_operation({
                        'operation': 'check_foreign_key_integrity',
                        'fk_column': fk_col,
                        'reference_column': reference_col,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info(f"外键完整性验证通过: 列'{fk_col}'的所有值都在参考列'{reference_col}'中")
                    else:
                        logger.warning(f"外键完整性验证失败: 发现{len(invalid_values)}个无效值")

                    return passed, result

                def check_data_types(self, df: pd.DataFrame, type_specs: Dict[str, str]) -> Tuple[bool, Dict]:
                    """
                    检查数据类型是否符合预期

                    Args:
                        df: 输入数据框
                        type_specs: 列名到预期类型的映射

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'type_mismatches': []}

                    for col, expected_type in type_specs.items():
                        if col in df.columns:
                            # 获取当前列的类型
                            current_type = str(df[col].dtype)

                            # 检查类型是否匹配
                            type_match = False

                            if expected_type == 'int' and ('int' in current_type or 'Int' in current_type):
                                type_match = True
                            elif expected_type == 'float' and ('float' in current_type):
                                type_match = True
                            elif expected_type == 'str' and ('object' in current_type or 'string' in current_type):
                                type_match = True
                            elif expected_type == 'bool' and ('bool' in current_type):
                                type_match = True
                            elif expected_type == 'datetime' and ('datetime' in current_type):
                                type_match = True
                            elif expected_type == 'category' and ('category' in current_type):
                                type_match = True
                            elif expected_type == current_type:
                                type_match = True

                            if not type_match:
                                result['type_mismatches'].append({
                                    'column': col,
                                    'expected_type': expected_type,
                                    'actual_type': current_type
                                })
                        else:
                            result['type_mismatches'].append({
                                'column': col,
                                'error': 'column_not_found'
                            })

                    # 验证结果
                    passed = len(result['type_mismatches']) == 0

                    # 记录操作
                    self.validation_results['format']['data_types'] = result
                    self.log_operation({
                        'operation': 'check_data_types',
                        'type_specs': type_specs,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("数据类型验证通过")
                    else:
                        logger.warning(f"数据类型验证失败: 发现{len(result['type_mismatches'])}个类型不匹配")

                    return passed, result

                def check_value_ranges(self, df: pd.DataFrame,
                                       range_specs: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
                    """
                    检查值是否在指定范围内

                    Args:
                        df: 输入数据框
                        range_specs: 列名到范围规范的映射，格式为
                                    {列名: {'min': 最小值, 'max': 最大值, 'inclusive': True/False}}

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'out_of_range': {}}

                    for col, range_spec in range_specs.items():
                        if col in df.columns:
                            min_val = range_spec.get('min')
                            max_val = range_spec.get('max')
                            inclusive = range_spec.get('inclusive', True)

                            # 检查最小值
                            if min_val is not None:
                                if inclusive:
                                    out_of_range = df[df[col] < min_val]
                                else:
                                    out_of_range = df[df[col] <= min_val]

                                if not out_of_range.empty:
                                    result['out_of_range'][f"{col}_below_min"] = {
                                        'count': len(out_of_range),
                                        'min_allowed': min_val,
                                        'inclusive': inclusive
                                    }

                            # 检查最大值
                            if max_val is not None:
                                if inclusive:
                                    out_of_range = df[df[col] > max_val]
                                else:
                                    out_of_range = df[df[col] >= max_val]

                                if not out_of_range.empty:
                                    result['out_of_range'][f"{col}_above_max"] = {
                                        'count': len(out_of_range),
                                        'max_allowed': max_val,
                                        'inclusive': inclusive
                                    }

                    # 验证结果
                    passed = len(result['out_of_range']) == 0

                    # 记录操作
                    self.validation_results['format']['value_ranges'] = result
                    self.log_operation({
                        'operation': 'check_value_ranges',
                        'range_specs': range_specs,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("值范围验证通过")
                    else:
                        logger.warning(f"值范围验证失败: 发现{len(result['out_of_range'])}个超出范围项")

                    return passed, result

                def check_regex_patterns(self, df: pd.DataFrame,
                                         pattern_specs: Dict[str, str]) -> Tuple[bool, Dict]:
                    """
                    使用正则表达式验证数据格式

                    Args:
                        df: 输入数据框
                        pattern_specs: 列名到正则表达式模式的映射

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'pattern_mismatches': {}}

                    for col, pattern in pattern_specs.items():
                        if col in df.columns:
                            # 应用正则表达式
                            mask = df[col].astype(str).str.match(pattern) == False
                            mismatches = df[mask]

                            if not mismatches.empty:
                                result['pattern_mismatches'][col] = {
                                    'count': len(mismatches),
                                    'pattern': pattern,
                                    'sample_values': mismatches[col].head(5).tolist()
                                }

                    # 验证结果
                    passed = len(result['pattern_mismatches']) == 0

                    # 记录操作
                    self.validation_results['format']['regex_patterns'] = result
                    self.log_operation({
                        'operation': 'check_regex_patterns',
                        'pattern_specs': pattern_specs,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("正则表达式模式验证通过")
                    else:
                        logger.warning(f"正则表达式模式验证失败: 发现{len(result['pattern_mismatches'])}个不匹配列")

                    return passed, result

                def check_structural_consistency(self, df: pd.DataFrame,
                                                 schema: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
                    """
                    检查数据结构一致性（列名、类型、约束等）

                    Args:
                        df: 输入数据框
                        schema: 数据架构定义

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {
                        'missing_columns': [],
                        'extra_columns': [],
                        'type_mismatches': []
                    }

                    # 检查所需的列是否都存在
                    schema_columns = set(schema.keys())
                    df_columns = set(df.columns)

                    result['missing_columns'] = list(schema_columns - df_columns)
                    result['extra_columns'] = list(df_columns - schema_columns)

                    # 检查列类型
                    for col, col_schema in schema.items():
                        if col in df.columns:
                            expected_type = col_schema.get('type')
                            if expected_type:
                                current_type = str(df[col].dtype)

                                # 检查类型是否匹配
                                type_match = False

                                if expected_type == 'int' and ('int' in current_type or 'Int' in current_type):
                                    type_match = True
                                elif expected_type == 'float' and ('float' in current_type):
                                    type_match = True
                                elif expected_type == 'str' and ('object' in current_type or 'string' in current_type):
                                    type_match = True
                                elif expected_type == 'bool' and ('bool' in current_type):
                                    type_match = True
                                elif expected_type == 'datetime' and ('datetime' in current_type):
                                    type_match = True
                                elif expected_type == 'category' and ('category' in current_type):
                                    type_match = True
                                elif expected_type == current_type:
                                    type_match = True

                                if not type_match:
                                    result['type_mismatches'].append({
                                        'column': col,
                                        'expected_type': expected_type,
                                        'actual_type': current_type
                                    })

                    # 验证结果
                    passed = (len(result['missing_columns']) == 0 and
                              len(result['type_mismatches']) == 0)

                    # 记录操作
                    self.validation_results['format']['structural_consistency'] = result
                    self.log_operation({
                        'operation': 'check_structural_consistency',
                        'schema': schema,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("结构一致性验证通过")
                    else:
                        logger.warning(f"结构一致性验证失败: 缺少{len(result['missing_columns'])}列, "
                                       f"类型不匹配{len(result['type_mismatches'])}列")

                    return passed, result

                def check_domain_rules(self, df: pd.DataFrame,
                                       rule_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> Tuple[bool, Dict]:
                    """
                    检查领域特定规则

                    Args:
                        df: 输入数据框
                        rule_specs: 规则名称到规则函数的映射，每个函数应返回一个布尔Series

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'rule_violations': {}}

                    for rule_name, rule_func in rule_specs.items():
                        # 应用规则函数
                        try:
                            violations = ~rule_func(df)
                            violation_count = violations.sum()

                            if violation_count > 0:
                                result['rule_violations'][rule_name] = {
                                    'count': int(violation_count),
                                    'first_few_indices': df[violations].index[:5].tolist()
                                }
                        except Exception as e:
                            result['rule_violations'][rule_name] = {
                                'error': str(e)
                            }
                            logger.error(f"应用规则'{rule_name}'时发生错误: {e}")

                    # 验证结果
                    passed = len(result['rule_violations']) == 0

                    # 记录操作
                    self.validation_results['business_rules']['domain_rules'] = result
                    self.log_operation({
                        'operation': 'check_domain_rules',
                        'rules': list(rule_specs.keys()),
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("领域规则验证通过")
                    else:
                        logger.warning(f"领域规则验证失败: 发现{len(result['rule_violations'])}个规则违反")

                    return passed, result

                def check_cross_field_relations(self, df: pd.DataFrame,
                                                relation_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> \
                Tuple[
                    bool, Dict]:
                    """
                    检查跨字段关系

                    Args:
                        df: 输入数据框
                        relation_specs: 关系名称到验证函数的映射，每个函数应返回一个布尔Series

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'relation_violations': {}}

                    for relation_name, relation_func in relation_specs.items():
                        # 应用关系函数
                        try:
                            violations = ~relation_func(df)
                            violation_count = violations.sum()

                            if violation_count > 0:
                                result['relation_violations'][relation_name] = {
                                    'count': int(violation_count),
                                    'first_few_indices': df[violations].index[:5].tolist()
                                }
                        except Exception as e:
                            result['relation_violations'][relation_name] = {
                                'error': str(e)
                            }
                            logger.error(f"应用关系验证'{relation_name}'时发生错误: {e}")

                    # 验证结果
                    passed = len(result['relation_violations']) == 0

                    # 记录操作
                    self.validation_results['business_rules']['cross_field_relations'] = result
                    self.log_operation({
                        'operation': 'check_cross_field_relations',
                        'relations': list(relation_specs.keys()),
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("跨字段关系验证通过")
                    else:
                        logger.warning(f"跨字段关系验证失败: 发现{len(result['relation_violations'])}个关系违反")

                    return passed, result

                def check_time_series_consistency(self, df: pd.DataFrame, time_col: str,
                                                  value_col: str, group_col: Optional[str] = None,
                                                  max_change_percent: float = 200.0) -> Tuple[bool, Dict]:
                    """
                    检查时序数据一致性（异常变化率）

                    Args:
                        df: 输入数据框
                        time_col: 时间列
                        value_col: 值列
                        group_col: 分组列（如不同产品、地区等）
                        max_change_percent: 允许的最大变化百分比

                    Returns:
                        检查结果元组: (通过验证?, 详细结果)
                    """
                    result = {'change_rate_violations': []}

                    # 确保时间列是日期时间类型
                    df_sorted = df.copy()
                    if not pd.api.types.is_datetime64_dtype(df_sorted[time_col]):
                        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors='coerce')

                    # 按时间排序
                    df_sorted = df_sorted.sort_values(time_col)

                    if group_col:
                        # 按组计算变化率
                        for group_name, group_df in df_sorted.groupby(group_col):
                            # 计算百分比变化
                            pct_change = group_df[value_col].pct_change() * 100

                            # 找出超过阈值的变化
                            violations = pct_change.abs() > max_change_percent

                            if violations.any():
                                violation_times = group_df[violations][time_col]
                                violation_values = group_df[violations][value_col]
                                previous_values = group_df[violations][value_col].shift(1)
                                change_percents = pct_change[violations]

                                for i, (time, value, prev_val, change_pct) in enumerate(zip(
                                        violation_times, violation_values, previous_values, change_percents)):
                                    result['change_rate_violations'].append({
                                        'group': group_name,
                                        'time': time,
                                        'value': value,
                                        'previous_value': prev_val,
                                        'change_percent': change_pct
                                    })
                    else:
                        # 计算整体变化率
                        pct_change = df_sorted[value_col].pct_change() * 100

                        # 找出超过阈值的变化
                        violations = pct_change.abs() > max_change_percent

                        if violations.any():
                            violation_times = df_sorted[violations][time_col]
                            violation_values = df_sorted[violations][value_col]
                            previous_values = df_sorted[violations][value_col].shift(1)
                            change_percents = pct_change[violations]

                            for i, (time, value, prev_val, change_pct) in enumerate(zip(
                                    violation_times, violation_values, previous_values, change_percents)):
                                result['change_rate_violations'].append({
                                    'time': time,
                                    'value': value,
                                    'previous_value': prev_val,
                                    'change_percent': change_pct
                                })

                    # 验证结果
                    passed = len(result['change_rate_violations']) == 0

                    # 记录操作
                    self.validation_results['business_rules']['time_series_consistency'] = result
                    self.log_operation({
                        'operation': 'check_time_series_consistency',
                        'time_col': time_col,
                        'value_col': value_col,
                        'group_col': group_col,
                        'max_change_percent': max_change_percent,
                        'result': result,
                        'passed': passed
                    })

                    if passed:
                        logger.info("时序数据一致性验证通过")
                    else:
                        logger.warning(f"时序数据一致性验证失败: 发现{len(result['change_rate_violations'])}个异常变化")

                    return passed, result

                    def check_workflow_compliance(self, df: pd.DataFrame,
                                                  state_col: str, timestamp_col: str, id_col: str,
                                                  valid_transitions: Dict[str, List[str]]) -> Tuple[bool, Dict]:
                        """
                        检查业务流程合规性（状态转换是否有效）

                        Args:
                            df: 输入数据框
                            state_col: 状态列
                            timestamp_col: 时间戳列
                            id_col: ID列（标识不同实体，如订单、客户等）
                            valid_transitions: 有效的状态转换词典 {当前状态: [有效的下一个状态列表]}

                        Returns:
                            检查结果元组: (通过验证?, 详细结果)
                        """
                        result = {'invalid_transitions': []}

                        # 确保时间列是日期时间类型
                        df_sorted = df.copy()
                        if not pd.api.types.is_datetime64_dtype(df_sorted[timestamp_col]):
                            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')

                        # 按ID和时间排序
                        df_sorted = df_sorted.sort_values([id_col, timestamp_col])

                        # 检查每个实体的状态转换
                        for entity_id, entity_df in df_sorted.groupby(id_col):
                            if len(entity_df) <= 1:
                                continue  # 只有一个状态，没有转换

                            # 获取状态序列
                            states = entity_df[state_col].tolist()
                            times = entity_df[timestamp_col].tolist()

                            # 检查转换有效性
                            for i in range(1, len(states)):
                                prev_state = states[i - 1]
                                curr_state = states[i]

                                if prev_state in valid_transitions:
                                    if curr_state not in valid_transitions[prev_state]:
                                        result['invalid_transitions'].append({
                                            'entity_id': entity_id,
                                            'from_state': prev_state,
                                            'to_state': curr_state,
                                            'from_time': times[i - 1],
                                            'to_time': times[i]
                                        })
                                else:
                                    # 前一个状态不在转换规则中
                                    result['invalid_transitions'].append({
                                        'entity_id': entity_id,
                                        'from_state': prev_state,
                                        'to_state': curr_state,
                                        'from_time': times[i - 1],
                                        'to_time': times[i],
                                        'error': 'undefined_origin_state'
                                    })

                        # 验证结果
                        passed = len(result['invalid_transitions']) == 0

                        # 记录操作
                        self.validation_results['business_rules']['workflow_compliance'] = result
                        self.log_operation({
                            'operation': 'check_workflow_compliance',
                            'state_col': state_col,
                            'timestamp_col': timestamp_col,
                            'id_col': id_col,
                            'valid_transitions': valid_transitions,
                            'result': result,
                            'passed': passed
                        })

                        if passed:
                            logger.info("业务流程合规性验证通过")
                        else:
                            logger.warning(
                                f"业务流程合规性验证失败: 发现{len(result['invalid_transitions'])}个无效转换")

                        return passed, result

                    def get_validation_summary(self) -> Dict:
                        """
                        获取验证结果摘要

                        Returns:
                            包含验证结果的字典
                        """
                        # 计算每个类别的通过率
                        summary = {category: {'total': 0, 'passed': 0} for category in self.validation_results.keys()}

                        for category, checks in self.validation_results.items():
                            for check_name, check_result in checks.items():
                                summary[category]['total'] += 1
                                if isinstance(check_result, dict) and check_result.get('passed', False):
                                    summary[category]['passed'] += 1

                        # 计算总体通过率
                        total_checks = sum(cat['total'] for cat in summary.values())
                        passed_checks = sum(cat['passed'] for cat in summary.values())

                        summary['overall'] = {
                            'total': total_checks,
                            'passed': passed_checks,
                            'pass_rate': passed_checks / total_checks if total_checks > 0 else 1.0
                        }

                        return summary

                ##############################################
                #          INTEGRATED DATA PROCESSOR         #
                ##############################################

                class DataProcessor:
                    """
                    集成的数据处理器，结合清洗、标准化、转换和验证功能
                    """

                    def __init__(self):
                        """初始化数据处理器"""
                        self.cleaner = DataCleaner()
                        self.normalizer = DataNormalizer()
                        self.transformer = DataTransformer()
                        self.validator = DataValidator()
                        self.processed_data = None
                        self.anomalies = None
                        self.processing_steps = []
                        self.input_stats = None
                        logger.info("初始化数据处理器")

                    def process(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
                        """
                        根据配置处理数据

                        Args:
                            df: 输入数据框
                            config: 处理配置

                        Returns:
                            处理后的数据框
                        """
                        result_df = df.copy()
                        self.processing_steps = []

                        # 记录输入数据统计
                        self.input_stats = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'memory_usage': df.memory_usage(deep=True).sum(),
                            'missing_values': df.isnull().sum().sum()
                        }
                        logger.info(f"开始处理数据: {self.input_stats['rows']}行, {self.input_stats['columns']}列")

                        # 1. 数据清洗
                        if 'cleaning' in config:
                            cleaning_config = config['cleaning']

                            # 处理缺失值
                            if 'missing_values' in cleaning_config:
                                missing_strategy = cleaning_config['missing_values'].get('strategy', 'auto')
                                result_df = self.cleaner.handle_missing_values(result_df, strategy=missing_strategy)
                                self.processing_steps.append(
                                    {'step': 'handle_missing_values', 'strategy': missing_strategy})

                            # 检测异常值
                            if 'outliers' in cleaning_config:
                                outlier_method = cleaning_config['outliers'].get('detection_method', 'zscore')
                                result_df, outliers_df = self.cleaner.detect_outliers(result_df, method=outlier_method)
                                self.anomalies = outliers_df

                                # 处理异常值
                                if cleaning_config['outliers'].get('handle', True):
                                    handle_strategy = cleaning_config['outliers'].get('handle_strategy', 'winsorize')
                                    result_df = self.cleaner.handle_outliers(result_df, outliers_df,
                                                                             strategy=handle_strategy)
                                    self.processing_steps.append(
                                        {'step': 'handle_outliers', 'strategy': handle_strategy})

                            # 移除重复数据
                            if cleaning_config.get('remove_duplicates', True):
                                subset = cleaning_config.get('duplicate_subset')
                                result_df = self.cleaner.remove_duplicates(result_df, subset=subset)
                                self.processing_steps.append({'step': 'remove_duplicates', 'subset': subset})

                            # 平滑噪声数据
                            if 'noise_filtering' in cleaning_config:
                                columns = cleaning_config['noise_filtering'].get('columns', [])
                                window_size = cleaning_config['noise_filtering'].get('window_size', 3)
                                if columns:
                                    result_df = self.cleaner.filter_noise(result_df, columns, window_size=window_size)
                                    self.processing_steps.append({'step': 'filter_noise', 'window_size': window_size})

                        # 2. 数据标准化/归一化
                        if 'normalization' in config:
                            norm_config = config['normalization']

                            # Z-Score标准化
                            if norm_config.get('z_score', False):
                                columns = norm_config.get('z_score_columns')
                                result_df = self.normalizer.z_score_normalize(result_df, columns=columns)
                                self.processing_steps.append({'step': 'z_score_normalize', 'columns': columns})

                            # Min-Max归一化
                            if norm_config.get('min_max', False):
                                columns = norm_config.get('min_max_columns')
                                feature_range = norm_config.get('min_max_range', (0, 1))
                                result_df = self.normalizer.min_max_normalize(result_df, columns=columns,
                                                                              feature_range=feature_range)
                                self.processing_steps.append(
                                    {'step': 'min_max_normalize', 'feature_range': feature_range})

                            # Robust缩放
                            if norm_config.get('robust', False):
                                columns = norm_config.get('robust_columns')
                                result_df = self.normalizer.robust_scale(result_df, columns=columns)
                                self.processing_steps.append({'step': 'robust_scale'})

                            # 对数变换
                            if norm_config.get('log_transform', False):
                                columns = norm_config.get('log_columns')
                                base = norm_config.get('log_base', np.e)
                                offset = norm_config.get('log_offset', 1.0)
                                result_df = self.normalizer.log_transform(result_df, columns=columns, base=base,
                                                                          offset=offset)
                                self.processing_steps.append({'step': 'log_transform', 'base': base, 'offset': offset})

                        # 3. 数据转换
                        if 'transformation' in config:
                            trans_config = config['transformation']

                            # 类型转换
                            if 'type_conversion' in trans_config:
                                type_map = trans_config['type_conversion']
                                result_df = self.transformer.convert_types(result_df, type_map)
                                self.processing_steps.append({'step': 'convert_types'})

                            # One-Hot编码
                            if 'one_hot_encoding' in trans_config:
                                columns = trans_config['one_hot_encoding'].get('columns', [])
                                drop_first = trans_config['one_hot_encoding'].get('drop_first', False)
                                max_categories = trans_config['one_hot_encoding'].get('max_categories')
                                result_df = self.transformer.one_hot_encode(result_df, columns, drop_first=drop_first,
                                                                            max_categories=max_categories)
                                self.processing_steps.append({'step': 'one_hot_encode'})

                            # 标签编码
                            if 'label_encoding' in trans_config:
                                columns = trans_config['label_encoding'].get('columns', [])
                                mapping = trans_config['label_encoding'].get('mapping')
                                result_df = self.transformer.label_encode(result_df, columns, mapping=mapping)
                                self.processing_steps.append({'step': 'label_encode'})

                            # 时间特征提取
                            if 'datetime_features' in trans_config:
                                date_columns = trans_config['datetime_features'].get('columns', [])
                                features = trans_config['datetime_features'].get('features', [])
                                result_df = self.transformer.extract_datetime_features(result_df, date_columns,
                                                                                       features=features)
                                self.processing_steps.append({'step': 'extract_datetime_features'})

                        # 4. 数据验证
                        if 'validation' in config:
                            valid_config = config['validation']

                            # 完整性检查
                            if 'completeness' in valid_config:
                                comp_config = valid_config['completeness']

                                # 必要字段检查
                                if 'required_fields' in comp_config:
                                    required_fields = comp_config['required_fields']
                                    self.validator.check_required_fields(result_df, required_fields)

                                # 数据聚合验证
                                if 'aggregation' in comp_config:
                                    agg_config = comp_config['aggregation']
                                    self.validator.check_aggregation(
                                        result_df,
                                        agg_config['group_col'],
                                        agg_config['sum_col'],
                                        agg_config['total_col'],
                                        agg_config.get('tolerance', 0.01)
                                    )

                                # 记录数量验证
                                if 'record_count' in comp_config:
                                    count_config = comp_config['record_count']
                                    self.validator.check_record_count(
                                        result_df,
                                        count_config['expected'],
                                        count_config.get('tolerance_percent', 5.0)
                                    )

                                # 外键完整性检查
                                if 'foreign_key' in comp_config:
                                    fk_config = comp_config['foreign_key']
                                    if 'reference_df' in fk_config and 'reference_col' in fk_config:
                                        self.validator.check_foreign_key_integrity(
                                            result_df,
                                            fk_config['fk_col'],
                                            fk_config['reference_df'],
                                            fk_config['reference_col']
                                        )

                            # 格式验证
                            if 'format' in valid_config:
                                format_config = valid_config['format']

                                # 数据类型校验
                                if 'data_types' in format_config:
                                    self.validator.check_data_types(result_df, format_config['data_types'])

                                # 值范围检查
                                if 'value_ranges' in format_config:
                                    self.validator.check_value_ranges(result_df, format_config['value_ranges'])

                                # 正则表达式匹配
                                if 'regex_patterns' in format_config:
                                    self.validator.check_regex_patterns(result_df, format_config['regex_patterns'])

                                # 结构一致性验证
                                if 'schema' in format_config:
                                    self.validator.check_structural_consistency(result_df, format_config['schema'])

                            # 业务规则验证
                            if 'business_rules' in valid_config:
                                rules_config = valid_config['business_rules']

                                # 领域特定规则检查
                                if 'domain_rules' in rules_config:
                                    self.validator.check_domain_rules(result_df, rules_config['domain_rules'])

                                # 跨字段关系验证
                                if 'cross_field_relations' in rules_config:
                                    self.validator.check_cross_field_relations(result_df,
                                                                               rules_config['cross_field_relations'])

                                # 时序数据一致性
                                if 'time_series_consistency' in rules_config:
                                    ts_config = rules_config['time_series_consistency']
                                    self.validator.check_time_series_consistency(
                                        result_df,
                                        ts_config['time_col'],
                                        ts_config['value_col'],
                                        ts_config.get('group_col'),
                                        ts_config.get('max_change_percent', 200.0)
                                    )

                                # 业务流程合规性
                                if 'workflow_compliance' in rules_config:
                                    wf_config = rules_config['workflow_compliance']
                                    self.validator.check_workflow_compliance(
                                        result_df,
                                        wf_config['state_col'],
                                        wf_config['timestamp_col'],
                                        wf_config['id_col'],
                                        wf_config['valid_transitions']
                                    )

                        # 保存处理后的数据
                        self.processed_data = result_df

                        # 记录输出数据统计
                        output_stats = {
                            'rows': len(result_df),
                            'columns': len(result_df.columns),
                            'memory_usage': result_df.memory_usage(deep=True).sum(),
                            'missing_values': result_df.isnull().sum().sum()
                        }
                        logger.info(f"数据处理完成: {output_stats['rows']}行, {output_stats['columns']}列")

                        return result_df

                    def get_processing_summary(self) -> Dict:
                        """
                        获取处理摘要

                        Returns:
                            处理摘要字典
                        """
                        summary = {
                            'processing_steps': self.processing_steps,
                            'validation_summary': self.validator.get_validation_summary() if hasattr(self.validator,
                                                                                                     'validation_results') else None,
                            'input_shape': None,
                            'output_shape': None,
                            'anomalies_count': 0
                        }

                        if self.input_stats:
                            summary['input_shape'] = (self.input_stats['rows'], self.input_stats['columns'])

                        if self.processed_data is not None:
                            summary['output_shape'] = self.processed_data.shape

                        if self.anomalies is not None:
                            summary['anomalies_count'] = len(self.anomalies)

                        return summary

                    def save_processed_data(self, filepath: str, format: str = 'csv', index: bool = False) -> None:
                        """
                        保存处理后的数据

                        Args:
                            filepath: 文件路径
                            format: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
                            index: 是否保存索引
                        """
                        if self.processed_data is None:
                            logger.error("没有可保存的处理后数据")
                            return

                        try:
                            if format.lower() == 'csv':
                                self.processed_data.to_csv(filepath, index=index)
                            elif format.lower() == 'parquet':
                                self.processed_data.to_parquet(filepath, index=index)
                            elif format.lower() == 'pickle':
                                self.processed_data.to_pickle(filepath)
                            elif format.lower() == 'excel':
                                self.processed_data.to_excel(filepath, index=index)
                            else:
                                logger.error(f"不支持的文件格式: {format}")
                                return

                            logger.info(f"处理后的数据已保存到: {filepath}")
                        except Exception as e:
                            logger.error(f"保存数据时出错: {e}")

                    def save_anomalies(self, filepath: str, format: str = 'csv', index: bool = False) -> None:
                        """
                        保存检测到的异常数据

                        Args:
                            filepath: 文件路径
                            format: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
                            index: 是否保存索引
                        """
                        if self.anomalies is None:
                            logger.error("没有可保存的异常数据")
                            return

                        try:
                            if format.lower() == 'csv':
                                self.anomalies.to_csv(filepath, index=index)
                            elif format.lower() == 'parquet':
                                self.anomalies.to_parquet(filepath, index=index)
                            elif format.lower() == 'pickle':
                                self.anomalies.to_pickle(filepath)
                            elif format.lower() == 'excel':
                                self.anomalies.to_excel(filepath, index=index)
                            else:
                                logger.error(f"不支持的文件格式: {format}")
                                return

                            logger.info(f"异常数据已保存到: {filepath}")
                        except Exception as e:
                            logger.error(f"保存异常数据时出错: {e}")

                # 使用示例
                if __name__ == "__main__":
                    # 创建示例数据
                    np.random.seed(42)
                    data = {
                        'id': range(1, 1001),
                        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
                        'value': np.random.normal(100, 20, 1000),
                        'date': pd.date_range(start='2023-01-01', periods=1000),
                        'status': np.random.choice(['pending', 'approved', 'rejected', 'completed'], 1000),
                        'email': [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, 1001)]
                    }

                    # 添加一些异常和缺失值
                    data['value'][10:20] = np.nan  # 缺失值
                    data['value'][30:40] = 500  # 异常值

                    # 创建数据框
                    df = pd.DataFrame(data)

                    # 创建处理配置
                    config = {
                        'cleaning': {
                            'missing_values': {
                                'strategy': 'auto'
                            },
                            'outliers': {
                                'detection_method': 'zscore',
                                'handle': True,
                                'handle_strategy': 'winsorize'
                            },
                            'remove_duplicates': True
                        },
                        'normalization': {
                            'z_score': True,
                            'z_score_columns': ['value']
                        },
                        'transformation': {
                            'type_conversion': {
                                'id': 'int',
                                'date': 'datetime'
                            },
                            'one_hot_encoding': {
                                'columns': ['category'],
                                'drop_first': True
                            },
                            'datetime_features': {
                                'columns': ['date'],
                                'features': ['year', 'month', 'dayofweek', 'is_weekend']
                            }
                        },
                        'validation': {
                            'completeness': {
                                'required_fields': ['id', 'value', 'date', 'status']
                            },
                            'format': {
                                'data_types': {
                                    'id': 'int',
                                    'value': 'float',
                                    'date': 'datetime'
                                },
                                'value_ranges': {
                                    'value': {'min': 0, 'max': 200}
                                },
                                'regex_patterns': {
                                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                                }
                            },
                            'business_rules': {
                                'domain_rules': {
                                    'positive_value': lambda df: df['value'] > 0
                                },
                                'cross_field_relations': {
                                    'status_valid': lambda df: df['status'].isin(
                                        ['pending', 'approved', 'rejected', 'completed'])
                                }
                            }
                        }
                    }

                    # 初始化处理器并处理数据
                    processor = DataProcessor()
                    processed_df = processor.process(df, config)

                    # 获取处理摘要
                    summary = processor.get_processing_summary()
                    print(f"处理摘要: {summary}")

                    # 保存处理后的数据
                    processor.save_processed_data('processed_data.csv')

                    # 保存异常数据
                    if processor.anomalies is not None:
                        processor.save_anomalies('anomalies.csv')
            # 4. 数据验证
            if 'validation' in config:
                valid_config = config['validation']

                # 完整性检查
                if 'completeness' in valid_config:
                    comp_config = valid_config['completeness']

                    # 必要字段检查
                    if 'required_fields' in comp_config:
                        required_fields = comp_config['required_fields']
                        self.validator.check_required_fields(result_df, required_fields)

                    # 数据聚合验证
                    if 'aggregation' in comp_config:
                        agg_config = comp_config['aggregation']
                        self.validator.check_aggregation(
                            result_df,
                            agg_config['group_col'],
                            agg_config['sum_col'],
                            agg_config['total_col'],
                            agg_config.get('tolerance', 0.01)
                        )

                    # 记录数量验证
                    if 'record_count' in comp_config:
                        count_config = comp_config['record_count']
                        self.validator.check_record_count(
                            result_df,
                            count_config['expected'],
                            count_config.get('tolerance_percent', 5.0)
                        )

                    # 外键完整性检查
                    if 'foreign_key' in comp_config:
                        fk_config = comp_config['foreign_key']
                        if 'reference_df' in fk_config and 'reference_col' in fk_config:
                            self.validator.check_foreign_key_integrity(
                                result_df,
                                fk_config['fk_col'],
                                fk_config['reference_df'],
                                fk_config['reference_col']
                            )

                # 格式验证
                if 'format' in valid_config:
                    format_config = valid_config['format']

                    # 数据类型校验
                    if 'data_types' in format_config:
                        self.validator.check_data_types(result_df, format_config['data_types'])

                    # 值范围检查
                    if 'value_ranges' in format_config:
                        self.validator.check_value_ranges(result_df, format_config['value_ranges'])

                    # 正则表达式匹配
                    if 'regex_patterns' in format_config:
                        self.validator.check_regex_patterns(result_df, format_config['regex_patterns'])

                    # 结构一致性验证
                    if 'schema' in format_config:
                        self.validator.check_structural_consistency(result_df, format_config['schema'])

                # 业务规则验证
                if 'business_rules' in valid_config:
                    rules_config = valid_config['business_rules']

                    # 领域特定规则检查
                    if 'domain_rules' in rules_config:
                        self.validator.check_domain_rules(result_df, rules_config['domain_rules'])

                    # 跨字段关系验证
                    if 'cross_field_relations' in rules_config:
                        self.validator.check_cross_field_relations(result_df, rules_config['cross_field_relations'])

                    # 时序数据一致性
                    if 'time_series_consistency' in rules_config:
                        ts_config = rules_config['time_series_consistency']
                        self.validator.check_time_series_consistency(
                            result_df,
                            ts_config['time_col'],
                            ts_config['value_col'],
                            ts_config.get('group_col'),
                            ts_config.get('max_change_percent', 200.0)
                        )

                    # 业务流程合规性
                    if 'workflow_compliance' in rules_config:
                        wf_config = rules_config['workflow_compliance']
                        self.validator.check_workflow_compliance(
                            result_df,
                            wf_config['state_col'],
                            wf_config['timestamp_col'],
                            wf_config['id_col'],
                            wf_config['valid_transitions']
                        )

            # 保存处理后的数据
            self.processed_data = result_df

            # 记录输出数据统计
            output_stats = {
                'rows': len(result_df),
                'columns': len(result_df.columns),
                'memory_usage': result_df.memory_usage(deep=True).sum(),
                'missing_values': result_df.isnull().sum().sum()
            }
            logger.info(f"数据处理完成: {output_stats['rows']}行, {output_stats['columns']}列")

            return result_df

        def get_processing_summary(self) -> Dict:
            """
            获取处理摘要

            Returns:
                处理摘要字典
            """
            summary = {
                'processing_steps': self.processing_steps,
                'validation_summary': self.validator.get_validation_summary() if hasattr(self.validator,
                                                                                         'validation_results') else None,
                'input_shape': None,
                'output_shape': None,
                'anomalies_count': 0
            }

            if self.input_stats:
                summary['input_shape'] = (self.input_stats['rows'], self.input_stats['columns'])

            if self.processed_data is not None:
                summary['output_shape'] = self.processed_data.shape

            if self.anomalies is not None:
                summary['anomalies_count'] = len(self.anomalies)

            return summary

        def save_processed_data(self, filepath: str, format: str = 'csv', index: bool = False) -> None:
            """
            保存处理后的数据

            Args:
                filepath: 文件路径
                format: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
                index: 是否保存索引
            """
            if self.processed_data is None:
                logger.error("没有可保存的处理后数据")
                return

            try:
                if format.lower() == 'csv':
                    self.processed_data.to_csv(filepath, index=index)
                elif format.lower() == 'parquet':
                    self.processed_data.to_parquet(filepath, index=index)
                elif format.lower() == 'pickle':
                    self.processed_data.to_pickle(filepath)
                elif format.lower() == 'excel':
                    self.processed_data.to_excel(filepath, index=index)
                else:
                    logger.error(f"不支持的文件格式: {format}")
                    return

                logger.info(f"处理后的数据已保存到: {filepath}")
            except Exception as e:
                logger.error(f"保存数据时出错: {e}")

        def save_anomalies(self, filepath: str, format: str = 'csv', index: bool = False) -> None:
            """
            保存检测到的异常数据

            Args:
                filepath: 文件路径
                format: 文件格式，支持 'csv', 'parquet', 'pickle', 'excel'
                index: 是否保存索引
            """
            if self.anomalies is None:
                logger.error("没有可保存的异常数据")
                return

            try:
                if format.lower() == 'csv':
                    self.anomalies.to_csv(filepath, index=index)
                elif format.lower() == 'parquet':
                    self.anomalies.to_parquet(filepath, index=index)
                elif format.lower() == 'pickle':
                    self.anomalies.to_pickle(filepath)
                elif format.lower() == 'excel':
                    self.anomalies.to_excel(filepath, index=index)
                else:
                    logger.error(f"不支持的文件格式: {format}")
                    return

                logger.info(f"异常数据已保存到: {filepath}")
            except Exception as e:
                logger.error(f"保存异常数据时出错: {e}")

    # 使用示例
    if __name__ == "__main__":
        # 创建示例数据
        np.random.seed(42)
        data = {
            'id': range(1, 1001),
            'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
            'value': np.random.normal(100, 20, 1000),
            'date': pd.date_range(start='2023-01-01', periods=1000),
            'status': np.random.choice(['pending', 'approved', 'rejected', 'completed'], 1000),
            'email': [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, 1001)]
        }

        # 添加一些异常和缺失值
        data['value'][10:20] = np.nan  # 缺失值
        data['value'][30:40] = 500  # 异常值

        # 创建数据框
        df = pd.DataFrame(data)

        # 创建处理配置
        config = {
            'cleaning': {
                'missing_values': {
                    'strategy': 'auto'
                },
                'outliers': {
                    'detection_method': 'zscore',
                    'handle': True,
                    'handle_strategy': 'winsorize'
                },
                'remove_duplicates': True
            },
            'normalization': {
                'z_score': True,
                'z_score_columns': ['value']
            },
            'transformation': {
                'type_conversion': {
                    'id': 'int',
                    'date': 'datetime'
                },
                'one_hot_encoding': {
                    'columns': ['category'],
                    'drop_first': True
                },
                'datetime_features': {
                    'columns': ['date'],
                    'features': ['year', 'month', 'dayofweek', 'is_weekend']
                }
            },
            'validation': {
                'completeness': {
                    'required_fields': ['id', 'value', 'date', 'status']
                },
                'format': {
                    'data_types': {
                        'id': 'int',
                        'value': 'float',
                        'date': 'datetime'
                    },
                    'value_ranges': {
                        'value': {'min': 0, 'max': 200}
                    },
                    'regex_patterns': {
                        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    }
                },
                'business_rules': {
                    'domain_rules': {
                        'positive_value': lambda df: df['value'] > 0
                    },
                    'cross_field_relations': {
                        'status_valid': lambda df: df['status'].isin(['pending', 'approved', 'rejected', 'completed'])
                    }
                }
            }
        }

        # 初始化处理器并处理数据
        processor = DataProcessor()
        processed_df = processor.process(df, config)

        # 获取处理摘要
        summary = processor.get_processing_summary()
        print(f"处理摘要: {summary}")

        # 保存处理后的数据
        processor.save_processed_data('processed_data.csv')

        # 保存异常数据
        if processor.anomalies is not None:
            processor.save_anomalies('anomalies.csv')