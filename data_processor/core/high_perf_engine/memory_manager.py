"""
Memory optimization manager for efficient data processing
"""

import gc
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from data_processor.utils.logging_utils import get_logger

logger = get_logger("MemoryManager")


class MemoryManager:
    """
    Memory optimization manager to reduce memory usage and improve efficiency

    This class provides:
    - Data type optimization to reduce memory footprint
    - Memory-mapped file operations for large datasets
    - Optimized categorical data handling
    - Garbage collection management
    """

    def __init__(self, optimize_dtypes: bool = True,
                 use_mmap: bool = True,
                 gc_threshold: float = 0.75,
                 string_to_category_threshold: float = 0.5):
        """
        Initialize memory manager

        Args:
            optimize_dtypes: Whether to optimize data types
            use_mmap: Whether to use memory mapping for file operations
            gc_threshold: Memory usage threshold to trigger garbage collection (0-1)
            string_to_category_threshold: Threshold for converting strings to category type
                                         (ratio of unique values to total values)
        """
        self.optimize_dtypes = optimize_dtypes
        self.use_mmap = use_mmap
        self.gc_threshold = gc_threshold
        self.string_to_category_threshold = string_to_category_threshold
        self._memory_usage_history = []

        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
            logger.warning("psutil not available, memory monitoring will be limited")

        logger.info(f"Initialized MemoryManager with optimize_dtypes={optimize_dtypes}, "
                    f"use_mmap={use_mmap}, gc_threshold={gc_threshold}")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics

        Returns:
            Dictionary with memory usage information
        """
        if self._psutil_available:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            system_memory = psutil.virtual_memory()

            memory_usage = {
                'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
                'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
                'percent': process.memory_percent(),
                'system_total_gb': system_memory.total / (1024 * 1024 * 1024),
                'system_available_gb': system_memory.available / (1024 * 1024 * 1024),
                'system_percent': system_memory.percent
            }
        else:
            # Fallback if psutil isn't available
            memory_usage = {
                'rss_mb': None,
                'vms_mb': None,
                'percent': None,
                'system_total_gb': None,
                'system_available_gb': None,
                'system_percent': None
            }

        self._memory_usage_history.append(memory_usage)
        return memory_usage

    def optimize_dataframe(self, df: pd.DataFrame, deep_optimization: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage

        Args:
            df: Input DataFrame
            deep_optimization: Whether to perform more intensive optimization

        Returns:
            Optimized DataFrame
        """
        if not self.optimize_dtypes:
            return df

        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        result_df = df.copy()

        logger.info(f"Optimizing DataFrame with shape {df.shape}, "
                    f"initial memory usage: {original_memory:.2f} MB")

        # Optimize integer columns
        for col in result_df.select_dtypes(include=['int']).columns:
            result_df[col] = self._optimize_integer_column(result_df[col])

        # Optimize float columns
        for col in result_df.select_dtypes(include=['float']).columns:
            result_df[col] = self._optimize_float_column(result_df[col])

        # Optimize string (object) columns
        for col in result_df.select_dtypes(include=['object']).columns:
            result_df[col] = self._optimize_object_column(result_df[col])

        # Check if garbage collection is needed
        self._check_gc_needed()

        # Log memory savings
        final_memory = result_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        memory_reduction = (1 - final_memory / original_memory) * 100 if original_memory > 0 else 0

        logger.info(f"Memory optimization complete. Original: {original_memory:.2f} MB, "
                    f"Final: {final_memory:.2f} MB, Reduction: {memory_reduction:.1f}%")

        return result_df

    def _optimize_integer_column(self, column: pd.Series) -> pd.Series:
        """
        Optimize integer column by downcasting to smallest possible type

        Args:
            column: Integer column

        Returns:
            Optimized column
        """
        # Skip optimization if the column contains NaN values and using Int64 already
        if column.isna().any() and str(column.dtype) == 'Int64':
            return column

        # Get column stats
        col_min, col_max = column.min(), column.max()

        # Check if column contains NaN values
        contains_na = column.isna().any()

        # Check if column contains unsigned values only
        unsigned = col_min >= 0

        # Choose appropriate data type based on range
        if contains_na:
            # For columns with NaNs, use pandas nullable integer types
            if col_min >= 0:
                if col_max <= 255:
                    return column.astype('UInt8')
                elif col_max <= 65535:
                    return column.astype('UInt16')
                elif col_max <= 4294967295:
                    return column.astype('UInt32')
                else:
                    return column.astype('UInt64')
            else:
                if col_min >= -128 and col_max <= 127:
                    return column.astype('Int8')
                elif col_min >= -32768 and col_max <= 32767:
                    return column.astype('Int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    return column.astype('Int32')
                else:
                    return column.astype('Int64')
        else:
            # For columns without NaNs, use numpy types
            if unsigned:
                if col_max <= 255:
                    return column.astype(np.uint8)
                elif col_max <= 65535:
                    return column.astype(np.uint16)
                elif col_max <= 4294967295:
                    return column.astype(np.uint32)
                else:
                    return column.astype(np.uint64)
            else:
                if col_min >= -128 and col_max <= 127:
                    return column.astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    return column.astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    return column.astype(np.int32)
                else:
                    return column.astype(np.int64)

    def _optimize_float_column(self, column: pd.Series) -> pd.Series:
        """
        Optimize float column by downcasting to smallest possible type

        Args:
            column: Float column

        Returns:
            Optimized column
        """
        # Try to detect if float32 precision is enough
        # Check if the column values can be represented in float32 without loss
        try:
            # Convert to float32 and back to float64 to see if we lose precision
            float32_col = column.astype(np.float32).astype(np.float64)

            # If the difference is small enough, use float32
            if np.allclose(column.fillna(0), float32_col.fillna(0), rtol=1e-5, atol=1e-8):
                return column.astype(np.float32)
        except Exception as e:
            logger.debug(f"Error during float optimization: {str(e)}")

        # Default to keeping as is
        return column

    def _optimize_object_column(self, column: pd.Series) -> pd.Series:
        """
        Optimize object (string) column by converting to more efficient types

        Args:
            column: Object column

        Returns:
            Optimized column
        """
        # Skip if the column is empty
        if column.empty:
            return column

        # Try to convert to datetime
        try:
            return pd.to_datetime(column)
        except (TypeError, ValueError):
            pass

        # Try to convert to numeric
        try:
            numeric_col = pd.to_numeric(column)
            return numeric_col
        except (TypeError, ValueError):
            pass

        # Check if this column would benefit from categorical type
        unique_ratio = column.nunique() / len(column)
        if unique_ratio < self.string_to_category_threshold:
            try:
                return column.astype('category')
            except (TypeError, ValueError):
                pass

        return column

    def _check_gc_needed(self) -> None:
        """Check if garbage collection is needed and trigger if necessary"""
        if self._psutil_available:
            import psutil
            memory_percent = psutil.virtual_memory().percent / 100

            if memory_percent > self.gc_threshold:
                logger.info(f"Memory usage ({memory_percent:.1%}) exceeds threshold "
                            f"({self.gc_threshold:.1%}), triggering garbage collection")
                gc.collect()

                # Log memory usage after garbage collection
                memory_percent_after = psutil.virtual_memory().percent / 100
                logger.info(f"Memory usage after garbage collection: {memory_percent_after:.1%}")

    def load_data_mmap(self, filepath: str, format: str = 'auto', **kwargs) -> pd.DataFrame:
        """
        Load data using memory mapping when possible

        Args:
            filepath: Path to data file
            format: File format ('csv', 'parquet', 'hdf', etc.)
            **kwargs: Additional arguments to pass to pandas read function

        Returns:
            Loaded DataFrame
        """
        if not self.use_mmap:
            return self._load_data_regular(filepath, format, **kwargs)

        # Detect format if not specified
        if format == 'auto':
            format = filepath.split('.')[-1].lower()

        logger.info(f"Loading data from {filepath} using format {format} with memory mapping")

        try:
            if format == 'csv':
                # CSV supports memory mapping
                df = pd.read_csv(filepath, memory_map=True, **kwargs)
            elif format == 'parquet':
                # Parquet doesn't directly support memory mapping but is efficient
                df = pd.read_parquet(filepath, **kwargs)
            elif format == 'hdf' or format == 'h5':
                # HDF5 supports memory mapping
                df = pd.read_hdf(filepath, mode='r', **kwargs)
            elif format == 'feather':
                # Feather can be efficient but doesn't use memory mapping
                df = pd.read_feather(filepath, **kwargs)
            else:
                logger.warning(f"Format {format} doesn't support memory mapping, using regular loading")
                df = self._load_data_regular(filepath, format, **kwargs)

            # Apply memory optimizations if enabled
            if self.optimize_dtypes:
                return self.optimize_dataframe(df)
            else:
                return df

        except Exception as e:
            logger.error(f"Error loading data with memory mapping: {str(e)}")
            logger.info("Falling back to regular data loading")
            return self._load_data_regular(filepath, format, **kwargs)

    def _load_data_regular(self, filepath: str, format: str = 'auto', **kwargs) -> pd.DataFrame:
        """
        Load data using regular pandas functions

        Args:
            filepath: Path to data file
            format: File format
            **kwargs: Additional arguments

        Returns:
            Loaded DataFrame
        """
        # Detect format if not specified
        if format == 'auto':
            format = filepath.split('.')[-1].lower()

        logger.info(f"Loading data from {filepath} using format {format} without memory mapping")

        try:
            if format == 'csv':
                df = pd.read_csv(filepath, **kwargs)
            elif format == 'parquet':
                df = pd.read_parquet(filepath, **kwargs)
            elif format == 'hdf' or format == 'h5':
                df = pd.read_hdf(filepath, **kwargs)
            elif format == 'excel' or format == 'xlsx' or format == 'xls':
                df = pd.read_excel(filepath, **kwargs)
            elif format == 'json':
                df = pd.read_json(filepath, **kwargs)
            elif format == 'feather':
                df = pd.read_feather(filepath, **kwargs)
            elif format == 'pickle' or format == 'pkl':
                df = pd.read_pickle(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {format}")

            # Apply memory optimizations if enabled
            if self.optimize_dtypes:
                return self.optimize_dataframe(df)
            else:
                return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise