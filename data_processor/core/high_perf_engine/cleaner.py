"""
High Performance Data Cleaner
Coordinates data cleaning operations using parallel processing, memory optimization, and IO optimization
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging

from data_processor.core.high_perf_engine.parallel_engine import ParallelEngine
from data_processor.core.high_perf_engine.memory_manager import MemoryManager
from data_processor.core.high_perf_engine.io_optimizer import IOOptimizer
from data_processor.utils.logging_utils import get_logger

logger = get_logger("HighPerformanceDataCleaner")


class HighPerformanceDataCleaner:
    """
    High performance data cleaner that leverages parallel processing,
    memory optimization, and IO optimization for efficient data cleaning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the high performance data cleaner

        Args:
            config: Configuration dictionary with settings for the cleaner
                - num_workers: Number of worker processes
                - scheduler: Parallelization strategy ('processes', 'threads', 'distributed')
                - chunk_size: Data chunk size for processing
                - optimize_dtypes: Whether to optimize data types
                - use_mmap: Whether to use memory mapping for file operations
                - use_async: Whether to use asynchronous IO
                - prefetch_size: Number of chunks to prefetch
                - compression: Compression format for saved files
        """
        self.config = config or {}

        # Initialize core components
        self.parallel_engine = ParallelEngine(
            num_workers=self.config.get('num_workers'),
            scheduler=self.config.get('scheduler', 'processes'),
            chunk_size=self.config.get('chunk_size', 'auto')
        )

        self.memory_manager = MemoryManager(
            optimize_dtypes=self.config.get('optimize_dtypes', True),
            use_mmap=self.config.get('use_mmap', True),
            gc_threshold=self.config.get('gc_threshold', 0.75)
        )

        self.io_optimizer = IOOptimizer(
            use_async=self.config.get('use_async', True),
            prefetch_size=self.config.get('prefetch_size', 2),
            compression=self.config.get('compression', 'snappy')
        )

        # Performance monitoring
        self.performance_metrics = {
            'start_time': None,
            'end_time': None,
            'memory_peak': 0,
            'processed_rows': 0,
            'io_time': 0,
            'processing_time': 0,
            'optimization_time': 0
        }

        logger.info("Initialized HighPerformanceDataCleaner")

    def load_data(self, source: Union[str, pd.DataFrame],
                  format: str = 'auto', **kwargs) -> pd.DataFrame:
        """
        Smart data loading with optimizations

        Args:
            source: Data source (file path or DataFrame)
            format: File format for loading
            **kwargs: Additional loading parameters

        Returns:
            Loaded DataFrame
        """
        # Start timing
        io_start = time.time()
        self.performance_metrics['start_time'] = io_start

        # Get system memory info if available
        try:
            import psutil
            initial_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        except ImportError:
            initial_memory = None

        if isinstance(source, str):
            # Load from file
            logger.info(f"Loading data from file: {source}")

            if format == 'auto':
                # Auto-detect format from file extension
                format = source.split('.')[-1].lower()

            # Use memory-mapped loading when possible
            df = self.memory_manager.load_data_mmap(source, format=format, **kwargs)

        else:
            # Use existing DataFrame
            logger.info("Using provided DataFrame")
            df = source.copy()

        # Record IO time
        self.performance_metrics['io_time'] = time.time() - io_start

        # Record data info
        self.performance_metrics['processed_rows'] = len(df)

        # Update memory peak if available
        try:
            import psutil
            current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            self.performance_metrics['memory_peak'] = current_memory - initial_memory
            logger.info(f"Memory after loading: {current_memory:.2f} MB (+{current_memory - initial_memory:.2f} MB)")
        except:
            pass

        return df

    def _initial_cleaning_op(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Basic cleaning operations applied to each chunk

        Args:
            df_chunk: Input DataFrame chunk

        Returns:
            Cleaned DataFrame chunk
        """
        # Drop rows where all values are missing
        df_chunk = df_chunk.dropna(how='all')

        # Optimize data types to save memory
        df_chunk = self.memory_manager.optimize_dataframe(df_chunk)

        # Basic type conversion for common patterns
        for col in df_chunk.columns:
            col_dtype = str(df_chunk[col].dtype)

            # Skip optimization for already optimized columns
            if 'category' in col_dtype or 'datetime' in col_dtype:
                continue

            # Try to convert object columns to numeric
            if df_chunk[col].dtype == 'object':
                # Check if the column might be numeric
                try:
                    numeric_col = pd.to_numeric(df_chunk[col], errors='coerce')
                    # If not too many NaNs were introduced, assume it's numeric
                    if numeric_col.isna().sum() <= 0.1 * len(numeric_col):
                        df_chunk[col] = numeric_col
                except:
                    pass

                # Check if the column might be datetime
                if df_chunk[col].dtype == 'object':  # Still object type
                    try:
                        datetime_col = pd.to_datetime(df_chunk[col], errors='coerce')
                        # If not too many NaTs were introduced, assume it's datetime
                        if datetime_col.isna().sum() <= 0.1 * len(datetime_col):
                            df_chunk[col] = datetime_col
                    except:
                        pass

        return df_chunk

    def process(self, data: Union[str, pd.DataFrame],
                custom_cleaning_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Process and clean data with high performance optimizations

        Args:
            data: Input data (file path or DataFrame)
            custom_cleaning_func: Optional custom function to apply after initial cleaning

        Returns:
            Cleaned DataFrame
        """
        # Load data if needed
        df = self.load_data(data) if isinstance(data, str) else data.copy()

        # Start processing timer
        processing_start = time.time()

        # Define processing function for each chunk
        def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
            # Apply initial cleaning operations
            processed = self._initial_cleaning_op(chunk)

            # Apply custom function if provided
            if custom_cleaning_func:
                processed = custom_cleaning_func(processed)

            return processed

        # Process data in parallel
        logger.info(f"Starting parallel processing on DataFrame with shape {df.shape}")
        result_df = self.parallel_engine.process_data(df, process_chunk)

        # Record processing time
        self.performance_metrics['processing_time'] = time.time() - processing_start

        # Start optimization timer
        optimization_start = time.time()

        # Final optimizations
        logger.info("Applying final memory optimizations")
        result_df = self.memory_manager.optimize_dataframe(result_df, deep_optimization=True)

        # Record optimization time
        self.performance_metrics['optimization_time'] = time.time() - optimization_start

        # Record end time
        self.performance_metrics['end_time'] = time.time()

        # Record final shape
        logger.info(f"Completed processing: original shape {df.shape}, final shape {result_df.shape}")

        # Record final memory usage if possible
        try:
            import psutil
            final_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            memory_per_row = final_memory / len(result_df) if len(result_df) > 0 else 0
            logger.info(f"Final memory usage: {final_memory:.2f} MB ({memory_per_row:.2f} KB/row)")
        except:
            pass

        return result_df

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance metrics report

        Returns:
            Dictionary with performance metrics
        """
        if not self.performance_metrics['end_time']:
            return {"status": "Processing not completed"}

        # Calculate derived metrics
        total_duration = self.performance_metrics['end_time'] - self.performance_metrics['start_time']

        rows_per_second = (self.performance_metrics['processed_rows'] / total_duration
                           if total_duration > 0 else 0)

        io_percent = (self.performance_metrics['io_time'] / total_duration * 100
                      if total_duration > 0 else 0)

        processing_percent = (self.performance_metrics['processing_time'] / total_duration * 100
                              if total_duration > 0 else 0)

        optimization_percent = (self.performance_metrics['optimization_time'] / total_duration * 100
                                if total_duration > 0 else 0)

        report = {
            'total_duration_seconds': total_duration,
            'io_time_seconds': self.performance_metrics['io_time'],
            'processing_time_seconds': self.performance_metrics['processing_time'],
            'optimization_time_seconds': self.performance_metrics['optimization_time'],
            'rows_processed': self.performance_metrics['processed_rows'],
            'rows_per_second': rows_per_second,
            'memory_peak_mb': self.performance_metrics['memory_peak'],
            'time_breakdown_percent': {
                'io': io_percent,
                'processing': processing_percent,
                'optimization': optimization_percent,
                'other': 100 - io_percent - processing_percent - optimization_percent
            }
        }

        return report

    def clean_and_forward_to_preprocessor(self, data: Union[str, pd.DataFrame],
                                          data_processor: Any,
                                          processor_config: Dict) -> pd.DataFrame:
        """
        Clean data and forward to existing data processor

        Args:
            data: Input data (file path or DataFrame)
            data_processor: Existing DataProcessor instance
            processor_config: Configuration for the processor

        Returns:
            Processed DataFrame
        """
        # First apply high-performance cleaning
        cleaned_df = self.process(data)

        logger.info(f"High-performance cleaning complete, forwarding to standard processor")

        # Forward to standard processor
        return data_processor.process(cleaned_df, processor_config)

    def save_cleaned_data(self, df: pd.DataFrame, filepath: str, format: str = 'parquet') -> None:
        """
        Save cleaned data with optimizations

        Args:
            df: DataFrame to save
            filepath: Output path
            format: Output format
        """
        logger.info(f"Saving cleaned data (shape: {df.shape}) to {filepath}")

        # Use IO optimizer to save efficiently
        self.io_optimizer.save_data_optimized(df, filepath, format)

        logger.info(f"Data successfully saved to {filepath}")