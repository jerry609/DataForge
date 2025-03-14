"""
Parallel processing engine for efficient data processing
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Callable, List, Optional, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

from data_processor.utils.logging_utils import get_logger

logger = get_logger("ParallelEngine")


class ParallelEngine:
    """
    Parallel processing engine for efficient data processing

    This engine provides different parallelization strategies:
    - Multi-process: Best for CPU-bound tasks
    - Multi-thread: Best for I/O-bound tasks
    - Distributed: For scaling across multiple machines (requires Ray)
    """

    def __init__(self, num_workers: Optional[int] = None,
                 scheduler: str = 'processes',
                 chunk_size: Union[str, int] = 'auto'):
        """
        Initialize parallel processing engine

        Args:
            num_workers: Number of worker processes, defaults to CPU count
            scheduler: Scheduler type ('processes', 'threads', 'distributed')
            chunk_size: Data chunk size for processing, 'auto' for automatic sizing
        """
        self.num_workers = num_workers or os.cpu_count()
        self.scheduler = scheduler
        self.chunk_size = chunk_size
        self._distributed_available = self._check_distributed_available()
        self._dask_available = self._check_dask_available()

        logger.info(f"Initialized ParallelEngine with {self.num_workers} workers "
                    f"using {self.scheduler} scheduler")

    def _check_distributed_available(self) -> bool:
        """Check if Ray is available for distributed processing"""
        try:
            import ray
            return True
        except ImportError:
            return False

    def _check_dask_available(self) -> bool:
        """Check if Dask is available for parallel processing"""
        try:
            import dask
            return True
        except ImportError:
            return False

    def _calculate_chunk_size(self, data_size: int) -> int:
        """
        Calculate optimal chunk size based on data size and workers

        Args:
            data_size: Size of the data to process

        Returns:
            Optimal chunk size
        """
        if isinstance(self.chunk_size, int):
            return self.chunk_size

        # Aim for at least 4 chunks per worker for better load balancing
        chunks_per_worker = 4
        chunk_size = max(1, data_size // (self.num_workers * chunks_per_worker))

        # Ensure chunk size isn't too small for very large datasets
        min_chunk_size = 1000  # Minimum rows per chunk
        return max(chunk_size, min_chunk_size)

    def _split_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks for parallel processing

        Args:
            df: Input DataFrame

        Returns:
            List of DataFrame chunks
        """
        chunk_size = self._calculate_chunk_size(len(df))
        logger.debug(f"Splitting DataFrame of size {len(df)} into chunks of {chunk_size}")
        return np.array_split(df, max(1, len(df) // chunk_size))

    def process_data(self, data: pd.DataFrame, operation_func: Callable) -> pd.DataFrame:
        """
        Process data in parallel

        Args:
            data: Input data (DataFrame or similar)
            operation_func: Function to apply to each partition

        Returns:
            Processed and merged data
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        if len(data) == 0:
            logger.warning("Empty DataFrame provided to parallel processing")
            return data

        logger.info(f"Starting parallel processing with {self.scheduler} scheduler")

        if self.scheduler == 'processes':
            return self._process_with_processes(data, operation_func)
        elif self.scheduler == 'threads':
            return self._process_with_threads(data, operation_func)
        elif self.scheduler == 'distributed':
            return self._process_distributed(data, operation_func)
        elif self.scheduler == 'dask':
            return self._process_with_dask(data, operation_func)
        else:
            logger.warning(f"Unknown scheduler '{self.scheduler}', falling back to processes")
            return self._process_with_processes(data, operation_func)

    def _process_with_processes(self, df: pd.DataFrame, func: Callable) -> pd.DataFrame:
        """
        Process data using multi-processing

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk

        Returns:
            Processed DataFrame
        """
        chunks = self._split_dataframe(df)

        logger.info(f"Processing {len(chunks)} chunks with {self.num_workers} processes")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(func, chunks))

        logger.info(f"Completed parallel processing, merging {len(results)} results")

        # Combine results
        try:
            result = pd.concat(results, ignore_index=True)
            return result
        except Exception as e:
            logger.error(f"Error combining parallel processing results: {str(e)}")
            raise

    def _process_with_threads(self, df: pd.DataFrame, func: Callable) -> pd.DataFrame:
        """
        Process data using multi-threading

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk

        Returns:
            Processed DataFrame
        """
        chunks = self._split_dataframe(df)

        logger.info(f"Processing {len(chunks)} chunks with {self.num_workers} threads")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(func, chunks))

        logger.info(f"Completed threaded processing, merging {len(results)} results")

        # Combine results
        try:
            result = pd.concat(results, ignore_index=True)
            return result
        except Exception as e:
            logger.error(f"Error combining threaded processing results: {str(e)}")
            raise

    def _process_distributed(self, df: pd.DataFrame, func: Callable) -> pd.DataFrame:
        """
        Process data using distributed computing (Ray)

        Args:
            df: Input DataFrame
            func: Function to apply to each chunk

        Returns:
            Processed DataFrame
        """
        if not self._distributed_available:
            logger.warning("Ray not available for distributed processing, falling back to processes")
            return self._process_with_processes(df, func)

        import ray

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create remote function
        @ray.remote
        def process_chunk(chunk):
            return func(chunk)

        chunks = self._split_dataframe(df)

        logger.info(f"Processing {len(chunks)} chunks with Ray distributed computing")

        # Submit tasks to Ray
        result_ids = [process_chunk.remote(chunk) for chunk in chunks]
        results = ray.get(result_ids)

        logger.info(f"Completed distributed processing, merging {len(results)} results")

        # Combine results
        try:
            result = pd.concat(results, ignore_index=True)
            return result
        except Exception as e:
            logger.error(f"Error combining distributed processing results: {str(e)}")
            raise

    def _process_with_dask(self, df: pd.DataFrame, func: Callable) -> pd.DataFrame:
        """
        Process data using Dask

        Args:
            df: Input DataFrame
            func: Function to apply to each partition

        Returns:
            Processed DataFrame
        """
        if not self._dask_available:
            logger.warning("Dask not available, falling back to processes")
            return self._process_with_processes(df, func)

        import dask.dataframe as dd

        logger.info(f"Processing with Dask using {self.num_workers} workers")

        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=self.num_workers)

        # Apply function to each partition
        result_df = dask_df.map_partitions(func).compute(scheduler='processes')

        logger.info("Completed Dask processing")

        return result_df