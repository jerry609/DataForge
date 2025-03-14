"""
I/O optimization for efficient data loading and saving
"""

import os
import io
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, BinaryIO, Iterator, Any
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from data_processor.utils.logging_utils import get_logger

logger = get_logger("IOOptimizer")

class IOOptimizer:
    """
    I/O optimization for efficient data loading and saving

    This class provides:
    - Asynchronous I/O operations
    - Data prefetching
    - Automatic chunk management
    - Optimized storage formats
    """

    def __init__(self, use_async: bool = True,
                 prefetch_size: int = 2,
                 compression: str = 'snappy',
                 chunk_size_mb: int = 64):
        """
        Initialize I/O optimizer

        Args:
            use_async: Whether to use asynchronous I/O
            prefetch_size: Number of chunks to prefetch
            compression: Compression format to use
            chunk_size_mb: Chunk size in MB for chunked operations
        """
        self.use_async = use_async
        self.prefetch_size = prefetch_size
        self.compression = compression
        self.chunk_size_mb = chunk_size_mb
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._stop_prefetch = threading.Event()

        # Check if asyncio and aiofiles are available
        self._async_available = self._check_async_available()
        if use_async and not self._async_available:
            logger.warning("Async libraries not available, falling back to synchronous I/O")
            self.use_async = False

        logger.info(f"Initialized IOOptimizer with use_async={use_async}, "
                    f"prefetch_size={prefetch_size}, compression={compression}")

    def _check_async_available(self) -> bool:
        """Check if asyncio and aiofiles are available"""
        try:
            import asyncio
            try:
                import aiofiles
                return True
            except ImportError:
                return False
        except ImportError:
            return False

    def get_optimal_chunk_size(self, file_size: int) -> int:
        """
        Calculate optimal chunk size for processing

        Args:
            file_size: Size of the file in bytes

        Returns:
            Optimal chunk size in bytes
        """
        # Try to use system information if available
        try:
            import psutil
            # Get available memory
            mem_available = psutil.virtual_memory().available
            # Get number of CPUs
            cpu_count = os.cpu_count() or 1

            # Base the chunk size on available memory and CPU count
            # Use at most 25% of available memory divided by CPU count
            max_chunk_size = mem_available // (4 * cpu_count)

            # Ensure chunk size is reasonable for the file
            chunk_size = min(file_size // cpu_count, max_chunk_size)

            # Ensure minimum chunk size (1MB) and maximum (self.chunk_size_mb)
            return max(min(chunk_size, self.chunk_size_mb * 1024 * 1024), 1024 * 1024)
        except ImportError:
            # If psutil is not available, use a fixed calculation
            return min(file_size // 10, self.chunk_size_mb * 1024 * 1024)

    def create_chunks(self, filepath: str) -> List[Tuple[int, int]]:
        """
        Split a file into chunks for efficient processing

        Args:
            filepath: Path to the file

        Returns:
            List of (start_position, end_position) tuples
        """
        file_size = os.path.getsize(filepath)
        chunk_size = self.get_optimal_chunk_size(file_size)

        chunks = []
        start = 0
        while start < file_size:
            end = min(start + chunk_size, file_size)
            chunks.append((start, end))
            start = end

        logger.info(f"Split file {filepath} ({file_size / (1024*1024):.2f} MB) "
                    f"into {len(chunks)} chunks of ~{chunk_size / (1024*1024):.2f} MB each")

        return chunks

    def setup_prefetch(self, filepath: str, chunks: List[Tuple[int, int]] = None) -> None:
        """
        Set up data prefetching for chunked reading

        Args:
            filepath: Path to the file
            chunks: List of chunks to prefetch, if None will be created
        """
        if not self.use_async:
            logger.warning("Prefetching requires async I/O which is not enabled")
            return

        if chunks is None:
            chunks = self.create_chunks(filepath)

        # Create a queue to hold prefetched data
        self._prefetch_queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_prefetch.clear()

        # Start prefetch thread
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(filepath, chunks),
            daemon=True
        )
        self._prefetch_thread.start()

        logger.info(f"Started prefetch thread for {filepath} with queue size {self.prefetch_size}")

    def _prefetch_worker(self, filepath: str, chunks: List[Tuple[int, int]]) -> None:
        """
        Worker function to prefetch chunks in the background

        Args:
            filepath: Path to the file
            chunks: List of chunks to prefetch
        """
        try:
            with open(filepath, 'rb') as f:
                for chunk_start, chunk_end in chunks:
                    if self._stop_prefetch.is_set():
                        logger.debug("Prefetch stopped")
                        break

                    # Read the chunk
                    f.seek(chunk_start)
                    data = f.read(chunk_end - chunk_start)

                    # Try to put the chunk in the queue
                    try:
                        self._prefetch_queue.put((chunk_start, chunk_end, data), timeout=5)
                    except queue.Full:
                        # If the queue is full, wait a bit and try again
                        if not self._stop_prefetch.is_set():
                            time.sleep(0.1)
                            self._prefetch_queue.put((chunk_start, chunk_end, data))

                # Signal that all chunks have been processed
                if not self._stop_prefetch.is_set():
                    self._prefetch_queue.put(None)

        except Exception as e:
            logger.error(f"Error in prefetch worker: {str(e)}")
            # Signal an error
            if not self._stop_prefetch.is_set():
                try:
                    self._prefetch_queue.put(("ERROR", str(e)))
                except:
                    pass

    def stop_prefetch(self) -> None:
        """Stop prefetch thread and clear queue"""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._stop_prefetch.set()

            # Clear the queue
            if self._prefetch_queue:
                while not self._prefetch_queue.empty():
                    try:
                        self._prefetch_queue.get_nowait()
                    except:
                        pass

            # Wait for thread to finish
            self._prefetch_thread.join(timeout=2)

            logger.info("Prefetch stopped")

    def read_in_chunks(self, filepath: str, chunk_processor: callable = None) -> Union[bytes, List[Any]]:
        """
        Read a file in chunks with optional processing

        Args:
            filepath: Path to the file
            chunk_processor: Function to process each chunk, if None returns concatenated data

        Returns:
            Processed data or raw concatenated data if no processor
        """
        chunks = self.create_chunks(filepath)

        if self.use_async and self._prefetch_queue is None:
            self.setup_prefetch(filepath, chunks)

        all_data = []
        processed_chunks = []

        try:
            if self.use_async and self._prefetch_queue is not None:
                # Use prefetched data
                while True:
                    # Get next chunk from queue
                    chunk_data = self._prefetch_queue.get()

                    # Check if done or error
                    if chunk_data is None:
                        break
                    if isinstance(chunk_data, tuple) and chunk_data[0] == "ERROR":
                        raise RuntimeError(f"Error in prefetch: {chunk_data[1]}")

                    # Process or collect the chunk
                    _, _, data = chunk_data
                    if chunk_processor:
                        result = chunk_processor(data)
                        processed_chunks.append(result)
                    else:
                        all_data.append(data)

                    # Mark as done
                    self._prefetch_queue.task_done()
            else:
                # Read directly
                with open(filepath, 'rb') as f:
                    for chunk_start, chunk_end in chunks:
                        f.seek(chunk_start)
                        data = f.read(chunk_end - chunk_start)

                        if chunk_processor:
                            result = chunk_processor(data)
                            processed_chunks.append(result)
                        else:
                            all_data.append(data)

            # Return results
            if chunk_processor:
                return processed_chunks
            else:
                return b''.join(all_data)

        finally:
            # Clean up
            if self.use_async:
                self.stop_prefetch()

    def load_csv_optimized(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with optimized settings

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame loaded from CSV
        """
        # Determine if we should use chunked loading
        use_chunks = False
        file_size = os.path.getsize(filepath)

        # For large files, use chunked loading
        if file_size > 100 * 1024 * 1024:  # > 100MB
            use_chunks = True

        if use_chunks:
            logger.info(f"Loading large CSV file {filepath} in chunks")

            # Define chunk processor
            chunk_dfs = []

            def process_chunk(chunk_data):
                chunk_df = pd.read_csv(io.BytesIO(chunk_data), **kwargs)
                return chunk_df

            # Read and process chunks
            chunks = self.read_in_chunks(filepath, process_chunk)

            # Combine all chunk dataframes
            return pd.concat(chunks, ignore_index=True)
        else:
            # For smaller files, load directly
            return pd.read_csv(filepath, **kwargs)

    def save_data_optimized(self, df: pd.DataFrame, filepath: str, format: str = 'parquet') -> None:
        """
        Save DataFrame with optimized settings

        Args:
            df: DataFrame to save
            filepath: Output path
            format: Output format ('parquet', 'csv', 'feather', etc.)
        """
        format = format.lower()

        logger.info(f"Saving DataFrame with shape {df.shape} to {filepath} in {format} format")

        try:
            if format == 'parquet':
                # Parquet is already optimized for storage and speed
                df.to_parquet(filepath, compression=self.compression, index=False)

            elif format == 'csv':
                # For large DataFrames, write in chunks
                if len(df) > 500000:
                    chunk_size = 100000
                    logger.info(f"Writing large CSV in chunks of {chunk_size} rows")

                    # Write chunks using ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = []

                        # First chunk with header
                        first_chunk = df.iloc[:chunk_size]
                        futures.append(executor.submit(
                            first_chunk.to_csv,
                            filepath,
                            index=False,
                            mode='w'
                        ))

                        # Remaining chunks
                        for i in range(chunk_size, len(df), chunk_size):
                            chunk = df.iloc[i:i+chunk_size]
                            chunk_file = f"{filepath}.part{i//chunk_size}"
                            futures.append(executor.submit(
                                chunk.to_csv,
                                chunk_file,
                                index=False,
                                header=False
                            ))

                        # Wait for all futures to complete
                        for future in futures:
                            future.result()

                        # Append all chunks to the main file
                        with open(filepath, 'a') as outfile:
                            for i in range(1, (len(df) + chunk_size - 1) // chunk_size):
                                chunk_file = f"{filepath}.part{i}"
                                with open(chunk_file, 'r') as infile:
                                    outfile.write(infile.read())
                                # Remove temp file
                                os.remove(chunk_file)
                else:
                    # For smaller DataFrames, write directly
                    df.to_csv(filepath, index=False)

            elif format == 'feather':
                # Feather is good for intermediate storage
                df.to_feather(filepath)

            elif format == 'hdf' or format == 'h5':
                # HDF5 with table format
                df.to_hdf(filepath, key='data', mode='w', format='table',
                          complevel=5, complib='blosc')

            elif format == 'pickle' or format == 'pkl':
                # Pickle with compression
                df.to_pickle(filepath, compression='xz')

            else:
                raise ValueError(f"Unsupported output format: {format}")

            logger.info(f"Successfully saved data to {filepath}")

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise

    def read_chunked_csv(self, filepath: str, chunk_size: int = None, **kwargs) -> Iterator[pd.DataFrame]:
        """
        Read CSV file in chunks as an iterator

        Args:
            filepath: Path to CSV file
            chunk_size: Number of rows per chunk, if None uses automatic sizing
            **kwargs: Additional arguments for read_csv

        Returns:
            Iterator of DataFrame chunks
        """
        if chunk_size is None:
            # Determine a reasonable chunk size based on file size
            file_size = os.path.getsize(filepath)

            # Estimate rows based on sampling first few lines
            with open(filepath, 'r') as f:
                # Read header and first line
                header = f.readline()
                first_line = f.readline()

                # Estimate bytes per row
                if first_line:
                    bytes_per_row = len(first_line)
                    estimated_rows = file_size / bytes_per_row

                    # Aim for chunks that process around 50MB of data
                    chunk_size = int(50 * 1024 * 1024 / bytes_per_row)

                    # Ensure chunk size is reasonable
                    chunk_size = max(1000, min(chunk_size, 500000))
                else:
                    # Default if we can't estimate
                    chunk_size = 100000

        logger.info(f"Reading CSV {filepath} in chunks of {chunk_size} rows")

        # Use pandas chunked reading
        return pd.read_csv(filepath, chunksize=chunk_size, **kwargs)