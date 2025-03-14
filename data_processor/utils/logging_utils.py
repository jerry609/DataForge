# data_processor/utils/logging_utils.py
"""Logging utilities for the data processing framework"""

import logging
import os
from typing import Optional
import sys

DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO


def setup_logging(log_level: int = DEFAULT_LOG_LEVEL,
                  log_file: Optional[str] = "data_processor.log",
                  log_format: str = DEFAULT_LOG_FORMAT) -> None:
    """
    Setup logging configuration for the application

    Args:
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Log file path, None to disable file logging
        log_format: Logging format string
    """
    handlers = []

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format=log_format
    )


def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name

    Args:
        name: Logger name
        log_level: Optional specific log level for this logger

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Set specific log level if provided
    if log_level is not None:
        logger.setLevel(log_level)

    return logger