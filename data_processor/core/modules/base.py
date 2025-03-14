# data_processor/core/modules/base.py
"""Base module for all data processing components"""

from datetime import datetime
from typing import Dict
import logging
from data_processor.utils.logging_utils import get_logger

logger = get_logger("BaseModule")


class BaseModule:
    """Base class for all data processing modules"""

    def __init__(self):
        """Initialize the base module"""
        self.operation_log = []

    def log_operation(self, operation_info: Dict):
        """
        Add timestamp and log an operation

        Args:
            operation_info: Dictionary containing operation details

        Returns:
            The operation info with added timestamp
        """
        operation_info['timestamp'] = datetime.now().isoformat()
        self.operation_log.append(operation_info)
        return operation_info