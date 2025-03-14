"""Integrated data processor, combining cleaning, normalization, transformation and validation functions"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import json

from data_processor.core.modules.cleaner import DataCleaner
from data_processor.core.modules.normalizer import DataNormalizer
from data_processor.core.modules.transformer import DataTransformer
from data_processor.core.modules.validator import DataValidator
from data_processor.utils.logging_utils import get_logger

# Import high-performance engine if available
try:
    from data_processor.core.high_perf_engine.cleaner import HighPerformanceDataCleaner

    HIGH_PERF_AVAILABLE = True
except ImportError:
    HIGH_PERF_AVAILABLE = False

logger = get_logger("DataProcessor")


class DataProcessor:
    """
    Integrated data processor, combining cleaning, normalization, transformation and validation functionality
    """

    def __init__(self, use_high_perf: bool = False, high_perf_config: Optional[Dict] = None):
        """
        Initialize the data processor

        Args:
            use_high_perf: Whether to use high-performance processing
            high_perf_config: Configuration for high-performance engine
        """
        # Initialize standard modules
        self.cleaner = DataCleaner()
        self.normalizer = DataNormalizer()
        self.transformer = DataTransformer()
        self.validator = DataValidator()

        # Initialize tracking variables
        self.processed_data = None
        self.anomalies = None
        self.processing_steps = []
        self.input_stats = None

        # High performance support
        self.use_high_perf = use_high_perf and HIGH_PERF_AVAILABLE
        self.high_perf_config = high_perf_config or {}

        if self.use_high_perf:
            if HIGH_PERF_AVAILABLE:
                self.high_perf_cleaner = HighPerformanceDataCleaner(self.high_perf_config)
                logger.info("Initialized DataProcessor with high-performance engine")
            else:
                logger.warning("High-performance engine requested but not available")
                self.use_high_perf = False
                logger.info("Initialized DataProcessor with standard engine")
        else:
            logger.info("Initialized DataProcessor with standard engine")

    def process(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Process data according to configuration

        Args:
            df: Input DataFrame
            config: Processing configuration

        Returns:
            Processed DataFrame
        """
        # Check if using high-performance mode
        if self.use_high_perf:
            logger.info("Using high-performance engine for initial cleaning")
            # Use high-performance engine for initial cleaning
            df = self.high_perf_cleaner.process(df)
            logger.info("High-performance cleaning complete, continuing with standard processing")

        result_df = df.copy()
        self.processing_steps = []

        # Record input data statistics
        self.input_stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().sum()
        }
        logger.info(f"Starting data processing: {self.input_stats['rows']} rows, {self.input_stats['columns']} columns")

        # 1. Data cleaning
        if 'cleaning' in config:
            cleaning_config = config['cleaning']

            # Handle missing values
            if 'missing_values' in cleaning_config:
                missing_strategy = cleaning_config['missing_values'].get('strategy', 'auto')
                result_df = self.cleaner.handle_missing_values(result_df, strategy=missing_strategy)
                self.processing_steps.append({'step': 'handle_missing_values', 'strategy': missing_strategy})

            # Detect outliers
            if 'outliers' in cleaning_config:
                outlier_method = cleaning_config['outliers'].get('detection_method', 'zscore')
                result_df, outliers_df = self.cleaner.detect_outliers(result_df, method=outlier_method)
                self.anomalies = outliers_df

                # Handle outliers
                if cleaning_config['outliers'].get('handle', True):
                    handle_strategy = cleaning_config['outliers'].get('handle_strategy', 'winsorize')
                    result_df = self.cleaner.handle_outliers(result_df, outliers_df, strategy=handle_strategy)
                    self.processing_steps.append({'step': 'handle_outliers', 'strategy': handle_strategy})

            # Remove duplicates
            if cleaning_config.get('remove_duplicates', True):
                subset = cleaning_config.get('duplicate_subset')
                result_df = self.cleaner.remove_duplicates(result_df, subset=subset)
                self.processing_steps.append({'step': 'remove_duplicates', 'subset': subset})

            # Smooth noisy data
            if 'noise_filtering' in cleaning_config:
                columns = cleaning_config['noise_filtering'].get('columns', [])
                window_size = cleaning_config['noise_filtering'].get('window_size', 3)
                if columns:
                    result_df = self.cleaner.filter_noise(result_df, columns, window_size=window_size)
                    self.processing_steps.append({'step': 'filter_noise', 'window_size': window_size})

        # 2. Data normalization/standardization
        if 'normalization' in config:
            norm_config = config['normalization']

            # Z-Score standardization
            if norm_config.get('z_score', False):
                columns = norm_config.get('z_score_columns')
                result_df = self.normalizer.z_score_normalize(result_df, columns=columns)
                self.processing_steps.append({'step': 'z_score_normalize', 'columns': columns})

            # Min-Max normalization
            if norm_config.get('min_max', False):
                columns = norm_config.get('min_max_columns')
                feature_range = norm_config.get('min_max_range', (0, 1))
                result_df = self.normalizer.min_max_normalize(result_df, columns=columns,
                                                              feature_range=feature_range)
                self.processing_steps.append({'step': 'min_max_normalize', 'feature_range': feature_range})

            # Robust scaling
            if norm_config.get('robust', False):
                columns = norm_config.get('robust_columns')
                result_df = self.normalizer.robust_scale(result_df, columns=columns)
                self.processing_steps.append({'step': 'robust_scale'})

            # Log transformation
            if norm_config.get('log_transform', False):
                columns = norm_config.get('log_columns')
                base = norm_config.get('log_base', np.e)
                offset = norm_config.get('log_offset', 1.0)
                result_df = self.normalizer.log_transform(result_df, columns=columns, base=base, offset=offset)
                self.processing_steps.append({'step': 'log_transform', 'base': base, 'offset': offset})

        # 3. Data transformation
        if 'transformation' in config:
            trans_config = config['transformation']

            # Type conversion
            if 'type_conversion' in trans_config:
                type_map = trans_config['type_conversion']
                result_df = self.transformer.convert_types(result_df, type_map)
                self.processing_steps.append({'step': 'convert_types'})

            # One-Hot encoding
            if 'one_hot_encoding' in trans_config:
                columns = trans_config['one_hot_encoding'].get('columns', [])
                drop_first = trans_config['one_hot_encoding'].get('drop_first', False)
                max_categories = trans_config['one_hot_encoding'].get('max_categories')
                result_df = self.transformer.one_hot_encode(result_df, columns, drop_first=drop_first,
                                                            max_categories=max_categories)
                self.processing_steps.append({'step': 'one_hot_encode'})

            # Label encoding
            if 'label_encoding' in trans_config:
                columns = trans_config['label_encoding'].get('columns', [])
                mapping = trans_config['label_encoding'].get('mapping')
                result_df = self.transformer.label_encode(result_df, columns, mapping=mapping)
                self.processing_steps.append({'step': 'label_encode'})

            # Datetime feature extraction
            if 'datetime_features' in trans_config:
                date_columns = trans_config['datetime_features'].get('columns', [])
                features = trans_config['datetime_features'].get('features', [])
                result_df = self.transformer.extract_datetime_features(result_df, date_columns, features=features)
                self.processing_steps.append({'step': 'extract_datetime_features'})

        # 4. Data validation
        if 'validation' in config:
            valid_config = config['validation']

            # Completeness checks
            if 'completeness' in valid_config:
                comp_config = valid_config['completeness']

                # Required fields check
                if 'required_fields' in comp_config:
                    required_fields = comp_config['required_fields']
                    self.validator.check_required_fields(result_df, required_fields)

                # Data aggregation validation
                if 'aggregation' in comp_config:
                    agg_config = comp_config['aggregation']
                    self.validator.check_aggregation(
                        result_df,
                        agg_config['group_col'],
                        agg_config['sum_col'],
                        agg_config['total_col'],
                        agg_config.get('tolerance', 0.01)
                    )

                # Record count validation
                if 'record_count' in comp_config:
                    count_config = comp_config['record_count']
                    self.validator.check_record_count(
                        result_df,
                        count_config['expected'],
                        count_config.get('tolerance_percent', 5.0)
                    )

                # Foreign key integrity check
                if 'foreign_key' in comp_config:
                    fk_config = comp_config['foreign_key']
                    if 'reference_df' in fk_config and 'reference_col' in fk_config:
                        self.validator.check_foreign_key_integrity(
                            result_df,
                            fk_config['fk_col'],
                            fk_config['reference_df'],
                            fk_config['reference_col']
                        )

            # Format validation
            if 'format' in valid_config:
                format_config = valid_config['format']

                # Data type validation
                if 'data_types' in format_config:
                    self.validator.check_data_types(result_df, format_config['data_types'])

                # Value range checks
                if 'value_ranges' in format_config:
                    self.validator.check_value_ranges(result_df, format_config['value_ranges'])

                # Regex pattern matching
                if 'regex_patterns' in format_config:
                    self.validator.check_regex_patterns(result_df, format_config['regex_patterns'])

                # Structural consistency validation
                if 'schema' in format_config:
                    self.validator.check_structural_consistency(result_df, format_config['schema'])

            # Business rule validation
            if 'business_rules' in valid_config:
                rules_config = valid_config['business_rules']

                # Domain-specific rule checks
                if 'domain_rules' in rules_config:
                    self.validator.check_domain_rules(result_df, rules_config['domain_rules'])

                # Cross-field relationship validation
                if 'cross_field_relations' in rules_config:
                    self.validator.check_cross_field_relations(result_df, rules_config['cross_field_relations'])

                # Time series data consistency
                if 'time_series_consistency' in rules_config:
                    ts_config = rules_config['time_series_consistency']
                    self.validator.check_time_series_consistency(
                        result_df,
                        ts_config['time_col'],
                        ts_config['value_col'],
                        ts_config.get('group_col'),
                        ts_config.get('max_change_percent', 200.0)
                    )

                # Business process compliance
                if 'workflow_compliance' in rules_config:
                    wf_config = rules_config['workflow_compliance']
                    self.validator.check_workflow_compliance(
                        result_df,
                        wf_config['state_col'],
                        wf_config['timestamp_col'],
                        wf_config['id_col'],
                        wf_config['valid_transitions']
                    )

        # Save processed data
        self.processed_data = result_df

        # Record output data statistics
        output_stats = {
            'rows': len(result_df),
            'columns': len(result_df.columns),
            'memory_usage': result_df.memory_usage(deep=True).sum(),
            'missing_values': result_df.isnull().sum().sum()
        }
        logger.info(f"Data processing completed: {output_stats['rows']} rows, {output_stats['columns']} columns")

        return result_df

    def get_processing_summary(self) -> Dict:
        """
        Get processing summary

        Returns:
            Processing summary dictionary
        """
        summary = {
            'processing_steps': self.processing_steps,
            'validation_summary': self.validator.get_validation_summary() if hasattr(self.validator,
                                                                                     'validation_results') else None,
            'input_shape': None,
            'output_shape': None,
            'anomalies_count': 0,
            'high_performance_mode': self.use_high_perf
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
        Save processed data

        Args:
            filepath: File path
            format: File format, supports 'csv', 'parquet', 'pickle', 'excel'
            index: Whether to save index
        """
        if self.processed_data is None:
            logger.error("No processed data to save")
            return

        try:
            if self.use_high_perf and HIGH_PERF_AVAILABLE:
                # Use optimized IO if high-performance mode is enabled
                self.high_perf_cleaner.io_optimizer.save_data_optimized(
                    self.processed_data, filepath, format=format
                )
                logger.info(f"Processed data saved to {filepath} using high-performance engine")
            else:
                # Use standard pandas saving
                if format.lower() == 'csv':
                    self.processed_data.to_csv(filepath, index=index)
                elif format.lower() == 'parquet':
                    self.processed_data.to_parquet(filepath, index=index)
                elif format.lower() == 'pickle':
                    self.processed_data.to_pickle(filepath)
                elif format.lower() == 'excel':
                    self.processed_data.to_excel(filepath, index=index)
                else:
                    logger.error(f"Unsupported file format: {format}")
                    return

                logger.info(f"Processed data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def save_anomalies(self, filepath: str, format: str = 'csv', index: bool = False) -> None:
        """
        Save detected anomalies data

        Args:
            filepath: File path
            format: File format, supports 'csv', 'parquet', 'pickle', 'excel'
            index: Whether to save index
        """
        if self.anomalies is None:
            logger.error("No anomalies data to save")
            return

        try:
            if self.use_high_perf and HIGH_PERF_AVAILABLE:
                # Use optimized IO if high-performance mode is enabled
                self.high_perf_cleaner.io_optimizer.save_data_optimized(
                    self.anomalies, filepath, format=format
                )
                logger.info(f"Anomalies data saved to {filepath} using high-performance engine")
            else:
                # Use standard pandas saving
                if format.lower() == 'csv':
                    self.anomalies.to_csv(filepath, index=index)
                elif format.lower() == 'parquet':
                    self.anomalies.to_parquet(filepath, index=index)
                elif format.lower() == 'pickle':
                    self.anomalies.to_pickle(filepath)
                elif format.lower() == 'excel':
                    self.anomalies.to_excel(filepath, index=index)
                else:
                    logger.error(f"Unsupported file format: {format}")
                    return

                logger.info(f"Anomalies data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving anomalies data: {e}")

    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from JSON file

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")

            # Extract high-performance config if available
            if self.use_high_perf and 'high_performance' in config:
                self.high_perf_config = config['high_performance']
                # Update high-performance cleaner if it exists
                if hasattr(self, 'high_perf_cleaner'):
                    self.high_perf_cleaner = HighPerformanceDataCleaner(self.high_perf_config)

            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def save_config(self, config: Dict, config_path: str) -> None:
        """
        Save configuration to JSON file

        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        try:
            # Add high-performance config if enabled
            if self.use_high_perf and 'high_performance' not in config:
                config['high_performance'] = self.high_perf_config

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def enable_high_performance(self, enable: bool = True, config: Optional[Dict] = None) -> bool:
        """
        Enable or disable high-performance processing mode

        Args:
            enable: Whether to enable high-performance mode
            config: Configuration for high-performance engine (if None, uses existing config)

        Returns:
            Whether high-performance mode is now enabled
        """
        if enable and not HIGH_PERF_AVAILABLE:
            logger.warning("Cannot enable high-performance mode: modules not available")
            return False

        self.use_high_perf = enable and HIGH_PERF_AVAILABLE

        if self.use_high_perf:
            # Update configuration if provided
            if config is not None:
                self.high_perf_config = config

            # Initialize high-performance cleaner
            self.high_perf_cleaner = HighPerformanceDataCleaner(self.high_perf_config)
            logger.info("High-performance processing mode enabled")
        else:
            logger.info("High-performance processing mode disabled")

        return self.use_high_perf

    def is_high_performance_available(self) -> bool:
        """
        Check if high-performance modules are available

        Returns:
            Whether high-performance modules are available
        """
        return HIGH_PERF_AVAILABLE