# data_processor/core/modules/validator.py
"""Data validation module for checking data integrity, format, and business rules"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from data_processor.core.modules.base import BaseModule
from data_processor.utils.logging_utils import get_logger

logger = get_logger("DataValidator")


class DataValidator(BaseModule):
    """Data validation module for checking data integrity, format, and business rules"""

    def __init__(self):
        """Initialize the data validator"""
        super().__init__()
        self.validation_results = {
            'completeness': {},
            'format': {},
            'business_rules': {}
        }
        logger.info("Initialized DataValidator")

    def check_required_fields(self, df: pd.DataFrame, required_fields: List[str]) -> Tuple[bool, Dict]:
        """
        Check if required fields exist and are non-null

        Args:
            df: Input DataFrame
            required_fields: List of required fields

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'missing_fields': [], 'null_fields': {}}

        # Check if fields exist
        for field in required_fields:
            if field not in df.columns:
                result['missing_fields'].append(field)

        # Check if required fields have null values
        for field in required_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    result['null_fields'][field] = null_count

        # Validation result
        passed = (len(result['missing_fields']) == 0 and len(result['null_fields']) == 0)

        # Log operation
        self.validation_results['completeness']['required_fields'] = result
        self.log_operation({
            'operation': 'check_required_fields',
            'required_fields': required_fields,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Required fields validation passed")
        else:
            logger.warning(f"Required fields validation failed: {result}")

        return passed, result

    def check_time_series_consistency(self, df: pd.DataFrame, time_col: str,
                                      value_col: str, group_col: Optional[str] = None,
                                      max_change_percent: float = 200.0) -> Tuple[bool, Dict]:
        """
        Check time series data consistency (anomalous change rates)

        Args:
            df: Input DataFrame
            time_col: Time column
            value_col: Value column
            group_col: Grouping column (e.g., different products, regions)
            max_change_percent: Maximum allowed percent change

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'change_rate_violations': []}

        # Ensure time column is datetime type
        df_sorted = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_sorted[time_col]):
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors='coerce')

        # Sort by time
        df_sorted = df_sorted.sort_values(time_col)

        if group_col:
            # Calculate change rate by group
            for group_name, group_df in df_sorted.groupby(group_col):
                # Calculate percent change
                pct_change = group_df[value_col].pct_change() * 100

                # Find changes exceeding threshold
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
            # Calculate overall change rate
            pct_change = df_sorted[value_col].pct_change() * 100

            # Find changes exceeding threshold
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

        # Validation result
        passed = len(result['change_rate_violations']) == 0

        # Log operation
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
            logger.info("Time series consistency validation passed")
        else:
            logger.warning(
                f"Time series consistency validation failed: found {len(result['change_rate_violations'])} anomalous changes")

        return passed, result

    def check_workflow_compliance(self, df: pd.DataFrame,
                                  state_col: str, timestamp_col: str, id_col: str,
                                  valid_transitions: Dict[str, List[str]]) -> Tuple[bool, Dict]:
        """
        Check business process compliance (valid state transitions)

        Args:
            df: Input DataFrame
            state_col: State column
            timestamp_col: Timestamp column
            id_col: ID column (identifying different entities like orders, customers)
            valid_transitions: Valid state transitions dictionary {current_state: [valid_next_states]}

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'invalid_transitions': []}

        # Ensure time column is datetime type
        df_sorted = df.copy()
        if not pd.api.types.is_datetime64_dtype(df_sorted[timestamp_col]):
            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')

        # Sort by ID and time
        df_sorted = df_sorted.sort_values([id_col, timestamp_col])

        # Check state transitions for each entity
        for entity_id, entity_df in df_sorted.groupby(id_col):
            if len(entity_df) <= 1:
                continue  # Only one state, no transitions

            # Get state sequence
            states = entity_df[state_col].tolist()
            times = entity_df[timestamp_col].tolist()

            # Check transition validity
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
                    # Previous state not in transition rules
                    result['invalid_transitions'].append({
                        'entity_id': entity_id,
                        'from_state': prev_state,
                        'to_state': curr_state,
                        'from_time': times[i - 1],
                        'to_time': times[i],
                        'error': 'undefined_origin_state'
                    })

        # Validation result
        passed = len(result['invalid_transitions']) == 0

        # Log operation
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
            logger.info("Business process compliance validation passed")
        else:
            logger.warning(
                f"Business process compliance validation failed: found {len(result['invalid_transitions'])} invalid transitions")

        return passed, result

    def get_validation_summary(self) -> Dict:
        """
        Get validation results summary

        Returns:
            Dictionary containing validation results
        """
        # Calculate pass rate for each category
        summary = {category: {'total': 0, 'passed': 0} for category in self.validation_results.keys()}

        for category, checks in self.validation_results.items():
            for check_name, check_result in checks.items():
                summary[category]['total'] += 1
                if isinstance(check_result, dict) and check_result.get('passed', False):
                    summary[category]['passed'] += 1

        # Calculate overall pass rate
        total_checks = sum(cat['total'] for cat in summary.values())
        passed_checks = sum(cat['passed'] for cat in summary.values())

        summary['overall'] = {
            'total': total_checks,
            'passed': passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 1.0
        }

        return summary

    def check_aggregation(self, df: pd.DataFrame,
                          group_col: str, sum_col: str, total_col: str,
                          tolerance: float = 0.01) -> Tuple[bool, Dict]:
        """
        Validate data aggregation (e.g., sum of items equals total)

        Args:
            df: Input DataFrame
            group_col: Grouping column (e.g., category, department)
            sum_col: Column to sum (e.g., sales, quantity)
            total_col: Total column
            tolerance: Allowed error margin

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'inconsistent_groups': []}

        # Calculate sum by group
        group_sums = df.groupby(group_col)[sum_col].sum().reset_index()

        # Compare with total column
        for group in group_sums[group_col].unique():
            group_sum = group_sums.loc[group_sums[group_col] == group, sum_col].values[0]
            total_value = df.loc[df[group_col] == group, total_col].values[0]

            # Check if sum is within tolerance
            if abs(group_sum - total_value) > tolerance:
                result['inconsistent_groups'].append({
                    'group': group,
                    'calculated_sum': group_sum,
                    'reported_total': total_value,
                    'difference': group_sum - total_value
                })

        # Validation result
        passed = len(result['inconsistent_groups']) == 0

        # Log operation
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
            logger.info("Data aggregation validation passed")
        else:
            logger.warning(
                f"Data aggregation validation failed: found {len(result['inconsistent_groups'])} inconsistent groups")

        return passed, result

    def check_record_count(self, df: pd.DataFrame, expected_count: int,
                           tolerance_percent: float = 5.0) -> Tuple[bool, Dict]:
        """
        Validate record count matches expected value

        Args:
            df: Input DataFrame
            expected_count: Expected record count
            tolerance_percent: Allowed percentage error

        Returns:
            Tuple of (passed validation?, detailed results)
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

        # Validation result
        passed = percent_diff <= tolerance_percent

        # Log operation
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
            logger.info(f"Record count validation passed: expected {expected_count}, actual {actual_count}")
        else:
            logger.warning(
                f"Record count validation failed: expected {expected_count}, actual {actual_count}, difference {percent_diff:.2f}%")

        return passed, result

    def check_foreign_key_integrity(self, df: pd.DataFrame, fk_col: str,
                                    reference_df: pd.DataFrame, reference_col: str) -> Tuple[bool, Dict]:
        """
        Check foreign key integrity

        Args:
            df: Input DataFrame (containing foreign key)
            fk_col: Foreign key column name
            reference_df: Reference DataFrame (containing primary key)
            reference_col: Primary key column name in reference table

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        # Get unique values from reference table
        reference_values = set(reference_df[reference_col].dropna().unique())

        # Check if foreign key values are in reference set
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

        # Validation result
        passed = len(invalid_values) == 0

        # Log operation
        self.validation_results['completeness']['foreign_key'] = result
        self.log_operation({
            'operation': 'check_foreign_key_integrity',
            'fk_column': fk_col,
            'reference_column': reference_col,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info(
                f"Foreign key integrity validation passed: all values in column '{fk_col}' exist in reference column '{reference_col}'")
        else:
            logger.warning(f"Foreign key integrity validation failed: found {len(invalid_values)} invalid values")

        return passed, result

    def check_data_types(self, df: pd.DataFrame, type_specs: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        Check if data types match expected types

        Args:
            df: Input DataFrame
            type_specs: Mapping from column names to expected types

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'type_mismatches': []}

        for col, expected_type in type_specs.items():
            if col in df.columns:
                # Get current column type
                current_type = str(df[col].dtype)

                # Check if type matches
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

        # Validation result
        passed = len(result['type_mismatches']) == 0

        # Log operation
        self.validation_results['format']['data_types'] = result
        self.log_operation({
            'operation': 'check_data_types',
            'type_specs': type_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Data types validation passed")
        else:
            logger.warning(f"Data types validation failed: found {len(result['type_mismatches'])} type mismatches")

        return passed, result

    def check_value_ranges(self, df: pd.DataFrame,
                           range_specs: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
        """
        Check if values are within specified ranges

        Args:
            df: Input DataFrame
            range_specs: Mapping from column names to range specifications, format:
                        {column_name: {'min': min_value, 'max': max_value, 'inclusive': True/False}}

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'out_of_range': {}}

        for col, range_spec in range_specs.items():
            if col in df.columns:
                min_val = range_spec.get('min')
                max_val = range_spec.get('max')
                inclusive = range_spec.get('inclusive', True)

                # Check minimum value
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

                # Check maximum value
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

        # Validation result
        passed = len(result['out_of_range']) == 0

        # Log operation
        self.validation_results['format']['value_ranges'] = result
        self.log_operation({
            'operation': 'check_value_ranges',
            'range_specs': range_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Value ranges validation passed")
        else:
            logger.warning(f"Value ranges validation failed: found {len(result['out_of_range'])} out-of-range items")

        return passed, result

    def check_regex_patterns(self, df: pd.DataFrame,
                             pattern_specs: Dict[str, str]) -> Tuple[bool, Dict]:
        """
        Validate data format using regular expressions

        Args:
            df: Input DataFrame
            pattern_specs: Mapping from column names to regex patterns

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'pattern_mismatches': {}}

        for col, pattern in pattern_specs.items():
            if col in df.columns:
                # Apply regex pattern
                mask = df[col].astype(str).str.match(pattern) == False
                mismatches = df[mask]

                if not mismatches.empty:
                    result['pattern_mismatches'][col] = {
                        'count': len(mismatches),
                        'pattern': pattern,
                        'sample_values': mismatches[col].head(5).tolist()
                    }

        # Validation result
        passed = len(result['pattern_mismatches']) == 0

        # Log operation
        self.validation_results['format']['regex_patterns'] = result
        self.log_operation({
            'operation': 'check_regex_patterns',
            'pattern_specs': pattern_specs,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Regex pattern validation passed")
        else:
            logger.warning(
                f"Regex pattern validation failed: found {len(result['pattern_mismatches'])} mismatched columns")

        return passed, result

    def check_structural_consistency(self, df: pd.DataFrame,
                                     schema: Dict[str, Dict[str, Any]]) -> Tuple[bool, Dict]:
        """
        Check data structure consistency (column names, types, constraints)

        Args:
            df: Input DataFrame
            schema: Data schema definition

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }

        # Check if required columns exist
        schema_columns = set(schema.keys())
        df_columns = set(df.columns)

        result['missing_columns'] = list(schema_columns - df_columns)
        result['extra_columns'] = list(df_columns - schema_columns)

        # Check column types
        for col, col_schema in schema.items():
            if col in df.columns:
                expected_type = col_schema.get('type')
                if expected_type:
                    current_type = str(df[col].dtype)

                    # Check if type matches
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

        # Validation result
        passed = (len(result['missing_columns']) == 0 and
                  len(result['type_mismatches']) == 0)

        # Log operation
        self.validation_results['format']['structural_consistency'] = result
        self.log_operation({
            'operation': 'check_structural_consistency',
            'schema': schema,
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Structural consistency validation passed")
        else:
            logger.warning(
                f"Structural consistency validation failed: missing {len(result['missing_columns'])} columns, "
                f"type mismatches in {len(result['type_mismatches'])} columns")

        return passed, result

    def check_domain_rules(self, df: pd.DataFrame,
                           rule_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> Tuple[bool, Dict]:
        """
        Check domain-specific rules

        Args:
            df: Input DataFrame
            rule_specs: Mapping from rule names to rule functions, each function should return a boolean Series

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'rule_violations': {}}

        for rule_name, rule_func in rule_specs.items():
            # Apply rule function
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
                logger.error(f"Error applying rule '{rule_name}': {e}")

        # Validation result
        passed = len(result['rule_violations']) == 0

        # Log operation
        self.validation_results['business_rules']['domain_rules'] = result
        self.log_operation({
            'operation': 'check_domain_rules',
            'rules': list(rule_specs.keys()),
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Domain rules validation passed")
        else:
            logger.warning(f"Domain rules validation failed: found {len(result['rule_violations'])} rule violations")

        return passed, result

    def check_cross_field_relations(self, df: pd.DataFrame,
                                    relation_specs: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> Tuple[
        bool, Dict]:
        """
        Check cross-field relationships

        Args:
            df: Input DataFrame
            relation_specs: Mapping from relation names to validation functions, each function should return a boolean Series

        Returns:
            Tuple of (passed validation?, detailed results)
        """
        result = {'relation_violations': {}}

        for relation_name, relation_func in relation_specs.items():
            # Apply relation function
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
                logger.error(f"Error applying relation validation '{relation_name}': {e}")

        # Validation result
        passed = len(result['relation_violations']) == 0

        # Log operation
        self.validation_results['business_rules']['cross_field_relations'] = result
        self.log_operation({
            'operation': 'check_cross_field_relations',
            'relations': list(relation_specs.keys()),
            'result': result,
            'passed': passed
        })

        if passed:
            logger.info("Cross-field relations validation passed")
        else:
            logger.warning(
                f"Cross-field relations validation failed: found {len(result['relation_violations'])} relation violations")

        return passed, result