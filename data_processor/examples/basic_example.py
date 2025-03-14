# data_processor/examples/basic_usage.py
"""Basic usage example for the data processing framework"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to system path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_processor import DataProcessor
from data_processor.utils import setup_logging

def create_sample_data(rows=1000):
    """Create sample data for demonstration"""
    np.random.seed(42)
    data = {
        'id': range(1, rows + 1),
        'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
        'value': np.random.normal(100, 20, rows),
        'date': pd.date_range(start='2023-01-01', periods=rows),
        'status': np.random.choice(['pending', 'approved', 'rejected', 'completed'], rows),
        'email': [f"user{i}@example.com" if i % 10 != 0 else None for i in range(1, rows + 1)]
    }

    # Add some anomalies and missing values
    data['value'][10:20] = np.nan  # Missing values
    data['value'][30:40] = 500  # Outliers

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    """Main function demonstrating the data processing framework"""
    parser = argparse.ArgumentParser(description='Data Processing Framework Demo')
    parser.add_argument('--config', type=str, default='../config/config_example.json',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default='processed_data.csv',
                        help='Output file path')
    parser.add_argument('--log', type=str, default='data_processor_demo.log',
                        help='Log file path')
    parser.add_argument('--rows', type=int, default=1000,
                        help='Number of sample rows to generate')
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log)

    # Create sample data
    print("Creating sample data...")
    df = create_sample_data(rows=args.rows)
    print(f"Created sample data with {len(df)} rows and {len(df.columns)} columns")

    # Initialize data processor
    processor = DataProcessor()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = processor.load_config(args.config)

    # Process data
    print("Processing data...")
    processed_df = processor.process(df, config)
    print(f"Processing completed. Output shape: {processed_df.shape}")

    # Get processing summary
    summary = processor.get_processing_summary()
    print("\nProcessing Summary:")
    print(f"- Input shape: {summary['input_shape']}")
    print(f"- Output shape: {summary['output_shape']}")
    print(f"- Processing steps: {len(summary['processing_steps'])}")
    print(f"- Anomalies detected: {summary['anomalies_count']}")

    if summary['validation_summary']:
        val_summary = summary['validation_summary']
        print("\nValidation Summary:")
        print(f"- Overall pass rate: {val_summary['overall']['pass_rate'] * 100:.1f}%")
        for category, stats in val_summary.items():
            if category != 'overall':
                print(f"- {category.capitalize()}: {stats['passed']}/{stats['total']} checks passed")

    # Save processed data
    print(f"\nSaving processed data to {args.output}")
    processor.save_processed_data(args.output)

    # Save anomalies if detected
    if processor.anomalies is not None and len(processor.anomalies) > 0:
        anomalies_output = os.path.splitext(args.output)[0] + '_anomalies.csv'
        print(f"Saving detected anomalies to {anomalies_output}")
        processor.save_anomalies(anomalies_output)

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()