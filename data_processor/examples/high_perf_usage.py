"""
Example demonstrating the use of high-performance data processing features
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import argparse
from data_processor.core.processor import DataProcessor
from data_processor.utils.performance_monitor import PerformanceMonitor

# Add parent directory to path if running from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_sample_data(rows=1000000, cols=20, output_file=None):
    """Create sample data for testing"""
    print(f"Generating sample data with {rows} rows and {cols} columns...")

    # Generate numeric columns
    data = {
        f'num_col_{i}': np.random.randn(rows) for i in range(cols-5)
    }

    # Add some categorical columns
    data['cat_col_1'] = np.random.choice(['A', 'B', 'C', 'D'], rows)
    data['cat_col_2'] = np.random.choice(['X', 'Y', 'Z'], rows)

    # Add a datetime column
    data['date_col'] = pd.date_range(start='2020-01-01', periods=rows)

    # Add missing values to some columns
    for col in list(data.keys())[:5]:
        mask = np.random.random(size=rows) < 0.05
        data[col] = pd.Series(data[col]).mask(mask)

    # Add outliers to some columns
    for col in list(data.keys())[5:10]:
        mask = np.random.random(size=rows) < 0.01
        data[col][mask] = data[col][mask] * 100

    # Create DataFrame
    df = pd.DataFrame(data)

    if output_file:
        # Save to file
        print(f"Saving data to {output_file}")
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.parquet'):
            df.to_parquet(output_file, index=False)
        else:
            print(f"Unsupported file format: {output_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description='High-performance data processing example')
    parser.add_argument('--input', help='Input data file path')
    parser.add_argument('--output', help='Output file path for processed data')
    parser.add_argument('--config', default='../config/high_perf_config.json', help='Configuration file path')
    parser.add_argument('--rows', type=int, default=1000000, help='Number of rows for sample data')
    parser.add_argument('--cols', type=int, default=20, help='Number of columns for sample data')
    parser.add_argument('--report', help='Output path for performance report')
    args = parser.parse_args()

    # Create a performance monitor
    monitor = PerformanceMonitor(interval=0.5)
    monitor.start_monitoring()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config = json.load(f)

        # Get data
        if args.input:
            print(f"Using input file: {args.input}")
            input_data = args.input
        else:
            print("No input file specified, generating sample data...")
            with monitor.time_operation("Generate Sample Data"):
                input_data = create_sample_data(args.rows, args.cols)

        # Initialize processors
        print("\nInitializing processors...")

        # Standard processor
        with monitor.time_operation("Initialize Standard Processor"):
            standard_processor = DataProcessor()

        # High-performance processor
        with monitor.time_operation("Initialize High-Performance Processor"):
            from data_processor.core.high_perf_engine import HighPerformanceDataCleaner
            high_perf_processor = HighPerformanceDataCleaner(config.get('high_performance', {}))

        # Process with standard processor
        print("\nProcessing with standard DataProcessor...")
        with monitor.time_operation("Standard Processing"):
            if isinstance(input_data, str):
                # Load file first
                with monitor.time_operation("Standard Load Data"):
                    df = pd.read_csv(input_data) if input_data.endswith('.csv') else pd.read_parquet(input_data)

                # Then process
                with monitor.time_operation("Standard Process Data"):
                    processed_df1 = standard_processor.process(df, config)
            else:
                # Already have DataFrame
                processed_df1 = standard_processor.process(input_data, config)

        # Get standard processing summary
        standard_summary = standard_processor.get_processing_summary()
        print(f"Standard processing completed:")
        print(f"  - Input shape: {standard_summary['input_shape']}")
        print(f"  - Output shape: {standard_summary['output_shape']}")
        print(f"  - Processing steps: {len(standard_summary['processing_steps'])}")

        # Process with high-performance processor
        print("\nProcessing with high-performance DataCleaner...")
        with monitor.time_operation("High-Performance Processing"):
            processed_df2 = high_perf_processor.process(input_data)

        # Get high-performance processing stats
        hp_report = high_perf_processor.get_performance_report()
        print(f"High-performance processing completed:")
        print(f"  - Total duration: {hp_report['total_duration_seconds']:.2f} seconds")
        print(f"  - Rows processed: {hp_report['rows_processed']}")
        print(f"  - Processing speed: {hp_report['rows_per_second']:.0f} rows/second")
        print(f"  - Memory peak: {hp_report.get('memory_peak_mb', 'N/A')} MB")

        # Compare results
        print("\nComparing results:")
        shape_match = processed_df1.shape == processed_df2.shape
        print(f"  - Shape match: {'Yes' if shape_match else 'No'}")

        if shape_match:
            # Compare statistics for numeric columns
            numeric_cols = processed_df1.select_dtypes(include=['number']).columns
            print(f"  - Comparing {len(numeric_cols)} numeric columns...")

            for col in numeric_cols[:5]:  # Show first 5 for brevity
                df1_mean = processed_df1[col].mean()
                df2_mean = processed_df2[col].mean()

                rel_diff = abs(df1_mean - df2_mean) / (abs(df1_mean) if abs(df1_mean) > 1e-10 else 1)

                print(f"    - Column '{col}' mean difference: {rel_diff:.6%}")

        # Save output if requested
        if args.output:
            print(f"\nSaving processed data to {args.output}")

            with monitor.time_operation("Save Processed Data"):
                if args.output.endswith('.csv'):
                    processed_df2.to_csv(args.output, index=False)
                elif args.output.endswith('.parquet'):
                    processed_df2.to_parquet(args.output, index=False)
                else:
                    print(f"Unsupported output format: {args.output}")

        # Generate performance report
        if args.report:
            print(f"\nGenerating performance report at {args.report}")
            monitor.generate_report(args.report)

        # Print overall comparison
        print("\nPerformance Comparison:")
        std_time = monitor.timers.get('Standard Processing', {}).get('duration', 0)
        hp_time = monitor.timers.get('High-Performance Processing', {}).get('duration', 0)

        if std_time and hp_time:
            speedup = std_time / hp_time if hp_time > 0 else float('inf')
            print(f"  - Standard processing time: {std_time:.2f} seconds")
            print(f"  - High-performance processing time: {hp_time:.2f} seconds")
            print(f"  - Speedup: {speedup:.2f}x")

            # Compare memory usage
            std_mem = monitor.timers.get('Standard Processing', {}).get('memory_diff_mb', 0)
            hp_mem = monitor.timers.get('High-Performance Processing', {}).get('memory_diff_mb', 0)

            if std_mem and hp_mem:
                mem_reduction = (1 - hp_mem / std_mem) * 100 if std_mem > 0 else 0
                print(f"  - Standard processing memory impact: {std_mem:.2f} MB")
                print(f"  - High-performance processing memory impact: {hp_mem:.2f} MB")
                print(f"  - Memory efficiency improvement: {mem_reduction:.2f}%")

        # Print timing breakdown
        print("\nTiming Breakdown:")
        all_timers = monitor.get_timer_stats()
        for op_name, stats in all_timers.items():
            if isinstance(stats, dict) and 'duration' in stats:
                print(f"  - {op_name}: {stats['duration']:.2f} seconds")

    finally:
        # Stop monitoring
        monitor.stop_monitoring()

        # Plot and save performance graphs if matplotlib is available
        try:
            import matplotlib.pyplot as plt

            # Create performance plots
            fig = monitor.plot_performance_summary()

            # Save if report was requested
            if args.report:
                plot_path = os.path.splitext(args.report)[0] + "_plot.png"
                fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                print(f"Performance plot saved to {plot_path}")

            # Show plot if running interactively
            plt.show()
        except ImportError:
            print("Matplotlib not available for performance visualization")


if __name__ == "__main__":
    main()