"""
Performance monitoring utilities for measuring and tracking resource usage
"""

import os
import time
import threading
import contextlib
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging
import pandas as pd
import matplotlib.pyplot as plt
from data_processor.utils.logging_utils import get_logger

logger = get_logger("PerformanceMonitor")


class PerformanceMonitor:
    """
    Performance monitoring for tracking CPU, memory and time usage

    This class provides:
    - Real-time CPU and memory monitoring
    - Time tracking for operations
    - Performance statistics
    - Visual reports
    """

    def __init__(self, interval: float = 0.5,
                 track_cpu: bool = True,
                 track_memory: bool = True,
                 track_disk: bool = False):
        """
        Initialize performance monitor

        Args:
            interval: Sampling interval in seconds
            track_cpu: Whether to track CPU usage
            track_memory: Whether to track memory usage
            track_disk: Whether to track disk I/O
        """
        self.interval = interval
        self.track_cpu = track_cpu
        self.track_memory = track_memory
        self.track_disk = track_disk

        # Data storage
        self.measurements = []
        self.timers = {}
        self.timer_history = {}

        # Monitoring thread
        self._monitor_thread = None
        self._stop_event = threading.Event()
        self._active = False

        # Check for psutil availability
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
            logger.warning("psutil not available, monitoring will be limited. Install with: pip install psutil")

        logger.info(f"Initialized PerformanceMonitor with interval={interval}s")

    def start_monitoring(self) -> None:
        """
        Start background monitoring thread
        """
        if self._active:
            logger.warning("Monitoring already active")
            return

        if not self._psutil_available:
            logger.warning("Cannot start monitoring without psutil")
            return

        self._stop_event.clear()
        self._active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Started resource monitoring")

    def stop_monitoring(self) -> None:
        """
        Stop background monitoring thread
        """
        if not self._active:
            return

        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.interval * 2)

        self._active = False
        logger.info("Stopped resource monitoring")

    def _monitor_resources(self) -> None:
        """
        Background thread function to monitor system resources
        """
        if not self._psutil_available:
            return

        import psutil
        process = psutil.Process(os.getpid())

        # Initial measurements
        start_time = time.time()

        while not self._stop_event.is_set():
            try:
                measurement = {
                    'timestamp': time.time() - start_time,
                    'wall_time': time.time()
                }

                # CPU usage
                if self.track_cpu:
                    measurement['cpu_percent'] = process.cpu_percent()
                    measurement['system_cpu_percent'] = psutil.cpu_percent()

                # Memory usage
                if self.track_memory:
                    mem_info = process.memory_info()
                    measurement['memory_rss_mb'] = mem_info.rss / (1024 * 1024)
                    measurement['memory_vms_mb'] = mem_info.vms / (1024 * 1024)

                    sys_mem = psutil.virtual_memory()
                    measurement['system_memory_used_percent'] = sys_mem.percent
                    measurement['system_memory_available_gb'] = sys_mem.available / (1024 * 1024 * 1024)

                # Disk I/O
                if self.track_disk:
                    io_counters = process.io_counters()
                    measurement['io_read_mb'] = io_counters.read_bytes / (1024 * 1024)
                    measurement['io_write_mb'] = io_counters.write_bytes / (1024 * 1024)

                # Store measurement
                self.measurements.append(measurement)

                # Sleep until next measurement
                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(self.interval)

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage

        Returns:
            Dictionary with current usage metrics
        """
        if not self._psutil_available:
            return {'error': 'psutil not available'}

        import psutil
        process = psutil.Process(os.getpid())

        usage = {}

        try:
            # CPU usage
            usage['cpu_percent'] = process.cpu_percent()
            usage['system_cpu_percent'] = psutil.cpu_percent()

            # Memory usage
            mem_info = process.memory_info()
            usage['memory_rss_mb'] = mem_info.rss / (1024 * 1024)
            usage['memory_vms_mb'] = mem_info.vms / (1024 * 1024)

            sys_mem = psutil.virtual_memory()
            usage['system_memory_used_percent'] = sys_mem.percent
            usage['system_memory_available_gb'] = sys_mem.available / (1024 * 1024 * 1024)

            return usage
        except Exception as e:
            logger.error(f"Error getting current usage: {e}")
            return {'error': str(e)}

    @contextlib.contextmanager
    def time_operation(self, operation_name: str) -> None:
        """
        Context manager for timing operations

        Args:
            operation_name: Name of the operation to time
        """
        start_time = time.time()
        current_usage = self.get_current_usage() if self._psutil_available else {}

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Store timer
            if operation_name not in self.timer_history:
                self.timer_history[operation_name] = []

            # Get final usage
            final_usage = self.get_current_usage() if self._psutil_available else {}

            # Calculate memory difference
            memory_diff = None
            if 'memory_rss_mb' in current_usage and 'memory_rss_mb' in final_usage:
                memory_diff = final_usage['memory_rss_mb'] - current_usage['memory_rss_mb']

            timer_info = {
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time,
                'initial_usage': current_usage,
                'final_usage': final_usage,
                'memory_diff_mb': memory_diff
            }

            self.timers[operation_name] = timer_info
            self.timer_history[operation_name].append(timer_info)

            logger.info(f"Operation '{operation_name}' took {duration:.2f} seconds" +
                        (f", memory change: {memory_diff:.2f} MB" if memory_diff is not None else ""))

    def get_timer_stats(self, operation_name: Optional[str] = None) -> Dict:
        """
        Get statistics about timed operations

        Args:
            operation_name: Specific operation to get stats for, None for all

        Returns:
            Dictionary with timing statistics
        """
        if operation_name:
            if operation_name not in self.timer_history:
                return {'error': f"Operation '{operation_name}' not found"}

            timer_data = self.timer_history[operation_name]

            return {
                'operation': operation_name,
                'count': len(timer_data),
                'total_duration': sum(t['duration'] for t in timer_data),
                'min_duration': min(t['duration'] for t in timer_data),
                'max_duration': max(t['duration'] for t in timer_data),
                'avg_duration': sum(t['duration'] for t in timer_data) / len(timer_data),
                'memory_diff': sum(t['memory_diff_mb'] for t in timer_data if t.get('memory_diff_mb') is not None)
            }
        else:
            # Stats for all operations
            all_stats = {}
            for op_name in self.timer_history:
                all_stats[op_name] = self.get_timer_stats(op_name)
            return all_stats

    def get_measurements_df(self) -> pd.DataFrame:
        """
        Get all measurements as a pandas DataFrame

        Returns:
            DataFrame with all measurements
        """
        return pd.DataFrame(self.measurements)

    def plot_memory_usage(self, ax=None, figsize=(10, 6)) -> plt.Figure:
        """
        Plot memory usage over time

        Args:
            ax: Matplotlib axis to plot on, None to create new
            figsize: Figure size if creating new

        Returns:
            Matplotlib figure
        """
        df = self.get_measurements_df()

        if len(df) == 0:
            logger.warning("No measurements available for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if 'memory_rss_mb' in df.columns:
            ax.plot(df['timestamp'], df['memory_rss_mb'], label='Process Memory (RSS)')

            # Add annotations for operations
            for op_name, timer_info in self.timers.items():
                start_time = timer_info['start_time'] - df['wall_time'].iloc[0] + df['timestamp'].iloc[0]
                end_time = timer_info['end_time'] - df['wall_time'].iloc[0] + df['timestamp'].iloc[0]

                # Find memory at these times
                memory_at_start = df['memory_rss_mb'][df['timestamp'] >= start_time].iloc[0] if any(
                    df['timestamp'] >= start_time) else None

                if memory_at_start is not None:
                    ax.annotate(op_name, xy=(start_time, memory_at_start),
                                xytext=(start_time, memory_at_start + 20),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                ha='center', va='bottom')

                    # Draw span
                    ax.axvspan(start_time, end_time, alpha=0.2)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Process Memory Usage Over Time')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No memory data available", ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_cpu_usage(self, ax=None, figsize=(10, 6)) -> plt.Figure:
        """
        Plot CPU usage over time

        Args:
            ax: Matplotlib axis to plot on, None to create new
            figsize: Figure size if creating new

        Returns:
            Matplotlib figure
        """
        df = self.get_measurements_df()

        if len(df) == 0:
            logger.warning("No measurements available for plotting")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        if 'cpu_percent' in df.columns:
            ax.plot(df['timestamp'], df['cpu_percent'], label='Process CPU')

            if 'system_cpu_percent' in df.columns:
                ax.plot(df['timestamp'], df['system_cpu_percent'], label='System CPU', linestyle='--')

            # Add annotations for operations
            for op_name, timer_info in self.timers.items():
                start_time = timer_info['start_time'] - df['wall_time'].iloc[0] + df['timestamp'].iloc[0]
                end_time = timer_info['end_time'] - df['wall_time'].iloc[0] + df['timestamp'].iloc[0]

                # Find CPU at these times
                cpu_at_start = df['cpu_percent'][df['timestamp'] >= start_time].iloc[0] if any(
                    df['timestamp'] >= start_time) else None

                if cpu_at_start is not None:
                    ax.annotate(op_name, xy=(start_time, cpu_at_start),
                                xytext=(start_time, cpu_at_start + 10),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                                ha='center', va='bottom')

                    # Draw span
                    ax.axvspan(start_time, end_time, alpha=0.2)

            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('CPU Usage (%)')
            ax.set_title('CPU Usage Over Time')
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No CPU data available", ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_performance_summary(self, figsize=(12, 10)) -> plt.Figure:
        """
        Plot a comprehensive performance summary

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # Memory plot
        self.plot_memory_usage(ax=axs[0])

        # CPU plot
        self.plot_cpu_usage(ax=axs[1])

        # Operation timing plot
        df = pd.DataFrame([
            {
                'operation': op,
                'duration': info['duration'],
                'start_time': info['start_time'] - self.measurements[0]['wall_time'] + self.measurements[0][
                    'timestamp'] if self.measurements else info['start_time'],
                'memory_change': info.get('memory_diff_mb', 0) or 0
            }
            for op, info in self.timers.items()
        ])

        if not df.empty:
            df = df.sort_values('start_time')

            # Plot timing bars
            bars = axs[2].barh(df['operation'], df['duration'], left=df['start_time'])

            # Color bars by memory change
            if 'memory_change' in df.columns:
                norm = plt.Normalize(df['memory_change'].min(), max(df['memory_change'].max(), 1))
                colors = plt.cm.RdYlGn_r(norm(df['memory_change']))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)

                # Add colorbar
                sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlGn_r)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axs[2])
                cbar.set_label('Memory Change (MB)')

            axs[2].set_xlabel('Time (seconds)')
            axs[2].set_ylabel('Operations')
            axs[2].set_title('Operation Timing')

            # Add duration text
            for i, (_, row) in enumerate(df.iterrows()):
                axs[2].text(row['start_time'] + row['duration'] / 2, i, f"{row['duration']:.2f}s",
                            ha='center', va='center', color='black', fontweight='bold')
        else:
            axs[2].text(0.5, 0.5, "No operation timing data available", ha='center', va='center',
                        transform=axs[2].transAxes)

        plt.tight_layout()
        return fig

    def generate_report(self, output_file: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive performance report

        Args:
            output_file: If provided, save report as HTML

        Returns:
            Dictionary with performance metrics
        """
        # Collect all stats
        stats = {
            'measurements_count': len(self.measurements),
            'total_monitoring_time': self.measurements[-1]['timestamp'] if self.measurements else 0,
            'peak_memory_mb': max(
                [m.get('memory_rss_mb', 0) for m in self.measurements]) if self.measurements else None,
            'avg_cpu_percent': sum([m.get('cpu_percent', 0) for m in self.measurements]) / len(
                self.measurements) if self.measurements else None,
            'operations': self.get_timer_stats()
        }

        # Generate plots if requested to save HTML
        if output_file:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend

                # Create plots
                fig = self.plot_performance_summary()

                # Save plot to file
                plot_file = f"{os.path.splitext(output_file)[0]}_plots.png"
                fig.savefig(plot_file, dpi=100, bbox_inches='tight')
                plt.close(fig)

                # Create HTML report
                html_content = f"""
                <html>
                <head>
                    <title>Performance Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .plot-image {{ max-width: 100%; }}
                    </style>
                </head>
                <body>
                    <h1>Performance Report</h1>

                    <h2>Summary</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Monitoring Time</td><td>{stats['total_monitoring_time']:.2f} seconds</td></tr>
                        <tr><td>Peak Memory Usage</td><td>{stats['peak_memory_mb']:.2f} MB</td></tr>
                        <tr><td>Average CPU Usage</td><td>{stats['avg_cpu_percent']:.2f}%</td></tr>
                        <tr><td>Number of Measurements</td><td>{stats['measurements_count']}</td></tr>
                    </table>

                    <h2>Operation Timing</h2>
                    <table>
                        <tr>
                            <th>Operation</th>
                            <th>Count</th>
                            <th>Total Time (s)</th>
                            <th>Avg Time (s)</th>
                            <th>Min Time (s)</th>
                            <th>Max Time (s)</th>
                            <th>Memory Change (MB)</th>
                        </tr>
                """

                # Add operation rows
                for op_name, op_stats in stats['operations'].items():
                    html_content += f"""
                        <tr>
                            <td>{op_name}</td>
                            <td>{op_stats['count']}</td>
                            <td>{op_stats['total_duration']:.2f}</td>
                            <td>{op_stats['avg_duration']:.2f}</td>
                            <td>{op_stats['min_duration']:.2f}</td>
                            <td>{op_stats['max_duration']:.2f}</td>
                            <td>{op_stats.get('memory_diff', 'N/A')}</td>
                        </tr>
                    """

                html_content += f"""
                    </table>

                    <h2>Performance Plots</h2>
                    <img src="{os.path.basename(plot_file)}" class="plot-image" alt="Performance Plots">
                </body>
                </html>
                """

                # Write HTML to file
                with open(output_file, 'w') as f:
                    f.write(html_content)

                logger.info(f"Performance report saved to {output_file}")
                stats['report_file'] = output_file
                stats['plot_file'] = plot_file

            except Exception as e:
                logger.error(f"Error generating HTML report: {e}")

        return stats