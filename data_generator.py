import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
from typing import Tuple, List
import seaborn as sns
from common.plot_features import plot_data
import os


class ServerFailureSimulator:
    def __init__(self, start_time: datetime = None, interval_ms: int = 300):
        """
        Initialize the server failure simulator with smooth patterns
        
        Args:
            start_time: Starting timestamp for telemetry data
            interval_ms: Interval between telemetry readings in milliseconds
        """
        self.start_time = start_time or datetime.now()
        self.interval_ms = interval_ms
        self.interval_seconds = interval_ms / 1000.0
        
        # Baseline normal operation ranges
        self.normal_ranges = {
            'cpu_usage': (5, 25),
            'memory_usage': (30, 60),
            'network_utilization': (2, 20),
            'ram_utilization': (40, 70),
            'disk_utilization': (1, 15)
        }
        
        # Smooth trigonometric pattern configurations (longer periods for smoothness)
        self.trig_patterns = {
            'daily_cycle': {'frequency': 2*np.pi/5760, 'amplitude': 12},      # 48-hour cycle for smoother daily pattern
            'business_cycle': {'frequency': 2*np.pi/2880, 'amplitude': 8},    # 24-hour business cycle
            'weekly_pattern': {'frequency': 2*np.pi/40320, 'amplitude': 10},  # 14-day cycle for ultra-smooth
            'shift_pattern': {'frequency': 2*np.pi/1440, 'amplitude': 6},     # 12-hour shifts
            'maintenance_cycle': {'frequency': 2*np.pi/10080, 'amplitude': 5} # 7-day maintenance
        }
        
        # Smooth failure scenarios with harmonic patterns
        self.failure_scenarios = {
            'cpu_thermal_rise': {
                'duration': 2400,  # Longer duration for smoothness
                'affected_metrics': ['cpu_usage'],
                'pattern': 'smooth_thermal_rise',
                'peak_range': (80, 95),
                'trig_component': 'smooth_sigmoid_sine'
            },
            'memory_gradual_leak': {
                'duration': 3600,
                'affected_metrics': ['memory_usage', 'ram_utilization'],
                'pattern': 'smooth_exponential_growth',
                'peak_range': (85, 98),
                'trig_component': 'smooth_exponential_harmonic'
            },
            'network_congestion_wave': {
                'duration': 1800,
                'affected_metrics': ['network_utilization'],
                'pattern': 'smooth_wave_buildup',
                'peak_range': (75, 90),
                'trig_component': 'smooth_low_frequency_wave'
            },
            'disk_io_plateau': {
                'duration': 2000,
                'affected_metrics': ['disk_utilization'],
                'pattern': 'smooth_plateau',
                'peak_range': (82, 96),
                'trig_component': 'smooth_plateau_harmonic'
            },
            'resource_cascade': {
                'duration': 3000,
                'affected_metrics': ['cpu_usage', 'memory_usage', 'ram_utilization'],
                'pattern': 'smooth_cascade',
                'peak_range': (78, 92),
                'trig_component': 'smooth_phase_cascade'
            },
            'system_overload': {
                'duration': 2800,
                'affected_metrics': ['cpu_usage', 'memory_usage', 'network_utilization', 'disk_utilization'],
                'pattern': 'smooth_system_stress',
                'peak_range': (75, 88),
                'trig_component': 'smooth_multi_harmonic'
            },
            'load_balancer_failure': {
                'duration': 2200,
                'affected_metrics': ['network_utilization', 'cpu_usage'],
                'pattern': 'smooth_load_redistribution',
                'peak_range': (70, 85),
                'trig_component': 'smooth_redistribution_wave'
            },
            'database_lock_contention': {
                'duration': 1600,
                'affected_metrics': ['disk_utilization', 'cpu_usage'],
                'pattern': 'smooth_lock_pattern',
                'peak_range': (73, 89),
                'trig_component': 'smooth_lock_harmonic'
            }
        }
    
    def create_smooth_base_pattern(self, length: int, base_value: float, 
                                 amplitude: float, frequency: float, 
                                 phase: float = 0) -> np.ndarray:
        """Create ultra-smooth base patterns using low-frequency harmonics"""
        time_indices = np.linspace(0, 2*np.pi*frequency*length, length)
        
        # Primary smooth wave
        primary_wave = amplitude * np.sin(time_indices + phase)
        
        # Add subtle harmonics for richness but maintain smoothness
        harmonic_2 = (amplitude * 0.3) * np.sin(2 * time_indices + phase + np.pi/4)
        harmonic_3 = (amplitude * 0.15) * np.sin(0.5 * time_indices + phase + np.pi/6)
        
        # Very gentle random walk component
        random_walk = np.cumsum(np.random.normal(0, 0.2, length))
        random_walk = random_walk - np.linspace(random_walk[0], random_walk[-1], length)  # Detrend
        
        smooth_pattern = base_value + primary_wave + harmonic_2 + harmonic_3 + random_walk * 0.5
        
        return smooth_pattern
    
    def apply_smooth_trigonometric_patterns(self, base_values: np.ndarray, 
                                          metric: str, start_idx: int = 0) -> np.ndarray:
        """Apply ultra-smooth trigonometric patterns to base telemetry data"""
        length = len(base_values)
        enhanced_values = base_values.copy()
        mean_base = np.mean(base_values)
        
        if metric == 'cpu_usage':
            # Ultra-smooth daily cycle with business hours
            daily_component = self.create_smooth_base_pattern(
                length, 0, 
                self.trig_patterns['daily_cycle']['amplitude'], 
                self.trig_patterns['daily_cycle']['frequency'],
                phase=np.pi/6  # Morning rise
            )
            
            # Smooth business hours pattern
            business_component = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['business_cycle']['amplitude'] * 0.6,
                self.trig_patterns['business_cycle']['frequency'],
                phase=np.pi/4
            )
            
            enhanced_values += daily_component + np.abs(business_component) * 0.7
            
        elif metric == 'memory_usage':
            # Very smooth memory accumulation pattern
            weekly_component = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['weekly_pattern']['amplitude'],
                self.trig_patterns['weekly_pattern']['frequency'],
                phase=0
            )
            
            # Smooth daily buildup
            daily_buildup = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['daily_cycle']['amplitude'] * 0.4,
                self.trig_patterns['daily_cycle']['frequency'],
                phase=np.pi/3
            )
            
            enhanced_values += weekly_component + daily_buildup + 2
            
        elif metric == 'network_utilization':
            # Smooth traffic patterns
            shift_pattern = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['shift_pattern']['amplitude'],
                self.trig_patterns['shift_pattern']['frequency'],
                phase=np.pi/8
            )
            
            # Very gentle variations
            maintenance_component = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['maintenance_cycle']['amplitude'] * 0.8,
                self.trig_patterns['maintenance_cycle']['frequency'],
                phase=np.pi/2
            )
            
            enhanced_values += shift_pattern + maintenance_component
            
        elif metric in ['ram_utilization', 'disk_utilization']:
            # Ultra-smooth long-term patterns
            long_term_component = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['weekly_pattern']['amplitude'] * 0.7,
                self.trig_patterns['weekly_pattern']['frequency'] * 0.5,  # Even longer period
                phase=np.pi/4
            )
            
            # Gentle daily variation
            daily_component = self.create_smooth_base_pattern(
                length, 0,
                self.trig_patterns['daily_cycle']['amplitude'] * 0.3,
                self.trig_patterns['daily_cycle']['frequency'],
                phase=np.pi/5
            )
            
            enhanced_values += long_term_component + daily_component
        
        return enhanced_values
    
    def generate_normal_telemetry(self, num_points: int) -> dict:
        """Generate normal telemetry data with ultra-smooth variations"""
        data = {}
        
        for metric, (min_val, max_val) in self.normal_ranges.items():
            # Create smooth base trend
            mean_val = (min_val + max_val) / 2
            base_values = np.full(num_points, mean_val)
            
            # Apply smooth trigonometric patterns
            enhanced_values = self.apply_smooth_trigonometric_patterns(base_values, metric)
            
            # Add very gentle noise (much reduced)
            gentle_noise = np.random.normal(0, 0.8, num_points)
            
            # Apply smoothing filter to noise
            from scipy import ndimage
            try:
                smooth_noise = ndimage.gaussian_filter1d(gentle_noise, sigma=2.0)
            except:
                # Fallback if scipy not available
                smooth_noise = np.convolve(gentle_noise, np.ones(5)/5, mode='same')
            
            # Combine all components
            values = enhanced_values + smooth_noise
            
            # Ensure values stay within reasonable bounds
            values = np.clip(values, max(0, min_val - 5), max_val + 15)
            data[metric] = values
            
        return data
    
    def create_smooth_failure_component(self, base_values: np.ndarray, 
                                      trig_component: str, 
                                      peak_range: tuple,
                                      start_val: float = None) -> np.ndarray:
        """Create ultra-smooth failure patterns using advanced trigonometric functions"""
        length = len(base_values)
        time_indices = np.linspace(0, 1, length)
        
        if trig_component == 'smooth_sigmoid_sine':
            # Smooth sigmoid rise with gentle sine modulation
            sigmoid_rise = 1 / (1 + np.exp(-8 * (time_indices - 0.3)))  # Smooth S-curve
            gentle_modulation = 0.1 * np.sin(2 * np.pi * time_indices * 0.5)  # Very gentle sine
            pattern_intensity = sigmoid_rise + gentle_modulation
            
        elif trig_component == 'smooth_exponential_harmonic':
            # Smooth exponential growth with harmonic overtones
            exp_growth = (np.exp(3 * time_indices) - 1) / (np.exp(3) - 1)  # Normalized exp
            harmonic_1 = 0.15 * np.sin(2 * np.pi * time_indices * 0.3)
            harmonic_2 = 0.08 * np.sin(2 * np.pi * time_indices * 0.7)
            pattern_intensity = exp_growth + harmonic_1 + harmonic_2
            
        elif trig_component == 'smooth_low_frequency_wave':
            # Ultra-smooth low frequency wave
            primary_wave = np.sin(2 * np.pi * time_indices * 0.4) ** 2  # Squared for smoothness
            secondary_wave = 0.3 * np.sin(2 * np.pi * time_indices * 0.2 + np.pi/3)
            pattern_intensity = primary_wave + secondary_wave
            
        elif trig_component == 'smooth_plateau_harmonic':
            # Smooth plateau with gentle harmonic variations
            plateau_shape = np.tanh(8 * (time_indices - 0.2)) - np.tanh(8 * (time_indices - 0.8))
            gentle_variation = 0.2 * np.sin(2 * np.pi * time_indices * 0.6)
            pattern_intensity = 0.5 * (plateau_shape + 1) + gentle_variation
            
        elif trig_component == 'smooth_phase_cascade':
            # Smooth cascading phases
            phase1 = 0.4 * (1 + np.sin(2 * np.pi * time_indices * 0.2 - np.pi/2))
            phase2 = 0.3 * (1 + np.sin(2 * np.pi * time_indices * 0.3 - np.pi/4))
            phase3 = 0.3 * (1 + np.sin(2 * np.pi * time_indices * 0.25 - 0))
            pattern_intensity = phase1 + phase2 + phase3
            
        elif trig_component == 'smooth_multi_harmonic':
            # Multiple smooth harmonics
            h1 = 0.5 * (1 + np.sin(2 * np.pi * time_indices * 0.15))
            h2 = 0.3 * (1 + np.sin(2 * np.pi * time_indices * 0.25 + np.pi/6))
            h3 = 0.2 * (1 + np.sin(2 * np.pi * time_indices * 0.35 + np.pi/4))
            pattern_intensity = (h1 + h2 + h3) / 3
            
        elif trig_component == 'smooth_redistribution_wave':
            # Smooth redistribution pattern
            redistribution = np.sin(2 * np.pi * time_indices * 0.3) ** 4  # Fourth power for ultra-smoothness
            gentle_drift = 0.2 * np.sin(2 * np.pi * time_indices * 0.1)
            pattern_intensity = redistribution + gentle_drift + 0.3
            
        elif trig_component == 'smooth_lock_harmonic':
            # Smooth lock contention pattern
            contention_wave = 0.6 * (1 + np.sin(2 * np.pi * time_indices * 0.4)) * np.exp(-time_indices)
            background_stress = 0.4 * (1 + np.sin(2 * np.pi * time_indices * 0.15))
            pattern_intensity = contention_wave + background_stress
            
        else:
            # Default smooth pattern
            pattern_intensity = 0.5 * (1 + np.sin(2 * np.pi * time_indices * 0.3))
        
        # Normalize pattern intensity to [0, 1]
        pattern_intensity = np.clip(pattern_intensity, 0, 1)
        
        # Calculate target peak values
        start_value = start_val if start_val is not None else np.mean(base_values)
        peak_value = np.random.uniform(*peak_range)
        
        # Create smooth transition
        failure_component = start_value + (peak_value - start_value) * pattern_intensity
        
        return failure_component
    
    def inject_failure_pattern(self, data: dict, failure_type: str, start_idx: int) -> dict:
        """Inject ultra-smooth failure patterns into telemetry data"""
        failure_config = self.failure_scenarios[failure_type]
        duration = failure_config['duration']
        affected_metrics = failure_config['affected_metrics']
        pattern = failure_config['pattern']
        peak_range = failure_config['peak_range']
        trig_component = failure_config.get('trig_component', 'smooth_sigmoid_sine')
        
        end_idx = min(start_idx + duration, len(data['cpu_usage']))
        failure_length = end_idx - start_idx
        
        for metric in affected_metrics:
            original_slice = data[metric][start_idx:end_idx]
            start_value = data[metric][start_idx]
            
            if pattern == 'smooth_thermal_rise':
                # Smooth thermal rise pattern
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_exponential_growth':
                # Smooth exponential growth (memory leak)
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_wave_buildup':
                # Smooth wave buildup pattern
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_plateau':
                # Smooth plateau pattern
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_cascade':
                # Smooth cascading failure
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_system_stress':
                # Smooth system-wide stress
                # Different intensities for different metrics
                if metric == 'cpu_usage':
                    adjusted_peak = (peak_range[0] + 10, peak_range[1])
                elif metric == 'memory_usage':
                    adjusted_peak = (peak_range[0] + 5, peak_range[1] - 3)
                else:
                    adjusted_peak = peak_range
                    
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, adjusted_peak, start_value
                )
                
            elif pattern == 'smooth_load_redistribution':
                # Smooth load redistribution
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
                
            elif pattern == 'smooth_lock_pattern':
                # Smooth lock contention pattern
                failure_values = self.create_smooth_failure_component(
                    original_slice, trig_component, peak_range, start_value
                )
            
            # Ensure smooth transitions at boundaries
            if len(failure_values) > 10:
                # Smooth entry transition
                entry_blend = np.linspace(0, 1, 20)
                failure_values[:20] = (entry_blend * failure_values[:20] + 
                                     (1 - entry_blend) * original_slice[:20])
                
                # Smooth exit transition  
                exit_blend = np.linspace(1, 0, 20)
                if end_idx < len(data[metric]):
                    exit_values = data[metric][end_idx:min(end_idx + 20, len(data[metric]))]
                    blend_length = min(20, len(failure_values), len(exit_values))
                    if blend_length > 0:
                        failure_values[-blend_length:] = (exit_blend[:blend_length] * failure_values[-blend_length:] + 
                                                        (1 - exit_blend[:blend_length]) * exit_values[:blend_length])
            
            # Apply values ensuring they stay within bounds
            failure_values = np.clip(failure_values, 0, 100)
            data[metric][start_idx:start_idx + len(failure_values)] = failure_values
        
        return data
    
    def generate_telemetry_data(self, total_points: int = 20000, num_failures: int = 6) -> pd.DataFrame:
        """Generate complete telemetry dataset with smooth patterns and failures"""
        print(f"Generating {total_points} smooth telemetry data points...")
        
        # Generate base smooth telemetry
        data = self.generate_normal_telemetry(total_points)
        
        # Generate timestamps
        timestamps = [
            self.start_time + timedelta(seconds=i * self.interval_seconds)
            for i in range(total_points)
        ]
        
        # Inject failures with adequate spacing for smoothness
        failure_types = list(self.failure_scenarios.keys())
        min_gap = 2000  # Minimum gap between failures for smoothness
        failure_positions = []
        
        current_pos = 1500
        for _ in range(num_failures):
            if current_pos < total_points - 4000:
                failure_positions.append(current_pos)
                current_pos += random.randint(min_gap, min_gap + 1000)
        
        print("Injecting smooth failure scenarios:")
        for i, start_pos in enumerate(failure_positions):
            failure_type = random.choice(failure_types)
            duration = self.failure_scenarios[failure_type]['duration']
            trig_pattern = self.failure_scenarios[failure_type].get('trig_component', 'smooth_sigmoid_sine')
            print(f"  {i+1}. {failure_type} (duration: {duration}, pattern: {trig_pattern}) at index {start_pos}")
            data = self.inject_failure_pattern(data, failure_type, start_pos)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': data['cpu_usage'],
            'memory_usage': data['memory_usage'],
            'network_utilization': data['network_utilization'],
            'ram_utilization': data['ram_utilization'],
            'disk_utilization': data['disk_utilization']
        })
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'server_telemetry_smooth.csv'):
        """Save telemetry data to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Smooth telemetry data saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    def plot_telemetry(self, df: pd.DataFrame, start_time: datetime = None, end_time: datetime = None):
        """Plot telemetry data within specified time range"""
        # Filter data by time range if specified
        if start_time or end_time:
            mask = pd.Series([True] * len(df))
            if start_time:
                mask &= df['timestamp'] >= start_time
            if end_time:
                mask &= df['timestamp'] <= end_time
            plot_df = df[mask].copy()
        else:
            plot_df = df.copy()
        
        if plot_df.empty:
            print("No data in specified time range")
            return
        
        # Set up the plot
        plt.style.use('default')
        fig, axes = plt.subplots(5, 1, figsize=(16, 20))
        fig.suptitle('Server Telemetry Data - Ultra-Smooth Patterns', fontsize=16, fontweight='bold')
        
        metrics = ['cpu_usage', 'memory_usage', 'network_utilization', 'ram_utilization', 'disk_utilization']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i]
            
            # Plot the smooth metric
            ax.plot(plot_df['timestamp'], plot_df[metric], 
                   color=color, linewidth=1.5, alpha=0.9, label=metric.replace('_', ' ').title())
            
            # Highlight elevated values (>75) with smooth regions
            elevated_mask = plot_df[metric] > 75
            if elevated_mask.any():
                ax.fill_between(plot_df['timestamp'], 0, plot_df[metric],
                               where=elevated_mask, alpha=0.2, color='orange',
                               label='Elevated (>75%)')
            
            # Highlight critical values (>90)
            critical_mask = plot_df[metric] > 90
            if critical_mask.any():
                ax.fill_between(plot_df['timestamp'], 0, plot_df[metric],
                               where=critical_mask, alpha=0.3, color='red',
                               label='Critical (>90%)')
            
            # Formatting
            ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontweight='bold')
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            
            # Add threshold lines
            ax.axhline(y=75, color='orange', linestyle='--', alpha=0.6, linewidth=1)
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        # Format x-axis for the last subplot
        axes[-1].set_xlabel('Timestamp', fontweight='bold')
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print smoothness statistics
        print(f"\nSmooth Pattern Statistics ({len(plot_df)} data points):")
        print("="*70)
        for metric in metrics:
            values = plot_df[metric].values
            
            # Calculate smoothness metrics
            gradients = np.diff(values)
            avg_gradient = np.mean(np.abs(gradients))
            max_jump = np.max(np.abs(gradients))
            
            elevated_count = (values > 75).sum()
            critical_count = (values > 90).sum()
            
            print(f"{metric.replace('_', ' ').title():20s}: "
                  f"Avg: {values.mean():.1f}%, "
                  f"Max: {values.max():.1f}%, "
                  f"Smooth(avg Δ): {avg_gradient:.2f}, "
                  f"Max jump: {max_jump:.2f}, "
                  f"Elevated: {elevated_count:4d}, "
                  f"Critical: {critical_count:4d}")

def main():
    """Main function to generate ultra-smooth telemetry data"""
    # Initialize simulator
    simulator = ServerFailureSimulator(
        start_time=datetime.now() - timedelta(hours=4),
        interval_ms=300
    )
    
    # Generate smooth telemetry data
    df = simulator.generate_telemetry_data(total_points=5000, num_failures=18)
    
    # Ensure "data" folder exists
    os.makedirs("data", exist_ok=True)
    
    # Split dataset (70% train, 15% eval, 15% test)
    train_size = int(len(df) * 0.7)
    eval_size = int(len(df) * 0.15)
    
    train_df = df.iloc[:train_size]
    eval_df = df.iloc[train_size:train_size + eval_size]
    test_df = df.iloc[train_size + eval_size:]
    
    # Save splits
    simulator.save_to_csv(train_df, "data/train.csv")
    simulator.save_to_csv(eval_df, "data/eval.csv")
    simulator.save_to_csv(test_df, "data/test.csv")
    
    print("\n✅ Files saved: data/train.csv, data/eval.csv, data/test.csv")
    
    # Plot full dataset
    print("\nPlotting full ultra-smooth telemetry dataset...")
    simulator.plot_telemetry(df)
    
    # Example: Plot specific time range (last 60 minutes)
    end_time = df['timestamp'].max()
    start_time = end_time - timedelta(minutes=14)
    
    print(f"\nPlotting smooth telemetry data from {start_time} to {end_time}...")
    simulator.plot_telemetry(df, start_time, end_time)

if __name__ == "__main__":
    main()
    thresholds = {
    "cpu": 85,
    "memory": 85,
    "disk_io": 85,
    "network": 85,
    "temperature": 85
    }   
    # plot_data(file_path="./data/train.csv", thresholds=thresholds)
    plot_data(file_path="ultra_smooth_server_telemetry.csv", thresholds=thresholds)