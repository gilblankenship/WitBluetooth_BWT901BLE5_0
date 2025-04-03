import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from datetime import datetime
import os
import re

def load_imu_data(file_path):
    """Load IMU data from a file and return a pandas DataFrame"""
    # Read the data with tab separator and proper column names
    column_names = [
        'time', 'DeviceName', 'AccX(g)', 'AccY(g)', 'AccZ(g)', 
        'AsX(°/s)', 'AsY(°/s)', 'AsZ(°/s)',
        'AngleX(°)', 'AngleY(°)', 'AngleZ(°)',
        'HX(uT)', 'HY(uT)', 'HZ(uT)',
        'Q0()', 'Q1()', 'Q2()', 'Q3()',
        'Temperature(°C)', 'Version()', 'Battery level(%)'
    ]
    
    df = pd.read_csv(file_path, sep='\t', names=column_names)
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['time'])
    
    # Calculate acceleration magnitude
    df['AccMag'] = np.sqrt(df['AccX(g)']**2 + df['AccY(g)']**2 + df['AccZ(g)']**2)
    
    return df

def split_by_device(df):
    """Split data by device name and return a dictionary of dataframes"""
    unique_devices = df['DeviceName'].unique()
    device_data = {}
    
    for device in unique_devices:
        device_data[device] = df[df['DeviceName'] == device].copy()
        device_data[device].sort_values('timestamp', inplace=True)
        device_data[device].reset_index(drop=True, inplace=True)
        
    return device_data

def detect_steps(device_df, window_size=5, threshold_factor=0.5, min_distance=5):
    """Detect steps based on acceleration magnitude peaks"""
    # Apply moving average to smooth the data
    device_df['AccMag_smooth'] = device_df['AccMag'].rolling(window=window_size, center=True).mean()
    device_df['AccMag_smooth'].fillna(device_df['AccMag'], inplace=True)
    
    # Calculate threshold based on mean and standard deviation
    mean_acc = device_df['AccMag_smooth'].mean()
    std_acc = device_df['AccMag_smooth'].std()
    threshold = mean_acc + threshold_factor * std_acc
    
    # Find peaks using scipy's find_peaks
    peaks, _ = find_peaks(device_df['AccMag_smooth'], height=threshold, distance=min_distance)
    
    return peaks

def analyze_step_consistency(peaks, timestamps):
    """Analyze consistency of steps"""
    if len(peaks) <= 1:
        return None
    
    # Calculate time differences between consecutive steps
    time_diffs = np.diff(timestamps[peaks].astype(np.int64)) / 1e6  # Convert to milliseconds
    
    mean_gap = time_diffs.mean()
    std_dev = time_diffs.std()
    variability_coef = (std_dev / mean_gap) * 100 if mean_gap > 0 else 0
    
    return {
        'mean_time_between_steps_ms': mean_gap,
        'std_time_between_steps_ms': std_dev,
        'variability_coefficient': variability_coef,
        'time_diffs': time_diffs
    }

def analyze_angle_changes(device_df, peaks, look_back=5):
    """Analyze angle changes during steps"""
    if len(peaks) <= 1:
        return None
    
    angle_changes = []
    
    for peak in peaks:
        prev_idx = max(0, peak - look_back)
        
        # Calculate angle changes from before the step to the step
        angle_x_change = device_df.iloc[peak]['AngleX(°)'] - device_df.iloc[prev_idx]['AngleX(°)']
        angle_y_change = device_df.iloc[peak]['AngleY(°)'] - device_df.iloc[prev_idx]['AngleY(°)']
        angle_z_change = device_df.iloc[peak]['AngleZ(°)'] - device_df.iloc[prev_idx]['AngleZ(°)']
        
        angle_changes.append({
            'X': angle_x_change,
            'Y': angle_y_change,
            'Z': angle_z_change
        })
    
    # Calculate means
    mean_x = np.mean([change['X'] for change in angle_changes])
    mean_y = np.mean([change['Y'] for change in angle_changes])
    mean_z = np.mean([change['Z'] for change in angle_changes])
    
    return {
        'mean_angle_changes': {'X': mean_x, 'Y': mean_y, 'Z': mean_z},
        'angle_changes': angle_changes
    }

def match_steps_between_devices(device1_df, device2_df, peaks1, peaks2, max_time_gap_ms=1000):
    """Match steps between two devices based on timestamps"""
    timestamps1 = device1_df['timestamp'].iloc[peaks1].values
    timestamps2 = device2_df['timestamp'].iloc[peaks2].values
    
    matched_pairs = []
    
    for i, t1 in enumerate(timestamps1):
        best_match = None
        min_diff = max_time_gap_ms * 1e6  # Convert to nanoseconds
        
        for j, t2 in enumerate(timestamps2):
            time_diff = abs(t1.astype(np.int64) - t2.astype(np.int64))
            
            if time_diff < min_diff:
                min_diff = time_diff
                best_match = j
        
        if best_match is not None and min_diff < max_time_gap_ms * 1e6:
            time_diff_ms = (t1.astype(np.int64) - timestamps2[best_match].astype(np.int64)) / 1e6
            matched_pairs.append({
                'device1_peak': i,
                'device2_peak': best_match,
                'time_diff_ms': time_diff_ms
            })
    
    return matched_pairs

def analyze_leg_differences(device1_df, device2_df, peaks1, peaks2, matched_pairs, angle_analysis1, angle_analysis2):
    """Analyze differences between left and right leg movements"""
    # Compare acceleration magnitudes
    mean_acc1 = device1_df['AccMag'].mean()
    mean_acc2 = device2_df['AccMag'].mean()
    acc_diff_pct = abs((mean_acc1 - mean_acc2) / ((mean_acc1 + mean_acc2) / 2)) * 100
    
    # Compare step counts
    step_count_diff_pct = abs((len(peaks1) - len(peaks2)) / ((len(peaks1) + len(peaks2)) / 2)) * 100
    
    # Compare angle changes
    mean_angle_x1 = abs(angle_analysis1['mean_angle_changes']['X'])
    mean_angle_x2 = abs(angle_analysis2['mean_angle_changes']['X'])
    angle_x_diff_pct = abs((mean_angle_x1 - mean_angle_x2) / ((mean_angle_x1 + mean_angle_x2) / 2)) * 100
    
    # Analyze which leg typically moves first
    first_leg_counts = {'device1': 0, 'device2': 0}
    
    for pair in matched_pairs:
        if pair['time_diff_ms'] < 0:  # device1 before device2
            first_leg_counts['device1'] += 1
        else:
            first_leg_counts['device2'] += 1
    
    dominant_leg = 'device1' if first_leg_counts['device1'] > first_leg_counts['device2'] else 'device2'
    dominance_pct = (max(first_leg_counts['device1'], first_leg_counts['device2']) / len(matched_pairs)) * 100 if matched_pairs else 0
    
    return {
        'acceleration_diff_pct': acc_diff_pct,
        'step_count_diff_pct': step_count_diff_pct,
        'angle_x_diff_pct': angle_x_diff_pct,
        'dominant_leg': dominant_leg,
        'leg_dominance_pct': dominance_pct,
        'higher_acc_device': 'device1' if mean_acc1 > mean_acc2 else 'device2',
        'more_steps_device': 'device1' if len(peaks1) > len(peaks2) else 'device2' if len(peaks2) > len(peaks1) else 'equal',
        'larger_angle_device': 'device1' if mean_angle_x1 > mean_angle_x2 else 'device2'
    }

def main(file_path):
    """Main function to run the full analysis pipeline"""
    print(f"Loading IMU data from: {file_path}")
    df = load_imu_data(file_path)
    print(f"Loaded {len(df)} data points.")
    
    # Split data by device
    print("Splitting data by device...")
    device_data = split_by_device(df)
    device_names = list(device_data.keys())
    
    if len(device_names) != 2:
        print(f"Warning: Expected 2 devices, but found {len(device_names)}")
    
    # For plotting and storage, shorten device names by extracting the ID part
    short_names = {}
    for name in device_names:
        match = re.search(r'\((.*?)\)', name)
        if match:
            short_id = match.group(1)[-4:]  # Take last 4 chars of the ID
            short_names[name] = f"Device {short_id}"
        else:
            short_names[name] = name
    
    # Prepare data structures for results
    peaks_data = {}
    consistency_data = {}
    angle_analysis_data = {}
    
    # Process each device
    print("Processing each device...")
    for device_name, device_df in device_data.items():
        print(f"  Detecting steps for {short_names[device_name]}...")
        peaks = detect_steps(device_df, window_size=5, threshold_factor=0.5, min_distance=5)
        peaks_data[device_name] = peaks
        
        print(f"    Detected {len(peaks)} steps.")
        
        print(f"  Analyzing step consistency for {short_names[device_name]}...")
        consistency = analyze_step_consistency(peaks, device_df['timestamp'].values)
        consistency_data[device_name] = consistency
        
        print(f"  Analyzing angle changes for {short_names[device_name]}...")
        angle_analysis = analyze_angle_changes(device_df, peaks)
        angle_analysis_data[device_name] = angle_analysis
    
    # Match steps between devices
    print("Matching steps between devices...")
    matched_pairs = []
    
    if len(device_names) >= 2:
        device1 = device_names[0]
        device2 = device_names[1]
        
        matched_pairs = match_steps_between_devices(
            device_data[device1], 
            device_data[device2], 
            peaks_data[device1], 
            peaks_data[device2]
        )
        
        print(f"Found {len(matched_pairs)} matching step pairs.")
    
    # Analyze leg differences
    print("Analyzing leg differences...")
    leg_diff_analysis = None
    stair_direction = "unknown"
    
    if len(device_names) >= 2:
        device1 = device_names[0]
        device2 = device_names[1]
        
        leg_diff_analysis = analyze_leg_differences(
            device_data[device1], 
            device_data[device2], 
            peaks_data[device1], 
            peaks_data[device2], 
            matched_pairs,
            angle_analysis_data[device1],
            angle_analysis_data[device2]
        )
        
        stair_direction = determine_stair_direction(
            angle_analysis_data[device1],
            angle_analysis_data[device2]
        )
        
        print(f"Determined stair direction: {stair_direction}")
    
    # Create plots
    print("Creating plots...")
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot acceleration and detected steps
    acc_fig = plot_acceleration_and_steps(device_data, peaks_data)
    acc_fig.savefig(os.path.join(output_dir, 'acceleration_steps.png'))
    
    # Plot angles
    angles_fig = plot_angles(device_data, peaks_data)
    angles_fig.savefig(os.path.join(output_dir, 'angles.png'))
    
    # Plot step timing
    timing_fig = plot_step_timing(device_data, peaks_data, matched_pairs)
    timing_fig.savefig(os.path.join(output_dir, 'step_timing.png'))
    
    # Plot angle changes
    if len(device_names) >= 2:
        angle_changes_fig = plot_angle_changes_at_steps(
            angle_analysis_data[device_names[0]],
            angle_analysis_data[device_names[1]],
            [short_names[device_names[0]], short_names[device_names[1]]]
        )
        angle_changes_fig.savefig(os.path.join(output_dir, 'angle_changes.png'))
    
    # Generate report
    print("Generating report...")
    report = create_report(
        df, device_data, peaks_data, consistency_data, angle_analysis_data,
        matched_pairs, leg_diff_analysis, stair_direction
    )
    
    with open(os.path.join(output_dir, 'report.md'), 'w') as f:
        f.write(report)
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    
    # Return results for further use if needed
    return {
        'df': df,
        'device_data': device_data,
        'peaks_data': peaks_data,
        'consistency_data': consistency_data,
        'angle_analysis_data': angle_analysis_data,
        'matched_pairs': matched_pairs,
        'leg_diff_analysis': leg_diff_analysis,
        'stair_direction': stair_direction,
        'report': report
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "paste.txt"  # Default filename
    
    main(file_path)

def determine_stair_direction(angle_analysis1, angle_analysis2):
    """Determine if the person is going up or down stairs based on angle patterns"""
    # For stair climbing, we expect significant forward/upward angle changes
    mean_x1 = angle_analysis1['mean_angle_changes']['X']
    mean_x2 = angle_analysis2['mean_angle_changes']['X']
    
    # Positive angle changes in X often indicate going up, negative going down
    # This is a simplification and might need adjustment based on sensor orientation
    combined_x_change = (mean_x1 + mean_x2) / 2
    
    if abs(combined_x_change) < 2:
        return "unknown (insufficient angle change)"
    
    return "up" if combined_x_change > 0 else "down"

def create_report(df, device_data, peaks_data, consistency_data, angle_analysis_data, 
                 matched_pairs, leg_diff_analysis, stair_direction):
    """Generate a detailed report of the analysis"""
    device_names = list(device_data.keys())
    
    report = []
    report.append("# IMU Stair Movement Analysis Report")
    report.append("\n## Dataset Overview")
    report.append(f"- Total data points: {len(df)}")
    report.append(f"- Time period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    report.append(f"- Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.2f} seconds")
    report.append(f"- Number of devices: {len(device_names)}")
    
    for i, device_name in enumerate(device_names):
        report.append(f"\n### Device {i+1}: {device_name}")
        report.append(f"- Number of data points: {len(device_data[device_name])}")
        report.append(f"- Number of detected steps: {len(peaks_data[device_name])}")
        
        if consistency_data[device_name]:
            report.append("\n#### Step Consistency")
            report.append(f"- Mean time between steps: {consistency_data[device_name]['mean_time_between_steps_ms']:.2f} ms")
            report.append(f"- Standard deviation: {consistency_data[device_name]['std_time_between_steps_ms']:.2f} ms")
            report.append(f"- Variability coefficient: {consistency_data[device_name]['variability_coefficient']:.2f}%")
        
        if angle_analysis_data[device_name]:
            report.append("\n#### Angle Changes During Steps")
            report.append(f"- Mean X angle change: {angle_analysis_data[device_name]['mean_angle_changes']['X']:.2f}°")
            report.append(f"- Mean Y angle change: {angle_analysis_data[device_name]['mean_angle_changes']['Y']:.2f}°")
            report.append(f"- Mean Z angle change: {angle_analysis_data[device_name]['mean_angle_changes']['Z']:.2f}°")
    
    report.append("\n## Step Matching Analysis")
    report.append(f"- Number of matched steps between devices: {len(matched_pairs)}")
    
    if matched_pairs:
        time_diffs = [pair['time_diff_ms'] for pair in matched_pairs]
        mean_diff = np.mean(time_diffs)
        median_diff = np.median(time_diffs)
        
        report.append(f"- Mean time difference: {mean_diff:.2f} ms")
        report.append(f"- Median time difference: {median_diff:.2f} ms")
        report.append(f"- Alternating leg pattern: {'Yes' if abs(median_diff) > 200 else 'No'}")
    
    report.append("\n## Left-Right Leg Differences")
    report.append(f"- Acceleration magnitude difference: {leg_diff_analysis['acceleration_diff_pct']:.2f}%")
    report.append(f"- Step count difference: {leg_diff_analysis['step_count_diff_pct']:.2f}%")
    report.append(f"- X angle change difference: {leg_diff_analysis['angle_x_diff_pct']:.2f}%")
    
    if matched_pairs:
        dom_leg_idx = 0 if leg_diff_analysis['dominant_leg'] == 'device1' else 1
        report.append(f"- Dominant leg (moves first): {device_names[dom_leg_idx]} ({leg_diff_analysis['leg_dominance_pct']:.2f}% of matched steps)")
    
    higher_acc_idx = 0 if leg_diff_analysis['higher_acc_device'] == 'device1' else 1
    report.append(f"- Device with higher acceleration: {device_names[higher_acc_idx]}")
    
    if leg_diff_analysis['more_steps_device'] != 'equal':
        more_steps_idx = 0 if leg_diff_analysis['more_steps_device'] == 'device1' else 1
        report.append(f"- Device with more steps: {device_names[more_steps_idx]}")
    
    larger_angle_idx = 0 if leg_diff_analysis['larger_angle_device'] == 'device1' else 1
    report.append(f"- Device with larger angle changes: {device_names[larger_angle_idx]}")
    
    report.append("\n## Stair Movement Analysis")
    report.append(f"- Determined stair direction: {stair_direction}")
    
    if stair_direction == "up":
        report.append("- The positive angle changes indicate the person is lifting their legs higher, consistent with climbing up stairs.")
    elif stair_direction == "down":
        report.append("- The negative angle changes indicate the person is lowering their legs, consistent with descending stairs.")
    
    return '\n'.join(report)

def plot_acceleration_and_steps(device_data, peaks_data, title_prefix=""):
    """Create plots for acceleration magnitude and detected steps"""
    plt.figure(figsize=(15, 10))
    
    # Plot for each device
    for i, (device_name, device_df) in enumerate(device_data.items()):
        plt.subplot(2, 1, i+1)
        
        # Convert timestamps to relative seconds from the start
        start_time = device_df['timestamp'].iloc[0]
        relative_times = [(t - start_time).total_seconds() for t in device_df['timestamp']]
        
        # Plot acceleration magnitude
        plt.plot(relative_times, device_df['AccMag'], label='Acceleration Magnitude', alpha=0.5)
        plt.plot(relative_times, device_df['AccMag_smooth'], label='Smoothed Acceleration', color='blue')
        
        # Plot detected steps
        peaks = peaks_data[device_name]
        peak_times = [relative_times[p] for p in peaks]
        peak_values = [device_df['AccMag_smooth'].iloc[p] for p in peaks]
        plt.scatter(peak_times, peak_values, color='red', label=f'Detected Steps ({len(peaks)})')
        
        # Plot threshold
        mean_acc = device_df['AccMag_smooth'].mean()
        std_acc = device_df['AccMag_smooth'].std()
        threshold = mean_acc + 0.5 * std_acc
        plt.axhline(y=threshold, color='green', linestyle='--', label=f'Peak Threshold ({threshold:.2f})')
        
        plt.title(f'{title_prefix} {device_name} - Acceleration and Steps')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration Magnitude (g)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def plot_angles(device_data, peaks_data, title_prefix=""):
    """Create plots for angles and detected steps"""
    plt.figure(figsize=(15, 15))
    
    # Plot for each device
    for i, (device_name, device_df) in enumerate(device_data.items()):
        # Convert timestamps to relative seconds from the start
        start_time = device_df['timestamp'].iloc[0]
        relative_times = [(t - start_time).total_seconds() for t in device_df['timestamp']]
        
        # Plot X angle
        plt.subplot(3, 2, i*3+1)
        plt.plot(relative_times, device_df['AngleX(°)'], label='X Angle (°)')
        
        # Plot detected steps
        peaks = peaks_data[device_name]
        peak_times = [relative_times[p] for p in peaks]
        peak_values = [device_df['AngleX(°)'].iloc[p] for p in peaks]
        plt.scatter(peak_times, peak_values, color='red', label=f'Steps')
        
        plt.title(f'{device_name} - X Angle (Pitch)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Y angle
        plt.subplot(3, 2, i*3+2)
        plt.plot(relative_times, device_df['AngleY(°)'], label='Y Angle (°)')
        peak_values = [device_df['AngleY(°)'].iloc[p] for p in peaks]
        plt.scatter(peak_times, peak_values, color='red', label=f'Steps')
        
        plt.title(f'{device_name} - Y Angle (Roll)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Z angle
        plt.subplot(3, 2, i*3+3)
        plt.plot(relative_times, device_df['AngleZ(°)'], label='Z Angle (°)')
        peak_values = [device_df['AngleZ(°)'].iloc[p] for p in peaks]
        plt.scatter(peak_times, peak_values, color='red', label=f'Steps')
        
        plt.title(f'{device_name} - Z Angle (Yaw)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()

def plot_step_timing(device_data, peaks_data, matched_pairs=None, title_prefix=""):
    """Create a plot showing step timing for both devices"""
    plt.figure(figsize=(15, 6))
    
    device_names = list(device_data.keys())
    colors = ['blue', 'red']
    
    # Plot timeline of steps for each device
    for i, device_name in enumerate(device_names):
        device_df = device_data[device_name]
        peaks = peaks_data[device_name]
        
        # Convert timestamps to relative seconds from the start of the entire dataset
        all_start_time = min([df['timestamp'].iloc[0] for df in device_data.values()])
        step_times = [(device_df['timestamp'].iloc[p] - all_start_time).total_seconds() for p in peaks]
        
        # Plot steps as vertical lines
        for step_time in step_times:
            plt.axvline(x=step_time, color=colors[i], alpha=0.5, linewidth=2)
        
        # Add scatter points for better visibility
        plt.scatter(step_times, [i+1] * len(step_times), color=colors[i], s=100, 
                   label=f'{device_name} Steps ({len(peaks)})')
    
    # If we have matched pairs, connect them with lines
    if matched_pairs:
        device1_df = device_data[device_names[0]]
        device2_df = device_data[device_names[1]]
        all_start_time = min([df['timestamp'].iloc[0] for df in device_data.values()])
        
        for pair in matched_pairs:
            peak1_idx = peaks_data[device_names[0]][pair['device1_peak']]
            peak2_idx = peaks_data[device_names[1]][pair['device2_peak']]
            
            time1 = (device1_df['timestamp'].iloc[peak1_idx] - all_start_time).total_seconds()
            time2 = (device2_df['timestamp'].iloc[peak2_idx] - all_start_time).total_seconds()
            
            plt.plot([time1, time2], [1, 2], 'k-', alpha=0.3)
    
    plt.title(f'{title_prefix} Step Timing Comparison')
    plt.xlabel('Time (seconds)')
    plt.yticks([1, 2], device_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()

def plot_angle_changes_at_steps(angle_analysis1, angle_analysis2, device_names, title_prefix=""):
    """Create bar charts showing angle changes during steps for both devices"""
    plt.figure(figsize=(15, 8))
    
    # Extract all angle changes for each axis and each device
    device1_angles = np.array([[change['X'], change['Y'], change['Z']] 
                              for change in angle_analysis1['angle_changes']])
    device2_angles = np.array([[change['X'], change['Y'], change['Z']] 
                              for change in angle_analysis2['angle_changes']])
    
    # Define positions for grouped bars
    bar_width = 0.35
    r1 = np.arange(3)  # 3 angles (X, Y, Z)
    r2 = [x + bar_width for x in r1]
    
    # Create bar plots
    plt.bar(r1, [np.mean(device1_angles[:, 0]), np.mean(device1_angles[:, 1]), np.mean(device1_angles[:, 2])], 
            width=bar_width, color='blue', yerr=[np.std(device1_angles[:, 0]), np.std(device1_angles[:, 1]), np.std(device1_angles[:, 2])],
            label=device_names[0], capsize=5)
    
    plt.bar(r2, [np.mean(device2_angles[:, 0]), np.mean(device2_angles[:, 1]), np.mean(device2_angles[:, 2])], 
            width=bar_width, color='red', yerr=[np.std(device2_angles[:, 0]), np.std(device2_angles[:, 1]), np.std(device2_angles[:, 2])],
            label=device_names[1], capsize=5)
    
    # Add labels and title
    plt.xlabel('Angle Axis')
    plt.ylabel('Mean Angle Change (°)')
    plt.title(f'{title_prefix} Angle Changes During Steps')
    plt.xticks([r + bar_width/2 for r in range(3)], ['X (Pitch)', 'Y (Roll)', 'Z (Yaw)'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a horizontal line at zero for reference
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    return plt.gcf()