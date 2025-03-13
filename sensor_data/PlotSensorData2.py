import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse
from datetime import datetime

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Additional format for your data
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%H")

def load_sensor_data(file_path):
    """Load sensor data from CSV file"""
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path, sep='\t')
    
    # Check if time column exists (using your format)
    if 'time' in df.columns:
        # Convert timestamp strings to datetime objects
        df['time'] = df['time'].apply(parse_timestamp)
        
        # Create a time delta column in seconds from the start
        start_time = df['time'].min()
        df['Time'] = (df['time'] - start_time).dt.total_seconds()
    else:
        # If no timestamp, use index as time
        df['Time'] = np.arange(len(df))
    
    # Map column names to expected format
    column_mapping = {
        'AccX(g)': 'AccX',
        'AccY(g)': 'AccY',
        'AccZ(g)': 'AccZ',
        'AsX(°/s)': 'AsX',
        'AsY(°/s)': 'AsY', 
        'AsZ(°/s)': 'AsZ',
        'AngleX(°)': 'AngX',
        'AngleY(°)': 'AngY',
        'AngleZ(°)': 'AngZ',
        'HX(uT)': 'HX',
        'HY(uT)': 'HY',
        'HZ(uT)': 'HZ',
        'Q0()': 'Q0',
        'Q1()': 'Q1',
        'Q2()': 'Q2',
        'Q3()': 'Q3',
        'Temperature(°C)': 'Temperature',
        'Battery level(%)': 'Battery'
    }
    
    # Rename columns according to mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    return df

def plot_sensor_data(df, output_dir=None):
    """Plot sensor data in logical groups, with separate plots for each device"""
    if df.empty:
        print("No data to plot")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set figure size and style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (12, 6)
    
    # Check if DeviceName column exists and plot by device
    if 'DeviceName' in df.columns:
        devices = df['DeviceName'].unique()
        
        for device in devices:
            device_df = df[df['DeviceName'] == device]
            device_name = device  # Use full device name including ID
            
            print(f"Plotting data for device: {device_name}")
            
            # Create device-specific output directory - use sanitized name for folder
            device_folder_name = device.replace('(', '_').replace(')', '_').replace(':', '_')
            if output_dir:
                device_dir = os.path.join(output_dir, device_folder_name)
                os.makedirs(device_dir, exist_ok=True)
            else:
                device_dir = None
            
            # Plot data for this device
            plot_device_data(device_df, device_name, device_dir, fig_size)
    else:
        # If no DeviceName column, plot all data together
        plot_device_data(df, "All Devices", output_dir, fig_size)
        
def plot_device_data(df, device_name, output_dir, fig_size):
    """Plot sensor data for a specific device"""
    
    # 1. Acceleration Data (AccX, AccY, AccZ)
    if all(col in df.columns for col in ['AccX', 'AccY', 'AccZ']):
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AccX'], label='AccX')
        plt.plot(df['Time'], df['AccY'], label='AccY')
        plt.plot(df['Time'], df['AccZ'], label='AccZ')
        plt.title(f'Acceleration Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration (g)')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acceleration.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 2. Angular Speed (AsX, AsY, AsZ)
    if all(col in df.columns for col in ['AsX', 'AsY', 'AsZ']):
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AsX'], label='AsX')
        plt.plot(df['Time'], df['AsY'], label='AsY')
        plt.plot(df['Time'], df['AsZ'], label='AsZ')
        plt.title(f'Angular Speed Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angular Speed (°/s)')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angular_speed.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 3. Angle/Orientation (AngX, AngY, AngZ)
    if all(col in df.columns for col in ['AngX', 'AngY', 'AngZ']):
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AngX'], label='AngX')
        plt.plot(df['Time'], df['AngY'], label='AngY')
        plt.plot(df['Time'], df['AngZ'], label='AngZ')
        plt.title(f'Angle/Orientation Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (°)')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angle_orientation.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 4. Magnetic Field (HX, HY, HZ)
    if all(col in df.columns for col in ['HX', 'HY', 'HZ']):
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['HX'], label='HX')
        plt.plot(df['Time'], df['HY'], label='HY')
        plt.plot(df['Time'], df['HZ'], label='HZ')
        plt.title(f'Magnetic Field Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnetic Field (μT)')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'magnetic_field.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 5. Quaternion Data (Q0, Q1, Q2, Q3)
    if all(col in df.columns for col in ['Q0', 'Q1', 'Q2', 'Q3']):
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['Q0'], label='Q0')
        plt.plot(df['Time'], df['Q1'], label='Q1')
        plt.plot(df['Time'], df['Q2'], label='Q2')
        plt.plot(df['Time'], df['Q3'], label='Q3')
        plt.title(f'Quaternion Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Quaternion Value')
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'quaternion.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    
    # 6. Combined Acceleration Magnitude
    if all(col in df.columns for col in ['AccX', 'AccY', 'AccZ']):
        plt.figure(figsize=fig_size)
        acc_magnitude = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
        plt.plot(df['Time'], acc_magnitude)
        plt.title(f'Acceleration Magnitude - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Magnitude (g)')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acc_magnitude.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    # 7. 3D Acceleration Trajectory
    if all(col in df.columns for col in ['AccX', 'AccY', 'AccZ']):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['AccX'], df['AccY'], df['AccZ'], label='Acceleration Trajectory')
        ax.set_xlabel('AccX (g)')
        ax.set_ylabel('AccY (g)')
        ax.set_zlabel('AccZ (g)')
        ax.set_title(f'3D Acceleration Trajectory - {device_name}')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acc_trajectory_3d.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    # 8. Temperature Data (if available)
    if 'Temperature' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['Temperature'])
        plt.title(f'Temperature Data - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'temperature.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    # 9. Battery Level (if available)
    if 'Battery' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['Battery'])
        plt.title(f'Battery Level - {device_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Battery Level (%)')
        plt.grid(True)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'battery.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot sensor data from CSV or TSV files')
    parser.add_argument('--file', type=str, help='Path to a specific data file')
    parser.add_argument('--dir', type=str, default='sensor_data', help='Directory containing data files')
    parser.add_argument('--output', type=str, help='Directory to save plots')
    parser.add_argument('--latest', action='store_true', help='Plot only the latest data file')
    parser.add_argument('--format', type=str, default='auto', choices=['auto', 'csv', 'tsv'], 
                        help='Force a specific file format (auto detects by default)')
    
    args = parser.parse_args()
    
    # If a specific file is provided, use it
    if args.file and os.path.exists(args.file):
        df = load_sensor_data(args.file)
        plot_sensor_data(df, args.output)
    
    # Otherwise, look in the specified directory
    elif os.path.isdir(args.dir):
        data_files = []
        
        if args.format == 'auto' or args.format == 'csv':
            data_files.extend(glob.glob(os.path.join(args.dir, '*.csv')))
        
        if args.format == 'auto' or args.format == 'tsv':
            data_files.extend(glob.glob(os.path.join(args.dir, '*.tsv')))
            data_files.extend(glob.glob(os.path.join(args.dir, '*.txt')))
        
        if not data_files:
            print(f"No data files found in {args.dir}")
            return
        
        if args.latest:
            # Find the most recent file
            latest_file = max(data_files, key=os.path.getctime)
            print(f"Using latest file: {latest_file}")
            df = load_sensor_data(latest_file)
            plot_sensor_data(df, args.output)
        else:
            # Process all files
            for file_path in data_files:
                df = load_sensor_data(file_path)
                
                # Create subdirectory with file name without extension
                file_name = os.path.splitext(os.path.basename(file_path))[0]
                if args.output:
                    file_output_dir = os.path.join(args.output, file_name)
                else:
                    file_output_dir = None
                
                plot_sensor_data(df, file_output_dir)
    
    else:
        print("Please provide a valid file path or directory")

if __name__ == "__main__":
    main()