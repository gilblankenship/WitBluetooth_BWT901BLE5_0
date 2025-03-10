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
        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

def load_sensor_data(file_path):
    """Load sensor data from CSV file"""
    print(f"Loading data from: {file_path}")
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Check if Timestamp column exists
    if 'Timestamp' in df.columns:
        # Convert timestamp strings to datetime objects
        df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)
        
        # Create a time delta column in seconds from the start
        start_time = df['Timestamp'].min()
        df['Time'] = (df['Timestamp'] - start_time).dt.total_seconds()
    else:
        # If no timestamp, use index as time
        df['Time'] = np.arange(len(df))
    
    return df

def plot_sensor_data(df, output_dir=None):
    """Plot sensor data in logical groups"""
    if df.empty:
        print("No data to plot")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set figure size and style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (12, 6)
    
    # 1. Acceleration Data (AccX, AccY, AccZ)
    if 'AccX' in df.columns and 'AccY' in df.columns and 'AccZ' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AccX'], label='AccX')
        plt.plot(df['Time'], df['AccY'], label='AccY')
        plt.plot(df['Time'], df['AccZ'], label='AccZ')
        plt.title('Acceleration Data')
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
    if 'AsX' in df.columns and 'AsY' in df.columns and 'AsZ' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AsX'], label='AsX')
        plt.plot(df['Time'], df['AsY'], label='AsY')
        plt.plot(df['Time'], df['AsZ'], label='AsZ')
        plt.title('Angular Speed Data')
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
    if 'AngX' in df.columns and 'AngY' in df.columns and 'AngZ' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['AngX'], label='AngX')
        plt.plot(df['Time'], df['AngY'], label='AngY')
        plt.plot(df['Time'], df['AngZ'], label='AngZ')
        plt.title('Angle/Orientation Data')
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
    if 'HX' in df.columns and 'HY' in df.columns and 'HZ' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['HX'], label='HX')
        plt.plot(df['Time'], df['HY'], label='HY')
        plt.plot(df['Time'], df['HZ'], label='HZ')
        plt.title('Magnetic Field Data')
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
    if 'Q0' in df.columns and 'Q1' in df.columns and 'Q2' in df.columns and 'Q3' in df.columns:
        plt.figure(figsize=fig_size)
        plt.plot(df['Time'], df['Q0'], label='Q0')
        plt.plot(df['Time'], df['Q1'], label='Q1')
        plt.plot(df['Time'], df['Q2'], label='Q2')
        plt.plot(df['Time'], df['Q3'], label='Q3')
        plt.title('Quaternion Data')
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
    if 'AccX' in df.columns and 'AccY' in df.columns and 'AccZ' in df.columns:
        plt.figure(figsize=fig_size)
        acc_magnitude = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
        plt.plot(df['Time'], acc_magnitude)
        plt.title('Acceleration Magnitude')
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
    if 'AccX' in df.columns and 'AccY' in df.columns and 'AccZ' in df.columns:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(df['AccX'], df['AccY'], df['AccZ'], label='Acceleration Trajectory')
        ax.set_xlabel('AccX (g)')
        ax.set_ylabel('AccY (g)')
        ax.set_zlabel('AccZ (g)')
        ax.set_title('3D Acceleration Trajectory')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acc_trajectory_3d.png'))
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot sensor data from CSV files')
    parser.add_argument('--file', type=str, help='Path to a specific CSV file')
    parser.add_argument('--dir', type=str, default='sensor_data', help='Directory containing CSV files')
    parser.add_argument('--output', type=str, help='Directory to save plots')
    parser.add_argument('--latest', action='store_true', help='Plot only the latest CSV file')
    
    args = parser.parse_args()
    
    # If a specific file is provided, use it
    if args.file and os.path.exists(args.file):
        df = load_sensor_data(args.file)
        plot_sensor_data(df, args.output)
    
    # Otherwise, look in the specified directory
    elif os.path.isdir(args.dir):
        csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {args.dir}")
            return
        
        if args.latest:
            # Find the most recent file
            latest_file = max(csv_files, key=os.path.getctime)
            print(f"Using latest file: {latest_file}")
            df = load_sensor_data(latest_file)
            plot_sensor_data(df, args.output)
        else:
            # Process all files
            for file_path in csv_files:
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