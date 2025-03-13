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
    """Plot sensor data in logical groups, with paired subplots for left and right legs"""
    if df.empty:
        print("No data to plot")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set figure size and style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_size = (18, 10)  # Larger figure size for side-by-side plots
    
    # Check if DeviceName column exists
    if 'DeviceName' in df.columns:
        devices = df['DeviceName'].unique()
        
        if len(devices) >= 2:
            # Identify which device is left leg and which is right leg
            # For now, we'll assume first device is left leg and second is right leg
            # You might want to add specific identification based on device ID
            left_leg_device = devices[0]
            right_leg_device = devices[1]
            
            print(f"Left leg device: {left_leg_device}")
            print(f"Right leg device: {right_leg_device}")
            
            left_leg_df = df[df['DeviceName'] == left_leg_device]
            right_leg_df = df[df['DeviceName'] == right_leg_device]
            
            # Plot paired data for both legs
            plot_paired_leg_data(left_leg_df, right_leg_df, left_leg_device, right_leg_device, output_dir, fig_size)
        else:
            print("Warning: Less than two devices found. Cannot create left-right leg comparison.")
            # Plot individual devices if we can't pair them
            for device in devices:
                device_df = df[df['DeviceName'] == device]
                device_name = device
                
                print(f"Plotting data for device: {device_name}")
                
                # Create device-specific output directory
                device_folder_name = device.replace('(', '_').replace(')', '_').replace(':', '_')
                if output_dir:
                    device_dir = os.path.join(output_dir, device_folder_name)
                    os.makedirs(device_dir, exist_ok=True)
                else:
                    device_dir = None
                
                plot_device_data(device_df, device_name, device_dir, fig_size)
    else:
        # If no DeviceName column, plot all data together
        plot_device_data(df, "All Devices", output_dir, fig_size)
        
def plot_paired_leg_data(left_leg_df, right_leg_df, left_leg_device, right_leg_device, output_dir, fig_size):
    """Plot sensor data comparing left and right legs in subplots"""
    
    # 1. Acceleration Data (AccX, AccY, AccZ)
    if all(col in left_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']) and \
       all(col in right_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']):
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Acceleration Data - Left vs Right Leg', fontsize=16)
        
        # Left leg - X, Y, Z
        axs[0, 0].plot(left_leg_df['Time'], left_leg_df['AccX'])
        axs[0, 0].set_title(f'Left Leg - AccX - {left_leg_device}')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Acceleration (g)')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(left_leg_df['Time'], left_leg_df['AccY'])
        axs[0, 1].set_title(f'Left Leg - AccY - {left_leg_device}')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Acceleration (g)')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(left_leg_df['Time'], left_leg_df['AccZ'])
        axs[0, 2].set_title(f'Left Leg - AccZ - {left_leg_device}')
        axs[0, 2].set_xlabel('Time (seconds)')
        axs[0, 2].set_ylabel('Acceleration (g)')
        axs[0, 2].grid(True)
        
        # Right leg - X, Y, Z
        axs[1, 0].plot(right_leg_df['Time'], right_leg_df['AccX'])
        axs[1, 0].set_title(f'Right Leg - AccX - {right_leg_device}')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('Acceleration (g)')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(right_leg_df['Time'], right_leg_df['AccY'])
        axs[1, 1].set_title(f'Right Leg - AccY - {right_leg_device}')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Acceleration (g)')
        axs[1, 1].grid(True)
        
        axs[1, 2].plot(right_leg_df['Time'], right_leg_df['AccZ'])
        axs[1, 2].set_title(f'Right Leg - AccZ - {right_leg_device}')
        axs[1, 2].set_xlabel('Time (seconds)')
        axs[1, 2].set_ylabel('Acceleration (g)')
        axs[1, 2].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate the suptitle
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acceleration_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    # 2. Angular Speed (AsX, AsY, AsZ)
    if all(col in left_leg_df.columns for col in ['AsX', 'AsY', 'AsZ']) and \
       all(col in right_leg_df.columns for col in ['AsX', 'AsY', 'AsZ']):
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Angular Speed Data - Left vs Right Leg', fontsize=16)
        
        # Left leg - X, Y, Z
        axs[0, 0].plot(left_leg_df['Time'], left_leg_df['AsX'])
        axs[0, 0].set_title(f'Left Leg - AsX - {left_leg_device}')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Angular Speed (°/s)')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(left_leg_df['Time'], left_leg_df['AsY'])
        axs[0, 1].set_title(f'Left Leg - AsY - {left_leg_device}')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Angular Speed (°/s)')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(left_leg_df['Time'], left_leg_df['AsZ'])
        axs[0, 2].set_title(f'Left Leg - AsZ - {left_leg_device}')
        axs[0, 2].set_xlabel('Time (seconds)')
        axs[0, 2].set_ylabel('Angular Speed (°/s)')
        axs[0, 2].grid(True)
        
        # Right leg - X, Y, Z
        axs[1, 0].plot(right_leg_df['Time'], right_leg_df['AsX'])
        axs[1, 0].set_title(f'Right Leg - AsX - {right_leg_device}')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('Angular Speed (°/s)')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(right_leg_df['Time'], right_leg_df['AsY'])
        axs[1, 1].set_title(f'Right Leg - AsY - {right_leg_device}')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Angular Speed (°/s)')
        axs[1, 1].grid(True)
        
        axs[1, 2].plot(right_leg_df['Time'], right_leg_df['AsZ'])
        axs[1, 2].set_title(f'Right Leg - AsZ - {right_leg_device}')
        axs[1, 2].set_xlabel('Time (seconds)')
        axs[1, 2].set_ylabel('Angular Speed (°/s)')
        axs[1, 2].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angular_speed_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    # 3. Angle/Orientation (AngX, AngY, AngZ)
    if all(col in left_leg_df.columns for col in ['AngX', 'AngY', 'AngZ']) and \
       all(col in right_leg_df.columns for col in ['AngX', 'AngY', 'AngZ']):
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Angle/Orientation Data - Left vs Right Leg', fontsize=16)
        
        # Left leg - X, Y, Z
        axs[0, 0].plot(left_leg_df['Time'], left_leg_df['AngX'])
        axs[0, 0].set_title(f'Left Leg - AngX - {left_leg_device}')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Angle (°)')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(left_leg_df['Time'], left_leg_df['AngY'])
        axs[0, 1].set_title(f'Left Leg - AngY - {left_leg_device}')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Angle (°)')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(left_leg_df['Time'], left_leg_df['AngZ'])
        axs[0, 2].set_title(f'Left Leg - AngZ - {left_leg_device}')
        axs[0, 2].set_xlabel('Time (seconds)')
        axs[0, 2].set_ylabel('Angle (°)')
        axs[0, 2].grid(True)
        
        # Right leg - X, Y, Z
        axs[1, 0].plot(right_leg_df['Time'], right_leg_df['AngX'])
        axs[1, 0].set_title(f'Right Leg - AngX - {right_leg_device}')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('Angle (°)')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(right_leg_df['Time'], right_leg_df['AngY'])
        axs[1, 1].set_title(f'Right Leg - AngY - {right_leg_device}')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Angle (°)')
        axs[1, 1].grid(True)
        
        axs[1, 2].plot(right_leg_df['Time'], right_leg_df['AngZ'])
        axs[1, 2].set_title(f'Right Leg - AngZ - {right_leg_device}')
        axs[1, 2].set_xlabel('Time (seconds)')
        axs[1, 2].set_ylabel('Angle (°)')
        axs[1, 2].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angle_orientation_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    # 4. Magnetic Field (HX, HY, HZ)
    if all(col in left_leg_df.columns for col in ['HX', 'HY', 'HZ']) and \
       all(col in right_leg_df.columns for col in ['HX', 'HY', 'HZ']):
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Magnetic Field Data - Left vs Right Leg', fontsize=16)
        
        # Left leg - X, Y, Z
        axs[0, 0].plot(left_leg_df['Time'], left_leg_df['HX'])
        axs[0, 0].set_title(f'Left Leg - HX - {left_leg_device}')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Magnetic Field (μT)')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(left_leg_df['Time'], left_leg_df['HY'])
        axs[0, 1].set_title(f'Left Leg - HY - {left_leg_device}')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Magnetic Field (μT)')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(left_leg_df['Time'], left_leg_df['HZ'])
        axs[0, 2].set_title(f'Left Leg - HZ - {left_leg_device}')
        axs[0, 2].set_xlabel('Time (seconds)')
        axs[0, 2].set_ylabel('Magnetic Field (μT)')
        axs[0, 2].grid(True)
        
        # Right leg - X, Y, Z
        axs[1, 0].plot(right_leg_df['Time'], right_leg_df['HX'])
        axs[1, 0].set_title(f'Right Leg - HX - {right_leg_device}')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('Magnetic Field (μT)')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(right_leg_df['Time'], right_leg_df['HY'])
        axs[1, 1].set_title(f'Right Leg - HY - {right_leg_device}')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Magnetic Field (μT)')
        axs[1, 1].grid(True)
        
        axs[1, 2].plot(right_leg_df['Time'], right_leg_df['HZ'])
        axs[1, 2].set_title(f'Right Leg - HZ - {right_leg_device}')
        axs[1, 2].set_xlabel('Time (seconds)')
        axs[1, 2].set_ylabel('Magnetic Field (μT)')
        axs[1, 2].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'magnetic_field_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    # 5. Quaternion Data (Q0, Q1, Q2, Q3)
    if all(col in left_leg_df.columns for col in ['Q0', 'Q1', 'Q2', 'Q3']) and \
       all(col in right_leg_df.columns for col in ['Q0', 'Q1', 'Q2', 'Q3']):
        
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Quaternion Data - Left vs Right Leg', fontsize=16)
        
        # Left leg - Q0, Q1, Q2, Q3
        axs[0, 0].plot(left_leg_df['Time'], left_leg_df['Q0'])
        axs[0, 0].set_title(f'Left Leg - Q0 - {left_leg_device}')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Quaternion Value')
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(left_leg_df['Time'], left_leg_df['Q1'])
        axs[0, 1].set_title(f'Left Leg - Q1 - {left_leg_device}')
        axs[0, 1].set_xlabel('Time (seconds)')
        axs[0, 1].set_ylabel('Quaternion Value')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(left_leg_df['Time'], left_leg_df['Q2'])
        axs[0, 2].set_title(f'Left Leg - Q2 - {left_leg_device}')
        axs[0, 2].set_xlabel('Time (seconds)')
        axs[0, 2].set_ylabel('Quaternion Value')
        axs[0, 2].grid(True)
        
        axs[0, 3].plot(left_leg_df['Time'], left_leg_df['Q3'])
        axs[0, 3].set_title(f'Left Leg - Q3 - {left_leg_device}')
        axs[0, 3].set_xlabel('Time (seconds)')
        axs[0, 3].set_ylabel('Quaternion Value')
        axs[0, 3].grid(True)
        
        # Right leg - Q0, Q1, Q2, Q3
        axs[1, 0].plot(right_leg_df['Time'], right_leg_df['Q0'])
        axs[1, 0].set_title(f'Right Leg - Q0 - {right_leg_device}')
        axs[1, 0].set_xlabel('Time (seconds)')
        axs[1, 0].set_ylabel('Quaternion Value')
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(right_leg_df['Time'], right_leg_df['Q1'])
        axs[1, 1].set_title(f'Right Leg - Q1 - {right_leg_device}')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Quaternion Value')
        axs[1, 1].grid(True)
        
        axs[1, 2].plot(right_leg_df['Time'], right_leg_df['Q2'])
        axs[1, 2].set_title(f'Right Leg - Q2 - {right_leg_device}')
        axs[1, 2].set_xlabel('Time (seconds)')
        axs[1, 2].set_ylabel('Quaternion Value')
        axs[1, 2].grid(True)
        
        axs[1, 3].plot(right_leg_df['Time'], right_leg_df['Q3'])
        axs[1, 3].set_title(f'Right Leg - Q3 - {right_leg_device}')
        axs[1, 3].set_xlabel('Time (seconds)')
        axs[1, 3].set_ylabel('Quaternion Value')
        axs[1, 3].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'quaternion_comparison.png'))
            plt.close()
        else:
            plt.show()
            
    # 6. Additional plots - Temperature and Battery
    if 'Temperature' in left_leg_df.columns and 'Temperature' in right_leg_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Temperature Comparison - Left vs Right Leg', fontsize=16)
        
        axs[0].plot(left_leg_df['Time'], left_leg_df['Temperature'])
        axs[0].set_title(f'Left Leg - {left_leg_device}')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Temperature (°C)')
        axs[0].grid(True)
        
        axs[1].plot(right_leg_df['Time'], right_leg_df['Temperature'])
        axs[1].set_title(f'Right Leg - {right_leg_device}')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'temperature_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    if 'Battery' in left_leg_df.columns and 'Battery' in right_leg_df.columns:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Battery Level Comparison - Left vs Right Leg', fontsize=16)
        
        axs[0].plot(left_leg_df['Time'], left_leg_df['Battery'])
        axs[0].set_title(f'Left Leg - {left_leg_device}')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Battery Level (%)')
        axs[0].grid(True)
        
        axs[1].plot(right_leg_df['Time'], right_leg_df['Battery'])
        axs[1].set_title(f'Right Leg - {right_leg_device}')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Battery Level (%)')
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'battery_comparison.png'))
            plt.close()
        else:
            plt.show()
    
    # 7. Acceleration Magnitude Comparison
    if all(col in left_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']) and \
       all(col in right_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']):
        
        left_acc_magnitude = np.sqrt(left_leg_df['AccX']**2 + left_leg_df['AccY']**2 + left_leg_df['AccZ']**2)
        right_acc_magnitude = np.sqrt(right_leg_df['AccX']**2 + right_leg_df['AccY']**2 + right_leg_df['AccZ']**2)
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Acceleration Magnitude Comparison', fontsize=16)
        
        axs[0].plot(left_leg_df['Time'], left_acc_magnitude)
        axs[0].set_title(f'Left Leg - {left_leg_device}')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('Acceleration Magnitude (g)')
        axs[0].grid(True)
        
        axs[1].plot(right_leg_df['Time'], right_acc_magnitude)
        axs[1].set_title(f'Right Leg - {right_leg_device}')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Acceleration Magnitude (g)')
        axs[1].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acc_magnitude_comparison.png'))
            plt.close()
        else:
            plt.show()
            
    # 8. 3D Trajectory Comparison (in separate plots)
    if all(col in left_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']) and \
       all(col in right_leg_df.columns for col in ['AccX', 'AccY', 'AccZ']):
        
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('3D Acceleration Trajectory Comparison', fontsize=16)
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(left_leg_df['AccX'], left_leg_df['AccY'], left_leg_df['AccZ'])
        ax1.set_title(f'Left Leg - {left_leg_device}')
        ax1.set_xlabel('AccX (g)')
        ax1.set_ylabel('AccY (g)')
        ax1.set_zlabel('AccZ (g)')
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(right_leg_df['AccX'], right_leg_df['AccY'], right_leg_df['AccZ'])
        ax2.set_title(f'Right Leg - {right_leg_device}')
        ax2.set_xlabel('AccX (g)')
        ax2.set_ylabel('AccY (g)')
        ax2.set_zlabel('AccZ (g)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'acc_trajectory_3d_comparison.png'))
            plt.close()
        else:
            plt.show()

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