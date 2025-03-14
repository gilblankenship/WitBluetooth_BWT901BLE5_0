from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import os

# Define the file path to our data (in the sensor_data subdirectory)
file_path = os.path.join("sensor_data", "20250313123959.txt")

# Create the sensor_data directory if it doesn't exist
os.makedirs("sensor_data", exist_ok=True)

# Check if the file exists in the sensor_data directory
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    print("Copying the file to the sensor_data directory...")
    
    # If the file doesn't exist in the subdirectory but exists in the current directory,
    # copy it to the subdirectory
    original_path = "20250313092021.txt"
    if os.path.exists(original_path):
        import shutil
        shutil.copy(original_path, file_path)
        print(f"File copied from {original_path} to {file_path}")
    else:
        print(f"Original file {original_path} not found either.")
        print("Please place the data file in the sensor_data directory.")
        exit(1)

# Read the data from the tab-delimited file
print(f"Reading data from {file_path}")
df = pd.read_csv(file_path, delimiter='\t')

# Process the data for each unique device
device_names = df['DeviceName'].unique()
print(f"Found {len(device_names)} devices: {device_names}")

# Create figure arrays
fig_sensors, axes_sensors = pyplot.subplots(2, 3, figsize=(15, 8), sharex=True)
fig_sensors.suptitle("Sensor Data by Device")

fig_orientation, axes_orientation = pyplot.subplots(2, 3, figsize=(15, 8), sharex=True)
fig_orientation.suptitle("Device Orientation")

# Format the timestamp column
df['time'] = pd.to_datetime(df['time'])
df['timestamp'] = (df['time'] - df['time'].min()).dt.total_seconds()

# Process each device
for device_idx, device in enumerate(device_names):
    print(f"Processing data for device: {device}")
    device_data = df[df['DeviceName'] == device].copy()
    
    # Sort by timestamp to ensure chronological order
    device_data = device_data.sort_values('timestamp')
    
    # Extract the necessary columns
    timestamp = device_data['timestamp'].values
    
    # Convert columns to correct data type, handling any missing values
    gyroscope = np.zeros((len(device_data), 3))
    gyroscope[:, 0] = device_data['AsX(°/s)'].values
    gyroscope[:, 1] = device_data['AsY(°/s)'].values
    gyroscope[:, 2] = device_data['AsZ(°/s)'].values
    
    accelerometer = np.zeros((len(device_data), 3))
    accelerometer[:, 0] = device_data['AccX(g)'].values
    accelerometer[:, 1] = device_data['AccY(g)'].values
    accelerometer[:, 2] = device_data['AccZ(g)'].values
    
    angles = np.zeros((len(device_data), 3))
    angles[:, 0] = device_data['AngleX(°)'].values
    angles[:, 1] = device_data['AngleY(°)'].values
    angles[:, 2] = device_data['AngleZ(°)'].values
    
    # Extract quaternions
    quaternions = np.zeros((len(device_data), 4))
    quaternions[:, 0] = device_data['Q0()'].values
    quaternions[:, 1] = device_data['Q1()'].values
    quaternions[:, 2] = device_data['Q2()'].values
    quaternions[:, 3] = device_data['Q3()'].values
    
    # Magnetometer data
    magnetometer = np.zeros((len(device_data), 3))
    magnetometer[:, 0] = device_data['HX(uT)'].values
    magnetometer[:, 1] = device_data['HY(uT)'].values
    magnetometer[:, 2] = device_data['HZ(uT)'].values
    
    # Sample rate calculation
    if len(timestamp) > 1:
        sample_rate = 1.0 / np.mean(np.diff(timestamp))
        print(f"Calculated sample rate: {sample_rate:.2f} Hz")
    else:
        sample_rate = 100.0  # Default if not enough data
        print("Using default sample rate: 100 Hz")
    
    # Plot sensor data
    row = device_idx // 2
    col = device_idx % 3
    
    # Create a nicer color palette
    colors = ['#E41A1C', '#377EB8', '#4DAF4A']
    
    device_name_short = device.split('(')[0]
    device_id = device.split('(')[1].rstrip(')')
    
    # Gyroscope data
    axes_sensors[row, col].plot(timestamp, gyroscope[:, 0], color=colors[0], label='X')
    axes_sensors[row, col].plot(timestamp, gyroscope[:, 1], color=colors[1], label='Y')
    axes_sensors[row, col].plot(timestamp, gyroscope[:, 2], color=colors[2], label='Z')
    axes_sensors[row, col].set_ylabel("Gyroscope (°/s)")
    axes_sensors[row, col].set_title(f"{device}")
    axes_sensors[row, col].grid(True)
    axes_sensors[row, col].legend()
    
    # Accelerometer data
    axes_orientation[row, col].plot(timestamp, angles[:, 0], color=colors[0], label='Roll')
    axes_orientation[row, col].plot(timestamp, angles[:, 1], color=colors[1], label='Pitch')
    axes_orientation[row, col].plot(timestamp, angles[:, 2], color=colors[2], label='Yaw')
    axes_orientation[row, col].set_ylabel("Angles (°)")
    axes_orientation[row, col].set_title(f"{device}")
    axes_orientation[row, col].grid(True)
    axes_orientation[row, col].legend()

# Add a common x-axis label
for fig in [fig_sensors, fig_orientation]:
    fig.text(0.5, 0.04, 'Time (seconds)', ha='center')
    fig.tight_layout()

# 3D visualization of device orientations
fig_3d = pyplot.figure(figsize=(15, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.set_title("3D Orientation of Devices")
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

# Function to create coordinate system for visualization
def create_coordinate_system(quaternion, scale=0.5):
    # Rotation matrix from quaternion
    w, x, y, z = quaternion
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Unit vectors
    origin = np.array([0, 0, 0])
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    
    # Rotate unit vectors
    x_rotated = np.dot(rotation_matrix, x_axis)
    y_rotated = np.dot(rotation_matrix, y_axis)
    z_rotated = np.dot(rotation_matrix, z_axis)
    
    return origin, x_rotated, y_rotated, z_rotated

# Initialize coordinate systems for each device
markers = []
lines = []

# Set initial positions for devices (spread them out in 3D space)
positions = [
    np.array([0, 0, 0]),
    np.array([2, 0, 0])
]

for i, device in enumerate(device_names):
    if i < len(positions):  # In case we have more devices than positions
        device_data = df[df['DeviceName'] == device]
        q_idx = min(1, len(device_data) - 1)  # Ensure we have at least one valid quaternion
        q = [
            device_data.iloc[q_idx]['Q0()'],
            device_data.iloc[q_idx]['Q1()'],
            device_data.iloc[q_idx]['Q2()'],
            device_data.iloc[q_idx]['Q3()']
        ]
        
        origin, x_axis, y_axis, z_axis = create_coordinate_system(q)
        
        # Shift the coordinate system to its position
        origin = positions[i]
        x_end = origin + x_axis
        y_end = origin + y_axis
        z_end = origin + z_axis
        
        # Plot coordinate system
        marker = ax_3d.scatter(*origin, color='black', s=50, label=device)
        
        x_line, = ax_3d.plot([origin[0], x_end[0]], [origin[1], x_end[1]], [origin[2], x_end[2]], 'r-', linewidth=2)
        y_line, = ax_3d.plot([origin[0], y_end[0]], [origin[1], y_end[1]], [origin[2], y_end[2]], 'g-', linewidth=2)
        z_line, = ax_3d.plot([origin[0], z_end[0]], [origin[1], z_end[1]], [origin[2], z_end[2]], 'b-', linewidth=2)
        
        markers.append(marker)
        lines.append((x_line, y_line, z_line))

# Set 3D plot limits and aspect ratio
ax_3d.set_xlim(-2, 4)
ax_3d.set_ylim(-2, 2)
ax_3d.set_zlim(-2, 2)
ax_3d.set_box_aspect([6, 4, 4])  # Adjust aspect ratio

# Add a legend
ax_3d.legend()

# Animation function to update the 3D visualization
def update_3d_plot(frame):
    for i, device in enumerate(device_names):
        if i >= len(positions):
            continue
            
        device_data = df[df['DeviceName'] == device]
        if frame < len(device_data):
            q = [
                device_data.iloc[frame]['Q0()'],
                device_data.iloc[frame]['Q1()'],
                device_data.iloc[frame]['Q2()'],
                device_data.iloc[frame]['Q3()']
            ]
            
            origin, x_axis, y_axis, z_axis = create_coordinate_system(q)
            
            # Shift the coordinate system to its position
            origin = positions[i]
            x_end = origin + x_axis
            y_end = origin + y_axis
            z_end = origin + z_axis
            
            # Update lines
            x_line, y_line, z_line = lines[i]
            x_line.set_data([origin[0], x_end[0]], [origin[1], x_end[1]])
            x_line.set_3d_properties([origin[2], x_end[2]])
            
            y_line.set_data([origin[0], y_end[0]], [origin[1], y_end[1]])
            y_line.set_3d_properties([origin[2], y_end[2]])
            
            z_line.set_data([origin[0], z_end[0]], [origin[1], z_end[1]])
            z_line.set_3d_properties([origin[2], z_end[2]])
    
    # Update the title with the current time
    current_time = df.iloc[min(frame, len(df)-1)]['time']
    ax_3d.set_title(f"3D Orientation of Devices - Time: {current_time}")
    
    return markers + [line for line_set in lines for line in line_set]

# Create animation
num_frames = min(50, len(df))  # Limit to 50 frames to keep it reasonable
step = max(1, len(df) // num_frames)

ani = animation.FuncAnimation(
    fig_3d, 
    update_3d_plot,
    frames=range(0, len(df), step),
    interval=100,
    blit=False
)

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save animation as GIF in the output directory
output_path = os.path.join(output_dir, 'device_orientation.gif')
ani.save(output_path, writer='pillow', fps=10)
print(f"Animation saved to {output_path}")

# Create combined visualization of orientation over time
fig_combined, axes_combined = pyplot.subplots(len(device_names), 3, figsize=(15, 10), sharex=True)
fig_combined.suptitle("Device Orientation Over Time")

for i, device in enumerate(device_names):
    device_data = df[df['DeviceName'] == device]
    
    # Roll, Pitch, Yaw
    axes_combined[i, 0].plot(device_data['timestamp'], device_data['AngleX(°)'], 'r-')
    axes_combined[i, 0].set_ylabel(f"{device}")
    axes_combined[i, 0].set_title('Roll')
    axes_combined[i, 0].grid(True)
    
    axes_combined[i, 1].plot(device_data['timestamp'], device_data['AngleY(°)'], 'g-')
    axes_combined[i, 1].set_title('Pitch')
    axes_combined[i, 1].grid(True)
    
    axes_combined[i, 2].plot(device_data['timestamp'], device_data['AngleZ(°)'], 'b-')
    axes_combined[i, 2].set_title('Yaw')
    axes_combined[i, 2].grid(True)

# Add a common x-axis label
fig_combined.text(0.5, 0.04, 'Time (seconds)', ha='center')
fig_combined.tight_layout()

# Show all plots
pyplot.show()

print("Data processing and visualization complete.")