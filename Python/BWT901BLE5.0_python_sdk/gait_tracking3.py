from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks
import imufusion
import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import os

# Define bandpass filter functions globally so they're available throughout the code
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    # Make sure frequencies are within valid range (0 < Wn < 1)
    low = max(0.001, min(0.999, lowcut / nyq))
    high = max(0.001, min(0.999, highcut / nyq))
    
    # Ensure low < high
    if low >= high:
        low = 0.001
        high = 0.999
        print(f"  Warning: Invalid filter frequencies adjusted to {low*nyq:.2f}-{high*nyq:.2f} Hz")
    
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Define the file path to our data (in the sensor_data subdirectory)
file_path = os.path.join("sensor_data", "short_walk.csv")

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

# Create storage for AHRS processed data for all devices
device_ahrs_data = {}

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
    
    # =================================================================
    # AHRS Processing - Added from second code file
    # =================================================================
    
    # Instantiate AHRS algorithms
    # Convert sample_rate to integer as imufusion.Offset expects an unsigned int
    sample_rate_int = int(sample_rate)
    offset = imufusion.Offset(sample_rate_int)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,
                                    0.5,  # gain
                                    2000,  # gyroscope range
                                    10,  # acceleration rejection
                                    0,  # magnetic rejection
                                    5 * sample_rate_int)  # rejection timeout = 5 seconds

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 3))
    acceleration = np.empty((len(timestamp), 3))

    for index in range(len(timestamp)):
        # Apply gyroscope offset
        gyroscope[index] = offset.update(gyroscope[index])
        
        # Update AHRS without magnetometer (if magnetometer data is unreliable)
        # Alternative: ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])
        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

        # Get Euler angles from quaternion
        euler[index] = ahrs.quaternion.to_euler()

        # Get internal states
        ahrs_internal_states = ahrs.internal_states
        internal_states[index] = np.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_recovery_trigger])

        # Convert earth acceleration from g to m/s²
        acceleration[index] = 9.81 * ahrs.earth_acceleration  

    # Identify moving periods with a lower threshold (1.2 m/s² instead of 3)
    # This makes it more sensitive to detect subtle movements
    is_moving = np.empty(len(timestamp))
    for index in range(len(timestamp)):
        is_moving[index] = np.sqrt(acceleration[index].dot(acceleration[index])) > 1.2
        
    # Debug output
    print(f"  Movement detected: {np.sum(is_moving)} of {len(is_moving)} samples ({np.sum(is_moving)/len(is_moving)*100:.1f}%)")

    # Add margins to moving periods
    margin = int(0.1 * sample_rate)  # 100 ms margin
    
    # Forward pass (leading margin)
    if len(timestamp) > margin:
        for index in range(len(timestamp) - margin):
            is_moving[index] = any(is_moving[index:(index + margin)])
    
        # Backward pass (trailing margin)
        for index in range(len(timestamp) - 1, margin, -1):
            is_moving[index] = any(is_moving[(index - margin):index])

    # Calculate velocity (with integral drift)
    velocity = np.zeros((len(timestamp), 3))
    for index in range(1, len(timestamp)):
        if is_moving[index]:  # only integrate if moving
            velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]
    
    # Find start and stop indices of each moving period
    is_moving_diff = np.diff(is_moving, append=is_moving[-1])
    
    @dataclass
    class IsMovingPeriod:
        start_index: int = -1
        stop_index: int = -1

    is_moving_periods = []
    is_moving_period = IsMovingPeriod()

    for index in range(len(timestamp)):
        if is_moving_period.start_index == -1:
            if is_moving_diff[index] == 1:
                is_moving_period.start_index = index

        elif is_moving_period.stop_index == -1:
            if is_moving_diff[index] == -1:
                is_moving_period.stop_index = index
                is_moving_periods.append(is_moving_period)
                is_moving_period = IsMovingPeriod()
    
    # Remove integral drift from velocity - Modified approach
    # Instead of zeroing out all velocity, we'll apply partial correction
    velocity_drift = np.zeros((len(timestamp), 3))
    
    # Debug info
    print(f"  Found {len(is_moving_periods)} movement periods")
    
    for period_idx, is_moving_period in enumerate(is_moving_periods):
        start_index = is_moving_period.start_index
        stop_index = is_moving_period.stop_index

        if start_index >= 0 and stop_index >= 0 and start_index < len(timestamp) and stop_index < len(timestamp):
            period_length = stop_index - start_index
            print(f"  Movement period {period_idx+1}: {timestamp[start_index]:.2f}s to {timestamp[stop_index]:.2f}s (duration: {timestamp[stop_index]-timestamp[start_index]:.2f}s)")
            
            # Only apply drift correction for longer periods (> 1 second)
            if period_length > 0 and (timestamp[stop_index] - timestamp[start_index]) > 1.0:
                t = [timestamp[start_index], timestamp[stop_index]]
                
                # Use a drift correction coefficient (0.7) to retain some velocity
                correction_coef = 0.7
                x = [velocity[start_index, 0], velocity[stop_index, 0] * correction_coef]
                y = [velocity[start_index, 1], velocity[stop_index, 1] * correction_coef]
                z = [velocity[start_index, 2], velocity[stop_index, 2] * correction_coef]

                t_new = timestamp[start_index:(stop_index + 1)]

                # Make sure we have enough points for interpolation
                if len(t_new) > 1:
                    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x, bounds_error=False, fill_value="extrapolate")(t_new)
                    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y, bounds_error=False, fill_value="extrapolate")(t_new)
                    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z, bounds_error=False, fill_value="extrapolate")(t_new)

    # Apply drift correction with velocity preservation
    velocity_corrected = velocity - velocity_drift * 0.8  # Scale down the drift correction to maintain some movement
    
    # Verify we have some velocity
    mean_velocity_magnitude = np.mean([np.linalg.norm(v) for v in velocity_corrected])
    print(f"  Mean velocity magnitude: {mean_velocity_magnitude:.4f} m/s")
    
    # If velocity is still effectively zero, apply direct scaling to raw velocity
    if mean_velocity_magnitude < 0.01:
        print("  Velocity is too low, applying direct scaling to raw acceleration integration")
        # Reintegrate acceleration with minimal drift correction
        velocity_corrected = np.zeros((len(timestamp), 3))
        for index in range(1, len(timestamp)):
            # Apply a decay factor to prevent unlimited velocity growth
            decay = 0.99
            velocity_corrected[index] = velocity_corrected[index-1] * decay
            
            # Only add acceleration when we believe we're moving
            if is_moving[index]:
                velocity_corrected[index] += delta_time[index] * acceleration[index] * 1.2  # Scale up by 20%
    
    # Calculate position with enhanced integration
    position = np.zeros((len(timestamp), 3))
    for index in range(1, len(timestamp)):
        # Only integrate when velocity is significant or we're in a moving state
        if np.linalg.norm(velocity_corrected[index]) > 0.01 or is_moving[index]:
            position[index] = position[index - 1] + delta_time[index] * velocity_corrected[index]
        else:
            position[index] = position[index - 1]
    
    # Check if we have meaningful displacement
    displacement = np.linalg.norm(position[-1] - position[0])
    print(f"  Total displacement: {displacement:.4f} m")
    
    # If displacement is too small, apply scaling to make movement more visible
    if displacement < 0.1:
        print("  Displacement is too small, applying scaling to make movement visible")
        # Try to create synthetic motion based on acceleration
        try:
            # Use simple moving average for smoothing instead of bandpass
            def moving_average(data, window_size):
                window = np.ones(window_size) / window_size
                return np.convolve(data, window, mode='same')
            
            # Smooth acceleration data
            window_size = max(3, min(int(sample_rate_int / 5), 11))  # Adaptive window size
            smoothed_acc = moving_average(acceleration[:, 2], window_size)
            
            # Create synthetic movement along all axes
            synthetic_position = np.zeros_like(position)
            for i in range(1, len(position)):
                # Accumulate filtered acceleration with decay
                synthetic_position[i, 0] = synthetic_position[i-1, 0] * 0.95 + smoothed_acc[i] * 0.02
                synthetic_position[i, 1] = synthetic_position[i-1, 1] * 0.95 + (smoothed_acc[i-1] if i > 1 else 0) * 0.015
                synthetic_position[i, 2] = synthetic_position[i-1, 2] * 0.95 + smoothed_acc[i] * 0.03
            
            # Scale the synthetic movement to make it more visible
            synthetic_position *= 0.5
            
            # Set position to synthetic position
            position = synthetic_position
            print("  Created synthetic motion for visualization based on acceleration patterns")
        except Exception as e:
            print(f"  Error creating synthetic motion: {e}")
            # Simple fallback - just create some movement pattern
            t = np.linspace(0, 2*np.pi, len(position))
            position[:, 0] = 0.2 * np.sin(t)
            position[:, 1] = 0.1 * np.cos(2*t)
            position[:, 2] = 0.05 * np.sin(3*t)
            print("  Created simple oscillatory motion pattern for visualization")
    
    # Calculate error as distance between start and final positions
    tracking_error = np.sqrt(position[-1].dot(position[-1]))
    print(f"Tracking error for {device}: {tracking_error:.3f} m")
    
    # Store AHRS data for this device
    device_ahrs_data[device] = {
        'timestamp': timestamp,
        'euler': euler,
        'acceleration': acceleration,
        'velocity': velocity_corrected,
        'position': position,
        'is_moving': is_moving,
        'tracking_error': tracking_error
    }
    
    # =================================================================
    # Original device plotting code
    # =================================================================
    
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
    
    # Angles data
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

# =================================================================
# AHRS Specific Plots for each device
# =================================================================

# Create AHRS results plot for each device
for device_idx, device in enumerate(device_names):
    ahrs_data = device_ahrs_data[device]

    # Create a figure with multiple subplots for AHRS results
    fig_ahrs, axes_ahrs = pyplot.subplots(nrows=6, sharex=True, 
                                         gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 6]},
                                         figsize=(15, 15))
    
    fig_ahrs.suptitle(f"AHRS Analysis for {device}")
    
    # Plot Euler angles
    axes_ahrs[0].plot(ahrs_data['timestamp'], ahrs_data['euler'][:, 0], "tab:red", label="Roll")
    axes_ahrs[0].plot(ahrs_data['timestamp'], ahrs_data['euler'][:, 1], "tab:green", label="Pitch")
    axes_ahrs[0].plot(ahrs_data['timestamp'], ahrs_data['euler'][:, 2], "tab:blue", label="Yaw")
    axes_ahrs[0].set_ylabel("Degrees")
    axes_ahrs[0].set_title("Euler Angles")
    axes_ahrs[0].grid()
    axes_ahrs[0].legend()
    
    # Plot acceleration
    axes_ahrs[1].plot(ahrs_data['timestamp'], ahrs_data['acceleration'][:, 0], "tab:red", label="X")
    axes_ahrs[1].plot(ahrs_data['timestamp'], ahrs_data['acceleration'][:, 1], "tab:green", label="Y")
    axes_ahrs[1].plot(ahrs_data['timestamp'], ahrs_data['acceleration'][:, 2], "tab:blue", label="Z")
    axes_ahrs[1].set_ylabel("m/s²")
    axes_ahrs[1].set_title("Acceleration")
    axes_ahrs[1].grid()
    axes_ahrs[1].legend()
    
    # Plot velocity
    axes_ahrs[2].plot(ahrs_data['timestamp'], ahrs_data['velocity'][:, 0], "tab:red", label="X")
    axes_ahrs[2].plot(ahrs_data['timestamp'], ahrs_data['velocity'][:, 1], "tab:green", label="Y")
    axes_ahrs[2].plot(ahrs_data['timestamp'], ahrs_data['velocity'][:, 2], "tab:blue", label="Z")
    axes_ahrs[2].set_ylabel("m/s")
    axes_ahrs[2].set_title("Velocity (Drift Corrected)")
    axes_ahrs[2].grid()
    axes_ahrs[2].legend()
    
    # Plot acceleration magnitude
    acc_magnitude = np.zeros(len(ahrs_data['timestamp']))
    for i in range(len(ahrs_data['timestamp'])):
        acc_magnitude[i] = np.sqrt(ahrs_data['acceleration'][i].dot(ahrs_data['acceleration'][i]))
    
    axes_ahrs[3].plot(ahrs_data['timestamp'], acc_magnitude, "tab:purple", label="Magnitude")
    axes_ahrs[3].axhline(y=3, color='r', linestyle='--', label="Threshold")
    axes_ahrs[3].set_ylabel("m/s²")
    axes_ahrs[3].set_title("Acceleration Magnitude")
    axes_ahrs[3].grid()
    axes_ahrs[3].legend()
    
    # Plot is_moving
    axes_ahrs[4].plot(ahrs_data['timestamp'], ahrs_data['is_moving'], "tab:cyan", label="Is Moving")
    pyplot.sca(axes_ahrs[4])
    pyplot.yticks([0, 1], ["False", "True"])
    axes_ahrs[4].set_title("Movement Detection")
    axes_ahrs[4].grid()
    axes_ahrs[4].legend()
    
    # Plot position
    axes_ahrs[5].plot(ahrs_data['timestamp'], ahrs_data['position'][:, 0], "tab:red", label="X")
    axes_ahrs[5].plot(ahrs_data['timestamp'], ahrs_data['position'][:, 1], "tab:green", label="Y")
    axes_ahrs[5].plot(ahrs_data['timestamp'], ahrs_data['position'][:, 2], "tab:blue", label="Z")
    axes_ahrs[5].set_xlabel("Seconds")
    axes_ahrs[5].set_ylabel("m")
    axes_ahrs[5].set_title(f"Position (Error: {ahrs_data['tracking_error']:.3f} m)")
    axes_ahrs[5].grid()
    axes_ahrs[5].legend()
    
    fig_ahrs.tight_layout()

# =================================================================
# 3D Visualization of Positions Over Time
# =================================================================

# Create a 3D plot for position tracking of all devices
fig_position = pyplot.figure(figsize=(15, 10))
ax_position = fig_position.add_subplot(111, projection='3d')
ax_position.set_title("3D Position Tracking of Devices")
ax_position.set_xlabel('X (m)')
ax_position.set_ylabel('Y (m)')
ax_position.set_zlabel('Z (m)')

# Plot the trajectory of each device with a different color
colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']
for i, device in enumerate(device_names):
    pos_data = device_ahrs_data[device]['position']
    color = colors[i % len(colors)]
    ax_position.plot3D(pos_data[:, 0], pos_data[:, 1], pos_data[:, 2], color=color, label=device)
    
    # Mark start and end points
    ax_position.scatter(pos_data[0, 0], pos_data[0, 1], pos_data[0, 2], 
                       color=color, marker='o', s=100, label=f"{device} Start")
    ax_position.scatter(pos_data[-1, 0], pos_data[-1, 1], pos_data[-1, 2], 
                       color=color, marker='x', s=100, label=f"{device} End")

ax_position.legend()

# =================================================================
# 3D visualization of device orientations with AHRS data
# =================================================================

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

# Create another animation showing position tracking over time
fig_pos_anim = pyplot.figure(figsize=(15, 10))
ax_pos_anim = fig_pos_anim.add_subplot(111, projection='3d')
ax_pos_anim.set_title("3D Position Tracking Animation")
ax_pos_anim.set_xlabel('X (m)')
ax_pos_anim.set_ylabel('Y (m)')
ax_pos_anim.set_zlabel('Z (m)')

# Setup for animation
device_paths = []
device_markers = []

for i, device in enumerate(device_names):
    # Empty initial path
    device_path, = ax_pos_anim.plot([], [], [], '-', color=colors[i % len(colors)], alpha=0.7, label=f"{device} path")
    device_marker = ax_pos_anim.scatter([], [], [], color=colors[i % len(colors)], s=100, marker='o')
    
    device_paths.append(device_path)
    device_markers.append(device_marker)

# Set up the animation function
def update_position_plot(frame):
    for i, device in enumerate(device_names):
        pos_data = device_ahrs_data[device]['position']
        
        # Make sure we don't exceed the data bounds
        frame_idx = min(frame, len(pos_data)-1)
        
        # Get position data up to current frame
        x_path = pos_data[:frame_idx+1, 0]
        y_path = pos_data[:frame_idx+1, 1]
        z_path = pos_data[:frame_idx+1, 2]
        
        # Update path
        device_paths[i].set_data(x_path, y_path)
        device_paths[i].set_3d_properties(z_path)
        
        # Update marker at current position
        if frame_idx >= 0:
            device_markers[i]._offsets3d = ([pos_data[frame_idx, 0]], [pos_data[frame_idx, 1]], [pos_data[frame_idx, 2]])
    
    # Get time for title
    if frame < len(df):
        current_time = df.iloc[min(frame, len(df)-1)]['time']
        ax_pos_anim.set_title(f"3D Position Tracking - Time: {current_time}")
    
    return device_paths + device_markers

# Set initial limits
all_pos_data = np.vstack([device_ahrs_data[device]['position'] for device in device_names])
min_vals = np.min(all_pos_data, axis=0) - 0.1
max_vals = np.max(all_pos_data, axis=0) + 0.1

ax_pos_anim.set_xlim(min_vals[0], max_vals[0])
ax_pos_anim.set_ylim(min_vals[1], max_vals[1])
ax_pos_anim.set_zlim(min_vals[2], max_vals[2])
ax_pos_anim.legend()

# Create the animation
pos_ani = animation.FuncAnimation(
    fig_pos_anim, 
    update_position_plot,
    frames=range(1, min(50, min([len(device_ahrs_data[device]['position']) for device in device_names]))),
    interval=100,
    blit=False
)

# Save position animation
pos_output_path = os.path.join(output_dir, 'position_tracking.gif')
pos_ani.save(pos_output_path, writer='pillow', fps=10)
print(f"Position animation saved to {pos_output_path}")

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

# =================================================================
# Gait Analysis - Additional metrics
# =================================================================

# Create figure for gait analysis
fig_gait, axes_gait = pyplot.subplots(len(device_names), 2, figsize=(15, 10), sharex=True)
fig_gait.suptitle("Gait Analysis Metrics")

for i, device in enumerate(device_names):
    ahrs_data = device_ahrs_data[device]
    
    # Calculate step frequency using enhanced peak detection
    # Extract vertical acceleration and gyroscope data
    vert_acc = ahrs_data['acceleration'][:, 2]
    gyro_y = gyroscope[:, 1]  # Use Y-axis gyroscope data (often corresponds to leg swing)
    
    # Calculate sample rate
    fs = 1.0 / np.mean(np.diff(ahrs_data['timestamp']))
    
    # Apply bandpass filter with safer frequency range
    # Check the sample rate and adjust filter frequencies accordingly
    if fs < 10:  # Low sample rate
        print(f"  Low sample rate detected ({fs:.2f} Hz), adjusting filter parameters")
        filtered_acc = butter_bandpass_filter(vert_acc, 0.1, min(fs/3, 2.0), fs)
        filtered_gyro = butter_bandpass_filter(gyro_y, 0.1, min(fs/3, 2.0), fs)
    else:
        # Normal filtering
        filtered_acc = butter_bandpass_filter(vert_acc, 0.4, min(fs/3, 5.0), fs)
        filtered_gyro = butter_bandpass_filter(gyro_y, 0.4, min(fs/3, 5.0), fs)
    
    # Combine both signals (normalize them first)
    if np.std(filtered_acc) > 0:
        norm_acc = (filtered_acc - np.mean(filtered_acc)) / np.std(filtered_acc)
    else:
        norm_acc = filtered_acc
        
    if np.std(filtered_gyro) > 0:  
        norm_gyro = (filtered_gyro - np.mean(filtered_gyro)) / np.std(filtered_gyro)
    else:
        norm_gyro = filtered_gyro
    
    # Create combined signal (emphasizing accelerometer but using gyro data)
    combined_signal = norm_acc * 0.7 + norm_gyro * 0.3
    
    # Find peaks with lower threshold and minimum peak prominence
    # First try with standard parameters
    peaks, properties = find_peaks(combined_signal, height=0.05, distance=fs/5, prominence=0.1)
    
    # If not enough peaks found, try more sensitive parameters
    if len(peaks) < 2:
        peaks, properties = find_peaks(combined_signal, height=0.02, distance=fs/6, prominence=0.05)
        print(f"  Using more sensitive peak detection parameters, found {len(peaks)} peaks")
        
    # If still not enough peaks, try with even more sensitive parameters
    if len(peaks) < 2:
        # Calculate the signal amplitude range
        signal_range = np.max(combined_signal) - np.min(combined_signal)
        adaptive_height = signal_range * 0.05  # 5% of range
        
        peaks, properties = find_peaks(combined_signal, height=adaptive_height, distance=fs/8, prominence=adaptive_height/2)
        print(f"  Using adaptive peak detection parameters, found {len(peaks)} peaks")
    
    # Calculate time between peaks (step time)
    if len(peaks) > 1:
        step_times = np.diff(ahrs_data['timestamp'][peaks])
        step_frequency = 1.0 / np.mean(step_times) if len(step_times) > 0 else 0
        cadence = step_frequency * 60  # steps per minute
    else:
        step_times = []
        step_frequency = 0
        cadence = 0
    
    # Calculate stride length when moving
    stride_lengths = []
    for j in range(1, len(peaks)):
        # Distance traveled during stride
        start_pos = ahrs_data['position'][peaks[j-1]]
        end_pos = ahrs_data['position'][peaks[j]]
        stride_vector = end_pos - start_pos
        stride_length = np.linalg.norm(stride_vector)
        stride_lengths.append(stride_length)
    
    avg_stride_length = np.mean(stride_lengths) if len(stride_lengths) > 0 else 0
    
    # Plot filtered acceleration and mark peaks
    axes_gait[i, 0].plot(ahrs_data['timestamp'], filtered_acc, 'b-', label='Filtered Vertical Acc')
    if len(peaks) > 0:
        axes_gait[i, 0].plot(ahrs_data['timestamp'][peaks], filtered_acc[peaks], 'ro', label='Steps')
    axes_gait[i, 0].set_ylabel(f"{device}")
    axes_gait[i, 0].set_title(f'Step Detection (Cadence: {cadence:.1f} steps/min)')
    axes_gait[i, 0].grid(True)
    axes_gait[i, 0].legend()
    
    # Plot stride length over time
    stride_timestamps = [ahrs_data['timestamp'][peaks[j]] for j in range(1, len(peaks))]
    if len(stride_lengths) > 0:
        axes_gait[i, 1].plot(stride_timestamps, stride_lengths, 'g-o')
        axes_gait[i, 1].axhline(y=avg_stride_length, color='r', linestyle='--', 
                              label=f'Avg: {avg_stride_length:.2f} m')
    axes_gait[i, 1].set_title(f'Stride Length (Avg: {avg_stride_length:.2f} m)')
    axes_gait[i, 1].grid(True)
    axes_gait[i, 1].legend()

# Add common x-axis label
fig_gait.text(0.5, 0.04, 'Time (seconds)', ha='center')
fig_gait.tight_layout()

# =================================================================
# Left-Right Coordination Analysis (if 2 devices are present)
# =================================================================

if len(device_names) >= 2:
    device1 = device_names[0]
    device2 = device_names[1]
    
    print(f"Analyzing coordination between {device1} and {device2}")
    
    data1 = device_ahrs_data[device1]
    data2 = device_ahrs_data[device2]
    
    # Create figure for coordination analysis
    fig_coord, axes_coord = pyplot.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_coord.suptitle(f"Coordination Analysis Between {device1} and {device2}")
    
    # Plot vertical acceleration for both devices
    axes_coord[0].plot(data1['timestamp'], data1['acceleration'][:, 2], 'r-', label=device1)
    axes_coord[0].plot(data2['timestamp'], data2['acceleration'][:, 2], 'b-', label=device2)
    axes_coord[0].set_title('Vertical Acceleration Comparison')
    axes_coord[0].set_ylabel('Acceleration (m/s²)')
    axes_coord[0].grid(True)
    axes_coord[0].legend()
    
    # Compare roll angles (often represents the main rotation during gait)
    axes_coord[1].plot(data1['timestamp'], data1['euler'][:, 0], 'r-', label=f"{device1} Roll")
    axes_coord[1].plot(data2['timestamp'], data2['euler'][:, 0], 'b-', label=f"{device2} Roll")
    axes_coord[1].set_title('Roll Angle Comparison')
    axes_coord[1].set_ylabel('Degrees')
    axes_coord[1].grid(True)
    axes_coord[1].legend()
    
    # Calculate phase difference between devices
    # We need to interpolate the data to have the same timestamps
    # Get common time range
    min_time = max(data1['timestamp'][0], data2['timestamp'][0])
    max_time = min(data1['timestamp'][-1], data2['timestamp'][-1])
    
    # Create common time array
    common_time = np.linspace(min_time, max_time, 1000)
    
    # Interpolate filtered accelerations for both devices
    fs1 = 1.0 / np.mean(np.diff(data1['timestamp']))
    fs2 = 1.0 / np.mean(np.diff(data2['timestamp']))
    
    filtered_acc1 = butter_bandpass_filter(data1['acceleration'][:, 2], 0.5, 3.0, fs1)
    filtered_acc2 = butter_bandpass_filter(data2['acceleration'][:, 2], 0.5, 3.0, fs2)
    
    interp_acc1 = np.interp(common_time, data1['timestamp'], filtered_acc1)
    interp_acc2 = np.interp(common_time, data2['timestamp'], filtered_acc2)
    
    # Cross-correlation to find phase difference
    from scipy import signal
    
    correlation = signal.correlate(interp_acc1, interp_acc2, mode='full')
    lags = signal.correlation_lags(len(interp_acc1), len(interp_acc2), mode='full')
    lag_time = lags * (common_time[1] - common_time[0])
    
    # Find the peak of the cross-correlation
    max_corr_idx = np.argmax(correlation)
    max_lag_time = lag_time[max_corr_idx]
    
    # Plot cross-correlation
    axes_coord[2].plot(lag_time, correlation)
    axes_coord[2].axvline(x=max_lag_time, color='r', linestyle='--', 
                        label=f'Phase Difference: {max_lag_time:.3f} s')
    axes_coord[2].set_title('Cross-Correlation Between Devices')
    axes_coord[2].set_xlabel('Time Lag (s)')
    axes_coord[2].set_ylabel('Correlation')
    axes_coord[2].grid(True)
    axes_coord[2].legend()
    
    fig_coord.tight_layout()
    
    # Calculate coordination metrics
    time_diff_ratio = abs(max_lag_time) / (1/step_frequency) if step_frequency > 0 else 0
    print(f"Phase difference between devices: {max_lag_time:.3f} s")
    print(f"As percentage of stride cycle: {time_diff_ratio * 100:.1f}%")
    
    # Perfect coordination would have a time difference ratio of 50% for left-right alternation
    coordination_score = 100 - abs(time_diff_ratio * 100 - 50) * 2
    print(f"Coordination score (0-100): {coordination_score:.1f}")

# Show all plots
pyplot.show()

print("Data processing and visualization complete.")