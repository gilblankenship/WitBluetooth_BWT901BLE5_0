#!/usr/bin/env python3
"""
MAXREFDES106 (Health Sensor Platform 2.0) Bluetooth Data Receiver

This script connects to the Analog Devices MAXREFDES106 Health Sensor Platform 2.0
via Bluetooth Low Energy (BLE) and receives data from the various sensors.

Requirements:
- Python 3.6+
- bleak (BLE library): pip install bleak
- numpy: pip install numpy
- matplotlib (for visualization): pip install matplotlib

The MAXREFDES106 platform includes sensors for:
- ECG (MAX30003)
- Optical Pulse Oximetry & Heart Rate (MAX30101)
- Temperature (MAX30205)
- Accelerometer (LIS2DH12)
- Biopotential (MAX30003)
"""

import asyncio
import struct
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner

# Define MAXREFDES106 UUIDs
# These UUIDs are specific to the MAXREFDES106 device
SERVICE_UUID = "46a970e0-0d79-11e6-bdf4-0800200c9a66"  # Main service UUID

# Characteristic UUIDs for different sensors
CHAR_UUID_ECG = "46a970e1-0d79-11e6-bdf4-0800200c9a66"  # ECG data
CHAR_UUID_PPG = "46a970e2-0d79-11e6-bdf4-0800200c9a66"  # Photoplethysmogram
CHAR_UUID_TEMP = "46a970e3-0d79-11e6-bdf4-0800200c9a66"  # Temperature
CHAR_UUID_ACCEL = "46a970e4-0d79-11e6-bdf4-0800200c9a66"  # Accelerometer data
CHAR_UUID_LED_CTRL = "46a970e5-0d79-11e6-bdf4-0800200c9a66"  # LED control
CHAR_UUID_BATTERY = "46a970e6-0d79-11e6-bdf4-0800200c9a66"  # Battery level

# Data configuration
MAX_SAMPLES = 2000  # Maximum number of samples to collect
SAMPLING_RATE = 125  # Default sampling rate in Hz
DEVICE_NAME_PREFIX = "MAXREFDES"  # Prefix for device name filtering

# Data storage
ecg_data = []
ppg_data = []
temperature_data = []
accel_data = []
battery_level = 0
timestamp_data = []

# Visualization flags
SHOW_PLOTS = True

def parse_ecg_data(data):
    """
    Parse ECG data from MAXREFDES106
    
    Format: 3 bytes per sample (24-bit signed integer)
    """
    samples = len(data) // 3
    result = []
    
    for i in range(samples):
        # Extract 24-bit sample (3 bytes) and convert to signed int
        value = (data[i*3] << 16) | (data[i*3+1] << 8) | data[i*3+2]
        # Convert 2's complement if necessary
        if value & 0x800000:
            value = value - 0x1000000
        result.append(value)
    
    return result

def parse_ppg_data(data):
    """
    Parse PPG (photoplethysmogram) data from MAXREFDES106
    
    Format: Multiple channels of data with headers
    """
    # Simplified parsing - actual format may be more complex
    samples = len(data) // 4  # Assuming 4 bytes per sample
    result = []
    
    for i in range(samples):
        value = struct.unpack(">I", data[i*4:(i*4)+4])[0]
        result.append(value)
    
    return result

def parse_temperature_data(data):
    """
    Parse temperature data from MAX30205 sensor
    
    Format: IEEE-754 floating point (4 bytes)
    """
    if len(data) >= 4:
        temperature = struct.unpack("<f", data[0:4])[0]
        return temperature
    return None

def parse_accelerometer_data(data):
    """
    Parse accelerometer data from LIS2DH12
    
    Format: 3 channels (X, Y, Z) of 16-bit signed integers
    """
    if len(data) >= 6:
        x = struct.unpack("<h", data[0:2])[0]
        y = struct.unpack("<h", data[2:4])[0]
        z = struct.unpack("<h", data[4:6])[0]
        return (x, y, z)
    return None

def ecg_notification_handler(sender, data):
    """Handle incoming ECG data notifications"""
    parsed_data = parse_ecg_data(data)
    ecg_data.extend(parsed_data)
    timestamp = time.time()
    timestamp_data.extend([timestamp] * len(parsed_data))
    print(f"Received ECG data: {len(parsed_data)} samples")

def ppg_notification_handler(sender, data):
    """Handle incoming PPG data notifications"""
    parsed_data = parse_ppg_data(data)
    ppg_data.extend(parsed_data)
    print(f"Received PPG data: {len(parsed_data)} samples")

def temperature_notification_handler(sender, data):
    """Handle incoming temperature data notifications"""
    temp = parse_temperature_data(data)
    if temp is not None:
        temperature_data.append(temp)
        print(f"Received temperature: {temp:.2f}°C")

def accelerometer_notification_handler(sender, data):
    """Handle incoming accelerometer data notifications"""
    accel = parse_accelerometer_data(data)
    if accel is not None:
        accel_data.append(accel)
        print(f"Received accelerometer data: X={accel[0]}, Y={accel[1]}, Z={accel[2]}")

def battery_notification_handler(sender, data):
    """Handle incoming battery level notifications"""
    global battery_level
    if len(data) >= 1:
        battery_level = data[0]
        print(f"Battery level: {battery_level}%")

async def plot_data():
    """Plot collected data"""
    if not SHOW_PLOTS:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot ECG data
    if ecg_data:
        plt.subplot(3, 1, 1)
        plt.plot(ecg_data[-500:])
        plt.title("ECG Data (last 500 samples)")
        plt.ylabel("Amplitude")
    
    # Plot PPG data
    if ppg_data:
        plt.subplot(3, 1, 2)
        plt.plot(ppg_data[-500:])
        plt.title("PPG Data (last 500 samples)")
        plt.ylabel("Amplitude")
    
    # Plot temperature data
    if temperature_data:
        plt.subplot(3, 1, 3)
        plt.plot(temperature_data)
        plt.title("Temperature Data")
        plt.ylabel("Temperature (°C)")
    
    plt.tight_layout()
    plt.show()

async def save_data(filename_prefix="maxrefdes106_data"):
    """Save collected data to CSV files"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save ECG data
    if ecg_data:
        filename = f"{filename_prefix}_ecg_{timestamp}.csv"
        with open(filename, 'w') as f:
            f.write("timestamp,ecg_value\n")
            for i, (ts, value) in enumerate(zip(timestamp_data, ecg_data)):
                f.write(f"{ts},{value}\n")
        print(f"ECG data saved to {filename}")
    
    # Save temperature data
    if temperature_data:
        filename = f"{filename_prefix}_temp_{timestamp}.csv"
        with open(filename, 'w') as f:
            f.write("temperature_celsius\n")
            for temp in temperature_data:
                f.write(f"{temp}\n")
        print(f"Temperature data saved to {filename}")
    
    # Save accelerometer data
    if accel_data:
        filename = f"{filename_prefix}_accel_{timestamp}.csv"
        with open(filename, 'w') as f:
            f.write("accel_x,accel_y,accel_z\n")
            for x, y, z in accel_data:
                f.write(f"{x},{y},{z}\n")
        print(f"Accelerometer data saved to {filename}")

async def find_device():
    """Scan for MAXREFDES106 device"""
    print("Scanning for MAXREFDES106 device...")
    devices = await BleakScanner.discover()
    
    for device in devices:
        if device.name and DEVICE_NAME_PREFIX in device.name:
            print(f"Found device: {device.name} ({device.address})")
            return device
    
    return None

async def main():
    """Main function to connect to device and receive data"""
    device = await find_device()
    
    if not device:
        print(f"No device with prefix '{DEVICE_NAME_PREFIX}' found. Please make sure the device is powered on and in range.")
        return
    
    print(f"Connecting to {device.name}...")
    
    async with BleakClient(device.address) as client:
        print(f"Connected: {client.is_connected}")
        
        # Subscribe to notifications
        await client.start_notify(CHAR_UUID_ECG, ecg_notification_handler)
        await client.start_notify(CHAR_UUID_PPG, ppg_notification_handler)
        await client.start_notify(CHAR_UUID_TEMP, temperature_notification_handler)
        await client.start_notify(CHAR_UUID_ACCEL, accelerometer_notification_handler)
        await client.start_notify(CHAR_UUID_BATTERY, battery_notification_handler)
        
        print("Subscribed to sensor notifications. Collecting data...")
        
        try:
            # Collect data for specified time or until max samples
            start_time = time.time()
            
            while len(ecg_data) < MAX_SAMPLES:
                # Check every second
                await asyncio.sleep(1)
                elapsed = time.time() - start_time
                print(f"Data collection in progress... ({len(ecg_data)} ECG samples, {elapsed:.1f}s elapsed)")
                
                # Stop after 30 seconds
                if elapsed > 30:
                    break
            
            # Stop notifications
            await client.stop_notify(CHAR_UUID_ECG)
            await client.stop_notify(CHAR_UUID_PPG)
            await client.stop_notify(CHAR_UUID_TEMP)
            await client.stop_notify(CHAR_UUID_ACCEL)
            await client.stop_notify(CHAR_UUID_BATTERY)
            
            print("Data collection complete.")
            print(f"Collected {len(ecg_data)} ECG samples")
            print(f"Collected {len(ppg_data)} PPG samples")
            print(f"Collected {len(temperature_data)} temperature readings")
            print(f"Collected {len(accel_data)} accelerometer readings")
            
            # Save data to files
            await save_data()
            
            # Plot data if requested
            await plot_data()
            
        except Exception as e:
            print(f"Error during data collection: {e}")
        
        print("Disconnecting...")
    
    print("Disconnected")

if __name__ == "__main__":
    asyncio.run(main())