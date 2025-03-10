import asyncio
import bleak
import device_model
import json
import datetime
import os

# Create a filename with date, time and MAC address
def create_data_file(mac_address):
    # Create a datetime string in the format: YYYYMMDD_HHMMSS
    now = datetime.datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    
    # Clean up the MAC address for use in filename (remove colons or other non-alphanumeric chars)
    clean_mac = "".join(c for c in mac_address if c.isalnum())
    
    # Create directory if it doesn't exist
    os.makedirs("sensor_data", exist_ok=True)
    
    # Create filename
    filename = f"sensor_data/sensor_{date_time_str}_{clean_mac}.csv"
    
    # Open file and write header
    data_file = open(filename, "w")
    data_file.write("Timestamp,AccX,AccY,AccZ,AsX,AsY,AsZ,AngX,AngY,AngZ,HX,HY,HZ,Q0,Q1,Q2,Q3\n")
    
    print(f"Data will be saved to: {filename}")
    return data_file

# Factory function to create a data update callback for a specific file
def create_update_callback(data_file):
    def update_data(device_model):
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Get the data dictionary
        data = device_model.deviceData
        
        # Print to console with device name/address for clarity
        print(f"{device_model.deviceName} ({device_model.BLEDevice.address}): {data}")
        
        # Write to file if it's open
        if data_file and not data_file.closed:
            # Create a CSV line
            line = f"{timestamp},{data.get('AccX', '')},{data.get('AccY', '')},{data.get('AccZ', '')}"
            line += f",{data.get('AsX', '')},{data.get('AsY', '')},{data.get('AsZ', '')}"
            line += f",{data.get('AngX', '')},{data.get('AngY', '')},{data.get('AngZ', '')}"
            line += f",{data.get('HX', '')},{data.get('HY', '')},{data.get('HZ', '')}"
            line += f",{data.get('Q0', '')},{data.get('Q1', '')},{data.get('Q2', '')},{data.get('Q3', '')}\n"
            
            data_file.write(line)
            data_file.flush()  # Ensure data is written immediately
    
    return update_data

# Scan Bluetooth devices and filter by WT in the name
async def scan_wt_devices():
    print("Searching for WT Bluetooth devices...")
    try:
        devices = await bleak.BleakScanner.discover(timeout=20.0)
        print(f"Search ended, discovered {len(devices)} total devices")
        
        # Filter for devices with "WT" in the name
        wt_devices = []
        for d in devices:
            if d.name is not None and "WT" in d.name:
                wt_devices.append(d)
                print(f"Found WT device: {d.name} ({d.address})")
        
        if len(wt_devices) == 0:
            print("No WT devices found in this search!")
        
        return wt_devices
            
    except Exception as ex:
        print("Bluetooth search failed")
        print(ex)
        return []

# Connect to a single device and start collecting data
async def connect_and_collect(ble_device):
    try:
        # Create data file for this device
        data_file = create_data_file(ble_device.address)
        
        # Create device with a unique callback for this device/file
        device = device_model.DeviceModel(
            f"WT-{ble_device.address[-5:]}",  # Create a short name based on the MAC
            ble_device, 
            create_update_callback(data_file)
        )
        
        # Start connecting to the device
        print(f"Connecting to device {ble_device.name} ({ble_device.address})...")
        await device.openDevice()
        
        return device, data_file
    except Exception as e:
        print(f"Error connecting to {ble_device.address}: {e}")
        return None, None

async def main():
    # Find all WT devices
    wt_devices = await scan_wt_devices()
    
    if not wt_devices:
        print("No WT devices found. Exiting.")
        return
    
    print(f"Preparing to connect to {len(wt_devices)} WT devices...")
    
    # Store connected devices and their data files
    connected_devices = []
    data_files = []
    
    try:
        # Connect to all devices simultaneously
        connection_tasks = [connect_and_collect(device) for device in wt_devices]
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Store successful connections
        for device, file in results:
            if device and file:
                connected_devices.append(device)
                data_files.append(file)
        
        # Keep the program running until interrupted
        print(f"Connected to {len(connected_devices)} devices. Collecting data. Press Ctrl+C to stop.")
        # Keep the program running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nData collection stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close all files
        for file in data_files:
            if file and not file.closed:
                file.close()
                print(f"Data file closed: {file.name}")
        
        # Disconnect from devices (if device_model has a close method)
        for device in connected_devices:
            try:
                # Check if the device model has a closeDevice method
                if hasattr(device, 'closeDevice'):
                    await device.closeDevice()
                    print(f"Disconnected from {device.deviceName}")
            except Exception as e:
                print(f"Error disconnecting from {device.deviceName}: {e}")

if __name__ == '__main__':
    # Use a single event loop for all operations
    asyncio.run(main())