import asyncio
import bleak
import device_model
import json
import datetime
import os

# Scanned devices
devices = []
# BLEDevice
BLEDevice = None
# File for saving data
data_file = None

# Scan Bluetooth devices and filter names
async def scan():
    global devices
    global BLEDevice
    find = []
    print("Searching for Bluetooth devices......")
    try:
        devices = await bleak.BleakScanner.discover(timeout=20.0)
        print("Search ended")
        for d in devices:
            if d.name is not None and "WT" in d.name:
                find.append(d)
                print(d)
        
        if len(find) == 0:
            print("No devices found in this search!")
        elif len(find) == 1:
            # If only one WT device found, select it automatically
            BLEDevice = find[0]
            print(f"Automatically selecting the only found device: {BLEDevice.name} ({BLEDevice.address})")
        else:
            # Multiple devices found, ask user to select
            print("\nMultiple WT devices found. Please select one:")
            for i, device in enumerate(find):
                print(f"{i+1}. {device.name} - {device.address}")
            
            user_input = input("Please enter the Mac address you want to connect to: ")
            for d in devices:
                if d.address == user_input:
                    BLEDevice = d
                    break
    except Exception as ex:
        print("Bluetooth search failed to start")
        print(ex)

# Specify MAC address to search and connect devices
async def scanByMac(device_mac):
    global BLEDevice
    print("Searching for Bluetooth devices......")
    BLEDevice = await bleak.BleakScanner.find_device_by_address(device_mac, timeout=20)

# Create a filename with date, time and MAC address
def create_data_file(mac_address):
    global data_file
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

# This method will be called when data is updated
def updateData(DeviceModel):
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Get the data dictionary
    data = DeviceModel.deviceData
    
    # Print to console
    print(data)
    
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

async def main():
    global BLEDevice, data_file
    
    # Method 1: Broadcast search and connect Bluetooth devices
    await scan()
    
    # # Method 2: Specify MAC address to search and connect devices
    # BLEDevice = await scanByMac("C6:46:21:41:0B:BD")
    
    if BLEDevice is not None:
        try:
            # Create data file
            create_data_file(BLEDevice.address)
            
            # Create device
            device = device_model.DeviceModel("MyBle5.0", BLEDevice, updateData)
            
            # Start connecting devices
            print("Connecting to device and collecting data. Press Ctrl+C to stop.")
            await device.openDevice()
        except KeyboardInterrupt:
            print("\nData collection stopped by user.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Close the file
            if data_file and not data_file.closed:
                data_file.close()
                print("Data file closed.")
    else:
        print("This BLEDevice was not found!!")

if __name__ == '__main__':
    # Use a single event loop for all operations
    asyncio.run(main())