import socket
import subprocess
import ipaddress

# Replace with your network range
network = "192.168.0.1"

for ip in ipaddress.IPv4Network(network):
    ip_str = str(ip)
    # Skip network and broadcast addresses
    if ip_str.endswith('.0') or ip_str.endswith('.255'):
        continue
    
    proc = subprocess.Popen(['ping', '-c', '1', '-W', '1', ip_str], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    
    if proc.returncode == 0:
        try:
            hostname = socket.gethostbyaddr(ip_str)[0]
        except socket.herror:
            hostname = "Unknown"
        print(f"IP: {ip_str} - Hostname: {hostname}")