"""Network diagnostics for camera connectivity"""
import socket
import time
from typing import Dict, Optional
import threading

from ..config import DEBUG

# ---------------- Network Diagnostics ----------------
def get_local_ip():
    """Get the local IP address of this machine"""
    import socket
    try:
        # Connect to a remote host to determine local IP (doesn't actually send data)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "Unable to determine"

def print_network_info():
    """Print network information for debugging"""
    import socket
    hostname = socket.gethostname()
    local_ip = get_local_ip()

    print("\n" + "="*60)
    print(" ANPR SYSTEM - NETWORK INFORMATION")
    print("="*60)
    print(f" Hostname: {hostname}")
    print(f" Local IP: {local_ip}")
    print(f" Port: 8000")
    print("-"*60)
    print(" Access URLs:")
    print(f"  - Local:    http://localhost:8000")
    print(f"  - Network:  http://{local_ip}:8000")
    print("-"*60)
    print(" Available Endpoints:")
    print("  - http://{ip}:8000/              (Status page)")
    print("  - http://{ip}:8000/health         (Health check)")
    print("  - http://{ip}:8000/cameras        (List cameras)")
    print("  - http://{ip}:8000/video_feed/webcam")
    print("  - http://{ip}:8000/api/video_feed/webcam")
    print("-"*60)
    print(" Firewall Configuration:")
    print("  On Windows, ensure port 8000 is allowed:")
    print('  netsh advfirewall firewall add rule name="ANPR Port 8000"')
    print("  dir=in action=allow protocol=TCP localport=8000")
    print("-"*60)
    print(" For External Access (using ngrok):")
    print("  ngrok http 8000")
    print("  Then use the ngrok URL from another PC")
    print("="*60 + "\n")

