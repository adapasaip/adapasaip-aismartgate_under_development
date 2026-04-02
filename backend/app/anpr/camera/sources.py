"""Camera detection and source initialization"""
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import threading
import time
from typing import Dict, Optional, List, Tuple

from ..config import DEBUG

def detect_oak_cameras():
    """
    Detect OAK cameras WITHOUT opening connections (just check availability).
    """
    if not DEPTHAI_AVAILABLE:
        return []
    try:
        # Don't actually connect, just list
        devices = dai.Device.getAllAvailableDevices()
        if devices:
            return [{'name': 'OAK-0', 'type': 'oak', 'device_info': None}]
        return []
    except Exception as e:
        print(f"[WARN] OAK detection: {e}")
        return []

def detect_usb_cameras():
    usb_cameras = []
    # Suppress OpenCV's C++ warnings during USB camera probing by redirecting stderr
    
    try:
        # Save original stderr and redirect to /dev/null
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        
        try:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    frame_found = False
                    for attempt in range(2):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame_found = True
                            break
                        time.sleep(0.1)
                    
                    if cap.isOpened():
                        usb_cameras.append({'index': i, 'type': 'usb', 'name': f'USB_Camera_{i}'})
                        if frame_found:
                            print(f"[INFO] Detected USB camera at index {i}")
                        else:
                            print(f"[INFO] ⚠ Detected USB camera at index {i} (no initial frame, may work during streaming)")
                    cap.release()
                else:
                    cap.release()
        finally:
            # Restore stderr
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)
            os.close(devnull)
    except:
        # Fallback: just skip USB detection if stderr manipulation fails
        pass
    
    return usb_cameras

def detect_ipwebcam_cameras():
    """
    Detect and list available IP Webcam streams (Android IP Webcam app).
    Returns list of IP webcam sources that can be connected to.
    
    IP Webcam app streams at: http://<phone_ip>:8080/video
    Example: http://192.168.1.100:8080/video
    
    This function checks if common local IP ranges are available for IP webcam.
    You can also manually add cameras using the camera configuration.
    """
    ipwebcam_cameras = []
    
    # Common mobile IP ranges to check (adjust based on your network)
    ip_ranges = [
        # Local network scanner - check x.x.x.200-250 range on common subnets
        '192.168.1',
        '192.168.0',
        '10.0.0',
    ]
    
    # Try to detect IP webcam on common ports/IPs
    # NOTE: This is a quick check - users should manually configure their IP webcam URLs
    # or you can add them via API/config file
    
    print("[INFO] IP Webcam detection: Users should manually configure IP Webcam URLs")
    print("[INFO] Example: http://<your_phone_ip>:8080/video")
    
    return ipwebcam_cameras

def initialize_ipwebcam(url):
    """
    Initialize IP Webcam stream from Android IP Webcam app.
    
    Args:
        url: Full URL to IP Webcam stream (e.g., http://192.168.1.100:8080/video)
    
    Returns:
        VideoCapture object if successful, None otherwise
    """
    if not url:
        print("[ERROR] IP Webcam URL is empty")
        return None
    
    try:
        print(f"[INFO] Connecting to IP Webcam: {url}")
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print(f"[ERROR] Failed to open IP Webcam stream: {url}")
            return None
        
        # Test frame read
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[INFO] IP Webcam connected successfully: {url}")
            return cap
        else:
            print(f"[WARN] IP Webcam opened but frame read failed: {url}")
            # Still return the cap - it might work during streaming
            return cap
            
    except Exception as e:
        print(f"[ERROR] IP Webcam initialization failed: {e}")
        return None

def initialize_oak_camera(device_info=None, direct=True):
    """
    Initialize OAK camera using STILL output mode (preview_oak_camera.py logic).
    NO DEMO MODE - Requires actual hardware to be connected.
    """
    print("[INFO] ====== OAK CAMERA INITIALIZATION ======")
    print("[INFO] Attempting OAK camera initialization (STILL output mode)...")
    
    if not DEPTHAI_AVAILABLE:
        print("[ERROR] DepthAI not available - OAK camera cannot be initialized")
        raise RuntimeError("DepthAI not installed. Cannot use OAK camera.")
    
    try:
        # Check if devices are available
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print("[ERROR] No OAK devices detected via DepthAI")
            raise RuntimeError("No OAK camera detected. Is it connected via USB?")
        
        print(f"[INFO] Found {len(devices)} OAK device(s)")
        
        # Initialize OAK camera with STILL output
        oak = OAKCamera()
        if oak.start():
            print(f"[INFO] OAK CAMERA READY (STILL OUTPUT MODE)")
            print(f"[INFO] =====================================\n")
            
            return {
                'type': 'oak_still',
                'device': oak,
                'pipeline': None,
                'queue_name': 'still',
                'is_mock': False
            }
        else:
            raise RuntimeError("OAK camera failed to start")
        
    except Exception as e:
        print(f"[ERROR] OAK initialization error: {type(e).__name__}: {e}")
        raise RuntimeError(f"OAK camera initialization failed: {e}")

def read_frame_from_source(source):
    try:
        # Handle OAK camera with STILL output mode
        if isinstance(source, dict) and source.get('type') == 'oak_still':
            oak_device = source.get('device')
            if oak_device is not None and isinstance(oak_device, OAKCamera):
                frame = oak_device.get_frame()
                if frame is not None:
                    return True, frame
            return False, None
        
        # Handle OAK camera (DepthAI)
        if isinstance(source, dict) and source.get('type') in ['oak_depthai_simple', 'oak_depthai', 'oak_usb']:
            if source.get('type') == 'oak_depthai_simple' or source.get('type') == 'oak_depthai':
                # DepthAI OAK camera with pipeline
                device = source.get('device')
                queue_name = source.get('queue_name', 'rgb')
                try:
                    q = device.getOutputQueue(name=queue_name, maxSize=4, blocking=False)
                    if q is not None:
                        in_data = q.get()
                        if in_data is not None:
                            frame = in_data.getCvFrame()
                            if frame is not None:
                                return True, frame
                except Exception as e:
                    print(f"[WARN] DepthAI queue error: {e}")
                return False, None
            elif source.get('type') == 'oak_usb':
                # USB video device fallback for OAK camera
                cap = source.get('device')
                if cap is not None:
                    try:
                        return cap.read()
                    except Exception as e:
                        print(f"[WARN] USB video device read error: {e}")
                return False, None
        # Handle standard VideoCapture (USB webcams, IP Webcams, RTSP, etc)
        elif isinstance(source, cv2.VideoCapture):
            try:
                ret, frame = source.read()
                # For IP Webcam and streaming sources, frame might be delayed
                # If frame is None but ret is True, it's likely a streaming issue
                if ret and frame is not None:
                    return True, frame
                elif ret and frame is None:
                    # Frame is delayed, return False to trigger retry
                    return False, None
                else:
                    # ret is False, stream might be closed or unavailable
                    return False, None
            except Exception as e:
                print(f"[WARN] VideoCapture read error: {e}")
                return False, None
    except Exception as e:
        print(f"[WARN] Error reading frame: {e}")
    return False, None
# -------- END Camera Detection --------

# Processes OCR requests asynchronously to prevent blocking the main detection pipeline
def ocr_worker_background():
    """
    Background worker thread for OCR processing

class OAKCamera:
    """OAK camera wrapper for VIDEO preview streaming."""
    def __init__(self):
        self.running = False
        self.pipeline = None
        self.device = None
        self.queue = None
        self.shutdown = False
        self.last_frame = None
        
    def start(self):
        """Initialize and start the OAK camera stream."""
        try:
            print("[INFO] Initializing OAK camera...")
            self.pipeline = dai.Pipeline()
            
            cam = self.pipeline.create(dai.node.ColorCamera)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setInterleaved(False)
            cam.setFps(30)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam.setPreviewSize(1280, 720)
            cam.setVideoSize(1280, 720)
            
            xout_video = self.pipeline.create(dai.node.XLinkOut)
            xout_video.setStreamName("video")
            cam.video.link(xout_video.input)
            
            print("[INFO] Starting device...")
            self.device = dai.Device(self.pipeline)
            print("[INFO] Creating queues...")
            self.queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
            
            self.running = True
            time.sleep(2)  # Warm up
            
            print("[INFO] Camera stream ready")
            
            time.sleep(1)
            test_frame = self.get_frame()
            if test_frame is not None:
                print(f"[INFO] Camera ready! Frame shape: {test_frame.shape}")
            else:
                print("[WARN] No test frame captured")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Camera init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame(self):
        """Get a frame from the camera stream."""
        if not self.running or self.shutdown:
            return None
        
        try:
            msg = self.queue.get()
            if msg is None:
                return None
            
            frame = msg.getCvFrame()
            return frame if frame is not None and frame.size > 0 else None
            
        except Exception as e:
            print(f"[WARN] Frame error: {e}")
            return None
    
    def stop(self):
        """Stop the camera."""
        print("[INFO] Stopping camera...")
        self.shutdown = True
        self.running = False
        time.sleep(0.5)
        
        if self.device:
