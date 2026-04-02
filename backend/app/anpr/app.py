"""Main ANPR application orchestrator and initialization"""
import threading
import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path

from .config import DEBUG, BASE_DIR
from .camera.sources import detect_oak_cameras, detect_usb_cameras, detect_ipwebcam_cameras, OAKCamera
from .detectors.onnx_wrapper import ONNXPlateDetector
from .ocr.engine import OCRWorker
from .storage.persistence import persist_data, load_vehicles
from .streaming.mjpeg import stream_generator
from .processing.pipeline import process_frame_wrapper
from .utils.frame import FrameDeduplicator
from .utils.bbox import PlateBBoxStabilizer

# Global state management
camera_frames = defaultdict(lambda: {"frame": None, "timestamp": 0})
camera_workers = {}
camera_threads = {}
ocr_workers = {}
bbox_stabilizers = {}
frame_deduplicators = {}

def initialize_application():
    """Initialize ANPR application with all modules"""
    global camera_frames, camera_workers, camera_threads
    
    if DEBUG:
        print("[INFO] Initializing ANPR application...")
    
    # Load configuration
    cameras = detect_available_cameras()
    
    # Initialize workers
    for camera_id in cameras:
        initialize_camera_worker(camera_id)
    
    if DEBUG:
        print(f"[INFO] Application initialized with {len(cameras)} cameras")
    
    return cameras

def detect_available_cameras():
    """Detect all available camera sources"""
    cameras = {}
    
    cameras.update(detect_oak_cameras())
    cameras.update(detect_usb_cameras())
    cameras.update(detect_ipwebcam_cameras())
    
    return cameras

def initialize_camera_worker(camera_id: str):
    """Initialize worker thread for a camera"""
    global camera_workers, camera_threads, ocr_workers, bbox_stabilizers, frame_deduplicators
    
    # Initialize OCR worker
    ocr_workers[camera_id] = OCRWorker()
    
    # Initialize bounding box stabilizer
    bbox_stabilizers[camera_id] = PlateBBoxStabilizer()
    
    # Initialize frame deduplicator
    frame_deduplicators[camera_id] = FrameDeduplicator()
    
    # Start detector thread
    detector = ONNXPlateDetector()
    worker_thread = threading.Thread(
        target=process_camera_stream,
        args=(camera_id,),
        daemon=True
    )
    worker_thread.start()
    camera_threads[camera_id] = worker_thread

def process_camera_stream(camera_id: str):
    """Process frames from a camera source"""
    while True:
        try:
            frame = camera_frames[camera_id].get("frame")
            if frame is not None:
                process_frame_wrapper(frame, camera_id)
        except Exception as e:
            if DEBUG:
                print(f"[ERROR] Error processing camera {camera_id}: {e}")
            time.sleep(0.1)

def cleanup_application():
    """Cleanup and shutdown application"""
    global camera_threads
    
    if DEBUG:
        print("[INFO] Shutting down ANPR application...")
    
    for thread in camera_threads.values():
        if thread.is_alive():
            thread.join(timeout=1.0)

