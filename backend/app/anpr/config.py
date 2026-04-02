"""Configuration, constants, and color mappings for ANPR system"""
import os
import time
from pathlib import Path

    DEPTHAI_AVAILABLE = False
    dai = None

# Configuration
DEBUG = False

JPEG_QUALITY = 80
JPEG_QUALITY_HIGH = 90

STREAM_WIDTH = 960
STREAM_HEIGHT = 540
DETECTION_WIDTH = 320  # OPTIMIZED: Reduced from 416 for faster YOLO (2.6x speedup)
DETECTION_HEIGHT = 320  # OPTIMIZED: Reduced from 416 for faster YOLO

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

ADAPTIVE_RESOLUTION = True

PROCESS_EVERY_N_FRAMES = 1
TARGET_FPS = 15
YOLO_PROCESS_EVERY_N = 1  # Run YOLO on every frame for real-time detection
FRAME_SKIP_THRESHOLD = 0
FORCE_YOLO_INITIAL_FRAMES = 8

FRAME_BUFFER_SIZE = 4
MAX_FRAME_QUEUE = 4

CONFIDENCE_THRESHOLD = 0.35  # LOWERED from 0.45 for small plates (bikes/rickshaws)
CONFIDENCE_THRESHOLD_SMALL_PLATE = 0.30  # NEW: Even lower for very small plates
PLATE_SCALE_FACTOR = 1.03
PLATE_MIN_NEIGHBORS = 3
BUFFER_SIZE = FRAME_BUFFER_SIZE
DUPLICATE_PLATE_INTERVAL = 10
CAMERA_COOLDOWN = 8
PLATE_OVERLAP_THRESHOLD = 0.7

MIN_PLATE_WIDTH = 40  # Minimum plate width in pixels (bikes: ~30-40px)
MIN_PLATE_HEIGHT = 20  # Minimum plate height in pixels
MAX_SMALL_PLATE_AREA = 3000  # Pixels² - plates smaller than this get special handling
OCR_UPSCALE_FACTOR = 2.0  # Upscale small plates 2x before OCR for better text recognition

STOPPED_FRAME_THRESHOLD = 3
STOPPED_MOVEMENT_PIXELS = 5

EXIT_TIME_BUFFER = 25
EXIT_DETECTION_CONFIDENCE = 0.50

OCR_CONFIDENCE_THRESHOLD = 0.65
OCR_CONSISTENCY_FRAMES = 3
OCR_CONSISTENCY_TIMEOUT = 6

PLATE_SIMILARITY_THRESHOLD = 0.85
PLATE_DUPLICATE_WINDOW = 50
PLATE_FUZZY_MATCH_ENABLED = True
# Entry cameras create new records, Exit cameras update existing records only

# Paths - using relative path from backend/app/anpr to root data/ folder (4 levels up)
BASE_DIR = Path(__file__).parent.parent.parent.parent / "data"
DIR_PLATES = BASE_DIR / "plates"

# Only create the plates directory (for plate images)
DIR_PLATES.mkdir(parents=True, exist_ok=True)

DETECTIONS_JSON = BASE_DIR / "detections.json"

def ensure_json_array(path: Path):
    """JSON LINES FORMAT: Initialize empty JSON Lines file (no data needed)."""
    if not path.exists():
        try:
            # Create empty file for JSON Lines format
            # JSON Lines is append-only, so we just need an empty file
            with path.open("w", encoding="utf-8") as f:
                pass  # Create empty file
        except Exception as e:
            print(f"[WARN] Could not create {path}: {e}")
    # If file exists, don't modify it - just use as-is in JSON Lines format

ensure_json_array(DETECTIONS_JSON)

VEHICLES_JSON = BASE_DIR / "vehicles.json"  # DEPRECATED: Use detections.json instead

HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
HAAR_PATH = str(BASE_DIR / "haarcascade_plate.xml")

# ---------------- Color mapping (BGR for OpenCV) ----------------
COLOR_FRONT = (0, 255, 255)     # Yellow  - Front view
COLOR_REAR = (255, 0, 255)      # Purple  - Rear view
COLOR_DETECTING = (0, 0, 255)   # Red     - Detecting / trying to detect
COLOR_PLATE = (0, 255, 0)       # Green   - Plate detected
COLOR_CAPTURED = (255, 0, 0)    # Blue    - Object/image captured
COLOR_TEXT_BG = (0, 0, 0)       # Black background for label contrast

def get_ocr_validation_report(raw_text, is_valid, normalized, confidence):
    """Generate validation report for logging."""
    return {
        'raw_text': raw_text,
        'is_valid': is_valid,
        'normalized_plate': normalized,
        'confidence': confidence,
        'validation_status': 'ACCEPT' if is_valid else 'REJECT',
        'timestamp': time.time()
    }

def log_ocr_validation(report, camera_id=None):
    """Log OCR validation result."""
    status = report['validation_status']
    plate = report['normalized_plate'] if report['is_valid'] else report['raw_text']
    conf = f"{report['confidence']:.2%}" if report['confidence'] > 0 else "N/A"
    
    if DEBUG:
        if report['is_valid']:
            print(f"[OCR_VALID] Camera: {camera_id or 'N/A'} | Plate: {plate} | Confidence: {conf}")
        else:
            print(f"[OCR_REJECT] Camera: {camera_id or 'N/A'} | Raw: {report['raw_text']} | Reason: Invalid format")

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
            try:
                self.device.close()
                print("[INFO] Device closed")
            except:
                pass
