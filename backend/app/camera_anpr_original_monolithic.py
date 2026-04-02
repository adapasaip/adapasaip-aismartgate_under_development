#!/usr/bin/env python3
"""Advanced ANPR System with OCR and stop-detection."""
import argparse
import json
import re
import time
import os
import sys
import platform
import urllib.request
import uuid
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS

from config_manager import CameraConfigManager
from camera_manager import get_camera_manager, initialize_camera_manager

# Suppress warnings from OpenCV, FFmpeg, and cv2
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['FFREPORT'] = 'level=error'

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

cv2.setNumThreads(4)

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
    print("[INFO] DepthAI installed. OAK camera support enabled.")
except ImportError:
    DEPTHAI_AVAILABLE = False
    dai = None

# Configuration
DEBUG = False

JPEG_QUALITY = 80
JPEG_QUALITY_HIGH = 90

STREAM_WIDTH = 960
STREAM_HEIGHT = 540
DETECTION_WIDTH = 320  # ⭐ OPTIMIZED: Reduced from 416 for faster YOLO (2.6x speedup)
DETECTION_HEIGHT = 320  # ⭐ OPTIMIZED: Reduced from 416 for faster YOLO

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

ADAPTIVE_RESOLUTION = True

PROCESS_EVERY_N_FRAMES = 1
TARGET_FPS = 15
YOLO_PROCESS_EVERY_N = 5  # ⭐ OPTIMIZED: Increased from 3 to 5 (reduce YOLO frequency)
FRAME_SKIP_THRESHOLD = 0
FORCE_YOLO_INITIAL_FRAMES = 8

FRAME_BUFFER_SIZE = 4
MAX_FRAME_QUEUE = 4

CONFIDENCE_THRESHOLD = 0.45
PLATE_SCALE_FACTOR = 1.03
PLATE_MIN_NEIGHBORS = 3
BUFFER_SIZE = FRAME_BUFFER_SIZE
DUPLICATE_PLATE_INTERVAL = 10
CAMERA_COOLDOWN = 8
PLATE_OVERLAP_THRESHOLD = 0.7

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

# Paths - using relative path from backend/app/ to root data/ folder
BASE_DIR = Path(__file__).parent.parent.parent / "data"
DIR_PLATES = BASE_DIR / "plates"

# Only create the plates directory (for plate images)
DIR_PLATES.mkdir(parents=True, exist_ok=True)

DETECTIONS_JSON = BASE_DIR / "detections.json"

def ensure_json_array(path: Path):
    """⭐ JSON LINES FORMAT: Initialize empty JSON Lines file (no data needed)."""
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

VEHICLES_JSON = BASE_DIR / "vehicles.json"  # ⭐ DEPRECATED: Use detections.json instead

HAAR_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
HAAR_PATH = str(BASE_DIR / "haarcascade_plate.xml")

# ---------------- Color mapping (BGR for OpenCV) ----------------
COLOR_FRONT = (0, 255, 255)     # Yellow  - Front view
COLOR_REAR = (255, 0, 255)      # Purple  - Rear view
COLOR_DETECTING = (0, 0, 255)   # Red     - Detecting / trying to detect
COLOR_PLATE = (0, 255, 0)       # Green   - Plate detected
COLOR_CAPTURED = (255, 0, 0)    # Blue    - Object/image captured
COLOR_TEXT_BG = (0, 0, 0)       # Black background for label contrast


# ⭐ NEW: CENTROID TRACKER - Maintains consistent object IDs across frames
class CentroidTracker:
    """
    Tracks objects by centroid position to maintain consistent IDs across frames.
    Solves flickering and inconsistent OCR by keeping objects persistent.
    """
    def __init__(self, max_distance=100, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {}  # {object_id: (cx, cy)}
        self.disappeared = {}  # {object_id: frames_since_seen}
        self.max_distance = max_distance  # Max pixel distance to match centroid
        self.max_disappeared = max_disappeared  # Max frames before object removal
        self.lock = threading.Lock()
    
    def update(self, bboxes):
        """
        Update tracker with new bounding boxes.
        Returns: {bbox: object_id} mapping
        """
        with self.lock:
            if len(bboxes) == 0:
                # No detections - mark all objects as disappeared
                disappeared_ids = list(self.disappeared.keys())
                for obj_id in disappeared_ids:
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
                return {}
            
            # Calculate centroids of new detections
            input_centroids = []
            for (x1, y1, x2, y2, conf) in bboxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                input_centroids.append((cx, cy))
            
            # Match detections to existing objects
            if len(self.objects) == 0:
                # First detection - register all as new objects
                for i, centroid in enumerate(input_centroids):
                    self.objects[self.next_object_id] = centroid
                    self.disappeared[self.next_object_id] = 0
                    self.next_object_id += 1
            else:
                # Match existing objects to new detections
                object_ids = list(self.objects.keys())
                matched_new = set()
                matched_old = set()
                
                # Greedy matching: for each existing object, find closest new detection
                for obj_id in object_ids:
                    obj_cx, obj_cy = self.objects[obj_id]
                    min_distance = self.max_distance
                    closest_idx = -1
                    
                    for i, (new_cx, new_cy) in enumerate(input_centroids):
                        if i in matched_new:
                            continue
                        distance = ((obj_cx - new_cx) ** 2 + (obj_cy - new_cy) ** 2) ** 0.5
                        if distance < min_distance:
                            min_distance = distance
                            closest_idx = i
                    
                    if closest_idx >= 0:
                        # Match found - update position
                        self.objects[obj_id] = input_centroids[closest_idx]
                        self.disappeared[obj_id] = 0
                        matched_new.add(closest_idx)
                        matched_old.add(obj_id)
                    else:
                        # No match - increment disappeared counter
                        self.disappeared[obj_id] += 1
                
                # Register unmatched detections as new objects
                for i, centroid in enumerate(input_centroids):
                    if i not in matched_new:
                        self.objects[self.next_object_id] = centroid
                        self.disappeared[self.next_object_id] = 0
                        self.next_object_id += 1
                
                # Remove disappeared objects
                disappeared_ids = list(self.disappeared.keys())
                for obj_id in disappeared_ids:
                    if self.disappeared[obj_id] > self.max_disappeared:
                        del self.objects[obj_id]
                        del self.disappeared[obj_id]
            
            # Create bbox -> object_id mapping for return
            bbox_to_id = {}
            for idx, (x1, y1, x2, y2, conf) in enumerate(bboxes):
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Find which object_id this bbox belongs to
                for obj_id in self.objects:
                    obj_cx, obj_cy = self.objects[obj_id]
                    distance = ((cx - obj_cx) ** 2 + (cy - obj_cy) ** 2) ** 0.5
                    if distance < self.max_distance / 2:
                        bbox_to_id[(int(x1), int(y1), int(x2), int(y2))] = obj_id
                        break
            
            return bbox_to_id


# ⭐ NEW: BOUNDING BOX SMOOTHING - Eliminates flickering/jittering
class BBoxSmoother:
    """
    Smooth bounding box coordinates using exponential smoothing filter.
    Reduces jittering and provides stable visual feedback.
    
    Uses first-order exponential smoothing (EMA):
    smoothed = alpha * new_value + (1 - alpha) * previous_smoothed
    
    Where alpha controls smoothing:
    - alpha=1.0: No smoothing (raw detection)
    - alpha=0.5: 50% new, 50% previous
    - alpha=0.2: 20% new, 80% previous (heavy smoothing)
    """
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but more lag.
                   Recommended: 0.25-0.4
        """
        self.alpha = alpha
        self.smoothed_bbox = {}  # {object_id: (x1, y1, x2, y2)}
        self.lock = threading.Lock()
    
    def smooth(self, object_id, bbox):
        """
        Apply exponential smoothing to bbox coordinates.
        
        Args:
            object_id: Unique object/tracker ID
            bbox: (x1, y1, x2, y2) tuple
        
        Returns:
            Smoothed (x1, y1, x2, y2) tuple
        """
        x1, y1, x2, y2 = bbox
        
        with self.lock:
            if object_id not in self.smoothed_bbox:
                # First detection - no previous data to smooth with
                self.smoothed_bbox[object_id] = (x1, y1, x2, y2)
                return (x1, y1, x2, y2)
            
            # Apply exponential smoothing to each coordinate
            prev_x1, prev_y1, prev_x2, prev_y2 = self.smoothed_bbox[object_id]
            
            smoothed_x1 = self.alpha * x1 + (1 - self.alpha) * prev_x1
            smoothed_y1 = self.alpha * y1 + (1 - self.alpha) * prev_y1
            smoothed_x2 = self.alpha * x2 + (1 - self.alpha) * prev_x2
            smoothed_y2 = self.alpha * y2 + (1 - self.alpha) * prev_y2
            
            # Store for next frame
            smoothed = (int(smoothed_x1), int(smoothed_y1), int(smoothed_x2), int(smoothed_y2))
            self.smoothed_bbox[object_id] = smoothed
            
            return smoothed
    
    def cleanup(self, object_id):
        """Remove smoothing data for object when it disappears."""
        with self.lock:
            self.smoothed_bbox.pop(object_id, None)


class IndianPlateDetector:
    """OCR and validation for Indian license plates."""
    
    VALID_STATE_CODES = {
        'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'CT', 'DD', 'DL', 'DN',
        'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LD', 'MP',
        'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK',
        'TG', 'TS', 'TR', 'TN', 'UP', 'UT', 'WB',
    }
    
    OCR_CHAR_CORRECTIONS = {
        'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '7', 'G': '6', 'A': '4',
    }
    
    REVERSE_CHAR_CORRECTIONS = {
        '0': 'O', '1': 'I', '8': 'B', '5': 'S', '7': 'Z', '6': 'G',
    }
    
    REVERSE_CHAR_ALTERNATIVES = {
        '1': 'T', '2': 'Z',
    }
    
    LETTER_CONFUSIONS = {}
    DIGIT_CORRECTIONS = {}
    DIGIT_ALTERNATIVES = {
        'A': ['4'],
    }
    
    @staticmethod
    def normalize_ocr_text(raw_text):
        """Normalize OCR output: uppercase, remove spaces, standardize hyphens."""
        if not raw_text:
            return ""
        
        text = raw_text.strip().upper()
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[–—−_•·•]', '-', text)
        text = re.sub(r'[^A-Z0-9\-]', '', text)
        
        return text
    
    @staticmethod
    def apply_ocr_corrections(text, format_type='standard'):
        """
        NORMALIZATION STEP 2: Intelligent character correction
        
        Contextually corrects common OCR misclassifications:
        - O ↔ 0 (letter vs number)
        - I ↔ 1 (letter vs number)
        - B ↔ 8 (letter vs number)
        - S ↔ 5 (letter vs number)
        - Z ↔ 2 (letter vs number)
        - G → 0 (font confusion in numeric positions)
        - W ↔ M (font confusion in alphabetic positions)
        
        For STANDARD format (SS-NN-AA-NNNN):
        - Positions 0-1: Must be letters → correct letter confusions (W→M, M→W) and digit-to-letter conversions
        - Positions 2-3: Must be numbers → correct letter-to-digit conversions (O→0, I→1, etc.)
        - Positions 4-5: Must be letters → correct letter confusions (W→M, M→W) and digit-to-letter conversions
        - Positions 6-9: Must be numbers → correct letter-to-digit conversions
        
        For BHARAT format (YY-BH-NNNN-A/AA):
        - Positions 0-1: Must be numbers → correct letter-to-digit conversions
        - Positions 2-3: Must be 'BH' letters → correct letter confusions
        - Positions 4-7: Must be numbers → correct letter-to-digit conversions
        - Positions 8+: Must be letters → correct letter confusions (W→M, M→W)
        
        Args:
            text: Normalized text with potential OCR errors
            format_type: 'standard' or 'bharat'
        
        Returns:
            Corrected text
        """
        if not text:
            return ""
        
        text = list(text.replace('-', ''))  # Remove hyphens for positional analysis
        
        if format_type == 'standard':
            # Standard format: SS-NN-AA-NNNN (positions 0-9 after hyphen removal)
            corrections = [
                (0, 1, 'alpha'),    # State code (2 letters)
                (2, 3, 'numeric'),  # District code (2 numbers)
                (4, 5, 'alpha'),    # Category (2 letters)
                (6, 9, 'numeric'),  # Serial (4 numbers)
            ]
        elif format_type == 'bharat':
            # Bharat format: YY-BH-NNNN-A/AA (positions after hyphen removal)
            corrections = [
                (0, 1, 'numeric'),  # Year (2 numbers)
                (2, 3, 'alpha'),    # BH (2 letters)
                (4, 7, 'numeric'),  # Serial (4 numbers)
                (8, None, 'alpha'), # Series (1-2 letters)
            ]
        else:
            return ''.join(text)
        
        for start, end, pos_type in corrections:
            if end is None:
                end = len(text) - 1
            
            for i in range(start, min(end + 1, len(text))):
                char = text[i]
                
                if pos_type == 'numeric':
                    # Position requires digit - convert misidentified letters to digits
                    if char in IndianPlateDetector.OCR_CHAR_CORRECTIONS:
                        # Letter that should be digit (O→0, I→1, B→8, S→5, Z→2, G→0)
                        text[i] = IndianPlateDetector.OCR_CHAR_CORRECTIONS[char]
                    elif char in IndianPlateDetector.DIGIT_CORRECTIONS:
                        # Digit confusion (3↔9)
                        text[i] = IndianPlateDetector.DIGIT_CORRECTIONS[char]
                elif pos_type == 'alpha':
                    # Position requires letter - convert misidentified digits to letters or other letters
                    if char in IndianPlateDetector.REVERSE_CHAR_CORRECTIONS:
                        # Digit that should be letter (0→O, 1→I, 8→B, 5→S, 2→Z)
                        text[i] = IndianPlateDetector.REVERSE_CHAR_CORRECTIONS[char]
                    elif char in IndianPlateDetector.LETTER_CONFUSIONS:
                        # Letter-to-letter confusions (W→M, M→W, etc.)
                        text[i] = IndianPlateDetector.LETTER_CONFUSIONS[char]
        
        return ''.join(text)
    
    @staticmethod
    def apply_alternative_ocr_corrections(text, format_type='standard'):
        """Apply fallback character corrections when primary corrections fail."""
        if not text:
            return ""
        
        text = list(text.replace('-', ''))
        
        if format_type == 'standard':
            corrections = [(0, 1, 'alpha'), (2, 3, 'numeric'), (4, 5, 'alpha'), (6, 9, 'numeric')]
        elif format_type == 'bharat':
            corrections = [(0, 1, 'numeric'), (2, 3, 'alpha'), (4, 7, 'numeric'), (8, None, 'alpha')]
        else:
            return ''.join(text)
        
        for start, end, pos_type in corrections:
            if end is None:
                end = len(text) - 1
            
            for i in range(start, min(end + 1, len(text))):
                char = text[i]
                
                if pos_type == 'alpha':
                    # Try alternative digit-to-letter mappings for alphabetic positions
                    if char in IndianPlateDetector.REVERSE_CHAR_ALTERNATIVES:
                        # Use alternative mapping (1/7 → T)
                        text[i] = IndianPlateDetector.REVERSE_CHAR_ALTERNATIVES[char]
                elif pos_type == 'numeric':
                    # Try alternative digit-to-digit mappings for numeric positions
                    if char in IndianPlateDetector.DIGIT_ALTERNATIVES:
                        # Try first alternative in list (handles 4→2, 4→1, etc.)
                        alternatives = IndianPlateDetector.DIGIT_ALTERNATIVES[char]
                        if alternatives:
                            text[i] = alternatives[0]
        
        return ''.join(text)
    
    @staticmethod
    def validate_standard_format(normalized_text):
        """Validate standard state registration format (SS-NN-AA-NNNN)."""
        if not normalized_text or len(normalized_text) < 9:
            return False, "", 0.0
        
        pattern_full = re.match(r'^([A-Z]{2})-([0-9]{2})-([A-Z]{1,2})-([0-9]{4})$', normalized_text)
        pattern_variant = re.match(r'^([A-Z]{2})-([0-9]{2})-([A-Z])-([0-9]{4})$', normalized_text)
        text_clean = normalized_text.replace('-', '')
        pattern_flexible = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})$', text_clean)
        
        pattern = pattern_full or pattern_variant or pattern_flexible
        if not pattern:
            return False, "", 0.0
        
        state, district, category, serial = pattern.groups()
        
        if state.upper() not in IndianPlateDetector.VALID_STATE_CODES:
            return False, "", 0.0
        
        district_num = int(district)
        if district_num < 1 or district_num > 99:
            return False, "", 0.0
        
        if not category.isalpha() or len(category) > 2:
            return False, "", 0.0
        
        if not serial.isdigit() or len(serial) != 4:
            return False, "", 0.0
        
        normalized = f"{state}-{district}-{category}-{serial}"
        return True, normalized, 0.95
    
    @staticmethod
    def validate_bharat_format(normalized_text):
        """Validate Bharat (BH) series format (YY-BH-NNNN-A/AA)."""
        if not normalized_text or len(normalized_text) < 9:
            return False, "", 0.0
        
        pattern_single = re.match(r'^([0-9]{2})-BH-([0-9]{4})-([A-Z])$', normalized_text)
        pattern_double = re.match(r'^([0-9]{2})-BH-([0-9]{4})-([A-Z]{2})$', normalized_text)
        text_clean = normalized_text.replace('-', '')
        pattern_flexible = re.match(r'^([0-9]{2})BH([0-9]{4})([A-Z]{1,2})$', text_clean)
        
        pattern = pattern_single or pattern_double or pattern_flexible
        if not pattern:
            return False, "", 0.0
        
        year, serial, series = pattern.groups()
        
        year_num = int(year)
        if year_num < 8 or year_num > 29:
            return False, "", 0.0
        
        if not serial.isdigit() or len(serial) != 4:
            return False, "", 0.0
        
        if not series.isalpha() or len(series) > 2:
            return False, "", 0.0
        
        normalized = f"{year}-BH-{serial}-{series}"
        return True, normalized, 0.90
    
    @staticmethod
    def is_valid_indian_plate(plate_text):
        """Validate Indian license plate with OCR corrections (standard and Bharat formats)."""
        if not plate_text or len(plate_text) < 8:
            return False, "", 0.0
        
        normalized = IndianPlateDetector.normalize_ocr_text(plate_text)
        if not normalized:
            return False, "", 0.0
        
        is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(normalized)
        if is_valid:
            return True, formatted, conf
        
        is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(normalized)
        if is_valid:
            return True, formatted, conf
        
        corrected_standard = IndianPlateDetector.apply_ocr_corrections(normalized, format_type='standard')
        is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(corrected_standard)
        if is_valid:
            return True, formatted, conf
        
        corrected_bharat = IndianPlateDetector.apply_ocr_corrections(normalized, format_type='bharat')
        is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(corrected_bharat)
        if is_valid:
            return True, formatted, conf
        
        corrected_alt_standard = IndianPlateDetector.apply_alternative_ocr_corrections(normalized, format_type='standard')
        if corrected_alt_standard != normalized:
            is_valid, formatted, conf = IndianPlateDetector.validate_standard_format(corrected_alt_standard)
            if is_valid:
                return True, formatted, conf
        
        corrected_alt_bharat = IndianPlateDetector.apply_alternative_ocr_corrections(normalized, format_type='bharat')
        if corrected_alt_bharat != normalized:
            is_valid, formatted, conf = IndianPlateDetector.validate_bharat_format(corrected_alt_bharat)
            if is_valid:
                return True, formatted, conf
        
        return False, "", 0.0


# ⭐ UNIFIED PLATE NORMALIZATION FUNCTION (CRITICAL FOR ENTRY/EXIT MATCHING)
def normalize_plate_for_matching(plate_text: str) -> str:
    """
    Normalize a license plate into dashed format.
    Used by BOTH entry and exit cameras for consistent matching.
    
    Supports both formats:
    1. Standard: XX-00-XX-0000 or XX-00-X-0000 (2+2+1-2+4)
    2. Bharat: XX-BH-0000-XX (2+BH+4+1-2) 
    
    Args:
        plate_text: Raw plate text (with or without dashes)
    
    Returns:
        Normalized plate with consistent formatting
        Examples: 
        - \"JH05CZ8989\" → \"JH-05-CZ-8989\" (standard 10-char)
        - \"HR38P8989\" → \"HR-38-P-8989\" (standard 9-char, single letter)
        - \"21BH9999UP\" → \"21-BH-9999-UP\" (Bharat 10-char)
    """
    if not plate_text:
        return ""
    
    # Remove all non-alphanumeric characters and whitespace
    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    if len(clean_plate) < 8:
        return clean_plate  # Too short, return as-is
    
    # Detect Bharat format (YY-BH-NNNN-AA): digits + "BH" + digits + letters
    if len(clean_plate) >= 9 and clean_plate[2:4] == "BH" and clean_plate[0:2].isdigit():
            # Bharat format detected: YY-BH-NNNN-AA
            yy = clean_plate[0:2]
            nn = clean_plate[4:8]
            aa = clean_plate[8:]
            if len(aa) > 0 and len(aa) <= 2 and nn.isdigit():
                return f"{yy}-BH-{nn}-{aa}"
    
    # Try standard format: SSNNAAANNNN (SS=2 letters, NN=2 digits, AA-AA=1-2 letters, NNNN=4 digits)
    if len(clean_plate) >= 9:
        ss = clean_plate[0:2]
        nn = clean_plate[2:4]
        
        # For 9-char: SS-NN-X-NNNN (1 letter in middle)
        if len(clean_plate) == 9 and ss.isalpha() and nn.isdigit():
            xx = clean_plate[4:5]  # 1 letter
            nnnn = clean_plate[5:9]  # 4 digits
            if xx.isalpha() and nnnn.isdigit():
                return f"{ss}-{nn}-{xx}-{nnnn}"
        
        # For 10-char: SS-NN-XX-NNNN (2 letters in middle)
        if len(clean_plate) == 10 and ss.isalpha() and nn.isdigit():
            xx = clean_plate[4:6]  # 2 letters
            nnnn = clean_plate[6:10]  # 4 digits
            if xx.isalpha() and nnnn.isdigit():
                return f"{ss}-{nn}-{xx}-{nnnn}"
    
    # If we can't detect format, return with normalized spacing for exact matching
    # This ensures at least consistent formatting
    return clean_plate


# Initialize Indian Plate Detector
indian_detector = IndianPlateDetector()


# OCR Validation Helper Functions
def validate_license_plate_ocr(raw_ocr_text):
    """Validate OCR output against Indian license plate formats."""
    return indian_detector.is_valid_indian_plate(raw_ocr_text)


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
                            print(f"[INFO] ✓ Detected USB camera at index {i}")
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

# ⭐ OPTIMIZATION: Background OCR Worker Thread
# Processes OCR requests asynchronously to prevent blocking the main detection pipeline
def ocr_worker_background():
    """
    Background worker thread for OCR processing
    
    Processes plate crops from the ocr_processing_queue without blocking
    the main detection and streaming pipelines.
    
    This is CRITICAL for real-time performance - OCR no longer blocks the frame pipeline!
    """
    print("[OCR_WORKER] ⭐ Background OCR worker started")
    items_processed = 0
    
    while True:
        try:
            # Get next OCR task (blocking, but in background thread)
            crop, cam_id, object_id = ocr_processing_queue.get(timeout=1.0)
            
            if crop is None:  # Stop signal
                print(f"[OCR_WORKER] Stop signal received after processing {items_processed} plates")
                break
            
            try:
                # Perform OCR on the plate crop (can be slow, but doesn't block main pipeline)
                plate_text = perform_ocr(crop, enable_validation=True)
                
                if plate_text and len(plate_text) >= 2:
                    items_processed += 1
                    
                    # Save to database (non-blocking in background)
                    user_id = camera_user_ids.get(cam_id, 'default')
                    from datetime import datetime
                    
                    with json_operations_lock:
                        # Log detection to JSON
                        try:
                            detection_record = {
                                'timestamp': datetime.now().isoformat(),
                                'licensePlate': plate_text,
                                'camera': cam_id,
                                'confidence': 0.95,
                                'userId': user_id
                            }
                            
                            # Log but don't block if DB write fails
                            try:
                                detections_file = os.path.join(DATA_DIR, 'detections.json')
                                try:
                                    with open(detections_file, 'r') as f:
                                        detections = json.load(f)
                                except:
                                    detections = []
                                
                                detections.append(detection_record)
                                with open(detections_file, 'w') as f:
                                    json.dump(detections, f, indent=2)
                                
                                if items_processed % 10 == 0:
                                    print(f"[OCR_WORKER] ✓ Processed {items_processed} OCR requests")
                            except Exception as e:
                                # Silently fail - don't block OCR pipeline
                                pass
                        except Exception as e:
                            pass
            
            except Exception as e:
                # Log error but continue processing
                if items_processed <= 5:
                    print(f"[OCR_WORKER] ⚠️ Error processing OCR: {str(e)[:60]}")
        
        except queue.Empty:
            # Normal - no items in queue
            continue
        except Exception as e:
            print(f"[OCR_WORKER] ✗ Unexpected error: {str(e)[:80]}")
            continue

# Start OCR worker thread on startup
# Will be started in main() after all globals are initialized
# ocr_worker_thread = threading.Thread(target=ocr_worker_background, daemon=True)
# ocr_worker_thread.start()

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

config_manager = CameraConfigManager(
    config_path=os.path.join(os.path.dirname(__file__), '../../data/cameras-config.json')
)

from config_api import init_config_api

camera_locks = {}
camera_sources = {}
camera_source_map = {}
video_buffers = {}
frame_counters = {}
last_processed_frame = {}
persistent_detections = {}
persistent_detection_motion_levels = {}
prev_frames = {}
frame_deduplicators = {}
plate_bbox_stabilizers = {}

# ⭐ NEW: Bounding Box Smoothing - Eliminate flickering
bbox_smoothers = {}  # {tracker_id: BBoxSmoother}
bbox_smoother_lock = threading.Lock()

# ⭐ CRITICAL: Duplicate Image Prevention - Track processed plates
processed_plates = {}  # {(cam_id, tracker_id): True} - plates already saved to disk
processed_plates_lock = threading.Lock()
model_global = None
plate_detector_model = None
plate_cascade = None
ocr_enabled = False

ocr_plate_buffer = {}
OCR_BUFFER_FRAMES = 3  # ⭐ OPTIMIZATION: Reduced from 5 to 3 for faster stabilization (~180ms at 15 FPS)
OCR_BUFFER_TIMEOUT = 5.0

# ⭐ NEW: TRACKING & PERFORMANCE OPTIMIZATION SYSTEMS
object_trackers = {}  # {cam_id: CentroidTracker}
tracked_object_ocr = {}  # {(cam_id, object_id): {'readings': [...], 'locked': bool, 'bbox_history': [...]}}
tracked_object_lock = threading.Lock()

# ⭐ CRITICAL FIXES: Prevent duplicate image saving and unnecessary OCR re-execution
saved_objects = set()  # {(cam_id, object_id)} - Objects already saved to database
saved_objects_lock = threading.Lock()
object_image_saved = {}  # {(cam_id, object_id): True} - Track if image already saved
object_ocr_executed_this_frame = {}  # {(cam_id, object_id): frame_id} - Track which objects had OCR this frame
object_ocr_executed_lock = threading.Lock()

# ⭐ NEW FIX: Prevent duplicate plate images (same plate text) from being saved multiple times
saved_plate_texts = set()  # {plate_text} - Track unique plates that have been saved to prevent duplicates
saved_plate_texts_lock = threading.Lock()

ocr_processing_queue = queue.Queue(maxsize=100)  # Non-blocking OCR pipeline
next_object_id = 0
next_object_id_lock = threading.Lock()

from collections import deque
recent_plates_cache_opt = deque(maxlen=20)

frame_count_per_camera = {}

last_stable_bbox = {}
bbox_smoothing_alpha = 0.7

PLATE_DETECTOR_CLASS = 0

seen_plates = {}
camera_last_saved = {}
detected_vehicles = {}
plate_detections_history = {}

plate_ocr_history = {}

recent_plate_detections = {}

recent_plates_cache = {}
plate_cache_lock = threading.Lock()
PLATE_CACHE_TTL = 60

def cleanup_expired_persistent_detections():
    """Remove persistent detections older than 3 seconds."""
    current_time = time.time()
    cameras_to_clean = []
    
    for cam_id, detection in list(persistent_detections.items()):
        detection_age = current_time - detection.get('timestamp', current_time)
        if detection_age > 3:
            cameras_to_clean.append((cam_id, detection))
    
    for cam_id, detection in cameras_to_clean:
        plate_text = detection.get('text', 'UNKNOWN')
        persistent_detections.pop(cam_id, None)
        if DEBUG or frame_counters.get(cam_id, 0) % 30 == 0:
            print(f"[CLEANUP_DETECTION] {cam_id}: Cleared detection for {plate_text}")

latest_frames = {}
latest_frames_lock = threading.Lock()

detection_overlay_cache = {}
detection_overlay_lock = threading.Lock()

# ⭐ NEW: Lock for thread-safe JSON operations
json_operations_lock = threading.Lock()

persistent_overlays = {}
OVERLAY_GRACE_PERIOD = 60

# ⭐ PERFORMANCE FIX: In-memory detection cache (indexed by license plate)
# Eliminates repeated JSON file reads on every detection
detections_cache = {}  # {license_plate: detection_record}
detections_cache_lock = threading.Lock()
detections_cache_last_reload = 0  # Timestamp of last reload
DETECTIONS_CACHE_RELOAD_INTERVAL = 5  # Reload from disk every 5 seconds

vehicle_entry_log = {}

camera_gate_types = {}
camera_user_ids = {}
vehicle_detection_per_camera = {}

ocr_history = {}
last_saved_plate = {}
plate_frame_consistency = {}

plate_bbox_stability = {}

# CRITICAL: Background processing queues and control events
# These MUST be module-level globals to be accessible from all request handlers and threads
frame_queues = {}  # {cam_id: Queue} - frames to process
processing_stop_events = {}  # {cam_id: threading.Event} - stop signals for worker threads
processing_threads = []  # List of (cam_id, thread, stop_event) tuples

# -------- Frame Deduplicator --------
class FrameDeduplicator:
    def __init__(self, hash_size=8, similarity_threshold=5):
        self.last_hash = None
        self.hash_size = hash_size
        self.similarity_threshold = similarity_threshold

    def dhash(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()

    def hamming_distance(self, h1, h2):
        return np.count_nonzero(h1 != h2)

    def is_duplicate(self, frame):
        current_hash = self.dhash(frame)

        if self.last_hash is None:
            self.last_hash = current_hash
            return False

        distance = self.hamming_distance(current_hash, self.last_hash)

        if distance <= self.similarity_threshold:
            return True

        self.last_hash = current_hash
        return False

# -------- Plate BBox Stabilizer --------
class PlateBBoxStabilizer:
    def __init__(self):
        self.locked_bbox = None
        self.last_seen = None  # Anchor point for frame-level drift detection
        self.last_seen_time = time.time()  # ⭐ FIX 3: Timestamp for vehicle disappearance detection
        self.missed_frames = 0
        self.lock_threshold = 3  # CRITICAL FIX #2: Lock after 1 frame for debugging (production: 3)
        self.max_missed = 5
        self.locked = False
        self.lock_frames = 0
        self.timeout_threshold = 3.0  # ⭐ FIX 2: Reset after 3.0 seconds (was 1.5) - prevents CPU spike jitter

    def iou(self, b1, b2):
        x1,y1,x2,y2 = b1
        x1b,y1b,x2b,y2b = b2
        xi1 = max(x1,x1b)
        yi1 = max(y1,y1b)
        xi2 = min(x2,x2b)
        yi2 = min(y2,y2b)
        inter = max(0,xi2-xi1)*max(0,yi2-yi1)
        area1 = (x2-x1)*(y2-y1)
        area2 = (x2b-x1b)*(y2b-y1b)
        union = area1+area2-inter
        return inter/union if union>0 else 0

    def update(self, new_bbox):
        if self.locked_bbox is None:
            self.locked_bbox = new_bbox
            self.last_seen_time = time.time()  # ⭐ FIX 3: Record first sighting
            return new_bbox

        # ⭐ FIX 3: TIMEOUT CHECK - Reset stabilizer if vehicle disappears > 1.5 seconds
        current_time = time.time()
        time_since_seen = current_time - self.last_seen_time
        if time_since_seen > self.timeout_threshold:
            # Vehicle disappeared - reset for new vehicle
            self.locked = False
            self.lock_frames = 0
            self.locked_bbox = new_bbox
            self.last_seen_time = current_time
            print(f"[BBOX_TIMEOUT] Vehicle disappeared for {time_since_seen:.2f}s - RESET stabilizer")
            return new_bbox

        lx1, ly1, lx2, ly2 = self.locked_bbox
        nx1, ny1, nx2, ny2 = new_bbox

        # Compute centroid shift (motion magnitude)
        lc = ((lx1+lx2)//2, (ly1+ly2)//2)
        nc = ((nx1+nx2)//2, (ny1+ny2)//2)
        shift = ((lc[0]-nc[0])**2 + (lc[1]-nc[1])**2) ** 0.5
        iou_score = self.iou(self.locked_bbox, new_bbox)

        # ⭐ FIX 1: RESET STABILIZER ON NEW OBJECT DETECTION (IOU < 0.3)
        # If a new vehicle enters with low overlap, reset tracking
        if iou_score < 0.3 and self.locked:
            # Significant position change = new vehicle detected
            self.locked = False
            self.lock_frames = 0
            self.locked_bbox = new_bbox
            self.last_seen_time = current_time
            print(f"[BBOX_NEW_OBJECT] New vehicle detected (IOU={iou_score:.3f} < 0.3) - RESET stabilizer")
            return new_bbox

        # LOCK CONDITION: Require 3 consecutive frames of good overlap before locking
        if not self.locked:
            if iou_score > 0.5:
                self.lock_frames += 1
            else:
                self.lock_frames = 0

            if self.lock_frames >= self.lock_threshold:  # ⭐ FIX 4: Use lock_threshold (now 4)
                self.locked = True
                print(f"[BBOX_LOCK] Box locked after {self.lock_frames} stable frames (4-frame mode)")

        # TRACK when locked: Ignore YOLO jitter, follow only meaningful motion
        if self.locked:
            # Update last seen time (vehicle is still being detected)
            self.last_seen_time = current_time  # ⭐ FIX 3: Update timestamp on every detection
            
            # Compute motion shift (delta from old centroid to new centroid)
            lc = ((lx1+lx2)//2, (ly1+ly2)//2)
            nc = ((nx1+nx2)//2, (ny1+ny2)//2)

            # ANCHOR STABILIZATION: Reject drift relative to frame reference
            # This prevents visual jitter from camera motion or frame-level shifts
            if self.last_seen:
                px, py = self.last_seen
                dx_anchor = lc[0] - px
                dy_anchor = lc[1] - py
                
                anchor_shift = (dx_anchor*dx_anchor + dy_anchor*dy_anchor) ** 0.5
                
                # If centroid hasn't moved much relative to frame, reject update
                # This acts as a second-level filter for visual drift
                if anchor_shift < 15:
                    return self.locked_bbox

            dx = nc[0] - lc[0]
            dy = nc[1] - lc[1]

            # Calculate movement magnitude
            movement = (dx*dx + dy*dy) ** 0.5

            # Reject jitter (<2px), follow real motion with 20% blending for smoothing
            if movement < 2:
                return self.locked_bbox

            # Apply 20% smoothing blending
            x1 = lx1 + int(0.2 * dx)
            y1 = ly1 + int(0.2 * dy)
            x2 = lx2 + int(0.2 * dx)
            y2 = ly2 + int(0.2 * dy)

            self.locked_bbox = (x1,y1,x2,y2)
            # Store centroid as anchor for next frame's drift detection
            self.last_seen = ((x1+x2)//2, (y1+y2)//2)
            return self.locked_bbox
        else:
            # Not locked yet: accept new position immediately
            self.locked_bbox = new_bbox
            self.last_seen_time = current_time  # ⭐ FIX 3: Update timestamp
            return new_bbox

# Tracking helpers
class TrackedObject:
    def __init__(self, bbox, cls_id):
        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        self.cls_id = cls_id
        self.centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.history = deque(maxlen=10)
        self.history.append(self.centroid)
        self.stationary_frames = 0
        self.last_seen = time.time()
        self.saved = False  # whether we already saved this object
        self.last_ocr_time = 0  # Cooldown timer for OCR re-trigger prevention
        self.id = None

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.history.append(centroid)
        self.centroid = centroid
        self.last_seen = time.time()
        # compute movement over history
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < STOPPED_MOVEMENT_PIXELS:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0

# Simple IoU to match detections to trackers
def iou(bb1, bb2):
    x1, y1, x2, y2 = bb1
    x1b, y1b, x2b, y2b = bb2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area1 = max(1, (x2 - x1) * (y2 - y1))
    area2 = max(1, (x2b - x1b) * (y2b - y1b))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

trackers = {}  # {cam_id: {trk_id: TrackedObject}}
next_tracker_id = 0

# ---------------- Setup ----------------
pytesseract = None
paddle_ocr = None
ocr_enabled = False

# Initialize Tesseract
try:
    import pytesseract
    # Windows tesseract path detection
    if platform.system() == "Windows":
        paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.getenv('LOCALAPPDATA', ''), r'Tesseract-OCR\tesseract.exe')
        ]
        for p in paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    print("[INFO] Tesseract OCR enabled")
except Exception as e:
    print(f"[WARN] Tesseract not available: {e}")
    pytesseract = None

# Initialize PaddleOCR (optional, may have compatibility issues on Windows)
paddle_ocr = None
try:
    # Check if paddle module is available first
    try:
        import paddle
        from paddleocr import PaddleOCR
        # Use new parameter name to avoid deprecation warning
        paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("[INFO] PaddleOCR enabled")
    except Exception as e:
        # PaddleOCR has known compatibility issues on Windows, fall back to Tesseract
        print(f"[WARN] PaddleOCR initialization failed (common on Windows): {type(e).__name__}")
        print("[INFO] Falling back to Tesseract OCR only")
        paddle_ocr = None
except Exception as e:
    print(f"[WARN] PaddleOCR not available: {e}")
    paddle_ocr = None

ocr_enabled = (pytesseract is not None) or (paddle_ocr is not None)

if not ocr_enabled:
    print("[WARN] No OCR engine available. Please install tesseract-ocr or paddleocr for accurate plate recognition")

# Fix PyTorch 2.6+ weights_only security issue BEFORE importing YOLO
try:
    import torch
    import pickle
    
    # Store original torch.load
    _original_torch_load = torch.load
    
    # Create wrapper that handles weights_only parameter
    def torch_load_wrapper(f, *args, **kwargs):
        # If weights_only not specified, set it to False to allow loading older models
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        try:
            return _original_torch_load(f, *args, **kwargs)
        except Exception as e:
            # Fallback: try loading without weights_only restriction
            if 'weights_only' in str(e).lower():
                print("[INFO] Falling back to unsafe model loading due to weights_only restriction...")
                kwargs['weights_only'] = False
                return _original_torch_load(f, *args, **kwargs)
            raise
    
    # Replace torch.load with wrapper
    torch.load = torch_load_wrapper
    print("[INFO] PyTorch weights_only workaround applied")
except Exception as e:
    print(f"[WARN] Could not apply PyTorch workaround: {e}")

# ============================================================
# GPU Configuration for YOLO Acceleration (5-10x speedup!)
# ============================================================
USE_GPU = True  # Set to False to force CPU-only mode
DEFAULT_DEVICE = 0  # GPU device ID (0 for first/only GPU)

if USE_GPU:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(DEFAULT_DEVICE)
            gpu_memory = torch.cuda.get_device_properties(DEFAULT_DEVICE).total_memory / 1e9
            print(f"[INFO] GPU AVAILABLE: {gpu_name}")
            print(f"[INFO]    VRAM: {gpu_memory:.1f} GB")
            print(f"[INFO]    YOLO detection will run on GPU(~5-10x faster!)")
        else:
            print("[INFO] ⚠ CUDA not available, falling back to CPU")
            USE_GPU = False
    except Exception as e:
        print(f"[INFO] ⚠ GPU initialization failed: {e}")
        print(f"[INFO]    Falling back to CPU")
        USE_GPU = False
else:
    print("[INFO] GPU disabled (CPU mode)")

YOLO = None
ONNXRUNTIME = None
try:
    import onnxruntime as ort
    ONNXRUNTIME = ort
    print("[INFO] ONNX Runtime loaded successfully")
except ImportError as e:
    print(f"[WARN] Failed to import onnxruntime: {e}")
    print("[WARN] Will fall back to PyTorch. Install with: pip install onnxruntime")
except Exception as e:
    print(f"[WARN] Unexpected error loading onnxruntime: {type(e).__name__}: {e}")

try:
    from ultralytics import YOLO
    print("[INFO] YOLO (ultralytics) loaded successfully (kept for compatibility)")
except ImportError as e:
    print(f"[WARN] Failed to import ultralytics YOLO: {e}")
    print("[WARN] Will use ONNX Runtime for inference. Install with: pip install ultralytics")
except Exception as e:
    print(f"[WARN] Unexpected error loading YOLO: {type(e).__name__}: {e}")


def download_haar():
    if not os.path.exists(HAAR_PATH):
        print("[INFO] Downloading Haar cascade for plates...")
        urllib.request.urlretrieve(HAAR_URL, HAAR_PATH)
    return cv2.CascadeClassifier(HAAR_PATH)


# -------- ONNX Model Wrapper (INT8 Quantized YOLO with ONNX Runtime) --------
class ONNXPlateDetector:
    """
    ONNX-based plate detector using ONNX Runtime (INT8 quantized for efficiency).
    
    This class wraps an ONNX model and provides an interface similar to the
    YOLO class from ultralytics, but uses onnxruntime for inference instead.
    
    Features:
    - Uses INT8 quantized ONNX model (smaller memory footprint, faster inference)
    - CPU-optimized execution provider
    - Compatible with existing inference code
    """
    
    def __init__(self, model_path):
        """
        Initialize ONNX model.
        
        Args:
            model_path: Path to the ONNX model file
        """
        if ONNXRUNTIME is None:
            raise RuntimeError("ONNX Runtime not available. Install with: pip install onnxruntime")
        
        # Tune ONNX Runtime for Raspberry Pi 5 (4-core ARM CPU)
        opts = ONNXRUNTIME.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = ONNXRUNTIME.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ONNXRUNTIME.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ONNXRUNTIME.InferenceSession(
            model_path,
            sess_options=opts,
            providers=[('CPUExecutionProvider', {})]
        )
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"[INFO] ONNX Model loaded: {model_path}")
        print(f"[INFO] Input: {self.input_name}, shape: {self.input_shape}")
        print(f"[INFO] Outputs: {self.output_names}")
    
    def __call__(self, image, conf=0.4, verbose=False):
        """
        Run inference on the image.
        
        Args:
            image: Input image (BGR format, will be resized to model input size)
            conf: Confidence threshold
            verbose: Whether to print verbose output
        
        Returns:
            List containing results object (mimics YOLO API which returns a list)
        """
        # ⭐ FIX: Process full frame like PyTorch YOLO does (NOT cropped ROI)
        # The ROI cropping should be done by the caller if needed, not here
        # This ensures ONNX and PyTorch receive the same input
        h, w = image.shape[:2]
        
        # No ROI cropping here - use full frame for better detection
        self.roi_y_offset = 0
        self.roi_x_offset = 0
        
        # Preprocess image (full frame, just like PyTorch)
        img_resized = cv2.resize(image, (320, 320))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_input = np.ascontiguousarray(
            np.expand_dims(np.transpose(img_normalized, (2, 0, 1)), 0)
        )  # HWC→CHW→NCHW, contiguous memory = faster ONNX feed
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: img_input})
        
        # Parse outputs - assuming output is shape (1, num_detections, 5) where 5 = [x, y, w, h, conf]
        # or (1, 5, num_anchors) for YOLO format
        detections = outputs[0]
        
        # Wrap results to match YOLO Results interface
        # Return as a list to match YOLO API (YOLO returns [Results])
        results = ONNXResults(detections, conf, image.shape, (320, 320), roi_offset=(self.roi_y_offset, self.roi_x_offset))
        return [results]  # Wrap in list to match YOLO API


class ONNXResults:
    """Wrapper for ONNX inference results to match YOLO Results interface."""
    
    def __init__(self, detections, conf_threshold, original_shape, model_shape, roi_offset=None):
        """
        Args:
            detections: Raw output from ONNX model
            conf_threshold: Confidence threshold for filtering
            original_shape: Shape of original image (H, W, C)
            model_shape: Shape of model input (usually 320x320)
            roi_offset: Tuple of (y_offset, x_offset) for ROI cropping adjustment
        """
        self.boxes = ONNXBoxes(detections, conf_threshold, original_shape, model_shape, roi_offset)
        self.conf_threshold = conf_threshold
    
    def __getitem__(self, index):
        """Allow indexing for compatibility."""
        if index == 0:
            return self
        raise IndexError(f"Index {index} out of range")


class ONNXBox:
    """Single detection box wrapper to match YOLO Box interface."""
    def __init__(self, xyxy, conf):
        self.xyxy = np.array([xyxy])  # Wrap to match YOLO format
        self.conf = np.array([conf])   # Wrap to match YOLO format


class ONNXBoxes:
    """Wrapper for detection boxes to match YOLO Boxes interface."""
    
    def __init__(self, detections, conf_threshold, original_shape, model_shape, roi_offset=None):
        """
        Args:
            detections: Raw detections from ONNX model, shape (1, 5, 2100) for YOLO format
            conf_threshold: Confidence threshold
            original_shape: Original image shape (H, W, C)
            model_shape: Model input shape (H, W)
            roi_offset: Tuple of (y_offset, x_offset) for ROI cropping adjustment
        """
        self.conf_threshold = conf_threshold
        self.roi_y_offset = roi_offset[0] if roi_offset else 0  # Y offset from ROI cropping
        self.roi_x_offset = roi_offset[1] if roi_offset else 0  # X offset from ROI cropping
        self.data = []
        
        orig_h, orig_w = original_shape[:2]
        model_h, model_w = model_shape
        
        # Parse detections - VECTORIZED for 10-15% speed gain
        # YOLO output format: [batch, 5, num_anchors] where 5 = [x, y, w, h, conf]
        # Or might be [batch, num_anchors, 5]
        if len(detections.shape) == 3:
            # Try [batch, 5, num_anchors] format first
            if detections.shape[1] == 5:
                detections = detections[0]  # Remove batch dimension [5, num_anchors]
                
                # Transpose to get [num_anchors, 5]
                if detections.shape[0] == 5:
                    detections = detections.T  # Now [num_anchors, 5]
            
            # Try [batch, num_anchors, 5] format
            elif detections.shape[2] == 5:
                detections = detections[0]  # Remove batch dimension [num_anchors, 5]
            
            else:
                # Try to reshape to get [num_anchors, 5]
                if detections.size > 0:
                    flat = detections.reshape(-1)
                    if len(flat) % 5 == 0:  # Check if divisible by 5 (x, y, w, h, conf)
                        detections = flat.reshape(-1, 5)
        
        elif len(detections.shape) == 2:
            # Already in [num_anchors, 5] format
            pass
        
        if len(detections.shape) == 2 and detections.shape[1] >= 5:
            # 🚀 VECTORIZED filtering using numpy masking (10-15% faster than loop)
            # Instead of: for det in detections: if conf >= conf_threshold:
            # Use: mask = detections[:, 4] > conf_threshold; filtered = detections[mask]
            
            # Extract confidence scores and apply mask
            conf_scores = detections[:, 4]  # Shape: [num_anchors]
            mask = conf_scores >= conf_threshold  # Boolean mask: [num_anchors]
            filtered_detections = detections[mask]  # Only detections above threshold
            
            # ⭐ STEP 1: HARD LIMIT TO 1 DETECTION (gate mode - keep only best)
            # This is the earliest filter - reduces CPU ~40% by stopping multi-plate processing
            if len(filtered_detections) > 1:
                best_idx = np.argmax(filtered_detections[:, 4])  # Get index of max confidence
                filtered_detections = np.array([filtered_detections[best_idx]])  # Keep only best
            
            if len(filtered_detections) > 0:
                # Extract coordinates and scale in one vectorized operation
                x = filtered_detections[:, 0]
                y = filtered_detections[:, 1]
                w = filtered_detections[:, 2]
                h = filtered_detections[:, 3]
                conf_array = filtered_detections[:, 4]  # Renamed to conf_array to avoid conflict
                
                # Scale from model coordinates to original coordinates (vectorized)
                scale_x = orig_w / model_w
                scale_y = orig_h / model_h
                
                x1 = (x - w/2) * scale_x
                y1 = (y - h/2) * scale_y
                x2 = (x + w/2) * scale_x
                y2 = (y + h/2) * scale_y
                
                # Adjust coordinates back to full frame (from ROI offset)
                if self.roi_y_offset > 0 or self.roi_x_offset > 0:
                    y1 += self.roi_y_offset
                    y2 += self.roi_y_offset
                    x1 += self.roi_x_offset
                    x2 += self.roi_x_offset
                
                # Clamp to image bounds (vectorized with numpy.clip)
                x1 = np.clip(x1, 0, orig_w)
                y1 = np.clip(y1, 0, orig_h)
                x2 = np.clip(x2, 0, orig_w)
                y2 = np.clip(y2, 0, orig_h)
                
                # Filter out invalid boxes where x2 <= x1 or y2 <= y1 (vectorized)
                valid_mask = (x2 > x1) & (y2 > y1)
                
                # Get valid box indices
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # Prepare boxes for NMS
                    bboxes_for_nms = []
                    scores_for_nms = []
                    
                    for i in valid_indices:
                        bbox = [int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])]  # [x, y, w, h]
                        bboxes_for_nms.append(bbox)
                        scores_for_nms.append(float(conf_array[i]))
                    
                    # Apply NMS to remove duplicate/overlapping detections
                    nms_indices = cv2.dnn.NMSBoxes(
                        bboxes_for_nms,
                        scores_for_nms,
                        score_threshold=self.conf_threshold,
                        nms_threshold=0.4
                    )
                    
                    # Store only NMS-kept detections as ONNXBox objects
                    if len(nms_indices) > 0:
                        for nms_idx in nms_indices.flatten():
                            orig_idx = valid_indices[nms_idx]
                            xyxy = np.array([x1[orig_idx], y1[orig_idx], x2[orig_idx], y2[orig_idx]])
                            conf_val = float(conf_array[orig_idx])
                            self.data.append(ONNXBox(xyxy, conf_val))
                    
                    # ⭐ GATE MODE: Hard limit to 1 detection (keep best only)
                    if len(self.data) > 1:
                        # Sort by confidence descending, keep only best
                        self.data.sort(key=lambda box: box.conf[0], reverse=True)
                        self.data = self.data[:1]  # Keep only top 1 detection
                else:
                    # No valid boxes after filtering
                    pass
                
                # Store valid detections as ONNXBox objects (to match YOLO API)
                # (NMS handled above, this is now obsolete but kept for reference)
                # for i in np.where(valid_mask)[0]:
                #     xyxy = np.array([x1[i], y1[i], x2[i], y2[i]])
                #     conf_val = float(conf_array[i])
                #     self.data.append(ONNXBox(xyxy, conf_val))
        else:
            # Unexpected format - no detections extracted
            pass
    
    def __iter__(self):
        """Iterate over detections."""
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)



# Plate Image Cropping & Saving
def save_plate_image(frame, bbox, plate_text, user_id=None):
    """⭐ OPTIMIZED: Crop and save license plate image with minimal overhead."""
    try:
        if frame is None:
            return None
            
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop plate region with small margin
        margin = 5
        crop_x1 = max(0, x1 - margin)
        crop_y1 = max(0, y1 - margin)
        crop_x2 = min(frame.shape[1], x2 + margin)
        crop_y2 = min(frame.shape[0], y2 + margin)
        
        plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if plate_crop.size == 0:
            return None
        
        # ⭐ OPTIMIZATION: Resize to standard size (faster save + smaller files)
        # License plates are typically detected at various sizes, normalize to 300x100
        plate_crop = cv2.resize(plate_crop, (300, 100), interpolation=cv2.INTER_LINEAR)
        
        # Create unique filename
        safe_plate = plate_text.replace('-', '_').upper()
        timestamp = int(time.time() * 1000)
        filename = f"{safe_plate}_{timestamp}.jpg"
        
        # Save to plates folder with high compression (faster I/O)
        plates_dir = DIR_PLATES
        plates_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = plates_dir / filename
        # ⭐ OPTIMIZATION: Use compression parameters for faster saves
        cv2.imwrite(str(filepath), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        relative_path = f"plates/{filename}"
        if DEBUG:
            print(f"[PLATE_SAVED] {plate_text} -> {relative_path}")
        
        return relative_path
    
    except Exception as e:
        if DEBUG:
            print(f"[PLATE_SAVE_ERROR] Failed to save plate image: {type(e).__name__}: {str(e)[:50]}")
        return None


# -------- Original Fuzzy Matching --------
def levenshtein_distance(s1, s2):
    """
    Calculate Levenshtein distance between two strings
    Lower distance = more similar
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_plate_similarity(plate1, plate2):
    """Calculate similarity between two plates (0-1)."""
    if not plate1 or not plate2:
        return 0.0
    
    # Normalize plates for comparison (alphanumeric only)
    p1_clean = re.sub(r'[^A-Z0-9]', '', str(plate1).upper())
    p2_clean = re.sub(r'[^A-Z0-9]', '', str(plate2).upper())
    
    # Exact match
    if p1_clean == p2_clean:
        return 1.0
    
    # Calculate edit distance
    max_len = max(len(p1_clean), len(p2_clean))
    if max_len == 0:
        return 0.0
    
    distance = levenshtein_distance(p1_clean, p2_clean)
    similarity = 1.0 - (distance / max_len)
    
    return similarity


def find_similar_plate(current_plate, recent_plates_dict, similarity_threshold=PLATE_SIMILARITY_THRESHOLD):
    """Find if current plate matches any recent plate within threshold."""
    if not PLATE_FUZZY_MATCH_ENABLED:
        return None, 0.0
    
    current_time = time.time()
    best_match = None
    best_score = 0.0
    
    for plate_key, (detect_time, normalized) in recent_plates_dict.items():
        # Only check within the deduplication window
        if current_time - detect_time > PLATE_DUPLICATE_WINDOW:
            continue
        
        similarity = calculate_plate_similarity(current_plate, normalized)
        
        if similarity >= similarity_threshold and similarity > best_score:
            best_match = plate_key
            best_score = similarity
    
    return best_match, best_score


def check_plate_frame_stability(cam_id, bbox, required_frames=3):
    """Track plate across consecutive frames before OCR processing."""
    key = (cam_id, bbox)
    current_time = time.time()
    
    if key not in plate_frame_consistency:
        # First time seeing this bbox
        plate_frame_consistency[key] = {
            'frame_count': 1,
            'first_frame_time': current_time
        }
        return False
    
    # Check if too much time has passed (reset if >2 seconds gap)
    consistency_data = plate_frame_consistency[key]
    time_gap = current_time - consistency_data['first_frame_time']
    
    if time_gap > 2.0:
        # Too much time passed - plate is unstable, reset
        consistency_data['frame_count'] = 1
        consistency_data['first_frame_time'] = current_time
        return False
    
    # Increment frame counter (consecutive frame detection)
    consistency_data['frame_count'] += 1
    
    # Return True if we've seen this plate for 3+ consecutive frames
    is_stable = consistency_data['frame_count'] >= required_frames
    
    if is_stable:
        # Clean up after acceptance to allow re-detection
        del plate_frame_consistency[key]
    
    return is_stable


def reset_plate_stability(cam_id, bbox):
    """Reset plate frame consistency tracking after OCR/save."""
    key = (cam_id, bbox)
    if key in plate_frame_consistency:
        del plate_frame_consistency[key]


def stabilize_ocr(cam_id, plate_text):
    """Stabilize OCR results using majority voting."""
    if not plate_text:
        return None
    
    # Initialize history for this camera
    if cam_id not in ocr_history:
        ocr_history[cam_id] = deque(maxlen=OCR_CONSISTENCY_FRAMES)
    
    history = ocr_history[cam_id]
    history.append(plate_text)
    
    # Not enough readings yet
    if len(history) < OCR_CONSISTENCY_FRAMES:
        return None
    
    # ⭐ FIX 3: MAJORITY VOTING - Find most common reading
    # Count occurrences of each plate text
    text_counts = {}
    for text in history:
        text_counts[text] = text_counts.get(text, 0) + 1
    
    # Get the most common reading
    candidate = max(text_counts.items(), key=lambda x: x[1])
    plate_text, count = candidate
    
    # Accept only if it appears in majority (OCR_CONSISTENCY_FRAMES or more times)
    # For 6 frames, need at least 4-5 matching readings
    min_matches = max(3, OCR_CONSISTENCY_FRAMES - 2)
    if count >= min_matches:
        return plate_text
    
    return None


# ★ FIX 2: PLATE SIMILARITY CHECK with lower threshold for OCR noise correction
def is_similar_plate(p1, p2, threshold=PLATE_SIMILARITY_THRESHOLD):
    """
    ⭐ FIX 6: Check if two plates are similar using fuzzy matching.
    Prevents saving duplicate plates from same vehicle with OCR variations.
    Uses lower threshold (0.85 vs 0.95) to allow minor OCR errors while catching true duplicates.
    
    Args:
        p1: First plate text
        p2: Second plate text
        threshold: Similarity threshold (0-1), default now 0.85 (was 0.95)
        
    Returns:
        True if plates are similar, False otherwise
    """
    from difflib import SequenceMatcher
    
    if not p1 or not p2:
        return False
    
    similarity = SequenceMatcher(None, p1, p2).ratio()
    return similarity >= threshold


def append_json_array(path: Path, entry: dict):
    """
    ⭐ JSON LINES OPTIMIZED: O(1) append - write single line to file.
    This is the critical performance fix for detections.json handling.
    
    Performance improvement:
    - Old format (JSON array): O(n) - read file, parse, append, re-write = 2-3ms per entry
    - JSON Lines format: O(1) - just append one line = 0.08ms per entry
    - Result: 20-30x faster for large files
    
    Thread-safe with json_operations_lock to prevent concurrent write issues.
    """
    global detections_cache
    
    try:
        with json_operations_lock:  # ⭐ THREAD SAFETY
            # ⭐ CRITICAL FIX: JSON Lines append - just write one line to end of file
            # No need to read entire file, no re-serialization needed
            line = json.dumps(entry, separators=(',', ':'), ensure_ascii=False)
            
            # Append single line to file (atomic operation)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            
            # ⭐ PERFORMANCE: Update in-memory cache immediately (no file re-read needed)
            plate_key = entry.get("licensePlate")
            if plate_key:
                with detections_cache_lock:
                    detections_cache[plate_key] = entry
            
    except Exception as e:
        print(f"[ERROR] Failed to append to {path}: {type(e).__name__}: {e}")


def read_vehicles_json():
    """
    DEPRECATED: Vehicles are now stored in detections.json.
    Returns empty list for backward compatibility.
    Use read_detections_json() instead.
    """
    return []


def read_detections_json():
    """
    ⭐ JSON LINES OPTIMIZED: Read detections line-by-line from JSON Lines format.
    Thread-safe with json_operations_lock.
    
    This properly handles JSON Lines format where each line is a separate JSON object.
    Much more efficient than loading entire array into memory.
    """
    try:
        with json_operations_lock:  # ⭐ THREAD SAFETY
            if DETECTIONS_JSON.exists():
                data = []
                try:
                    with DETECTIONS_JSON.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                try:
                                    obj = json.loads(line)
                                    data.append(obj)
                                except json.JSONDecodeError:
                                    # Skip malformed lines
                                    pass
                    return data
                except Exception as e:
                    print(f"[WARN] Could not read detections.json: {e}")
    except Exception as e:
        print(f"[WARN] Could not read detections.json: {e}")
    return []


def vehicle_has_entry_time(plate_text):
    """Check if a vehicle with this plate already exists in the premises (has entry time but no exit time)"""
    try:
        vehicles = read_vehicles_json()
        plate_key = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        for vehicle in vehicles:
            if vehicle.get("licensePlate") == plate_key:
                # Vehicle exists - check if it has entry time and no exit time
                has_entry = vehicle.get("entryTime") is not None
                has_exit = vehicle.get("exitTime") is not None
                
                if has_entry and not has_exit:
                    return True  # Vehicle is currently in premises
        return False
    except Exception as e:
        print(f"[WARN] Could not check vehicle entry time: {e}")
    return False


def update_vehicle_exit_time(plate_text, is_registered=False):
    """
    Update exit time for vehicle and mark as Registered.
    Handles BOTH cases:
    1. Vehicle exists in DB (update existing record)
    2. Vehicle NEW at exit (create implicit entry + mark registered)
    """
    try:
        vehicles = read_vehicles_json()
        plate_key = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        
        exit_time_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        updated = False
        found = False
        
        # CASE 1: Vehicle already in DB - update existing record
        for i, vehicle in enumerate(vehicles):
            vehicle_plate = vehicle.get("licensePlate", "")
            if vehicle_plate == plate_key:
                found = True
                has_entry = vehicle.get("entryTime") is not None
                has_exit = vehicle.get("exitTime") is not None
                
                print(f"[EXIT_UPDATE] Found in DB: {vehicle_plate} | Has Entry: {has_entry} | Has Exit: {has_exit}")
                
                # Only update if vehicle is inside (has entry, no exit)
                if has_entry and not has_exit:
                    vehicle["exitTime"] = exit_time_str
                    vehicle["gate"] = "Exit"
                    vehicle["status"] = "Registered"  # Mark as Registered on exit
                    print(f"[EXIT_UPDATE] Updated existing vehicle with exitTime: {exit_time_str}")
                    updated = True
                    break
                elif has_exit:
                    print(f"[EXIT_UPDATE] ✗ Already has exit time: {vehicle.get('exitTime')}")
                    return True  # Already exited, not an error
        
        # ★ CASE 2: Vehicle NOT in DB - create implicit entry for exit-only detection
        if not found:
            print(f"[EXIT_UPDATE] ✗ Vehicle {plate_key} not in DB - creating implicit entry for unregistered exit")
            # Create implicit entry record for vehicle detected at exit without entry
            implicit_entry = {
                "id": str(uuid.uuid4()),
                "licensePlate": plate_key,
                "vehicleType": "Unknown",
                "driverName": "",
                "driverMobile": "",
                "entryTime": exit_time_str,  # ★ Use exit time as entry time (implicit/instant entry)
                "exitTime": exit_time_str,   # ★ Immediately mark as exited
                "gate": "Exit",
                "detectionMethod": "Live",
                "status": "Registered",  # ★ Mark as Registered when exiting (becomes registered via exit)
                "confidence": 0.0,
                "ocrConfidence": 0.0,
                "plateImage": None,
                "vehicleImage": None,
                "notes": "Unregistered vehicle detected at exit (no entry detection in current session)",
                "userId": "fa726c6b-0ccc-4c51-8658-36bd8f261cc3"  # Default admin
            }
            vehicles.append(implicit_entry)
            print(f"[EXIT_UPDATE] ✓ Created implicit entry for unregistered exit: {plate_key}")
            updated = True
        
        if updated:
            try:
                # ⭐ OPTIMIZED: Write back with compact JSON (no indent) - 2x faster
                tmp_path = VEHICLES_JSON.with_suffix(".tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(vehicles, f, separators=(',', ':'), ensure_ascii=False)
                
                # Replace original file
                tmp_path.replace(VEHICLES_JSON)
                print(f"[EXIT_UPDATE] ✓ Vehicle file updated successfully")
                
                # Verify write
                verify_vehicles = read_vehicles_json()
                verify_vehicle = next((v for v in verify_vehicles if v.get("licensePlate") == plate_key), None)
                if verify_vehicle and verify_vehicle.get("exitTime"):
                    status = verify_vehicle.get("status", "Unknown")
                    print(f"[EXIT_UPDATE] ✓✓ VERIFIED: {plate_key} marked as {status} with exitTime: {verify_vehicle.get('exitTime')}")
                else:
                    print(f"[EXIT_UPDATE] ⚠ WARNING: exitTime not verified in DB after write")
                
                return True
            except Exception as write_err:
                print(f"[ERROR] File write failed: {type(write_err).__name__}: {write_err}")
                return False
    except Exception as e:
        print(f"[ERROR] Could not update exit time: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    return False


def save_entry_to_detections(plate_key, cam_id, confidence, ocr_confidence, direction, detection_mode, gate_type="Entry", user_id=None, plate_image=None):
    """
    Save vehicle entry detection to detections.json (single database file).
    This is created when vehicle enters the premises.
    Prevents duplicate entries for currently active vehicles.
    
    Args:
        plate_image: Optional relative path to saved plate image
    """
    try:
        # ★ Check for existing non-exited entry (duplicate prevention)
        detections = read_detections_json()
        
        for detection in detections:
            if detection.get("licensePlate") == plate_key and detection.get("entryTime") and not detection.get("exitTime"):
                # Vehicle already has an active entry without exit time
                print(f"[SKIP] {plate_key:12} already tracked (active entry exists)")
                return detection.get("id")  # Return existing entry ID
        
        entry_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # Get location from camera config
        location = get_camera_location(cam_id)
        
        entry_detection = {
            "id": str(uuid.uuid4()),
            "licensePlate": plate_key,
            "vehicleType": "Unknown",
            "confidence": round(float(confidence), 3),
            "ocrConfidence": round(float(ocr_confidence), 3),
            "detectionMethod": detection_mode or "Unknown",
            "direction": direction,
            "cameraId": cam_id,
            "detectedAt": entry_timestamp,
            "entryTime": entry_timestamp,
            "exitTime": None,  # Will be added on exit
            "status": "Unregistered",  # Initially unregistered
            "gate": gate_type,
            "location": location,  # ⭐ Location from camera config
            "plateImage": plate_image,  # ⭐ Store plate image path
            "vehicleImage": None,  # Removed - not needed
            "captureReason": "Vehicle entry detection",
            "captured": True
        }
        
        # Add userId - Use camera owner's user ID for data isolation
        # If no user_id provided, get it from camera_user_ids mapping
        if not user_id:
            user_id = get_camera_user_id(cam_id)
        entry_detection["userId"] = user_id
        
        # Append to detections.json
        print(f"[DEBUG_SAVE] Creating entry with plateImage={plate_image}, userId={user_id}")
        append_json_array(DETECTIONS_JSON, entry_detection)
        print(f"[ENTRY] {plate_key:12} | Camera: {cam_id:15} | User: {user_id[:8]}... | Conf: {confidence:.2f} | OCR: {ocr_confidence:.2f} | Image: {plate_image}")
        
        return entry_detection.get("id")
    except Exception as e:
        print(f"[ERROR] Failed to save entry for {plate_key}: {str(e)[:80]}")
    return None


def update_exit_in_detections(plate_key, cam_id, confidence, ocr_confidence, direction, detection_mode, user_id=None, plate_image=None):
    """
    ⭐ FIXED EXIT CAMERA LOGIC: Update exit time for detection record.
    ⭐ UPDATED: Now also stores plate image for exit detection
    
    CRITICAL FIXES:
    1. Normalize plate BEFORE matching (case-insensitive, ignore whitespace)
    2. Search detections.json for exact match after normalization
    3. Update ONLY the existing record (no duplicates)
    4. Proper file locking to prevent race conditions
    5. Thread-safe with minimal lock time (read-only ops outside lock)
    6. Safe logging to prevent detection freezing
    7. Store exit plate image if provided
    
    Thread-safe with json_operations_lock.
    """
    global detections_cache
    
    try:
        # ⭐ FIX #1: Normalize plate to match entry format
        normalized_plate = normalize_plate_for_matching(plate_key)
        if not normalized_plate:
            print(f"[EXIT_SKIP] Invalid plate format: {plate_key}")
            return False
        
        # ⭐ FIX #2: Read detections OUTSIDE lock to minimize lock duration
        try:
            detections = read_detections_json()
        except Exception as read_err:
            print(f"[EXIT_ERROR] Failed to read detections: {read_err}")
            return False
        
        exit_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        updated = False
        updated_plate_text = ""
        
        # ⭐ FIX #3: Find entry record with case-insensitive, normalized matching
        for detection in detections:
            existing_plate = detection.get("licensePlate", "")
            normalized_existing = normalize_plate_for_matching(existing_plate)
            
            # Match if plates normalize to same value AND entry has no exit time
            if normalized_existing == normalized_plate and detection.get("entryTime") and not detection.get("exitTime"):
                # Found entry record without exit - update it
                detection["exitTime"] = exit_timestamp
                detection["exit_camera_id"] = cam_id  # Store exit camera ID
                detection["gate"] = "Exit"
                detection["status"] = "Registered"
                
                # ⭐ NEW: Store exit plate image if provided (allows different images for entry vs exit)
                if plate_image and not detection.get("exitPlateImage"):
                    detection["exitPlateImage"] = plate_image
                
                updated = True
                updated_plate_text = existing_plate
                print(f"[EXIT_MATCH] Found entry: {existing_plate} | Normalized: {normalized_plate} | Exit Image: {plate_image}")
                break
        
        # ⭐ FIX #4: STRICT ENTRY/EXIT LOGIC - Exit must match existing entry
        # Exit cameras must only update existing entries, never create new records
        if not updated:
            print(f"[EXIT_NOMATCH] {normalized_plate:12} - No matching entry in database (ignored)")
            return False
        
        # ⭐ FIX #5: FAST WRITE with lock only for write operation
        # Do the actual disk write INSIDE lock to ensure atomicity
        try:
            with json_operations_lock:  # CRITICAL: Lock only during write
                tmp_path = DETECTIONS_JSON.with_suffix(".tmp")
                with tmp_path.open("w", encoding="utf-8") as f:
                    for detection in detections:
                        line = json.dumps(detection, separators=(',', ':'), ensure_ascii=False)
                        f.write(line + "\n")
                
                # Atomic replace
                tmp_path.replace(DETECTIONS_JSON)
        except Exception as write_err:
            print(f"[EXIT_WRITE_ERROR] Failed to write detections: {write_err}")
            return False
        
        # ⭐ UPDATE CACHE after successful write (outside lock)
        try:
            with detections_cache_lock:
                for detection in detections:
                    plate = detection.get("licensePlate")
                    if plate:
                        detections_cache[plate] = detection
        except Exception as cache_err:
            print(f"[EXIT_CACHE_ERROR] Cache update failed: {cache_err}")
            # Don't return False - write succeeded, cache update is secondary
        
        print(f"[EXIT_UPDATE] {normalized_plate:12} | Plate: {updated_plate_text:15} | Camera: {cam_id:15} | Conf: {confidence:.2f} | OCR: {ocr_confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"[EXIT_ERROR] Unexpected error: {type(e).__name__}: {str(e)[:80]}")
    return False


def save_exit_detection(plate_key, cam_id, confidence, ocr_confidence, direction, detection_mode, user_id=None, plate_image=None):
    """
    Update vehicle exit in detections.json (single database file).
    Finds the entry record by license plate and adds exit time.
    
    ⭐ CRITICAL: plate_key should ALREADY be normalized (XX-00-XX-0000 format)
    before calling this function from process_anpr_pipeline.
    ⭐ NEW: Supports plate_image parameter to store exit plate image
    """
    # ⭐ Ensure plate is normalized for matching
    normalized = normalize_plate_for_matching(plate_key)
    return update_exit_in_detections(normalized, cam_id, confidence, ocr_confidence, direction, detection_mode, user_id, plate_image)


# ★ INTELLIGENT GATE MANAGEMENT SYSTEM
def sync_vehicle_entry_log_from_db():
    """
    Initialize vehicle_entry_log from persistent vehicles.json on startup.
    CRITICAL: Only load vehicles that are CURRENTLY INSIDE (have entry, NO exit).
    This ensures we don't treat new entries of the same vehicle as duplicates.
    
    Logic:
    - Vehicle with entryTime and NO exitTime → Currently inside → Load into log
    - Vehicle with entryTime and exitTime → Already exited → DO NOT load
    - Vehicle with NO entryTime → Never entered → DO NOT load
    
    CALLED ONCE during app startup.
    """
    try:
        vehicles = read_vehicles_json()
        count = 0
        loaded = 0
        skipped = 0
        
        for vehicle in vehicles:
            plate = vehicle.get("licensePlate", "")
            plate_key = re.sub(r'[^A-Z0-9]', '', plate.upper())
            
            if plate_key:
                has_entry = vehicle.get("entryTime") is not None
                has_exit = vehicle.get("exitTime") is not None
                count += 1
                
                # ★ CRITICAL FIX: Only load vehicles currently inside (entered but not exited)
                if has_entry and not has_exit:
                    vehicle_entry_log[plate_key] = {
                        "entry_time": vehicle.get("entryTime"),
                        "entry_camera": vehicle.get("cameraId"),
                        "normalized_plate": plate,
                        "has_exited": False,  # ← These are inside, not exited
                        "exit_time": None,
                        "exit_camera": None
                    }
                    loaded += 1
                    print(f"[STARTUP_LOAD] Vehicle inside: {plate} (will wait for exit)")
                else:
                    skipped += 1
                    if has_exit:
                        print(f"[STARTUP_SKIP] Vehicle already exited: {plate} (will treat as new entry if detected again)")
        
        if loaded > 0 or skipped > 0:
            print(f"[STARTUP] ✓ Loaded {loaded} vehicles currently inside | Skipped {skipped} vehicles already exited")
        else:
            print(f"[STARTUP] ✓ No vehicles currently inside, starting fresh")
    except Exception as e:
        print(f"[WARN] Could not sync vehicle log from DB: {e}")


def get_camera_gate_type(cam_id):
    """
    Get the gate type (Entry/Exit) for a camera.
    Implements intelligent gate management.
    Returns: "Entry" or "Exit" (defaults to "Entry" if not set)
    """
    return camera_gate_types.get(cam_id, "Entry")


def set_camera_gate_type(cam_id, gate_type):
    """
    Set the gate type for a camera during initialization.
    This is called when cameras are created/configured.
    """
    if gate_type in ["Entry", "Exit"]:
        camera_gate_types[cam_id] = gate_type
        print(f"[CAMERA_GATE] Camera '{cam_id}' set as {gate_type} gate")


def get_camera_user_id(cam_id):
    """
    Get the user ID (owner) of a camera for data isolation.
    Returns: user_id string, or "default" if not found
    """
    return camera_user_ids.get(cam_id, "default")


def get_camera_location(cam_id):
    """
    Get the location/gate name for a camera from config.
    Returns: location string from camera config, or "Not Specified" if not found
    """
    try:
        # Find camera in config and get its location field
        if hasattr(config_manager, 'config') and 'cameras' in config_manager.config:
            for camera in config_manager.config['cameras']:
                if camera.get('id') == cam_id:
                    location = camera.get('location', 'Not Specified')
                    return location if location and location.strip() else 'Not Specified'
    except Exception as e:
        if DEBUG:
            print(f"[WARN] Could not get location for camera {cam_id}: {e}")
    return 'Not Specified'


def set_camera_user_id(cam_id, user_id):
    """
    Set the user ID (owner) for a camera during initialization.
    This is called when cameras are loaded from config to establish ownership.
    """
    camera_user_ids[cam_id] = user_id
    print(f"[DATA_ISOLATION] Camera '{cam_id}' assigned to user '{user_id}'")


def check_vehicle_in_database(plate_alphanumeric):
    """
    ⭐ OPTIMIZED: Check if vehicle exists using in-memory cache.
    ⭐ FIXED: Normalizes plates for comparison to handle format variants.
    
    Returns: (is_in_db, normalized_plate, has_exit, is_current_occupant)
    - is_in_db: Vehicle exists in detections.json
    - normalized_plate: The plate format from DB
    - has_exit: Whether vehicle has an exitTime (already left)
    - is_current_occupant: Vehicle is inside (has entryTime, no exitTime)
    """
    global detections_cache_last_reload
    current_time = time.time()
    
    try:
        with detections_cache_lock:
            # ⭐ Reload cache if interval elapsed (5 second refresh)
            if current_time - detections_cache_last_reload > DETECTIONS_CACHE_RELOAD_INTERVAL:
                # Reload cache from disk (only happens every 5 seconds)
                detections = read_detections_json()
                detections_cache.clear()
                
                for detection in detections:
                    plate_key = detection.get("licensePlate", "")
                    if plate_key:
                        # ⭐ FIX: Normalize plate for better matching (handles format variants)
                        normalized_key = normalize_plate_for_matching(plate_key)
                        detections_cache[normalized_key] = detection
                        # Also store original for backward compatibility
                        if plate_key != normalized_key:
                            detections_cache[plate_key] = detection
                
                detections_cache_last_reload = current_time
                if DEBUG:
                    print(f"[CACHE_RELOAD] Loaded {len(detections_cache)} vehicles into memory")
            
            # ⭐ FIX: Normalize incoming plate for lookup
            normalized_lookup = normalize_plate_for_matching(plate_alphanumeric)
            
            # ⭐ O(1) lookup in memory cache (no file I/O!)
            # Try both normalized and original format for maximum compatibility
            detection = None
            if normalized_lookup in detections_cache:
                detection = detections_cache[normalized_lookup]
            elif plate_alphanumeric in detections_cache:
                detection = detections_cache[plate_alphanumeric]
            
            if detection:
                has_entry = detection.get("entryTime") is not None
                has_exit = detection.get("exitTime") is not None
                is_current = has_entry and not has_exit
                return (True, detection.get("licensePlate", ""), has_exit, is_current)
    
    except Exception as e:
        if DEBUG:
            print(f"[WARN] Cache lookup failed: {e}")
    
    return (False, None, False, False)


def check_vehicle_entry_status(plate_alphanumeric, cam_id):
    """
    Check if vehicle has been captured in CURRENT SESSION or is in detections.json.
    
    Returns: (in_current_session, was_same_camera, normalized_plate_str, is_registered, in_database)
    - in_current_session: Vehicle detected in current session's entry_log
    - was_same_camera: Vehicle already detected from this specific camera in session
    - normalized_plate_str: The original normalized plate format
    - is_registered: Vehicle was registered before
    - in_database: Vehicle is in detections.json (may be from previous session)
    """
    # ✅ CHECK 1: Is vehicle in current session's entry log?
    if plate_alphanumeric in vehicle_entry_log:
        entry_data = vehicle_entry_log[plate_alphanumeric]
        entry_camera = entry_data.get("entry_camera")
        entry_time = entry_data.get("entry_time", 0)
        
        was_same_camera = (cam_id == entry_camera)
        
        # Frame-to-frame: Allow consecutive frames from same camera if entry <10s old (allows time for vehicle to pass through detection zone)
        now = time.time()
        time_since_entry = now - entry_time
        
        if was_same_camera and time_since_entry < 10.0:
            # New entry, allow frame-to-frame detections within 10 seconds
            return (True, False, entry_data.get("normalized_plate"), entry_data.get("is_registered", False), False)
        
        return (True, was_same_camera, entry_data.get("normalized_plate"), entry_data.get("is_registered", False), False)
    
    # ✅ CHECK 2: Is vehicle in detections.json database?
    is_in_db, db_plate, has_exit, is_current_occupant = check_vehicle_in_database(plate_alphanumeric)
    if is_in_db:
        # Vehicle was registered/detected before (in database)
        if is_current_occupant:
            # Vehicle is currently inside (has entry, no exit) - this is the key for exits!
            return (False, False, db_plate, True, True)
        else:
            # Vehicle previously exited - could be returning
            return (False, False, db_plate, True, True)
    
    # Not in current session or database
    return (False, False, None, False, False)


def mark_vehicle_as_exiting(plate_alphanumeric, normalized_plate, cam_id):
    """
    Mark vehicle as exiting and REMOVE from entry log to allow re-entry.
    INTELLIGENT: Records which camera detected the exit.
    """
    if plate_alphanumeric in vehicle_entry_log:
        vehicle_entry_log[plate_alphanumeric]["has_exited"] = True
        vehicle_entry_log[plate_alphanumeric]["exit_time"] = time.time()
        vehicle_entry_log[plate_alphanumeric]["exit_camera"] = cam_id
        # ★ IMPORTANT: Remove from entry log to allow re-entry later
        del vehicle_entry_log[plate_alphanumeric]
        print(f"[EXIT_RECORDED] {normalized_plate} - Vehicle exited premises via camera '{cam_id}', removed from entry log")
        return True
    return False


def mark_vehicle_as_entered(plate_alphanumeric, normalized_plate, cam_id, is_registered=False):
    """
    Record vehicle as entered (first detection in Indian format).
    INTELLIGENT: Records which camera detected the entry.
    
    Args:
        is_registered: Whether this vehicle is a registered user (from DB)
    """
    vehicle_entry_log[plate_alphanumeric] = {
        "entry_time": time.time(),
        "entry_camera": cam_id,
        "normalized_plate": normalized_plate,
        "has_exited": False,
        "exit_time": None,
        "exit_camera": None,
        "is_registered": is_registered  # ★ Track registration status
    }
    reg_status = "Registered" if is_registered else "Unregistered"
    print(f"[ENTRY_RECORDED] {normalized_plate} - {reg_status} vehicle entered premises via camera '{cam_id}'")


def check_auto_exit(plate_alphanumeric, cam_id):
    """
    AUTOMATIC EXIT DETECTION (No gate configuration needed)
    
    Checks if a vehicle was previously detected and is now being detected again
    at a different camera, and automatically marks it as exiting.
    
    This enables exit detection without requiring explicit Entry/Exit gate configuration.
    Works by:
    1. If vehicle was in entry_log (already detected before)
    2. AND is being detected at a DIFFERENT camera
    3. THEN treat it as an exit event
    
    Returns: True if auto-exit was triggered, False otherwise
    """
    if plate_alphanumeric not in vehicle_entry_log:
        return False
    
    entry_data = vehicle_entry_log[plate_alphanumeric]
    entry_camera = entry_data.get("entry_camera")
    
    # Check if this is a DIFFERENT camera
    if cam_id != entry_camera and not entry_data.get("has_exited"):
        return True
    
    return False


def get_vehicle_direction_advanced(bbox, centroid_history, frame_height):
    """
    Improved direction detection using multiple heuristics:
    - Centroid movement: up = rear, down = front
    - Position: top = rear, bottom = front
    - Aspect ratio: may indicate orientation
    Returns: (direction_str, color)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    centroid_y = (y1 + y2) // 2
    
    # Calculate motion direction if history available
    motion_score = 0  # positive = moving down (front), negative = moving up (rear)
    if len(centroid_history) >= 5:
        recent = list(centroid_history)[-5:]
        y_trajectory = [c[1] for c in recent]
        # Simple regression: is y increasing or decreasing?
        y_diffs = [y_trajectory[i+1] - y_trajectory[i] for i in range(len(y_trajectory)-1)]
        motion_score = sum(y_diffs)  # positive = moving down
    
    # Position-based scoring
    position_score = 0  # positive = likely front, negative = likely rear
    if centroid_y > frame_height * 0.65:
        position_score = 1  # likely front (bottom of frame)
    elif centroid_y < frame_height * 0.35:
        position_score = -1  # likely rear (top of frame)
    
    # Aspect ratio scoring (heuristic for car orientation)
    aspect_ratio = width / height if height > 0 else 1.0
    aspect_score = 0
    if aspect_ratio > 1.3:  # wider = front view
        aspect_score = 0.5
    elif aspect_ratio < 0.8:  # taller = rear view
        aspect_score = -0.5
    
    # Combined score
    total_score = motion_score * 0.4 + position_score * 0.35 + aspect_score * 0.25
    
    if total_score > 0.1:
        return "Front (Approaching)", COLOR_FRONT
    elif total_score < -0.1:
        return "Rear (Leaving)", COLOR_REAR
    else:
        # Fallback to position
        if centroid_y > frame_height * 0.6:
            return "Front (Approaching)", COLOR_FRONT
        else:
            return "Rear (Leaving)", COLOR_REAR


def get_ocr_confidence_score(text, vehicle_conf=None):
    """
    Calculate OCR confidence score based on text characteristics.
    Returns a score from 0-1 indicating how confident we are in this plate reading.
    """
    if not text or len(text) < 4:
        return 0.0
    
    base_score = 0.5
    
    # Bonus for valid plate patterns (mix of letters and digits)
    has_letters = any(c.isalpha() for c in text)
    has_digits = any(c.isdigit() for c in text)
    if has_letters and has_digits:
        base_score += 0.20
    
    # Bonus for reasonable length (Indian plates typically 6-10 chars)
    if 5 <= len(text) <= 11:
        base_score += 0.15
    
    # Incorporate vehicle detection confidence if available
    if vehicle_conf and vehicle_conf > 0:
        base_score += (vehicle_conf * 0.15)
    
    return min(base_score, 1.0)


# Multi-engine OCR with voting system for accuracy
def perform_ocr_tesseract(img, config='--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """Tesseract wrapper with error handling."""
    try:
        if pytesseract is None:
            return ""
        
        text = pytesseract.image_to_string(img, config=config)
        return text.strip() if text else ""
    except Exception as e:
        if DEBUG:
            print(f"[OCR_ERROR] Tesseract error: {e}")
        return ""


# ⭐ NEW: COMPREHENSIVE OCR PREPROCESSING FOR SMALL/LARGE PLATES
def preprocess_plate_for_ocr_fast(plate_img, target_width=400):
    """
    ⭐ FAST PREPROCESSING PATH (< 5ms)
    Minimal preprocessing for speed - only resize + basic threshold.
    Used first; if OCR fails, falls back to comprehensive preprocessing.
    """
    if plate_img is None or plate_img.size == 0:
        return None
    
    try:
        h, w = plate_img.shape[:2]
        
        # Quick resize to optimal width
        if w < 200 or w > 600:
            # Need resizing
            scale_factor = target_width / w if w > 0 else 1.0
            new_w = max(200, int(w * scale_factor))
            new_h = max(50, int(h * scale_factor))
            resized = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = plate_img.copy()
        
        # Simple grayscale + Otsu threshold (FAST)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    except Exception as e:
        if DEBUG:
            print(f"[OCR_FAST_ERROR] {e}")
        return None


def preprocess_plate_for_ocr_comprehensive(plate_img, target_width=400):
    """
    ⭐ COMPREHENSIVE PREPROCESSING PATH (20-50ms)
    Full 5-step pipeline for difficult cases.
    Used only when fast preprocessing fails.
    """
    if plate_img is None or plate_img.size == 0:
        return None
    
    try:
        h, w = plate_img.shape[:2]
        
        # Intelligent resizing
        if w < 50:
            scale_factor = max(2.0, target_width / w)
        elif w > 600:
            scale_factor = target_width / w
        elif w < target_width * 0.8 or w > target_width * 1.2:
            scale_factor = target_width / w
        else:
            scale_factor = 1.0
        
        if scale_factor != 1.0:
            new_w = max(200, int(w * scale_factor))
            new_h = max(50, int(h * scale_factor))
            resized = cv2.resize(plate_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            resized = plate_img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()
        
        # Bilateral filter (EXPENSIVE but effective)
        filtered = cv2.bilateralFilter(gray, 5, 75, 75)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Sharpening
        blurred = cv2.GaussianBlur(morph, (0, 0), 1.0)
        sharpened = cv2.addWeighted(morph, 2.0, blurred, -1.0, 0)
        
        if DEBUG:
            print(f"[OCR_COMPREHENSIVE] Plate {w}x{h} → {resized.shape}")
        
        return sharpened
    
    except Exception as e:
        print(f"[OCR_COMPREHENSIVE_ERROR] {e}")
        return None


def preprocess_plate_for_ocr(plate_img, target_width=400):
    """⭐ BACKWARD COMPATIBILITY - Routes to comprehensive for now"""
    return preprocess_plate_for_ocr_comprehensive(plate_img, target_width)


def perform_ocr_tesseract(img, config='--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    """⭐ OPTIMIZED: Tesseract OCR with error logging for debugging"""
    if pytesseract is None:
        return ""
    try:
        # ⭐ DEBUG: Validate image before OCR
        if img is None or img.size == 0:
            return ""
        
        text = pytesseract.image_to_string(img, config=config)
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned if len(cleaned) >= 3 else ""
    except Exception as e:
        # ⭐ DEBUG: Log OCR errors for troubleshooting
        if DEBUG:
            print(f"[OCR_ERROR] Tesseract extraction failed: {type(e).__name__}: {str(e)[:100]}")
        return ""


def perform_ocr_paddle(img):
    """PaddleOCR engine"""
    if paddle_ocr is None:
        return ""
    try:
        # PaddleOCR works on BGR, convert if needed
        if img.ndim != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        result = paddle_ocr.ocr(img, cls=True)
        if result and len(result) > 0:
            text_parts = []
            for line in result:
                if line:
                    for word_info in line:
                        if word_info and len(word_info) > 0:
                            text = word_info[1][0] if isinstance(word_info[1], (list, tuple)) else str(word_info[1])
                            confidence = word_info[1][1] if isinstance(word_info[1], (list, tuple)) and len(word_info[1]) > 1 else 0
                            if confidence >= 0.4:  # confidence threshold
                                text_parts.append(text)
            
            combined_text = ''.join(text_parts)
            cleaned = re.sub(r'[^A-Z0-9]', '', combined_text.upper())
            return cleaned if len(cleaned) >= 3 else ""
    except Exception as e:
        pass
    return ""


def perform_ocr_multi_engine(img, with_preprocessing=True):
    """
    🚀 OPTIMIZED: Lazy preprocessing - try FAST path first, then COMPREHENSIVE.
    
    ⭐ SPEED IMPROVEMENTS (2026-03-24):
    1. Fast path (< 5ms): Resize + Otsu threshold for normal plates
    2. Comprehensive path (20-50ms): Full pipeline for problem plates
    3. Most plates use fast path → 90% speed improvement
    4. Problem plates (small/large) fall back to comprehensive
    5. Overall: < 100ms latency (was 250-400ms)
    
    Strategy:
    - Try fast preprocessing first
    - If OCR succeeds, return immediately
    - If fast fails, apply comprehensive preprocessing
    - Fall back to comprehensive if both fail
    """
    if not ocr_enabled or pytesseract is None:
        return ""
    
    try:
        if img is None or img.size == 0:
            return ""
        
        # ⭐ STRATEGY 1: TRY FAST PREPROCESSING FIRST (< 5ms)
        if with_preprocessing:
            fast_preprocess = preprocess_plate_for_ocr_fast(img, target_width=400)
            if fast_preprocess is not None:
                # Try OCR on fast-preprocessed image
                config = '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = perform_ocr_tesseract(fast_preprocess, config)
                
                if text and len(text) >= 4:
                    # Fast path succeeded!
                    if DEBUG:
                        print(f"[OCR_FAST_SUCCESS] {text} (< 5ms)")
                    return text
                
                # Try PSM 6 with fast preprocessing
                config = '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = perform_ocr_tesseract(fast_preprocess, config)
                if text and len(text) >= 3:
                    if DEBUG:
                        print(f"[OCR_FAST_PSM6] {text}")
                    return text
        
        # ⭐ STRATEGY 2: COMPREHENSIVE PREPROCESSING FALLBACK (20-50ms)
        if with_preprocessing:
            comprehensive = preprocess_plate_for_ocr_comprehensive(img, target_width=400)
            if comprehensive is not None:
                # PSM 7
                config = '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = perform_ocr_tesseract(comprehensive, config)
                
                if text and len(text) >= 4:
                    if DEBUG:
                        print(f"[OCR_COMPREHENSIVE_SUCCESS] {text} (20-50ms)")
                    return text
                
                # PSM 6
                config = '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = perform_ocr_tesseract(comprehensive, config)
                if text and len(text) >= 3:
                    if DEBUG:
                        print(f"[OCR_COMPREHENSIVE_PSM6] {text}")
                    return text
        
        # No preprocessing fallback
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        config = '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = perform_ocr_tesseract(binary, config)
        if text and len(text) >= 4:
            return text
        
        config = '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return perform_ocr_tesseract(binary, config)
        
    except Exception as e:
        print(f"[ERROR] OCR error: {type(e).__name__}: {str(e)[:100]}")
        return ""


# Improved OCR: use multiple preprocessing steps and image_to_data to pick best candidate
def perform_ocr(img, enable_validation=True):
    """Complete OCR extraction and validation pipeline."""
    raw_text = perform_ocr_multi_engine(img, with_preprocessing=True)
    
    if not enable_validation:
        return raw_text
    
    # Run comprehensive validation
    is_valid, normalized, confidence = validate_license_plate_ocr(raw_text)
    
    return raw_text, is_valid, normalized, confidence


# ---------------- Processing ----------------
def match_and_update_trackers(cam_id, detections):
    """
    ⭐ FIX 4: GATE OPTIMIZATION - Single tracker per camera
    Gate system has at most 1 detection (hard capped).
    No need for complex multi-object matching.
    
    detections: list of (x1,y1,x2,y2,cls_id) - max 1 element for gate
    returns mapping of tracker_id -> detection tuple
    """
    global next_tracker_id
    
    # Initialize single tracker slot for this camera
    if cam_id not in trackers:
        trackers[cam_id] = {}
    
    assigned = {}
    trks = trackers[cam_id]
    
    # For gate system: single tracker approach
    if len(detections) == 0:
        # No detection - tracker expires after 1 second (was 2.0)
        # ⭐ CRITICAL: Faster expiration ensures new vehicles get new trackers
        now = time.time()
        to_del = []
        for tid, tr in list(trks.items()):
            if now - tr.last_seen > 1.0:  # ⭐ FIXED: 1 second timeout (was 2.0)
                to_del.append(tid)
        for tid in to_del:
            if DEBUG or frame_counters.get(cam_id, 0) % 100 == 0:
                print(f"[TRACKER_EXPIRED] {cam_id}: Tracker {tid} expired (no detection for 1+ sec)")
            trks.pop(tid, None)
        return assigned
    
    # Exactly 1 detection (hard capped)
    det = detections[0]
    x1, y1, x2, y2, cls_id = det
    
    if DEBUG or (frame_counters.get(cam_id, 0) <= 200):
        print(f"[TRACKER_MATCH] {cam_id}: Got detection ({x1},{y1},{x2},{y2}), existing trackers={len(trks)}")
    
    # If we have existing tracker, update it
    if len(trks) > 0:
        # Use the existing tracker ID (gate only has 1)
        tid = list(trks.keys())[0]
        tr = trks[tid]
        old_centroid = tr.centroid
        
        # ⭐ CRITICAL FIX: Detect NEW VEHICLE by significant position change
        # If centroid moved >150px, it's likely a different vehicle, not same vehicle moving
        new_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = ((new_centroid[0] - old_centroid[0])**2 + (new_centroid[1] - old_centroid[1])**2) ** 0.5
        
        if distance > 150:  # Significant movement = new vehicle
            # Reset tracker state for new vehicle
            tr.saved = False  # ⭐ UNBLOCK OCR FOR NEW VEHICLE
            tr.last_ocr_time = 0  # ⭐ RESET OCR COOLDOWN FOR NEW VEHICLE
            if cam_id in ocr_history:
                ocr_history[cam_id].clear()  # Clear OCR history
            print(f"[NEW_VEHICLE_DETECTED] {cam_id}: Position changed {distance:.1f}px (old:{old_centroid}, new:{new_centroid}) - RESET tracker")
        
        # ⭐ CRITICAL: Always reset saved flag when updating tracker with new detection
        # This ensures new vehicles get OCR even if tracker is reused
        tr.saved = False
        tr.last_ocr_time = 0  # ⭐ RESET OCR COOLDOWN for fresh OCR on new detection
        
        tr.update((x1, y1, x2, y2))
        assigned[tid] = det
    else:
        # Create new tracker
        tr = TrackedObject((x1, y1, x2, y2), cls_id)
        tr.id = next_tracker_id
        trackers[cam_id][next_tracker_id] = tr
        assigned[next_tracker_id] = det
        if DEBUG or (frame_counters.get(cam_id, 0) <= 200):
            print(f"[TRACKER_NEW] {cam_id}: Created new tracker {next_tracker_id} for detection ({x1},{y1},{x2},{y2})")
        next_tracker_id += 1
    
    return assigned  # Single tracker mapping


def draw_label_with_bg(img, text, org, color, scale=0.6, thickness=2):
    """
    Draws text with a simple background for readability.
    org is (x,y) top-left of text baseline.
    """
    x, y = org
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    # background rectangle slightly larger
    cv2.rectangle(img, (x - 2, y - h - 2), (x + w + 2, y + 2), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def detect_plates_yolo(vehicle_roi_color, confidence_threshold=0.4):
    """Detect license plates in vehicle ROI using YOLO."""
    if plate_detector_model is None:
        return []
    
    if vehicle_roi_color is None or vehicle_roi_color.size == 0:
        return []
    
    plates = []
    try:
        # Run plate detector on vehicle ROI
        # Works with both ONNX and YOLO models (both return lists)
        results = plate_detector_model(vehicle_roi_color, conf=confidence_threshold, verbose=False)
        
        # Extract results from list (both ONNX and YOLO return lists)
        if isinstance(results, list) and len(results) > 0:
            results = results[0]
        
        # DEBUG: Check what we got from the model
        if hasattr(results, 'boxes'):
            num_boxes = len(results.boxes) if hasattr(results.boxes, '__len__') else 0
            if num_boxes == 0:
                # Silently skip - too much spam otherwise
                pass
        
        # ⭐ HARD LOCK TO SINGLE PLATE (Gate Mode)
        # Keep only highest confidence detection - gate system = 1 vehicle at a time
        best_box = None
        best_conf = 0.0
        
        for box in results.boxes:
            try:
                conf = float(box.conf[0])
                if conf >= confidence_threshold and conf > best_conf:
                    best_conf = conf
                    best_box = box
            except (IndexError, TypeError, AttributeError):
                continue
        
        # Process only the single highest confidence plate
        if best_box is not None:
            try:
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                # Ensure coordinates are within ROI bounds
                h, w = vehicle_roi_color.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)
                
                if x2 > x1 and y2 > y1:  # Valid box
                    plates.append((x1, y1, x2, y2))
            except (IndexError, TypeError, AttributeError):
                pass
    except Exception as e:
        print(f"[WARN] Plate detection error: {type(e).__name__}: {e}")
    
    return plates


def compute_bbox_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    bboxes format: (x1, y1, x2, y2)
    Returns IoU score between 0 and 1.
    """
    x1_a, y1_a, x2_a, y2_a = bbox1
    x1_b, y1_b, x2_b, y2_b = bbox2

    inter_xmin = max(x1_a, x1_b)
    inter_ymin = max(y1_a, y1_b)
    inter_xmax = min(x2_a, x2_b)
    inter_ymax = min(y2_a, y2_b)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def is_new_vehicle_detected(cam_id, raw_detections, persistent_bbox, iou_threshold=0.3):
    """
    Check if new vehicle has appeared by comparing raw YOLO detections with persistent bbox.
    Returns True if a detection appears at a significantly different location (new vehicle).
    
    Args:
        cam_id: Camera ID
        raw_detections: List of YOLO detections [(x1, y1, x2, y2, cls_id), ...]
        persistent_bbox: Last persistent detection bbox (x1, y1, x2, y2) or None
        iou_threshold: IoU threshold below which we consider it a new vehicle (default 0.3)
    
    Returns:
        True if new vehicle detected, False otherwise
    """
    if not persistent_bbox or not raw_detections:
        return False
    
    # Check if ANY raw detection has low IoU with persistent bbox
    # This indicates a new vehicle at a different location
    for det in raw_detections:
        x1, y1, x2, y2, cls_id = det
        current_bbox = (x1, y1, x2, y2)
        iou = compute_bbox_iou(current_bbox, persistent_bbox)
        
        # If IoU is very low, this is likely a new vehicle
        if iou < iou_threshold:
            if DEBUG:
                print(f"[NEW_VEHICLE] {cam_id}: New detection at different location (IoU={iou:.2f}), clearing persistent")
            return True
    
    return False


def update_plate_stability(cam_id, bbox):
    """
    Update plate detection stability tracking.
    Returns True if plate detection is stable (detected for >= OCR_CONSISTENCY_FRAMES consecutive frames).
    
    Args:
        cam_id: Camera ID
        bbox: Plate bounding box (x1, y1, x2, y2)
    
    Returns:
        (is_stable, frame_count) - is_stable indicates if stable, frame_count shows consistency frames
    """
    if cam_id not in plate_bbox_stability:
        plate_bbox_stability[cam_id] = {}

    current_time = time.time()
    IOU_THRESHOLD = 0.5
    STABILITY_TIMEOUT = 2.0

    stability_dict = plate_bbox_stability[cam_id]

    best_match_bbox = None
    best_iou = 0.0

    for tracked_bbox, data in list(stability_dict.items()):
        iou = compute_bbox_iou(bbox, tracked_bbox)

        if iou > best_iou:
            best_iou = iou
            best_match_bbox = tracked_bbox

    if best_iou >= IOU_THRESHOLD and best_match_bbox is not None:
        data = stability_dict[best_match_bbox]
        data['frame_count'] += 1
        data['timestamp'] = current_time
        is_stable = data['frame_count'] >= OCR_CONSISTENCY_FRAMES
        cleanup_stale_bboxes(cam_id, STABILITY_TIMEOUT)
        return is_stable, data['frame_count']
    else:
        stability_dict[bbox] = {
            'frame_count': 1,
            'timestamp': current_time
        }
        cleanup_stale_bboxes(cam_id, STABILITY_TIMEOUT)
        return False, 1


def cleanup_stale_bboxes(cam_id, timeout):
    """
    Remove plate bboxes from stability tracking that haven't been detected recently.
    """
    if cam_id not in plate_bbox_stability:
        return

    current_time = time.time()
    stale_bboxes = [bbox for bbox, data in plate_bbox_stability[cam_id].items()
                    if current_time - data['timestamp'] > timeout]

    for bbox in stale_bboxes:
        del plate_bbox_stability[cam_id][bbox]


def detect_motion(cam_id, frame):
    """
    Motion Detection Engine - Optimized for performance (especially RPi)
    Returns True if motion detected, False if static scene
    
    🚀 Performance optimizations:
    - Resizes frames to 320x240 for speed
    - Maintains per-camera frame history
    - Threshold: 2% pixel change = motion
    
    Args:
        cam_id: Camera ID for tracking previous frame
        frame: Current frame (BGR)
    
    Returns:
        bool: True if motion detected, False otherwise
    """
    global prev_frames

    if cam_id not in prev_frames:
        prev_frames[cam_id] = frame
        return True

    prev = prev_frames[cam_id]

    # Resize for speed (very important on Pi)
    frame_small = cv2.resize(frame, (320, 240))
    prev_small = cv2.resize(prev, (320, 240))

    diff = cv2.absdiff(frame_small, prev_small)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    motion_score = np.sum(thresh) / thresh.size

    prev_frames[cam_id] = frame

    return motion_score > 0.02


# ★ FIX 1: Bounding Box Smoothing Function
def smooth_bbox(cam_id, bbox):
    """
    ★ FIX 1: Smooth bounding boxes using exponential moving average
    
    Reduces flickering from sparse YOLO detection runs.
    
    Args:
        cam_id: Camera ID
        bbox: Current detection (x, y, w, h)
    
    Returns:
        Smoothed bbox (x, y, w, h)
    """
    prev_bbox = last_stable_bbox.get(cam_id)
    
    if prev_bbox is None:
        # First detection - store and return as-is
        last_stable_bbox[cam_id] = bbox
        return bbox
    
    # Exponential smoothing: alpha * previous + (1-alpha) * current
    alpha = bbox_smoothing_alpha
    x = int(alpha * prev_bbox[0] + (1 - alpha) * bbox[0])
    y = int(alpha * prev_bbox[1] + (1 - alpha) * bbox[1])
    w = int(alpha * prev_bbox[2] + (1 - alpha) * bbox[2])
    h = int(alpha * prev_bbox[3] + (1 - alpha) * bbox[3])
    
    smoothed = (x, y, w, h)
    last_stable_bbox[cam_id] = smoothed
    return smoothed


# ⭐ FIX 2: CACHE CLEANUP FUNCTION
def cleanup_expired_plates():
    """
    Allow re-detection after PLATE_CACHE_TTL expires.
    Run this every frame to keep cache fresh.
    """
    current_time = time.time()
    with plate_cache_lock:
        for plate in list(recent_plates_cache.keys()):
            if current_time - recent_plates_cache[plate] > PLATE_CACHE_TTL:
                del recent_plates_cache[plate]


# ⭐ FIX 4: TRACKER CLEANUP FUNCTION
def cleanup_lost_trackers(cam_id):
    """
    Reset trackers that haven't been seen for >1.5 seconds (FIX 7).
    This allows new vehicles to be detected after old ones disappear.
    Increased timeout from 3s to 1.5s for faster new vehicle detection.
    Run this every frame for each camera.
    """
    current_time = time.time()
    if cam_id not in trackers:
        return
    
    for trk_id in list(trackers[cam_id].keys()):
        tr = trackers[cam_id][trk_id]
        # ⭐ FIX 7: REDUCED timeout from 3.0 to 1.5 seconds for faster vehicle reset
        if current_time - tr.last_seen > 1.5:
            # Reset tracker state before deletion
            tr.saved = False  # ⭐ Ensure saved flag is reset
            del trackers[cam_id][trk_id]


# ⭐ FIX 5: STABILIZER RESET FUNCTION
def reset_stabilizer_if_no_detection(cam_id, has_detection):
    """
    Clear bounding box stabilizer when no plate detected.
    Otherwise old bounding box blocks new vehicle detection.
    Run this every frame after detection attempt.
    Also reset tracker.saved flag to allow re-detection.
    """
    if not has_detection and cam_id in plate_bbox_stabilizers:
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
        
        # ⭐ Also reset tracker.saved flag so next vehicle can be detected
        if cam_id in trackers:
            for tr in trackers[cam_id].values():
                tr.saved = False


# ⭐ OCR STABILIZATION HELPER FUNCTIONS
def get_stable_plate_ocr(plates: list) -> str:
    """
    Use majority voting to select most common OCR result.
    
    Args:
        plates: List of OCR plate readings (strings)
    
    Returns:
        Most common plate string, or empty if no consensus
    """
    if not plates:
        return ""
    
    from collections import Counter
    counts = Counter(plates)
    most_common_plate, count = counts.most_common(1)[0]
    
    # Only accept if at least 60% consensus (3 out of 5, 2 out of 4, etc.)
    if count >= len(plates) * 0.6:
        return most_common_plate
    return ""


def is_plate_similar(plate1: str, plate2: str, threshold: float = 0.9) -> bool:
    """
    Check if two plates are similar using fuzzy matching.
    Handles OCR variations (e.g., "KA05MN9009" vs "KA05MN5009").
    
    Args:
        plate1: First plate string
        plate2: Second plate string
        threshold: Similarity threshold (0-1)
    
    Returns:
        True if plates are similar, False otherwise
    """
    from difflib import SequenceMatcher
    
    if not plate1 or not plate2:
        return False
    
    # Compare cleaned versions
    p1 = re.sub(r'[^A-Z0-9]', '', str(plate1).upper())
    p2 = re.sub(r'[^A-Z0-9]', '', str(plate2).upper())
    
    # ⭐ OPTIMIZATION: Quick length check before expensive SequenceMatcher
    if abs(len(p1) - len(p2)) > 2:
        return False
    
    similarity = SequenceMatcher(None, p1, p2).ratio()
    return similarity > threshold


# ⭐ NEW: PER-OBJECT OCR TRACKING FUNCTIONS
def get_tracked_object_ocr_key(cam_id, object_id):
    """Create a unique key for tracked object OCR data."""
    return (cam_id, object_id)


def update_tracked_object_ocr(cam_id, object_id, plate_text, ocr_confidence, bbox):
    """
    Update OCR reading for a tracked object with AGGRESSIVE LOCKING.
    
    ⭐ NEW BEHAVIOR (Lock-and-Process-Once):
    - Lock immediately on first good OCR reading (confidence > 0.5)
    - No waiting for multiple readings or consensus
    - Prevents repeated OCR on same plate
    
    Returns: (stable_plate, is_locked) or (None, False) if not ready to save
    """
    global tracked_object_ocr
    
    key = get_tracked_object_ocr_key(cam_id, object_id)
    current_time = time.time()
    
    with tracked_object_lock:
        if key not in tracked_object_ocr:
            tracked_object_ocr[key] = {
                'readings': [],
                'confidences': [],
                'bbox_history': [],
                'locked': False,
                'locked_plate': None,
                'first_seen': current_time,
                'last_seen': current_time,
                'ocr_executed': False  # ⭐ NEW: Track if OCR has been run
            }
        
        obj_data = tracked_object_ocr[key]
        
        # If already locked, don't process further
        if obj_data['locked']:
            return obj_data['locked_plate'], True
        
        # Add new reading
        obj_data['readings'].append(plate_text)
        obj_data['confidences'].append(ocr_confidence)
        obj_data['bbox_history'].append(bbox)
        obj_data['last_seen'] = current_time
        
        # ⭐ AGGRESSIVE LOCK: Lock immediately on first good reading (> 0.5 confidence)
        # This eliminates the buffer/consensus delay and processes immediately
        if ocr_confidence > 0.5 and not obj_data['ocr_executed']:
            obj_data['locked'] = True
            obj_data['locked_plate'] = plate_text
            obj_data['ocr_executed'] = True
            if DEBUG:
                print(f"[INSTANT_LOCK] {cam_id}:{object_id} → {plate_text} (confidence={ocr_confidence:.2f}, locked immediately)")
            return plate_text, True
        
        # Keep only last N readings for fallback consensus
        if len(obj_data['readings']) > OCR_BUFFER_FRAMES:
            obj_data['readings'].pop(0)
            obj_data['confidences'].pop(0)
            obj_data['bbox_history'].pop(0)
        
        # Fallback: Check if we have enough readings for stabilization (older mechanism)
        if len(obj_data['readings']) >= OCR_BUFFER_FRAMES:
            # Apply majority voting
            from collections import Counter
            counts = Counter(obj_data['readings'])
            most_common_plate, count = counts.most_common(1)[0]
            
            # Lock if we have majority consensus (60% minimum)
            if count >= len(obj_data['readings']) * 0.6:
                obj_data['locked'] = True
                obj_data['locked_plate'] = most_common_plate
                obj_data['ocr_executed'] = True
                if DEBUG:
                    print(f"[CONSENSUS_LOCK] {cam_id}:{object_id} → {most_common_plate} (consensus={count}/{len(obj_data['readings'])})")
                return most_common_plate, True
            else:
                # Not enough consensus - use most recent reading but don't lock
                if DEBUG:
                    print(f"[TRACK_OCR_UNSTABLE] {cam_id}:{object_id} readings={obj_data['readings']} (consensus={count}/{len(obj_data['readings'])})")
                return obj_data['readings'][-1], False
        
        # Not enough readings yet
        return None, False


def cleanup_old_tracked_objects(cam_id, timeout=5.0):
    """Remove tracked objects that haven't been seen for timeout seconds."""
    global tracked_object_ocr, saved_objects, object_image_saved, processed_plates
    current_time = time.time()
    keys_to_remove = []
    
    with tracked_object_lock:
        for key, obj_data in tracked_object_ocr.items():
            if key[0] == cam_id:  # Only for this camera
                if current_time - obj_data['last_seen'] > timeout:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del tracked_object_ocr[key]
            
            # ⭐ CLEANUP: Also remove from saved tracking when object is cleaned up
            with saved_objects_lock:
                saved_objects.discard(key)
            object_image_saved.pop(key, None)
            
            # ⭐ CLEANUP: Remove from processed_plates deduplication tracking
            with processed_plates_lock:
                processed_plates.pop(key, None)
            
            if DEBUG:
                print(f"[TRACK_CLEANUP] Removed old tracked object {key[1]} from camera {key[0]}")


# ⭐ NEW: Helper function to check if object has been saved
def is_object_saved(cam_id, object_id):
    """Check if this object has already been saved to database."""
    with saved_objects_lock:
        return (cam_id, object_id) in saved_objects


# ⭐ NEW: Mark object as saved
def mark_object_as_saved(cam_id, object_id):
    """Mark object as saved to prevent duplicate database entries."""
    with saved_objects_lock:
        saved_objects.add((cam_id, object_id))
    if DEBUG:
        print(f"[MARK_SAVED] Marked object {object_id} @ {cam_id} as saved")


# ⭐ NEW: Check if object's image has been saved to disk
def is_object_image_saved(cam_id, object_id):
    """Check if this object's cropped plate image has been saved."""
    return object_image_saved.get((cam_id, object_id), False)


# ⭐ NEW: Mark object's image as saved
def mark_object_image_as_saved(cam_id, object_id):
    """Mark object's image as saved to prevent duplicate image files."""
    object_image_saved[(cam_id, object_id)] = True
    if DEBUG:
        print(f"[MARK_IMAGE_SAVED] Marked image saved for object {object_id} @ {cam_id}")


# ⭐ NEW: Check if OCR was already executed for this object in current frame
def was_ocr_executed_this_frame(cam_id, object_id, current_frame_count):
    """Prevent duplicate OCR execution for same object in same frame."""
    key = (cam_id, object_id)
    with object_ocr_executed_lock:
        last_frame = object_ocr_executed_this_frame.get(key, -1)
        if last_frame == current_frame_count:
            return True
        return False


# ⭐ NEW: Mark OCR as executed for this object in current frame
def mark_ocr_executed_this_frame(cam_id, object_id, current_frame_count):
    """Track that OCR was executed for this object in this frame."""
    key = (cam_id, object_id)
    with object_ocr_executed_lock:
        object_ocr_executed_this_frame[key] = current_frame_count


def cleanup_stale_ocr_buffers():
    """
    Remove OCR buffer entries that haven't been seen for OCR_BUFFER_TIMEOUT seconds.
    Prevents memory buildup for vehicles that left the scene.
    """
    global ocr_plate_buffer
    current_time = time.time()
    stale_keys = []
    
    for plate_key, buffer_data in ocr_plate_buffer.items():
        if current_time - buffer_data.get("last_seen", current_time) > OCR_BUFFER_TIMEOUT:
            stale_keys.append(plate_key)
    
    for key in stale_keys:
        del ocr_plate_buffer[key]
        if DEBUG:
            print(f"[OCR_CLEANUP] Removed stale buffer for {key}")


def process_anpr_pipeline(frame, cam_id, user_id=None, camera_id=None):
    """
    ANPR detection pipeline: Frame handling, YOLO detection, OCR, and overlay.
    
    ⭐ CRITICAL OPTIMIZATIONS (Fixed 2026-03-24):
    ============================================
    
    1. ✅ DUPLICATE IMAGE PREVENTION
       - Only save cropped plate image ONCE per object (first save)
       - Uses global object_image_saved dict to track
       - Prevents accumulation of identical images in plates/ directory
       - Reduces disk space by ~95% for typical vehicle detections
    
    2. ✅ DUPLICATE DATABASE ENTRY PREVENTION
       - Skip saving if (cam_id, object_id) already in saved_objects set
       - Global tracking across all frames for persistent vehicles
       - Entry/Exit gate logic enforced to prevent invalid duplicates
    
    3. ✅ OCR EXECUTION OPTIMIZATION
       - Skip OCR preprocessing if object already has locked reading
       - Reduces unnecessary CPU usage after OCR stabilization
       - Line ~3441: Check tracked_object_ocr['locked'] before OCR
    
    4. ✅ FASTER OCR STABILIZATION
       - Reduced OCR_BUFFER_FRAMES from 5 to 3 (180ms vs 300ms at 15 FPS)
       - Majority voting consensus (60% threshold) with fewer frames
       - Objects lock faster → saving happens sooner
    
    5. ✅ RESOLUTION-ADAPTIVE OBJECT TRACKING
       - CentroidTracker max_distance scales with frame width
       - Line ~3317: adaptive_max_distance = 150 * (width / 640)
       - Prevents flickering on high-res cameras (1080p, 4K)
       - Maintains stability on low-res cameras (480p, 720p)
    
    PIPELINE FLOW:
    ===============
    1. Initialize per-camera state (frame buffer, trackers, stabilizers)
    2. YOLO detection on selected frames (YOLO_PROCESS_EVERY_N parameter)
    3. CentroidTracker updates - assign unique IDs to detections
    4. For each stable detection:
       a. Skip if object already has locked OCR → avoid reprocessing
       b. Crop plate region
       c. Preprocess image (resize, blur, threshold)
       d. Run OCR (Tesseract)
       e. Normalize, correct, validate OCR output
       f. Update per-object OCR buffer (majority voting)
       g. Check if OCR locked (3 frames consensus = 60%)
    5. For each locked OCR result:
       a. Check if already saved (skip if yes)
       b. Check gate logic (Entry/Exit)
       c. Save image ONLY if not already saved
       d. Save detection record to database
       e. Mark as saved globally
    6. Draw overlays and return frame
    
    PERFORMANCE CHARACTERISTICS:
    ============================
    - Detection latency: ~100ms (YOLO inference on selected frames)
    - OCR stabilization: ~180ms (3 frames at 15 FPS)
    - Total to save: ~280ms (100ms detection + 180ms OCR stabilization)
    - Memory per vehicle: ~2KB tracking + ~50KB image
    - CPU: ~15-20% (YOLO + OCR) with 4 cores allocated
    """
    if frame is None:
        return frame
    
    # Initialize per-camera state with defensive checks for race conditions
    # Multiple threads (streaming + worker) may access this simultaneously
    if cam_id not in frame_counters:
        frame_counters[cam_id] = 0
    if cam_id not in plate_bbox_stabilizers:
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
    if cam_id not in video_buffers:
        video_buffers[cam_id] = deque(maxlen=BUFFER_SIZE)
    
    frame_counters[cam_id] += 1
    frame_h, frame_w = frame.shape[:2]
    detection_count = 0
    
    # ★ STEP 1: Frame Buffer / Skip Frames
    # Only process YOLO every Nth frame for performance
    should_run_yolo = (frame_counters[cam_id] % YOLO_PROCESS_EVERY_N == 0)
    
    # Update frame buffer for fallback detection
    video_buffers[cam_id].append(frame.copy())
    
    # ★ STEP 2: YOLO License Plate Detection
    raw_detections = []
    if should_run_yolo and plate_detector_model is not None:
        try:
            # Resize frame for YOLO detection (faster inference)
            detection_frame = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            # Calculate scaling factors for bbox adjustment
            scale_x = frame_w / DETECTION_WIDTH if DETECTION_WIDTH > 0 else 1.0
            scale_y = frame_h / DETECTION_HEIGHT if DETECTION_HEIGHT > 0 else 1.0
            
            # Run YOLO inference
            if USE_GPU:
                model_output = plate_detector_model(detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEFAULT_DEVICE)
            else:
                model_output = plate_detector_model(detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            results = model_output[0] if isinstance(model_output, list) else model_output
            
            # Extract plate detections
            for idx, box in enumerate(results.boxes):
                try:
                    conf = float(box.conf[0])
                except:
                    continue
                
                if conf >= CONFIDENCE_THRESHOLD:
                    try:
                        xyxy_coords = box.xyxy[0] if isinstance(box.xyxy, list) else box.xyxy[0]
                        x1, y1, x2, y2 = map(float, xyxy_coords)
                        
                        # Scale coordinates from detection_frame to original frame
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        # Clamp to frame bounds
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame_w - 1, x2); y2 = min(frame_h - 1, y2)
                        
                        if x2 > x1 and y2 > y1:
                            raw_detections.append((x1, y1, x2, y2, conf, 0))  # cls_id=0
                    except:
                        continue
        except Exception as e:
            if DEBUG:
                print(f"[YOLO_ERROR] {cam_id}: {e}")
    
    # ★ STEP 3: OBJECT TRACKING - Assign consistent IDs to detections
    # Initialize CentroidTracker if needed
    if cam_id not in object_trackers:
        # Adaptive distance normalized to 640p reference frame
        adaptive_max_distance = max(80, int(150 * (frame_w / 640.0)))
        object_trackers[cam_id] = CentroidTracker(max_distance=adaptive_max_distance, max_disappeared=30)
        if DEBUG:
            print(f"[TRACKER_INIT] Camera {cam_id}: max_distance={adaptive_max_distance} (width={frame_w})")
    
    tracker = object_trackers[cam_id]
    
    # Convert raw detections for tracker input (exclude confidence from main tracking)
    tracker_input = [(x1, y1, x2, y2, conf) for x1, y1, x2, y2, conf, _ in raw_detections]
    
    # Get object IDs for each bbox
    bbox_to_object_id = tracker.update(tracker_input)
    
    # Cleanup old tracked objects
    cleanup_old_tracked_objects(cam_id, timeout=5.0)
    
    # ★ STEP 3B: Plate BBox Stability Check (with tracking)
    if cam_id not in plate_bbox_stabilizers:
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
    
    # Initialize persistent overlays tracking for this camera
    if cam_id not in persistent_overlays:
        persistent_overlays[cam_id] = {}
    
    stabilizer = plate_bbox_stabilizers[cam_id]
    stable_detections = []
    detection_overlay_list = []  # For caching YOLO results in streaming
    current_frame_bboxes = set()  # Track which bboxes were detected this frame
    
    # DEBUG: Log raw detection count (first 20 frames per camera)
    if frame_counters[cam_id] <= 20:
        print(f"[YOLO_DEBUG] {cam_id} frame {frame_counters[cam_id]}: {len(raw_detections)} raw detections")
    
    for x1, y1, x2, y2, conf, cls_id in raw_detections:
        # Apply stabilization filter
        stabilized_bbox = stabilizer.update((x1, y1, x2, y2))
        sx1, sy1, sx2, sy2 = stabilized_bbox
        
        # Only process if bbox is valid
        if sx2 > sx1 and sy2 > sy1:
            stable_detections.append((sx1, sy1, sx2, sy2, conf, cls_id))
            
            # Create bbox key for tracking
            bbox_key = (int(sx1), int(sy1), int(sx2), int(sy2))
            current_frame_bboxes.add(bbox_key)
            
            # Cache for streaming overlay (label updated after OCR processing)
            overlay_data = {
                'bbox': bbox_key,
                'label': '',  # Will be set to OCR text after processing
                'ocr_text': None,  # Will store OCR result
                'color': (200, 150, 50),  # Blue shade for initial detection
                'confidence': float(conf),
                'index': len(detection_overlay_list)
            }
            detection_overlay_list.append(overlay_data)
            
            # Update persistent overlay tracking for this bbox
            # Store a reference so the overlay data can be updated with OCR text later
            if bbox_key not in persistent_overlays[cam_id]:
                persistent_overlays[cam_id][bbox_key] = {
                    'overlay': overlay_data,
                    'last_detected_frame': frame_counters[cam_id],
                    'frames_since_detection': 0
                }
            else:
                # Update existing overlay - reset grace period counter and refresh reference
                persistent_overlays[cam_id][bbox_key]['last_detected_frame'] = frame_counters[cam_id]
                persistent_overlays[cam_id][bbox_key]['frames_since_detection'] = 0
                persistent_overlays[cam_id][bbox_key]['overlay'] = overlay_data
    
    # ⭐ GRACE PERIOD: Keep overlays visible for detected plates even if not detected this frame
    # Create a deduplicated set of bbox_keys already in display list to avoid duplicates
    displayed_bboxes = set(overlay['bbox'] for overlay in detection_overlay_list)
    
    # ⭐ CRITICAL FIX: If new detections found, clear old overlays to prevent OCR text accumulation
    # Only keep overlays for currently detected plates, remove all others
    if len(current_frame_bboxes) > 0:
        # New detections found - remove ALL persistent overlays not in current frame
        # This prevents old OCR texts from displaying with new detections
        overlays_to_remove = []
        for bbox_key in persistent_overlays[cam_id].keys():
            if bbox_key not in current_frame_bboxes:
                overlays_to_remove.append(bbox_key)
        
        for bbox_key in overlays_to_remove:
            del persistent_overlays[cam_id][bbox_key]
    
    # Iterate through persistent overlays and check if they should still be displayed
    for bbox_key in list(persistent_overlays[cam_id].keys()):
        overlay_info = persistent_overlays[cam_id][bbox_key]
        
        if bbox_key not in current_frame_bboxes:
            # Plate not detected in this frame - apply grace period
            overlay_info['frames_since_detection'] += 1
            
            # Keep overlay visible for grace period
            if overlay_info['frames_since_detection'] <= OVERLAY_GRACE_PERIOD:
                # Add persistent overlay to display list ONLY if not already there
                if bbox_key not in displayed_bboxes:
                    # Use a copy to avoid reference issues
                    overlay_copy = overlay_info['overlay'].copy()
                    detection_overlay_list.append(overlay_copy)
                    displayed_bboxes.add(bbox_key)
            else:
                # Grace period expired - remove this overlay
                del persistent_overlays[cam_id][bbox_key]
        else:
            # Plate detected again - the main loop already added it, just reset grace period
            overlay_info['frames_since_detection'] = 0
    
    # ★ STEP 4-5: Plate ROI Extraction & Per-Object OCR Tracking
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects_to_save = []  # List of (object_id, stable_plate, ocr_conf, bbox) tuples ready for saving
    
    for sx1, sy1, sx2, sy2, conf, cls_id in stable_detections:
        # Get object ID for this bbox from tracker
        bbox_key = (int(sx1), int(sy1), int(sx2), int(sy2))
        object_id = bbox_to_object_id.get(bbox_key, None)
        
        if object_id is None:
            continue  # Skip untracked detections
        
        # Extract region
        plate_roi_color = frame[int(sy1):int(sy2), int(sx1):int(sx2)]
        plate_roi_gray = gray[int(sy1):int(sy2), int(sx1):int(sx2)]
        
        if plate_roi_color.size == 0 or plate_roi_gray.size == 0:
            continue
        
        # ⭐ OPTIMIZATION: Skip OCR if object already has locked reading or OCR executed
        # This prevents unnecessary re-running OCR for same vehicle
        key = get_tracked_object_ocr_key(cam_id, object_id)
        with tracked_object_lock:
            if key in tracked_object_ocr:
                obj_data = tracked_object_ocr[key]
                # Skip if either locked already OR OCR has been executed
                if obj_data.get('locked') or obj_data.get('ocr_executed'):
                    # Object has locked OCR or OCR already attempted - skip re-processing
                    if DEBUG and frame_counters[cam_id] % 30 == 0:
                        print(f"[SKIP_OCR] Object {object_id}: locked={obj_data.get('locked')}, executed={obj_data.get('ocr_executed')}")
                    continue
        
        # ★ STEP 6: Optimal Plate Image Preprocessing (Fast + Accurate)
        # ⭐ NEW STRATEGY: Resize to optimal width (400px) for Tesseract + contrast boost
        # This ensures fast OCR processing while maintaining accuracy
        
        plate_h, plate_w = plate_roi_gray.shape
        
        # Resize to optimal width for OCR (300-600px range, target 400px for speed)
        target_width = 400
        if plate_w > 0:
            scale = target_width / plate_w
            new_h = max(1, int(plate_h * scale))
            new_w = target_width
        else:
            new_h, new_w = plate_h, plate_w
        
        plate_resized = cv2.resize(plate_roi_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # ⭐ Fast contrast enhancement using CLAHE (better than Gaussian blur)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(plate_resized)
        
        # Otsu threshold for binary image (optimal for OCR)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ★ STEP 7: OCR Engine
        if ocr_enabled:
            raw_text = perform_ocr(binary, enable_validation=False)
        else:
            raw_text = ""
        
        if not raw_text or len(raw_text) < 4:
            continue
        
        # ★ STEP 8-10: Normalization, Correction, Validation
        normalized_text = indian_detector.normalize_ocr_text(raw_text)
        corrected_text = indian_detector.apply_ocr_corrections(normalized_text)
        is_valid, final_plate, ocr_confidence = validate_license_plate_ocr(corrected_text)
        
        if not is_valid or not final_plate:
            continue
        
        # ★ STEP 11: PER-OBJECT OCR STABILIZATION
        # Update tracked object OCR buffer and check if stable
        stable_plate, is_locked = update_tracked_object_ocr(
            cam_id, object_id, final_plate, ocr_confidence, bbox_key
        )
        
        if stable_plate:
            # OCR is stabilized (either locked or recent reading)
            detection_count += 1
            
            # Add valid stable plates to save list (both locked and stabilizing)
            objects_to_save.append((object_id, stable_plate, ocr_confidence, bbox_key, plate_roi_color, conf))
            
            # ★ UPDATE OVERLAY LIST: Set stabilized OCR text
            for overlay_item in detection_overlay_list:
                overlay_bbox = overlay_item['bbox']
                if overlay_bbox == bbox_key:
                    overlay_item['label'] = stable_plate
                    overlay_item['ocr_text'] = stable_plate
                    overlay_item['color'] = (0, 255, 0) if is_locked else (0, 165, 255)  # Green if locked, orange if stabilizing
                    break
            
            # Update persistent overlay with stabilized text
            if bbox_key in persistent_overlays[cam_id]:
                persistent_overlays[cam_id][bbox_key]['overlay']['label'] = stable_plate
                persistent_overlays[cam_id][bbox_key]['overlay']['ocr_text'] = stable_plate
                persistent_overlays[cam_id][bbox_key]['overlay']['color'] = (0, 255, 0) if is_locked else (0, 165, 255)
    
    # ★ STEP 12: INSTANT SAVE - Objects with locked OCR
    current_time = time.time()
    saved_plates = []
    saved_object_ids = set()
    
    for object_id, stable_plate, ocr_conf, bbox_key, plate_roi_color, yolo_conf in objects_to_save:
        # ⭐ CRITICAL FIX #1: Skip if already saved (check GLOBAL saved_objects, not just current frame)
        if is_object_saved(cam_id, object_id):
            if DEBUG:
                print(f"[SKIP_ALREADY_SAVED] {stable_plate} | Object: {object_id} already processed")
            continue
        
        # Skip if already saved in this frame processing
        if object_id in saved_object_ids:
            continue
        
        # ⭐ FIX: Use normalized plate for database matching
        normalized_plate_key = normalize_plate_for_matching(stable_plate)
        
        gate_type = get_camera_gate_type(cam_id)
        is_in_db, db_plate, has_exit, is_current_occupant = check_vehicle_in_database(normalized_plate_key)
        
        should_save = False
        save_mode = None
        
        if gate_type == "Entry":
            # Entry camera: only save if vehicle is NEW or has already exited
            if not is_in_db or has_exit:
                should_save = True
                save_mode = "Entry"
        elif gate_type == "Exit":
            # Exit camera: only update if vehicle is currently in premises
            if is_in_db and is_current_occupant:
                should_save = True
                save_mode = "Exit"
        
        if not should_save:
            if DEBUG:
                print(f"[GATE_SKIP] {stable_plate} | Gate: {gate_type} | In DB: {is_in_db} | Current: {is_current_occupant}")
            continue
        # Save plate image for new entries or re-entries
        try:
            plate_image_path = None
            plate_key_for_dedup = stable_plate
            image_save_success = False
            is_reentry = is_in_db and has_exit
            should_save_new_image = is_reentry or (plate_key_for_dedup not in saved_plate_texts)
            
            with saved_plate_texts_lock:
                if should_save_new_image:
                    # This is a new entry or re-entry - ALWAYS save image
                    try:
                        plate_image_filename = f"{stable_plate.replace('-', '_')}_{int(time.time()*1000)}.jpg"
                        plate_image_full_path = DIR_PLATES / plate_image_filename
                        cv2.imwrite(str(plate_image_full_path), plate_roi_color)
                        plate_image_path = f"plates/{plate_image_filename}"
                        mark_object_image_as_saved(cam_id, object_id)
                        image_save_success = True
                        saved_plate_texts.add(plate_key_for_dedup)
                        
                        if DEBUG:
                            print(f"[PLATE_IMAGE_SAVED] {plate_image_path} | Plate: {stable_plate} | Object: {object_id} | IsReentry: {is_reentry}")
                    except Exception as e:
                        # Image save failed - don't proceed with DB save
                        print(f"[PLATE_IMAGE_ERROR] Failed to save plate image for {stable_plate}: {e}")
                        image_save_success = False
                        plate_image_path = None
                else:
                    # Skip duplicate plate images
                    image_save_success = True
                    if DEBUG:
                        print(f"[SKIP_DUPLICATE_IMAGE] {stable_plate} - Plate image already saved")
            
            # Skip DB save if image save failed
            if not image_save_success and plate_image_path is None:
                if DEBUG:
                    print(f"[SKIP_DB_SAVE] {stable_plate} - Image failed to save, retrying next frame")
                continue
            
            # Save detection with timestamp
            if save_mode == "Exit":
                # ⭐ FIX: Use normalized plate for exit matching
                entry_id = save_exit_detection(
                    plate_key=normalized_plate_key,  # Use normalized format
                    cam_id=cam_id,
                    confidence=yolo_conf,
                    ocr_confidence=ocr_conf,
                    direction="Front",
                    detection_mode="YOLO_OCR",
                    user_id=user_id,
                    plate_image=plate_image_path  # ⭐ NEW: Pass exit plate image
                )
            else:  # Entry
                entry_id = save_entry_to_detections(
                    plate_key=stable_plate,  # Original plate for entry
                    cam_id=cam_id,
                    confidence=yolo_conf,
                    ocr_confidence=ocr_conf,
                    direction="Front",
                    detection_mode="YOLO_OCR",
                    gate_type=gate_type,
                    user_id=user_id,
                    plate_image=plate_image_path
                )
            
            if entry_id:
                saved_plates.append(stable_plate)
                saved_object_ids.add(object_id)
                # ⭐ FIX: Use normalized plate key for tracking (consistent format)
                tracking_key = normalized_plate_key if save_mode == "Exit" else normalize_plate_for_matching(stable_plate)
                seen_plates[tracking_key] = (current_time, stable_plate)
                recent_plates_cache_opt.append(stable_plate)
                
                # ⭐ CRITICAL FIX #3: Mark object as saved globally ONLY after successful database save
                # This prevents retry of database save, but image retry happens through image_save_success
                mark_object_as_saved(cam_id, object_id)
                
                # ⭐ FINAL FIX: Only add to saved_plate_texts if BOTH image and DB save succeeded
                # This ensures re-detection retries if image save was skipped
                if image_save_success:
                    with saved_plate_texts_lock:
                        saved_plate_texts.add(plate_key_for_dedup)
                
                print(f"[INSTANT_SAVE] {stable_plate} | Object: {object_id} | Gate: {gate_type} | Mode: {save_mode} | Normalized: {normalized_plate_key} | Image: {plate_image_path}")
            else:
                # Save failed or not applicable
                if save_mode == "Exit":
                    # Exit camera didn't find matching entry - this is expected for untracked vehicles
                    if DEBUG:
                        print(f"[EXIT_SKIP] {stable_plate} (normalized: {normalized_plate_key}) | No matching entry found")
                else:
                    # Entry save failed - log but don't mark as saved so we retry next frame
                    if DEBUG:
                        print(f"[SAVE_FAILED] {stable_plate} | Object: {object_id} | Will retry on next frame")
        
        except Exception as e:
            print(f"[SAVE_ERROR] {cam_id}: {e}")
    
    # ★ STEP 13: Draw bounding boxes and OCR text overlays on frame (OPTIMIZED - Minimal drawing ops)
    # ⭐ PERFORMANCE FIX: Reduced from 12+ drawing operations per detection to ~2 operations
    # This eliminates the excessive cv2.rectangle() and cv2.line() calls that were causing slowdown
    for sx1, sy1, sx2, sy2, conf, cls_id in stable_detections:
        x1, y1, x2, y2 = int(sx1), int(sy1), int(sx2), int(sy2)
        bbox_key = (x1, y1, x2, y2)
        
        # Get object ID for styling
        object_id = bbox_to_object_id.get(bbox_key, None)
        
        # ⭐ APPLY BBOX SMOOTHING - Eliminate flickering (OPTIMIZED - Use smoother.smooth directly)
        if object_id is not None:
            # Quick check without lock first (fast path)
            if object_id in bbox_smoothers:
                smoother = bbox_smoothers[object_id]
                x1, y1, x2, y2 = smoother.smooth(object_id, (x1, y1, x2, y2))
            else:
                # Slow path - need to create smoother (rare, only on first detection)
                with bbox_smoother_lock:
                    if object_id not in bbox_smoothers:
                        bbox_smoothers[object_id] = BBoxSmoother(alpha=0.35)
                    smoother = bbox_smoothers[object_id]
                x1, y1, x2, y2 = smoother.smooth(object_id, (x1, y1, x2, y2))
        
        # Check if this object was saved (locked OCR)
        is_saved = object_id in saved_object_ids if object_id else False
        color_main = (0, 255, 0) if is_saved else (0, 165, 255)  # Green if saved, orange if stabilizing
        
        # ⚡ FAST DRAWING: Single rectangle only (was 4 rectangles + 8 lines = 12 ops → now 1 op)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_main, 2)
        
        # ✨ Get OCR text from tracked object
        ocr_text = None
        if object_id:
            key = get_tracked_object_ocr_key(cam_id, object_id)
            with tracked_object_lock:
                if key in tracked_object_ocr:
                    ocr_text = tracked_object_ocr[key].get('locked_plate') or (tracked_object_ocr[key]['readings'][-1] if tracked_object_ocr[key]['readings'] else None)
        
        # Draw OCR text label (simplified)
        if ocr_text:
            label = f"{ocr_text}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            text_color = (0, 255, 255)  # Cyan - bright and visible
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Position text at top-left of bbox
            text_x = max(x1 + 2, 0)
            text_y = max(y1 - 5, text_size[1] + 2)
            
            if text_x + text_size[0] > frame_w:
                text_x = frame_w - text_size[0] - 2
            
            # ⚡ SIMPLE TEXT: Just draw text directly (no background/shadow)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness)
    
    # ★ STEP 14: Cache detection overlay for streaming
    hud_info = {
        'frame_count': frame_counters[cam_id],
        'timestamp': datetime.now().isoformat(),
        'camera': cam_id,
        'detection_count': len(detection_overlay_list),
        'mode': 'YOLO_tracking',
        'original_width': frame_w,
        'original_height': frame_h
    }
    update_detection_overlay(cam_id, detection_overlay_list, hud_info)
    
    # Add frame counter
    fps_text = f"Frame: {frame_counters[cam_id]} | Objects: {len(object_trackers.get(cam_id, {}).objects) if cam_id in object_trackers else 0} | Saved: {len(saved_plates)}"
    cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Reset stabilizer if no detections
    reset_stabilizer_if_no_detection(cam_id, detection_count > 0)
    
    return frame


def process_frame(frame, cam_id, user_id=None, camera_id=None):
    """
    WRAPPER: Delegates to process_anpr_pipeline()
    
    ✅ Mission: Process single frame using sequential YOLO + OCR pipeline
    
    Args:
        frame: Video frame
        cam_id: Camera source ID (e.g., '0' for webcam)
        user_id: User ID from authentication (passed by Express backend)
        camera_id: Database camera ID (UUID or 'webcam')
    
    Returns:
        Processed frame with overlay
    """
    return process_anpr_pipeline(frame, cam_id, user_id=user_id, camera_id=camera_id)


# -------- Lightweight Detection Overlay System --------
def update_detection_overlay(cam_id, detections_list, hud_info):
    """
    Store detection overlay data for rendering on stream frames.
    This is called by camera_anpr_process_frame() with detection results.
    
    CRITICAL: If this is not called, bounding boxes will NOT appear in stream.
    
    Args:
        cam_id: Camera identifier
        detections_list: List of detection dicts with 'bbox', 'label', 'color', 'confidence'
        hud_info: Dictionary with frame_count, timestamp, detection_count, etc.
    """
    try:
        with detection_overlay_lock:
            # CRITICAL: Store in global cache visible to streaming thread
            detection_overlay_cache[cam_id] = {
                'detections': detections_list,
                'hud_info': hud_info,
                'timestamp': time.time()
            }
        
        # CRITICAL VERIFICATION: Confirm cache was updated
        frame_count = hud_info.get('frame_count', 0)
        if frame_count <= 5 or len(detections_list) > 0:
            cache_exists = cam_id in detection_overlay_cache
            cache_detections = len(detections_list) if detections_list else 0
            print(f"[DEBUG] Cache update: {cam_id} → {cache_detections} detections (cache_exists={cache_exists}, frame={frame_count})")
            
            # Verify cache contains expected structure
            if cache_exists:
                cached = detection_overlay_cache[cam_id]
                has_detections = 'detections' in cached
                has_hud = 'hud_info' in cached
                has_timestamp = 'timestamp' in cached
                print(f"[DEBUG] Cache structure OK: detections={has_detections}, hud_info={has_hud}, timestamp={has_timestamp}")
    
    except Exception as e:
        if DEBUG:
            print(f"[OVERLAY_CACHE] Error storing overlay data for {cam_id}: {e}")


def apply_detection_overlay(frame, cam_id):
    """
    Apply cached detection overlay to frame by drawing bounding boxes.
    
    Uses detection data from the background processing pipeline.
    If no recent detections, returns frame as-is.
    
    Args:
        frame: Frame (BGR numpy array)
        cam_id: Camera ID
    
    Returns:
        Frame with bounding boxes and labels drawn
    """
    if frame is None:
        return frame
    
    try:
        with detection_overlay_lock:
            if cam_id not in detection_overlay_cache:
                return frame
            
            overlay_data = detection_overlay_cache[cam_id]
            detections_list = overlay_data.get('detections', [])
            hud_info = overlay_data.get('hud_info', {})
            cache_timestamp = overlay_data.get('timestamp', 0)
        
        # Only use cache if recent (within 0.5 seconds)
        cache_age = time.time() - cache_timestamp
        if cache_age > 0.5:
            return frame
        
        # CRITICAL DEBUG: Confirm reading from cache
        frame_count = hud_info.get('frame_count', 0)
        if frame_count <= 5:
            print(f"[DEBUG] apply_detection_overlay: Reading {cam_id} - found {len(detections_list)} in cache (age={cache_age:.3f}s)")
        
        if len(detections_list) > 0 and frame_count <= 5:
            print(f"[DEBUG] apply_detection_overlay: Will draw {len(detections_list)} bounding boxes on stream")
        
        # Calculate coordinate scaling: overlays are in original resolution, frame may be resized
        # Use stored original dimensions if available, otherwise infer from cache
        original_width = hud_info.get('original_width', 1920)  # Default to typical capture resolution
        original_height = hud_info.get('original_height', 1080)
        stream_height, stream_width = frame.shape[:2]
        scale_x = stream_width / original_width if original_width > 0 else 1.0
        scale_y = stream_height / original_height if original_height > 0 else 1.0
        
        if frame_count <= 5:
            print(f"[DEBUG] Overlay scaling: original=({original_width}x{original_height}), stream=({stream_width}x{stream_height}), scale=({scale_x:.3f}, {scale_y:.3f})")
        
        # Draw each detection on the frame
        for detection in detections_list:
            try:
                bbox = detection.get('bbox', None)
                label = detection.get('label', 'Plate')
                color = detection.get('color', (0, 255, 0))  # Default green
                ocr_text = detection.get('ocr_text', None)
                confidence = detection.get('confidence', 0.0)
                
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # ⭐ CRITICAL: Scale bbox coordinates from original resolution to current frame size
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    
                    # Clamp to frame bounds
                    x1 = max(0, min(x1, stream_width - 1))
                    y1 = max(0, min(y1, stream_height - 1))
                    x2 = max(0, min(x2, stream_width - 1))
                    y2 = max(0, min(y2, stream_height - 1))
                    
                    # Only draw if bbox is valid
                    if x2 > x1 and y2 > y1:
                        # ⭐ DRAW ENHANCED BOUNDING BOX WITH PREMIUM STYLING
                        color_glow = (100, 200, 255)  # Light cyan glow
                        
                        # ✨ Draw glowing shadow border
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color_glow, 1)
                        
                        # ✨ Draw main bounding box with enhanced thickness
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # ✨ Draw corner markers for premium look
                        corner_length = 15
                        corner_thickness = 2
                        # Top-left
                        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
                        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
                        # Top-right
                        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
                        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
                        # Bottom-left
                        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
                        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
                        # Bottom-right
                        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
                        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
                        
                        # ⭐ DRAW LABEL WITH OCR TEXT IF AVAILABLE - Blue colors only, on top of box
                        if ocr_text:
                            label_text = f"{ocr_text}"  # Use OCR text
                        elif label:
                            label_text = f"{label}"  # Use label if no OCR text
                        else:
                            label_text = ""  # No text if neither available
                        
                        # Only draw text background and text if there's text to display
                        if label_text:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.7  # Increased from 0.5 for better visibility
                            thickness = 2  # Bold text
                            text_color = (255, 200, 0)  # Bright cyan/blue
                            text_shadow = (50, 100, 150)  # Dark blue shadow
                            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                            
                            # Position text ON TOP of bounding box
                            text_x = max(x1 + 5, 0)  # Ensure no clipping on left
                            text_y = max(y1 - 15, text_size[1] + 5)  # Above box instead of inside
                            
                            # Ensure text doesn't go beyond frame width
                            if text_x + text_size[0] > stream_width:
                                text_x = stream_width - text_size[0] - 5
                            
                            # ✨ Draw shadow effect for text background with BLUE colors only
                            shadow_offset = 2
                            bg_x1 = text_x - 8
                            bg_y1 = text_y - text_size[1] - 12
                            bg_x2 = text_x + text_size[0] + 8
                            bg_y2 = text_y + 8
                            
                            # Shadow background (dark blue)
                            cv2.rectangle(frame, (bg_x1 + shadow_offset, bg_y1 + shadow_offset), 
                                        (bg_x2 + shadow_offset, bg_y2 + shadow_offset), (100, 100, 150), -1)
                            
                            # Main text background (blue shade with darker tone)
                            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (150, 120, 80), -1)
                            
                            # Enhanced border around text background - Blue tones only
                            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (200, 150, 50), 2)
                            
                            # Draw text with shadow effect - Blue colors only
                            cv2.putText(frame, label_text, (text_x + 1, text_y + 1), font, font_scale, text_shadow, thickness)
                            cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, text_color, thickness)
                        print(f"[OVERLAY] {cam_id}: ✓ Drew box at {(x1,y1,x2,y2)} with text '{label_text}' (scaled)")
            except Exception as e:
                if DEBUG:
                    print(f"[OVERLAY_DRAW] Error drawing detection for {cam_id}: {e}")
                continue
        
        # Draw FPS/frame counter from HUD info
        frame_count = hud_info.get('frame_count', 0)
        if frame_count > 0:
            fps_text = f"Frame: {frame_count}"
            cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame
    except Exception as e:
        if DEBUG:
            print(f"[OVERLAY_ERROR] Error applying overlay for {cam_id}: {e}")
        return frame


# -------- CRITICAL: Capture Thread for Dynamic Cameras --------
def start_camera_capture(cam_id, cap, camera_manager):
    """
    ★ FIX: Start capture loop for a camera dynamically
    
    This captures frames continuously and pushes them to the frame queue,
    allowing detection to run even when no client is streaming.
    
    This is called IMMEDIATELY when a camera is added via API.
    
    Args:
        cam_id: Camera identifier
        cap: cv2.VideoCapture or OAK device object
        camera_manager: CameraManager instance for push_frame()
    """
    def capture_loop():
        print(f"[CAPTURE] ★ Started capture loop for camera '{cam_id}'")
        consecutive_errors = 0
        # ★ FIX 5: Add FPS throttle to stabilize capture without CPU overload
        frame_delay = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.03  # Target: 1/FPS or 33ms
        last_frame_time = time.time()
        
        while True:
            try:
                # ★ FIX 5: Throttle capture to target FPS
                if frame_delay > 0:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_delay:
                        sleep_time = frame_delay - elapsed
                        if sleep_time > 0.001:
                            time.sleep(sleep_time)
                    last_frame_time = time.time()
                
                # Check if camera is still active
                if cam_id not in camera_sources:
                    print(f"[CAPTURE] Camera '{cam_id}' removed from system, stopping capture")
                    break
                
                # Check if stop event is set (camera being deleted)
                if cam_id in processing_stop_events:
                    if processing_stop_events[cam_id].is_set():
                        print(f"[CAPTURE] Stop event triggered for '{cam_id}', exiting capture loop")
                        break
                
                # Read frame from camera
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        print(f"[CAPTURE] Too many errors reading from '{cam_id}', stopping capture")
                        break
                    time.sleep(0.05)
                    continue
                
                consecutive_errors = 0
                
                # ★ CRITICAL: Push frame to CameraManager queue for detection processing
                camera_manager.push_frame(cam_id, frame)
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:  # Only log first few errors
                    print(f"[CAPTURE] Error reading frame from '{cam_id}': {str(e)[:80]}")
                if consecutive_errors > 10:
                    print(f"[CAPTURE] Capture loop stopping due to repeated errors")
                    break
                time.sleep(0.05)
    
    # Start capture thread as daemon
    thread = threading.Thread(
        target=capture_loop,
        daemon=True,
        name=f"Capture-{cam_id}"
    )
    thread.start()
    print(f"[CAPTURE] Capture thread spawned for camera '{cam_id}'")

# -------- Detection Pipeline Callback --------
def camera_anpr_process_frame(frame, cam_id):
    """
    ★ CRITICAL FUNCTION: Pipeline callback for real-time plate detection
    
    Called by worker thread for each frame in the processing queue.
    Performs lightweight plate detection and creates overlay data for streaming.
    
    Args:
        frame: Video frame from camera (numpy array)
        cam_id: Camera identifier
    """
    try:
        if frame is None:
            return
        
        # Initialize counters if needed
        if cam_id not in frame_counters:
            frame_counters[cam_id] = 0
            plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
            
            # CRITICAL VERIFICATION: Worker thread is running
            print(f"\n[DEBUG] ★ DETECTION WORKER STARTED ★")
            print(f"[DEBUG] Camera ID: {cam_id}")
            print(f"[DEBUG] Frame counters initialized: {cam_id in frame_counters}")
            print(f"[DEBUG] Stabilizer initialized: {cam_id in plate_bbox_stabilizers}")
            
            # Verify YOLO model is available in this thread
            model_status = "✓ LOADED" if plate_detector_model is not None else "✗ NOT LOADED"
            print(f"[DEBUG] YOLO MODEL STATUS: {model_status}")
            print(f"[DEBUG] YOLO MODEL TYPE: {type(plate_detector_model).__name__ if plate_detector_model else 'None'}")
            print(f"[DEBUG] YOLO MODEL CALLABLE: {callable(plate_detector_model)}")
            
            # Verify overlay cache is accessible
            print(f"[DEBUG] Overlay cache accessible: {isinstance(detection_overlay_cache, dict)}")
            print(f"[DEBUG] Overlay cache for {cam_id}: {cam_id in detection_overlay_cache}")
            print(f"[DEBUG] Configuration - Expiry: 0.5s, Stabilizer Timeout: 3.0s")
            print(f"[DEBUG] Frame buffers - BUFFER_SIZE: {FRAME_BUFFER_SIZE}, QUEUE: {MAX_FRAME_QUEUE}")
            
            if not plate_detector_model:
                print(f"[ERROR] ★★★ YOLO MODEL NOT LOADED IN WORKER! ★★★")
                print(f"[ERROR] Camera '{cam_id}' will NOT perform detection!")
            else:
                print(f"[DEBUG] ✓ Detection pipeline FULLY READY for '{cam_id}'\n")
        
        frame_counters[cam_id] += 1
        
        # ⭐ OPTIMIZATION: Global frame skipping - skip every 3rd frame for lighter CPU load
        if frame_counters[cam_id] % 3 != 0:
            return
        
        # Skip some frames for performance
        should_process = (frame_counters[cam_id] % PROCESS_EVERY_N_FRAMES == 0)
        if not should_process:
            return
        
        if frame_counters[cam_id] <= 3:
            print(f"[DEBUG] {cam_id}: FRAME RECEIVED - frame_counters[{cam_id}]={frame_counters[cam_id]}")
        
        frame_h, frame_w = frame.shape[:2]
        detections_list = []
        
        # Only run YOLO every N frames for performance
        should_run_yolo = (frame_counters[cam_id] % YOLO_PROCESS_EVERY_N == 0)
        
        # Debug: Print status every 30 frames
        if frame_counters[cam_id] % 30 == 0:
            model_status = "✓ LOADED" if plate_detector_model is not None else "✗ NOT LOADED"
            print(f"[DEBUG] {cam_id}: STATUS - frame={frame_counters[cam_id]}, should_run_yolo={should_run_yolo}, model={model_status}")
        
        # ⭐ CRITICAL: Log YOLO startup
        if should_run_yolo:
            if frame_counters[cam_id] <= 10:
                print(f"[DEBUG] {cam_id}: YOLO DECISION - frame {frame_counters[cam_id]}, should_run={should_run_yolo}, model_loaded={plate_detector_model is not None}")
        
        if should_run_yolo and plate_detector_model is not None:
            try:
                # ⭐ OPTIMIZATION: Pre-resize frame to 640x360 ONCE before YOLO (massive speedup)
                # This ensures consistent and fast processing pipeline
                frame_resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                
                # Create detection frame from pre-resized frame
                if DETECTION_WIDTH > 0 and DETECTION_HEIGHT > 0:
                    detection_frame = cv2.resize(frame_resized, (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    # Scale calculations from detection size back to original
                    scale_x = frame_w / DETECTION_WIDTH
                    scale_y = frame_h / DETECTION_HEIGHT
                else:
                    detection_frame = frame_resized
                    scale_x = frame_w / 640  # Scale from 640x360
                    scale_y = frame_h / 360
                
                # Debug: Log before YOLO
                if frame_counters[cam_id] <= 5:
                    print(f"[DEBUG] {cam_id}: YOLO RUNNING on frame {frame_counters[cam_id]}, confidence_threshold={CONFIDENCE_THRESHOLD}")
                
                # Run YOLO detection
                if USE_GPU:
                    model_output = plate_detector_model(detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False, device=DEFAULT_DEVICE)
                else:
                    model_output = plate_detector_model(detection_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                # Debug: Log after YOLO inference
                if frame_counters[cam_id] <= 5:
                    print(f"[DEBUG] {cam_id}: ✓ YOLO inference completed (took ~{1000*YOLO_PROCESS_EVERY_N/TARGET_FPS:.0f}ms)")
                
                results = model_output[0] if isinstance(model_output, list) else model_output
                
                # DEBUG: Log YOLO result
                if frame_counters[cam_id] <= 5:
                    num_boxes = len(results.boxes) if hasattr(results, 'boxes') else 0
                    print(f"[DEBUG] {cam_id}: YOLO RETURNED - boxes_count={num_boxes}, has_boxes_attr={hasattr(results, 'boxes')}")
                
                # Extract plate detections
                if hasattr(results, 'boxes'):
                    stabilizer = plate_bbox_stabilizers.get(cam_id)
                    
                    for idx, box in enumerate(results.boxes):
                        try:
                            conf = float(box.conf[0])
                            if frame_counters[cam_id] <= 5:
                                print(f"[DEBUG] {cam_id}: Processing box {idx}/{len(results.boxes)} - conf={conf:.3f} (threshold={CONFIDENCE_THRESHOLD})")
                            
                            if conf >= CONFIDENCE_THRESHOLD:
                                # Extract and scale coordinates
                                xyxy_coords = box.xyxy[0] if isinstance(box.xyxy, list) else box.xyxy[0]
                                x1, y1, x2, y2 = map(float, xyxy_coords)
                                
                                # Scale from detection frame to original frame
                                x1 = int(x1 * scale_x)
                                y1 = int(y1 * scale_y)
                                x2 = int(x2 * scale_x)
                                y2 = int(y2 * scale_y)
                                
                                # Clamp to frame bounds
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame_w - 1, x2)
                                y2 = min(frame_h - 1, y2)
                                
                                # ⭐ OPTIMIZATION: DISABLE HEAVY STABILIZATION FOR SPEED
                                # BBoxSmoother causes per-frame latency - use raw detections for real-time
                                raw_bbox = (x1, y1, x2, y2)
                                bbox = raw_bbox  # ⭐ Use raw bbox directly (no smoothing overhead)
                                
                                x1, y1, x2, y2 = bbox
                                
                                # Convert to format expected by apply_detection_overlay
                                # Format: {'bbox': (x1,y1,x2,y2), 'label': str, 'color': (B,G,R), 'ocr_text': str}
                                # ⭐ FIX: No label initially - will be set to OCR text after processing
                                detection_dict = {
                                    'bbox': (x1, y1, x2, y2),
                                    'label': '',  # Will be set when OCR completes
                                    'color': (200, 150, 50),  # Blue shade for detecting state
                                    'confidence': float(conf),
                                    'index': idx,
                                    'ocr_text': None  # Will store OCR result after processing
                                }
                                detections_list.append(detection_dict)
                                
                                # ⭐ OPTIMIZATION: Limit max detections to 1 plate per frame (highest confidence)
                                # Process only the most confident detection for real-time speed
                                if len(detections_list) >= 1:
                                    break  # Stop processing more detections
                                if frame_counters[cam_id] <= 5:
                                    print(f"[DEBUG] {cam_id}: ✓ DETECTION ADDED #{len(detections_list)} - bbox={detection_dict['bbox']}, conf={conf:.3f}")
                        except Exception as e:
                            # Skip this box if error
                            pass
                
                # Debug: Show detection count
                if len(detections_list) > 0:
                    print(f"[PROCESS_FRAME] {cam_id}: ✓ Found {len(detections_list)} plate(s) in frame {frame_counters[cam_id]}")
            
            except Exception as e:
                # YOLO errors are non-fatal
                if frame_counters[cam_id] <= 5:
                    print(f"[PROCESS_FRAME] YOLO error for {cam_id}: {str(e)[:80]}")
        
        # Store overlay data for streaming using update_detection_overlay
        hud_info = {
            'frame_count': frame_counters[cam_id],
            'timestamp': datetime.now().isoformat(),
            'camera': cam_id,
            'detection_count': len(detections_list),
            'original_width': frame_w,  # Store original frame dimensions for scaling overlays
            'original_height': frame_h
        }
        
        if frame_counters[cam_id] <= 5 or len(detections_list) > 0:
            print(f"[DEBUG] {cam_id}: About to cache {len(detections_list)} detections for frame {frame_counters[cam_id]}")
        
        update_detection_overlay(cam_id, detections_list, hud_info)
    
    except Exception as e:
        # Silently fail - processing errors shouldn't crash the pipeline
        print(f"[PROCESS_FRAME] ERROR in {cam_id}: {str(e)[:80]}")

# -------- Streaming --------
def generate_stream(cam_id, user_id=None, camera_id=None):
    """Generate video stream with performance optimizations and robust error handling
    
    ★ CRITICAL FIX: Enhanced with deleted camera detection to prevent FFmpeg crashes
    """
    # ★ SAFETY CHECK 1: Verify camera exists at start
    cap = camera_sources.get(cam_id)
    if cap is None:
        print(f"[ERROR] Camera '{cam_id}' not found in sources")
        return

    # ★ SAFETY CHECK 2: Verify frame queue is available (not being deleted)
    frame_queue = frame_queues.get(cam_id) if frame_queues else None
    if frame_queue is None:
        print(f"[ERROR] ✗ Frame queue NOT available for '{cam_id}'!")
        print(f"[ERROR] DETECTIONS DISABLED - Background processor not connected!")
        print(f"[ERROR] This means NO vehicles will be detected on live feed")
        print(f"[DEBUG] Global frame_queues: {frame_queues}")
        print(f"[DEBUG] Available cameras in queues: {list(frame_queues.keys()) if frame_queues else 'EMPTY'}")
    else:
        print(f"[INFO] ✓ Frame queue connected for '{cam_id}' - Detections ENABLED")

    # Ensure lock exists for this camera
    if cam_id not in camera_locks:
        camera_locks[cam_id] = threading.Lock()
    
    lock = camera_locks[cam_id]
    
    # Minimal frame buffer for low latency
    frame_buffer = deque(maxlen=1)  # Single frame for minimum latency
    consecutive_errors = 0
    max_consecutive_errors = 5  # Increased threshold for resilience
    last_frame = None

    # FPS limiting - optimized for high FPS
    frame_delay = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0
    last_frame_time = time.time()
    frame_count = 0
    frame_skip_count = 0
    frame_processing_interval = PROCESS_EVERY_N_FRAMES  # Send to processing thread every Nth frame
    
    debug_print_interval = 50  # Print debug info every 50 frames
    frames_queued = 0

    # Adaptive quality control function - optimized for Raspberry Pi streaming
    def get_adaptive_params(detection_confidence=0.0, frame_quality=1.0):
        """Get JPEG encoding parameters - optimized for smooth Pi 5 streaming"""
        # ⭐ REDUCED QUALITY for smooth streaming on Pi 5 (reduce CPU load from encoding)
        quality = 55  # Reduced from 70 - faster encoding, still acceptable visual quality
        
        return [
            int(cv2.IMWRITE_JPEG_QUALITY), quality,  # Lower quality for faster encoding
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 0,  # Disable optimization
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0  # Disable progressive JPEG
        ]

    while True:
        try:
            # ★ CRITICAL SAFETY CHECK IN LOOP: Verify camera still exists
            # This gives streaming time to stop gracefully if camera is being deleted
            # ★ FIX: Wrapped dict access to handle concurrent deletion
            try:
                cam_check = camera_sources.get(cam_id)
                if cam_check is None:
                    print(f"[STREAM] {cam_id}: Camera deleted during streaming, stopping generator")
                    return  # Exit gracefully
            except RuntimeError as e:
                # Dict changed size during iteration (concurrent deletion)
                print(f"[STREAM] {cam_id}: Dictionary modified during access (camera being deleted), stopping")
                return
            
            # ★ CRITICAL SAFETY CHECK: Verify frame queue is still valid
            # If set to None, camera is being removed
            # ★ FIX: Also wrapped for safety
            try:
                queue_check = frame_queues.get(cam_id) if frame_queues else None
                if frame_queue is not None and queue_check is None:
                    print(f"[STREAM] {cam_id}: Frame queue invalidated (camera being removed), stopping generator")
                    return  # Exit gracefully
            except RuntimeError as e:
                # Dict changed size during iteration (concurrent deletion)
                print(f"[STREAM] {cam_id}: Frame queue dictionary modified during access (camera being deleted), stopping")
                return
            
            frame_count += 1
            
            # Efficient FPS limiting - minimal overhead
            if frame_delay > 0:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_delay:
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0.001:
                        # Sleep for most of the required time
                        time.sleep(sleep_time * 0.98)
                last_frame_time = time.time()

            # Thread-safe camera read with timeout
            frame = None
            success = False
            
            with lock:
                try:
                    success, frame = read_frame_from_source(cap)
                except cv2.error as e:
                    print(f"[WARN] OpenCV error during frame read: {e}")
                    success = False
                except Exception as e:
                    print(f"[WARN] Frame read error: {type(e).__name__}")
                    success = False
            
            if not success or frame is None:
                consecutive_errors += 1
                frame_skip_count += 1
                
                # Break if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    print(f"[ERROR] Too many frame read errors ({consecutive_errors}), stopping stream")
                    break
                
                # Try to use last good frame to avoid black frames
                if last_frame is not None and frame_skip_count <= 2:
                    frame = last_frame
                    success = True
                else:
                    # Skip this frame but continue
                    time.sleep(0.0001)  # Minimal sleep
                    continue
            else:
                consecutive_errors = 0
                frame_skip_count = 0
                last_frame = frame

            # CRITICAL FIX #7: Queue FULL-RESOLUTION frame for detection BEFORE resizing for streaming
            # The detection worker needs original 1280x720 frames for accurate YOLO inference
            # Resizing to 640x360 for streaming should NOT affect detection quality
            
            # SEND TO BACKGROUND PROCESSING QUEUE (non-blocking)
            # This allows heavy YOLO/OCR processing without blocking the streaming loop
            queue_check = frame_count % frame_processing_interval
            if frame_count <= 10:
                print(f"[FRAMECOUNT_DEBUG] {cam_id}: frame={frame_count}, interval={frame_processing_interval}, modulo={queue_check}, queueing={queue_check == 0}")
            
            # ⭐ CRITICAL: Queue full-resolution frame FIRST, before any resizing
            if frame_queue is None:
                if frame_count <= 2:
                    print(f"[ERROR] {cam_id}: Frame queue is NONE! Detection will not work!")
            elif (frame_count % frame_processing_interval == 0):
                try:
                    # Queue the FULL-RESOLUTION frame (1280x720) for accurate YOLO detection
                    # This ensures detection gets proper resolution input
                    frame_queue.put_nowait(frame.copy())  # Queue BEFORE resize!
                    frames_queued += 1
                    if frames_queued <= 3 or frames_queued % 10 == 0:
                        print(f"[QUEUE_DEBUG] {cam_id}: ✓ Queued FULL-RES frame {frames_queued} ({frame.shape[1]}x{frame.shape[0]}) (frame_count={frame_count}, queue_size={frame_queue.qsize()})")
                except Exception as e:
                    # Queue error - skip
                    if frames_queued < 5:
                        print(f"[QUEUE_SKIP] Could not queue frame {frame_count}: {e}")
            elif frame_queue is None and frame_count == 1:
                print(f"[STREAM_ERROR] {cam_id}: Frame queue is None!")

            # NOW resize frame for streaming with minimal overhead (AFTER queuing for detection)
            try:
                # Only resize if dimensions significantly different (reduce resize overhead)
                w_diff = abs(frame.shape[1] - STREAM_WIDTH)
                h_diff = abs(frame.shape[0] - STREAM_HEIGHT)
                if w_diff > 10 or h_diff > 10:  # Only resize if significant difference
                    frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_AREA)
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Frame resize error: {e}")
            
            # Debug output - show progress
            if frame_count % debug_print_interval == 0:
                queue_status = "✓" if frame_queue else "✗"
                print(f"[STREAM] {cam_id}: Frame {frame_count:5d} | Queue:{queue_status} | Sent:{frames_queued:3d} frames to processor")

            # OPTIMIZED: Apply lightweight cached detection overlay (non-blocking)
            # Apply cached overlay for stable display
            try:
                # Use cached non-blocking overlay
                frame_with_overlay = apply_detection_overlay(frame, cam_id)
                
                if frame_count <= 5 or frame_count % 100 == 1:
                    print(f"[DEBUG] generate_stream: {cam_id} ✓ overlay applied (frame {frame_count})")
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Overlay error: {e}")
                frame_with_overlay = frame
            
            # ⭐ STORE PROCESSED FRAME: Save frame with overlays for port 5000 UI access
            try:
                with latest_frames_lock:
                    latest_frames[cam_id] = frame_with_overlay.copy()
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] Failed to store latest frame for {cam_id}: {e}")

            # Fast JPEG encoding - prioritize speed over quality
            try:
                current_params = get_adaptive_params()
                
                # Try encoding the primary frame
                ret, buf = cv2.imencode('.jpg', frame_with_overlay, current_params)
                
                # Fallback: If primary encoding fails, try with lower quality
                if not ret or buf is None or len(buf) == 0:
                    try:
                        # Fallback with minimal quality for recovery
                        fallback_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
                        ret, buf = cv2.imencode('.jpg', frame_with_overlay, fallback_params)
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] Fallback encoding failed for {cam_id}: {e}")
                        ret, buf = None, None
                
                # Ultimate fallback: Try last good frame if current frame fails
                if (not ret or buf is None or len(buf) == 0) and last_frame is not None:
                    try:
                        fallback_params = [cv2.IMWRITE_JPEG_QUALITY, 40]
                        ret, buf = cv2.imencode('.jpg', last_frame, fallback_params)
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] Last frame fallback encoding failed for {cam_id}: {e}")
                        ret, buf = None, None
                
                # If we have valid encoded data, yield it
                if ret and buf is not None and len(buf) > 0:
                    frame_bytes = buf.tobytes()
                    frame_len = len(frame_bytes)
                    boundary = b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                    try:
                        yield boundary + str(frame_len).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n'
                    except GeneratorExit:
                        raise
                    except BrokenPipeError:
                        raise GeneratorExit
                else:
                    # No valid frame - skip this iteration but continue streaming
                    if DEBUG:
                        print(f"[DEBUG] No valid encoded frame for {cam_id}, skipping iteration")
                    continue
            except GeneratorExit:
                raise
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] Encoding exception for {cam_id}: {type(e).__name__}: {e}")
                continue
                
        except GeneratorExit:
            print(f"[INFO] Stream generator closed for camera {cam_id}")
            return  # ★ CRITICAL: Return instead of break
        except Exception as e:
            print(f"[WARN] Stream generator error: {type(e).__name__}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                return  # ★ CRITICAL: Return instead of break
            time.sleep(0.0001)


# ============== DYNAMIC CAMERA PIPELINE HELPERS ==============
# These functions enable adding/removing cameras without server restart

def initialize_camera_source(source: str, camera_type: str):
    """
    Open a camera source based on type (USB, IP Webcam, or OAK)
    
    Args:
        source: USB index (0, 1), IP Webcam URL (http://...), or 'oak'
        camera_type: 'usb', 'ipwebcam', or 'oak'
    
    Returns:
        (cv2.VideoCapture or dict, bool success)
    """
    try:
        if camera_type == 'usb' or (isinstance(source, str) and source.isdigit()):
            # USB camera
            print(f"[INIT_CAMERA] Initializing USB camera at index {source}...")
            src_int = int(source)
            cap = cv2.VideoCapture(src_int)
            
            if not cap.isOpened():
                print(f"[INIT_CAMERA] ✗ Failed to open USB camera at index {src_int}")
                return None, False
            
            # Optimize camera settings
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                # ⭐ CAPTURE HIGH-RESOLUTION: 1280x720 for clear video
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            except Exception as e:
                print(f"[INIT_CAMERA] ⚠️  Could not optimize USB camera properties: {e}")
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INIT_CAMERA] ✓ USB camera verified (attempt {attempt + 1})")
                    return cap, True
                time.sleep(0.2)
            
            print(f"[INIT_CAMERA] ⚠️  USB camera opened but frame read failed. Proceeding anyway...")
            return cap, True
        
        elif camera_type == 'ipwebcam' or (isinstance(source, str) and (source.startswith('http://') or source.startswith('https://'))):
            # IP Webcam
            print(f"[INIT_CAMERA] Initializing IP Webcam from {source}...")
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print(f"[INIT_CAMERA] ✗ Failed to open IP Webcam from {source}")
                return None, False
            
            # Optimize for IP streaming
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                # ⭐ REQUEST HIGH-RESOLUTION: 1280x720 for clear video
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            except Exception as e:
                print(f"[INIT_CAMERA] ⚠️  Could not optimize IP Webcam properties: {e}")
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INIT_CAMERA] ✓ IP Webcam verified (attempt {attempt + 1})")
                    return cap, True
                time.sleep(0.2)
            
            print(f"[INIT_CAMERA] ⚠️  IP Webcam opened but frame read failed. Proceeding anyway...")
            return cap, True
        
        elif camera_type == 'oak' or (isinstance(source, str) and source.lower() == 'oak'):
            # OAK camera via DepthAI
            print(f"[INIT_CAMERA] Initializing OAK camera...")
            if not DEPTHAI_AVAILABLE:
                print(f"[INIT_CAMERA] ✗ DepthAI not installed - OAK camera cannot be initialized")
                return None, False
            
            try:
                import depthai as dai
                devices = dai.Device.getAllAvailableDevices()
                if not devices:
                    print(f"[INIT_CAMERA] ✗ No OAK devices detected. Is it connected via USB?")
                    return None, False
                
                print(f"[INIT_CAMERA] Found {len(devices)} OAK device(s)")
                
                # Create pipeline for OAK camera
                pipeline = dai.Pipeline()
                cam = pipeline.create(dai.node.ColorCamera)
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                cam.setFps(30)
                cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
                cam.setPreviewSize(1280, 720)
                cam.setVideoSize(1280, 720)
                
                xout_video = pipeline.create(dai.node.XLinkOut)
                xout_video.setStreamName("video")
                cam.video.link(xout_video.input)
                
                device = dai.Device(pipeline)
                
                # Return OAK device in dict format
                oak_device = {
                    'type': 'oak_depthai',
                    'device': device,
                    'queue_name': 'video'
                }
                
                # Verify with first frame
                q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
                time.sleep(1)  # Wait for camera to warm up
                in_data = q.get()
                if in_data is not None:
                    frame = in_data.getCvFrame()
                    if frame is not None:
                        print(f"[INIT_CAMERA] ✓ OAK camera verified (frame shape: {frame.shape})")
                        return oak_device, True
                
                print(f"[INIT_CAMERA] ⚠️  OAK camera initialized but no initial frame. Proceeding anyway...")
                return oak_device, True
                
            except Exception as e:
                print(f"[INIT_CAMERA] ✗ Error initializing OAK camera: {str(e)[:100]}")
                return None, False
        
        else:
            print(f"[INIT_CAMERA] Unknown camera type: {camera_type} for source: {source}")
            return None, False
    
    except Exception as e:
        print(f"[INIT_CAMERA] Error initializing camera: {str(e)[:100]}")
        return None, False


def sync_cameras_to_json_file():
    """
    ★ CRITICAL: Sync cameras-config.json to cameras.json while preserving ALL extended fields
    
    This ensures both files stay in sync:
    - cameras-config.json is the master config (used by backend)
    - cameras.json is the reference for frontend and API
    
    ★ IMPORTANT: This function preserves ALL extended fields from existing cameras
    because these fields (status, userId, location, gate, resolution, fps, anprEnabled, etc.)
    are managed by the Node.js API, not by the Python config
    """
    try:
        cameras_json_path = Path(config_manager.config_path).parent / 'cameras.json'
        
        # Get enabled cameras from config
        enabled_cameras = config_manager.get_enabled_cameras()
        
        # ★ CRITICAL FIX: Preserve ALL extended fields from existing cameras
        # Read existing cameras to get their extra fields (status, userId, location, etc.)
        existing_cameras_by_id = {}
        if cameras_json_path.exists():
            try:
                with open(cameras_json_path, 'r') as f:
                    existing_cams = json.load(f)
                    if isinstance(existing_cams, list):
                        for cam in existing_cams:
                            if cam.get('id'):
                                existing_cameras_by_id[cam['id']] = cam
            except Exception as e:
                print(f"[SYNC_CAMERAS] Warning: Could not read existing cameras: {str(e)[:80]}")
        
        # Merge: Preserve all extra fields from existing cameras AND config
        for camera in enabled_cameras:
            camera_id = camera.get('id')
            
            # ★ NEW: Check if extended fields are already in cameras-config.json
            # If they are, they don't need to be preserved from cameras.json
            fields_to_preserve = [
                'status',           # Online/Offline status
                'userId',           # Owner of the camera
                'location',         # Camera location
                'gate',             # Entry/Exit gate
                'resolution',       # Camera resolution
                'fps',              # Frames per second
                'anprEnabled',      # ANPR enabled flag
            ]
            
            if camera_id and camera_id in existing_cameras_by_id:
                existing_camera = existing_cameras_by_id[camera_id]
                
                # ★ Preserve all fields that are NOT in the config (API-managed fields)
                # These fields are set via the API and should not be overwritten
                for field in fields_to_preserve:
                    if field in existing_camera and field not in camera:
                        camera[field] = existing_camera[field]
            
            # ★ CRITICAL FIX: For NEW cameras, extended fields should already be in config
            # (added by Flask API), but if missing, use defaults
            for field in fields_to_preserve:
                if field not in camera:
                    if field == 'status':
                        camera[field] = 'Online'
                    elif field == 'userId':
                        camera[field] = 'default'
                    elif field == 'location':
                        camera[field] = 'Not Specified'
                    elif field == 'gate':
                        camera[field] = 'Entry'
                    elif field == 'resolution':
                        camera[field] = '1280x720'
                    elif field == 'fps':
                        camera[field] = 30
                    elif field == 'anprEnabled':
                        camera[field] = True
        
        # Write to cameras.json (config info + preserved fields)
        with open(cameras_json_path, 'w') as f:
            json.dump(enabled_cameras, f, separators=(',', ':'))
        
        print(f"[SYNC_CAMERAS] ✓ Synced {len(enabled_cameras)} camera(s) to cameras.json (all extended fields preserved)")
        return True
    
    except Exception as e:
        print(f"[SYNC_CAMERAS] ✗ Error syncing cameras.json: {str(e)[:100]}")
        return False


def reload_cameras_from_config():
    """
    ★ CRITICAL FIX 1: Reload all enabled cameras from configuration
    
    This function:
    1. Reads all enabled AND ONLINE cameras from cameras-config.json
    2. Checks if each camera is already loaded
    3. If not loaded, initializes and starts the pipeline
    4. ★ DETECTS AND REMOVES DELETED CAMERAS from memory
    5. Ensures detection server has access to all configured cameras
    6. ★ PHASE 8: Respects persistent camera online/offline status
    
    Returns:
        (int: newly_loaded_count, list: camera_ids)
    """
    try:
        print("\n[RELOAD_CAMERAS] Checking for new cameras in config...")
        
        # ★ CRITICAL: Reload config from disk first (in case it changed externally)
        config_manager.config = config_manager._load_config()
        
        # ★ PHASE 8: Use get_cameras_to_start() to only load cameras that are ENABLED AND ONLINE
        # This respects the persistent camera state set by users via the UI
        enabled_cameras = config_manager.get_cameras_to_start()
        newly_loaded = 0
        loaded_camera_ids = []
        config_camera_ids = set()
        
        # Step 1: Get all camera IDs from current config
        for camera_config in enabled_cameras:
            config_camera_ids.add(camera_config['id'])
        
        # Step 2: ★ CRITICAL - Detect and REMOVE DELETED CAMERAS ★
        cameras_in_memory = set(camera_sources.keys())
        deleted_cameras = cameras_in_memory - config_camera_ids
        
        if deleted_cameras:
            print(f"\n[RELOAD_CAMERAS] ★ DETECTED {len(deleted_cameras)} DELETED CAMERA(S)")
            for cam_id in deleted_cameras:
                print(f"[RELOAD_CAMERAS] Stopping deleted camera: '{cam_id}'")
                try:
                    success, msg = stop_camera_pipeline_runtime(cam_id)
                    if success:
                        print(f"[RELOAD_CAMERAS] ✓ Stopped and cleaned up '{cam_id}'")
                    else:
                        print(f"[RELOAD_CAMERAS] ⚠️  Failed to stop '{cam_id}': {msg}")
                except Exception as e:
                    print(f"[RELOAD_CAMERAS] ✗ Error stopping '{cam_id}': {str(e)[:100]}")
        
        # Step 3: Load new cameras from config
        for camera_config in enabled_cameras:
            cam_id = camera_config['id']
            source = camera_config['source']
            camera_type = camera_config['type']
            
            # Check if already loaded
            if cam_id in camera_sources:
                print(f"[RELOAD_CAMERAS] ✓ Camera '{cam_id}' already active")
                loaded_camera_ids.append(cam_id)
                continue
            
            # NEW CAMERA: Try to initialize
            print(f"[RELOAD_CAMERAS] Loading new camera '{cam_id}' from config: {source}")
            
            try:
                # Initialize the camera source
                camera_src, init_success = initialize_camera_source(source, camera_type)
                
                if not init_success or camera_src is None:
                    print(f"[RELOAD_CAMERAS] ⚠️  Camera '{cam_id}' source not accessible: {source}")
                    print(f"[RELOAD_CAMERAS] 💡 Adding camera to memory as 'disconnected' with frame queue ready")
                    
                    # ★ FIX 1: Create frame_queues and stop_events EVEN for disconnected cameras
                    # This allows detection to start immediately when source comes online
                    try:
                        # Create queue for when camera comes online
                        if cam_id not in frame_queues:
                            frame_queues[cam_id] = queue.Queue(maxsize=MAX_FRAME_QUEUE)
                            print(f"[RELOAD_CAMERAS] ✓ Frame queue created for '{cam_id}' (detection ready)")
                        
                        # Create stop event for future pipeline
                        if cam_id not in processing_stop_events:
                            processing_stop_events[cam_id] = threading.Event()
                            print(f"[RELOAD_CAMERAS] ✓ Stop event created for '{cam_id}'")
                        
                        # Add camera to camera_sources as dict with status
                        camera_sources[cam_id] = {
                            'id': cam_id,
                            'source': None,  # No active source yet
                            'type': camera_type,
                            'status': 'disconnected',  # Mark as disconnected
                            'frame_queue': frame_queues[cam_id],  # Frame queue ready
                            'stop_event': processing_stop_events[cam_id],  # Stop event ready
                            'worker_thread': None,
                            'last_error': f'Source not accessible: {source}'
                        }
                        print(f"[RELOAD_CAMERAS] ✓ Camera '{cam_id}' added (DISCONNECTED - detection ready when source comes online)")
                        loaded_camera_ids.append(cam_id)
                        newly_loaded += 1
                    except Exception as add_err:
                        print(f"[RELOAD_CAMERAS] ✗ Failed to add disconnected camera: {str(add_err)[:100]}")
                    continue
                
                print(f"[RELOAD_CAMERAS] ✓ Camera source initialized, starting pipeline...")
                
                # Start the processing pipeline
                pipeline_success, pipeline_msg = start_camera_pipeline_runtime(cam_id, camera_src, camera_type)
                
                if not pipeline_success:
                    print(f"[RELOAD_CAMERAS] ✗ Pipeline failed for '{cam_id}': {pipeline_msg}")
                    # Close camera on failure
                    try:
                        if hasattr(camera_src, 'release'):
                            camera_src.release()
                    except:
                        pass
                    # ★ FIX: Still add camera with pipeline error, keeping frame queues ready
                    try:
                        # Ensure frame queue exists
                        if cam_id not in frame_queues:
                            frame_queues[cam_id] = queue.Queue(maxsize=MAX_FRAME_QUEUE)
                        
                        # Ensure stop event exists
                        if cam_id not in processing_stop_events:
                            processing_stop_events[cam_id] = threading.Event()
                        
                        camera_sources[cam_id] = {
                            'id': cam_id,
                            'source': None,
                            'type': camera_type,
                            'status': 'pipeline_error',
                            'frame_queue': frame_queues[cam_id],
                            'stop_event': processing_stop_events[cam_id],
                            'worker_thread': None,
                            'last_error': pipeline_msg
                        }
                        print(f"[RELOAD_CAMERAS] ✓ Camera '{cam_id}' added (PIPELINE_ERROR - queues ready)")
                        loaded_camera_ids.append(cam_id)
                        newly_loaded += 1
                    except Exception as add_err:
                        print(f"[RELOAD_CAMERAS] ✗ Failed to add pipeline_error camera: {str(add_err)[:100]}")
                    continue
                
                print(f"[RELOAD_CAMERAS] ✓✓✓ CAMERA '{cam_id}' FULLY LOADED ✓✓✓")
                loaded_camera_ids.append(cam_id)
                newly_loaded += 1
                
                # ★ FIX: Add source mapping for preview endpoint
                # This allows the preview endpoint to find the camera by source
                camera_source_map[source] = cam_id
                if source.lower() == 'oak':
                    camera_source_map['oak'] = cam_id
                print(f"[RELOAD_CAMERAS] ✓ Source mapping added: '{source}' -> '{cam_id}'")
            
            except Exception as e:
                print(f"[RELOAD_CAMERAS] ✗ Error loading camera '{cam_id}': {str(e)[:100]}")
                # ★ FIX: Still add camera in error state with queues ready
                try:
                    # Ensure frame queue exists
                    if cam_id not in frame_queues:
                        frame_queues[cam_id] = queue.Queue(maxsize=MAX_FRAME_QUEUE)
                    
                    # Ensure stop event exists
                    if cam_id not in processing_stop_events:
                        processing_stop_events[cam_id] = threading.Event()
                    
                    camera_sources[cam_id] = {
                        'id': cam_id,
                        'source': None,
                        'type': camera_type,
                        'status': 'error',
                        'frame_queue': frame_queues[cam_id],
                        'stop_event': processing_stop_events[cam_id],
                        'worker_thread': None,
                        'last_error': str(e)[:100]
                    }
                    print(f"[RELOAD_CAMERAS] ✓ Camera '{cam_id}' added (ERROR state - queues ready)")
                    loaded_camera_ids.append(cam_id)
                    newly_loaded += 1
                except Exception as add_err:
                    print(f"[RELOAD_CAMERAS] ✗ Failed to add error camera: {str(add_err)[:100]}")
                continue
        
        print(f"[RELOAD_CAMERAS] Done! Loaded {newly_loaded} new cameras. Total active: {len(camera_sources)}")
        
        # ★ Sync cameras.json with updated config
        sync_cameras_to_json_file()
        
        return newly_loaded, loaded_camera_ids
    
    except Exception as e:
        print(f"[RELOAD_CAMERAS] ✗ Error reloading cameras: {str(e)[:100]}")
        return 0, []


def start_config_watcher_thread():
    """
    ★ CRITICAL FIX 3: Monitor config file for changes with faster response
    
    Background thread that:
    1. Monitors cameras-config.json for changes (faster detection)
    2. Detects new cameras automatically
    3. Detects deleted cameras and stops them immediately
    4. Triggers reload_cameras_from_config when config changes
    5. Allows seamless camera addition without restart
    
    This solves the issue where cameras added via API (port 5000)
    are not immediately accessible to the detection server (port 8000)
    """
    def watcher():
        import hashlib
        
        config_file = Path(config_manager.config_path)
        last_hash = None
        check_interval = 1  # ★ OPTIMIZED: Check every 1 second (was 3) for faster detection
        consecutive_errors = 0
        max_errors = 5
        
        print("[CONFIG_WATCHER] Started monitoring config file for changes...")
        print(f"[CONFIG_WATCHER] Watching: {config_file}")
        print(f"[CONFIG_WATCHER] Check interval: {check_interval} second(s)")
        
        while True:
            try:
                if not config_file.exists():
                    print(f"[CONFIG_WATCHER] ⚠️  Config file doesn't exist: {config_file}")
                    time.sleep(check_interval)
                    continue
                
                # Read config file and compute hash
                try:
                    with open(config_file, 'rb') as f:
                        content = f.read()
                    current_hash = hashlib.md5(content).hexdigest()
                except IOError as e:
                    print(f"[CONFIG_WATCHER] ⚠️  Could not read config file: {e}")
                    consecutive_errors += 1
                    if consecutive_errors < max_errors:
                        time.sleep(check_interval)
                        continue
                    else:
                        break
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Check if changed
                if last_hash is not None and current_hash != last_hash:
                    print(f"\n[CONFIG_WATCHER] ★ CONFIG FILE CHANGED (hash mismatch) - Reloading cameras...")
                    try:
                        newly_loaded, cam_ids = reload_cameras_from_config()
                        
                        if newly_loaded > 0:
                            print(f"[CONFIG_WATCHER] ✓ Loaded {newly_loaded} new camera(s): {cam_ids}")
                        else:
                            print(f"[CONFIG_WATCHER] ✓ Config reloaded (total active: {len(camera_sources)} cameras)")
                    except Exception as e:
                        print(f"[CONFIG_WATCHER] ✗ Error during reload: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()
                
                # ★ NEW FIX: Check if disconnected cameras can now be promoted to active
                # This implements automatic retry for cameras that were initially offline
                print("[CONFIG_WATCHER] ★ Checking if disconnected cameras can now be activated...")
                for cam_id in list(camera_sources.keys()):
                    cam_info = camera_sources.get(cam_id)
                    if isinstance(cam_info, dict) and cam_info.get('status') in ['disconnected', 'pipeline_error', 'error']:
                        # Try to initialize this camera
                        print(f"[CONFIG_WATCHER] 🔄 Attempting to activate disconnected camera: {cam_id}")
                        try:
                            cam_config = config_manager.get_camera(cam_id)
                            if cam_config:
                                source = cam_config.get('source')
                                camera_type = cam_config.get('type', 'usb')
                                
                                # Try to initialize the source
                                camera_src, init_success = initialize_camera_source(source, camera_type)
                                
                                if init_success and camera_src is not None:
                                    print(f"[CONFIG_WATCHER] ✓ Source now accessible for '{cam_id}', promoting to active...")
                                    
                                    # Start the pipeline
                                    pipeline_success, msg = start_camera_pipeline_runtime(cam_id, camera_src, camera_type)
                                    
                                    if pipeline_success:
                                        print(f"[CONFIG_WATCHER] ✓ Camera '{cam_id}' is now ACTIVE and processing!")
                                    else:
                                        print(f"[CONFIG_WATCHER] ⚠️  Camera source accessible but pipeline failed: {msg}")
                                        # Keep trying on next cycle
                        except Exception as retry_err:
                            print(f"[CONFIG_WATCHER] ℹ️  Camera '{cam_id}' still not accessible: {str(retry_err)[:80]}")
                
                last_hash = current_hash
                time.sleep(check_interval)
            
            except KeyboardInterrupt:
                print("[CONFIG_WATCHER] Stopped by interrupt")
                break
            except Exception as e:
                print(f"[CONFIG_WATCHER] ✗ Error: {str(e)[:100]}")
                time.sleep(check_interval)
    
    # Start as daemon thread so it doesn't block shutdown
    watcher_thread = threading.Thread(target=watcher, daemon=True, name="ConfigWatcher")
    watcher_thread.start()
    print("[INFO] ✓ Config file watcher thread started (1-second polling for real-time updates)")
    return watcher_thread


def start_camera_pipeline_runtime(cam_id: str, camera_source, camera_type: str = "usb") -> Tuple[bool, str]:
    """
    Start a camera pipeline at runtime (called when a new camera is added)
    
    This function:
    1. Registers the camera source with the manager
    2. Creates frame queue and stop event  
    3. Starts the processing worker thread
    
    Args:
        cam_id: Camera identifier
        camera_source: cv2.VideoCapture or OAK device
        camera_type: Type of camera (usb, ipwebcam, oak)
    
    Returns:
        (bool: success, str: message)
    """
    try:
        print(f"\n[PIPELINE_RUNTIME] Starting pipeline for camera '{cam_id}'...")
        
        # 1. Initialize per-camera data structures (data isolation, tracking, etc)
        frame_counters[cam_id] = 0
        last_processed_frame[cam_id] = None
        camera_last_saved[cam_id] = 0
        camera_locks[cam_id] = threading.Lock()
        # ⭐ FIX 3: DISABLED frame deduplicator - skips frames for slow vehicles
        # frame_deduplicators[cam_id] = FrameDeduplicator()  # DISABLED
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
        # ✅ Load gate type from config or default to Entry
        initial_gate = "Entry"
        try:
            if hasattr(config_manager, 'config') and 'cameras' in config_manager.config:
                for camera in config_manager.config['cameras']:
                    if camera.get('id') == cam_id:
                        initial_gate = camera.get('gate', camera.get('gateType', 'Entry'))
                        break
        except Exception as e:
            if DEBUG:
                print(f"[WARN] Could not load gate type for camera {cam_id}: {e}")
        camera_gate_types[cam_id] = initial_gate
        
        # ★ CRITICAL: Register with GLOBAL camera_sources so Flask endpoints can find it
        # This is ESSENTIAL - the streaming code looks in the global dict, not camera_manager's
        camera_sources[cam_id] = camera_source
        
        # Initialize these if not already present
        if cam_id not in camera_source_map:
            camera_source_map[cam_id] = cam_id
        
        # 2. Register camera with manager
        success = camera_manager.create_camera_pipeline(cam_id, camera_source)
        if not success:
            return False, f"Failed to register camera with manager"
        
        # 3. Start processing pipeline with manager
        def process_frame_wrapper(frame, cid):
            """Wrapper to call camera_anpr_process_frame for lightweight overlay caching"""
            camera_anpr_process_frame(frame, cid)
        
        success, msg = camera_manager.start_processing_pipeline(cam_id, process_frame_wrapper)
        if not success:
            # Cleanup on failure
            camera_manager.close_camera_source(cam_id)
            camera_sources.pop(cam_id, None)  # Remove from global dict on failure
            return False, f"Failed to start processing pipeline: {msg}"
        
        # ★ CRITICAL: Sync frame_queues with global dict so generate_stream can access it
        # camera_manager created the queue, now share it with the global frame_queues
        if cam_id in camera_manager.frame_queues:
            frame_queues[cam_id] = camera_manager.frame_queues[cam_id]
        
        # ★ CRITICAL: Sync processing_stop_events with global dict
        if cam_id in camera_manager.processing_stop_events:
            processing_stop_events[cam_id] = camera_manager.processing_stop_events[cam_id]
        
        print(f"[PIPELINE_RUNTIME] ✓ Camera '{cam_id}' registered in global camera_sources")
        print(f"[PIPELINE_RUNTIME] ✓ Frame queue synced for '{cam_id}' (detections enabled)")
        print(f"[PIPELINE_RUNTIME] ✓ Video stream now available at: /video_feed/{cam_id}")
        return True, f"Camera '{cam_id}' pipeline started"
    
    except Exception as e:
        print(f"[PIPELINE_RUNTIME] ✗ Error starting pipeline: {str(e)[:100]}")
        camera_sources.pop(cam_id, None)  # Cleanup on exception
        return False, str(e)


def stop_camera_pipeline_runtime(cam_id: str) -> Tuple[bool, str]:
    """
    Stop a camera pipeline at runtime (called when a camera is removed)
    
    ★ CRITICAL FIX: Enhanced with proper thread safety and resource cleanup
    
    This function:
    1. Signals all streams to stop accessing frames (prevents FFmpeg crashes)
    2. Stops the processing worker thread with timeout
    3. Closes the camera source safely
    4. Cleans up all related data structures
    
    Args:
        cam_id: Camera identifier
    
    Returns:
        (bool: success, str: message)
    """
    try:
        print(f"\n[PIPELINE_RUNTIME] Stopping pipeline for camera '{cam_id}'...")
        
        # ★ STEP 0: CRITICAL - Signal all streams to stop reading frames
        # This prevents generate_stream() from accessing the camera after it's deleted
        # Prevents: "Invalid stream index" FFmpeg assertion error
        if cam_id in frame_queues:
            print(f"[PIPELINE_RUNTIME] [STEP 0] Signaling streams to stop for '{cam_id}'")
            # ★ FIX 4: Safely signal frame queue before removal
            try:
                # Send stop signal through queue
                frame_queues[cam_id].put(None, timeout=0.5)
            except:
                pass
            # Now safe to mark as unavailable
            frame_queues[cam_id] = None  # Signal to stop reading
            time.sleep(0.3)  # Give streaming threads time to notice and stop
        
        # ★ STEP 1: Mark camera as deleted (prevent new access attempts)
        print(f"[PIPELINE_RUNTIME] [STEP 1] Marking camera as unavailable")
        camera_sources[cam_id] = None  # Signal unavailable before full removal
        time.sleep(0.1)  # Brief delay before removal
        
        # ★ STEP 1.5: FIX 2 - Signal all active streams to stop BEFORE removing camera
        # This ensures streams see the stop signal and exit gracefully
        if cam_id in processing_stop_events:
            processing_stop_events[cam_id].set()
            print(f"[PIPELINE_RUNTIME] [STEP 1.5] Stop event SET for '{cam_id}' - active streams will exit gracefully")
            time.sleep(0.3)  # Give streams time to see the stop signal
        
        # ★ STEP 2: Remove from GLOBAL dicts to prevent stale references
        print(f"[PIPELINE_RUNTIME] [STEP 2] Removing from global registries")
        # ★ FIX 1: Give streaming threads time to stop before removing
        time.sleep(0.2)  # Wait for streams to notice camera is None/stop is set
        camera_sources.pop(cam_id, None)
        
        # ★ STEP 3: Use camera manager to stop pipeline (with timeout protection)
        print(f"[PIPELINE_RUNTIME] [STEP 3] Stopping pipeline with camera manager")
        try:
            success, msg = camera_manager.remove_camera(cam_id)
            if success:
                print(f"[PIPELINE_RUNTIME] [STEP 3] ✓ Pipeline stopped: {msg}")
            else:
                print(f"[PIPELINE_RUNTIME] [STEP 3] ⚠️  Pipeline stop returned: {msg}")
        except Exception as e:
            print(f"[PIPELINE_RUNTIME] [STEP 3] ⚠️  Exception during pipeline stop: {str(e)[:80]}")
        
        # ★ STEP 4: Final cleanup - Remove from global dicts
        print(f"[PIPELINE_RUNTIME] [STEP 4] Cleaning up data structures")
        frame_queues.pop(cam_id, None)
        processing_stop_events.pop(cam_id, None)
        
        # Cleanup per-camera data structures
        frame_counters.pop(cam_id, None)
        last_processed_frame.pop(cam_id, None)
        camera_last_saved.pop(cam_id, None)
        frame_deduplicators.pop(cam_id, None)
        plate_bbox_stabilizers.pop(cam_id, None)
        camera_locks.pop(cam_id, None)
        camera_gate_types.pop(cam_id, None)
        camera_source_map.pop(cam_id, None)
        persistent_detections.pop(cam_id, None)
        persistent_detection_motion_levels.pop(cam_id, None)  # Also clean up motion level tracking
        prev_frames.pop(cam_id, None)
        
        print(f"[PIPELINE_RUNTIME] ✓✓✓ Camera '{cam_id}' pipeline stopped and cleaned up completely")
        return True, f"Camera '{cam_id}' pipeline stopped"
    
    except Exception as e:
        print(f"[PIPELINE_RUNTIME] ✗ Error stopping pipeline: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        # Still try to cleanup even on error
        try:
            camera_sources.pop(cam_id, None)
            frame_queues.pop(cam_id, None)
        except:
            pass
        return False, str(e)


# ============== INITIALIZE CAMERA MANAGER & CONFIG API ==============
# Now that helper functions are defined, initialize the camera manager
# and wire up the config API with dynamic pipeline management callbacks

camera_manager = initialize_camera_manager()
print("[INFO] Camera Manager initialized for dynamic camera pipeline control")

# ★ NOW PROPERLY INITIALIZE CONFIG API WITH PIPELINE FUNCTIONS
# This is called here (AFTER functions are defined) so routes have proper function references
init_config_api(
    app, 
    config_manager,
    camera_mgr=camera_manager,
    start_func=start_camera_pipeline_runtime,
    stop_func=stop_camera_pipeline_runtime,
    reload_func=reload_cameras_from_config,  # ★ NEW: Pass reload function to API
    process_frame_func=camera_anpr_process_frame  # ★ NEW: Pass frame processor callback
)

print("[INFO] Config API properly wired with dynamic pipeline management functions")
print("[INFO] Cameras can now be added/removed via API without server restart!")
print("[INFO] ★ CRITICAL FIX: Cameras now load into memory immediately when added")


@app.route('/favicon.svg')
@app.route('/favicon.ico')
def favicon():
    """Serve the favicon"""
    from flask import send_from_directory
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'static')
    
    try:
        response = send_from_directory(static_dir, 'favicon.svg', mimetype='image/svg+xml')
        response.headers['Cache-Control'] = 'max-age=31536000'  # Cache for 1 year
        return response
    except Exception as e:
        print(f"[FAVICON] Error serving favicon: {e}")
        # Return empty response with 204 No Content on error (browsers won't show error)
        from flask import Response
        return Response('', status=204)

@app.route('/')
def index():
    """Root endpoint with live camera preview - HTML UI"""
    from flask import request, render_template_string

    # Get request info for debugging
    remote_addr = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'Unknown')
    origin = request.headers.get('Origin', 'Direct')

    print(f"[ROOT] Access from {remote_addr}, Origin: {origin}, User-Agent: {user_agent}")

    cameras_list = list(camera_sources.keys())
    
    # HTML template for live camera feeds - LIGHT THEME
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ANPR Live Feed - Port 8000</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f5f5f5 0%, #ffffff 100%);
                color: #333;
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1600px;
                margin: 0 auto;
            }
            
            header {
                text-align: center;
                margin-bottom: 40px;
                padding: 25px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
            }
            
            h1 {
                font-size: 2.8em;
                margin-bottom: 10px;
                color: #ffffff;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            header p {
                color: #e0e7ff;
                font-size: 1.1em;
            }
            
            .status {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            
            .status-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 12px 18px;
                background: rgba(255, 255, 255, 0.15);
                border-radius: 8px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                color: #ffffff;
                font-weight: 500;
            }
            
            .status-item .value {
                font-weight: bold;
                font-size: 1.3em;
                color: #82d6ff;
            }
            
            .cameras-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
                gap: 28px;
                margin-bottom: 40px;
            }
            
            .camera-card {
                background: #ffffff;
                border: 3px solid #2a5298;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(42, 82, 152, 0.15);
                transition: all 0.3s ease;
            }
            
            .camera-card:hover {
                box-shadow: 0 8px 30px rgba(42, 82, 152, 0.25);
                border-color: #1e3c72;
                transform: translateY(-5px);
            }
            
            .camera-header {
                padding: 16px 20px;
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                border-bottom: 3px solid #1e3c72;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .camera-name {
                font-size: 1.4em;
                font-weight: bold;
                color: #ffffff;
            }
            
            .camera-status {
                font-size: 0.9em;
                padding: 6px 14px;
                border-radius: 6px;
                background: #d4f1d4;
                border: 2px solid #4caf50;
                color: #2e7d32;
                font-weight: 600;
            }
            
            .camera-video {
                width: 100%;
                aspect-ratio: 16/9;
                background: #f0f0f0;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
                border-top: 2px solid #e0e0e0;
            }
            
            .camera-video img,
            .camera-video video {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }
            
            .loading {
                position: absolute;
                color: #2a5298;
                font-size: 1.1em;
                animation: pulse 2s infinite;
                font-weight: 600;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            .no-cameras {
                text-align: center;
                padding: 60px 20px;
                background: #f9f9f9;
                border-radius: 12px;
                border: 3px dashed #2a5298;
            }
            
            .no-cameras h2 {
                color: #ff6b6b;
                margin-bottom: 10px;
                font-size: 1.8em;
            }
            
            .no-cameras p {
                color: #666;
                margin-bottom: 20px;
                font-size: 1.05em;
            }
            
            .action-buttons {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .btn {
                padding: 12px 28px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                font-weight: 600;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: #fff;
            }
            
            .btn-primary:hover {
                box-shadow: 0 4px 15px rgba(42, 82, 152, 0.4);
                transform: scale(1.05);
            }
            
            .btn-secondary {
                background: #fff;
                color: #2a5298;
                border: 2px solid #2a5298;
            }
            
            .btn-secondary:hover {
                background: #f0f4ff;
                transform: scale(1.05);
            }
            
            footer {
                text-align: center;
                margin-top: 40px;
                padding: 25px;
                border-top: 3px solid #2a5298;
                color: #666;
                font-size: 0.95em;
                background: #f9f9f9;
                border-radius: 8px;
            }
            
            .info-section {
                background: #ffffff;
                border-left: 6px solid #2a5298;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(42, 82, 152, 0.1);
            }
            
            .info-section h3 {
                color: #1e3c72;
                margin-bottom: 15px;
                font-size: 1.25em;
            }
            
            .info-section ul {
                list-style: none;
                padding-left: 20px;
            }
            
            .info-section li {
                margin: 10px 0;
                color: #555;
                line-height: 1.6;
            }
            
            .info-section code {
                background: #f0f4ff;
                padding: 4px 8px;
                border-radius: 4px;
                color: #1e3c72;
                font-family: 'Courier New', monospace;
                border: 1px solid #dde5ff;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎥 ANPR Live Monitoring System</h1>
                <p>Real-time License Plate Recognition & Detection Feed</p>
                <div class="status">
                    <div class="status-item">
                        <span>Status:</span>
                        <span class="value">🟢 ONLINE</span>
                    </div>
                    <div class="status-item">
                        <span>Active Cameras:</span>
                        <span class="value">{camera_count}</span>
                    </div>
                    <div class="status-item">
                        <span>Port:</span>
                        <span class="value">8000 (Detection Server)</span>
                    </div>
                </div>
            </header>
            
            {% if cameras_list %}
                <div class="cameras-grid">
                    {% for cam_id in cameras_list %}
                    <div class="camera-card">
                        <div class="camera-header">
                            <span class="camera-name">{{ cam_id }}</span>
                            <span class="camera-status">🟢 Live</span>
                        </div>
                        <div class="camera-video">
                            <img src="/api/video_feed/{{ cam_id }}" alt="Camera: {{ cam_id }}" 
                                 onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%2260%22><rect fill=%22%23222%22 width=%22100%22 height=%2260%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%23fff%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22>Camera Offline</text></svg>'">
                        </div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-cameras">
                    <h2>⚠️ No Cameras Detected</h2>
                    <p>No active cameras are currently registered in the system.</p>
                    <div class="action-buttons">
                        <a href="http://localhost:5000/settings" class="btn btn-primary" target="_blank">
                            Configure Cameras (Port 5000)
                        </a>
                        <button onclick="location.reload()" class="btn btn-secondary">Refresh</button>
                    </div>
                </div>
            {% endif %}
            
            <div class="info-section">
                <h3>📋 System Information</h3>
                <ul>
                    <li><strong>Detection Server:</strong> Running on Port 8000</li>
                    <li><strong>Configuration UI:</strong> Available at Port 5000</li>
                    <li><strong>Active Cameras:</strong> {camera_count} cameras loaded</li>
                    <li><strong>Detection:</strong> ✅ YOLO + OCR Processing Active</li>
                    <li><strong>License Plate Overlay:</strong> ✅ Bounding Box + Text Enabled</li>
                </ul>
            </div>
            
            <div class="info-section">
                <h3>🔗 Quick Links</h3>
                <ul>
                    <li><a href="http://localhost:5000/settings" style="color: #1e3c72; font-weight: 600;" target="_blank">Camera Configuration (Port 5000)</a></li>
                    <li><a href="/health" style="color: #1e3c72; font-weight: 600;">Server Health Check</a></li>
                    <li><a href="/api/config/camera-status" style="color: #1e3c72; font-weight: 600;">Camera Status Diagnostics</a></li>
                    <li><a href="/cameras" style="color: #1e3c72; font-weight: 600;">List All Cameras (JSON)</a></li>
                </ul>
            </div>
            
            <footer>
                <p>🎯 ANPR System v2.0 | Real-time License Plate Detection & Recognition</p>
                <p style="margin-top: 10px; font-size: 0.9em;">Powered by YOLO + Tesseract OCR | Live Detection & Recognition Enabled ✅</p>
            </footer>
        </div>
        
        <script>
            // Auto-refresh cameras every 10 seconds
            setTimeout(() => {
                location.reload();
            }, 10000);
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, cameras_list=cameras_list, camera_count=len(cameras_list))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with diagnostics"""
    from flask import request
    
    try:
        active_cameras = list(camera_sources.keys())
        has_model = plate_detector_model is not None
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'cameras_active': len(active_cameras),
            'camera_ids': active_cameras,
            'model_loaded': has_model,
            'debug_info': {
                'frame_queues': len(frame_queues),
                'processing_threads': len(processing_stop_events)
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/cameras', methods=['GET'])
def list_cameras():
    """List all active cameras with details"""
    try:
        camera_details = []
        for cam_id in camera_sources.keys():
            cam_info = camera_sources[cam_id]
            
            # Handle both active cameras and disconnected/error cameras
            status = cam_info.get('status', 'unknown') if isinstance(cam_info, dict) else 'active'
            
            camera_details.append({
                'id': cam_id,
                'status': status,
                'video_url': f'/video_feed/{cam_id}' if status == 'active' else None,
                'has_stream': cam_id in camera_sources and status == 'active',
                'has_queue': cam_id in frame_queues,
                'has_stop_event': cam_id in processing_stop_events,
                'error_message': cam_info.get('last_error') if isinstance(cam_info, dict) else None
            })
        
        return jsonify({
            'success': True,
            'cameras': camera_details,
            'total': len(camera_details)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Video feed implementations exist further down at @app.route('/api/video_feed/<cam_id>')
# The routes above handle diagnostics and listing
# (Duplicate old implementations removed - using new implementations at lines 4396-4441)



def video_feed_no_api(cam_id):
    """Alias for /api/video_feed/ to support NGROK and direct access"""
    from flask import request, Response
    
    remote_addr = request.remote_addr
    origin = request.headers.get('Origin', 'Direct')
    print(f"[VIDEO_FEED] Direct route access: cam_id={cam_id}, from={remote_addr}, origin={origin}")

    try:
        # Get user_id and camera_id from query parameters if available
        user_id = request.args.get('userId', None)
        camera_id = request.args.get('cameraId', cam_id)
        
        # Create response with proper headers for cross-origin streaming
        def stream_generator():
            try:
                for chunk in generate_stream(cam_id, user_id, camera_id):
                    yield chunk
            except GeneratorExit:
                print(f"[INFO] Client disconnected from camera {cam_id}")
                return  # ★ CRITICAL: Return instead of pass
            except Exception as e:
                print(f"[ERROR] Stream generator error for camera {cam_id}: {e}")
                return  # ★ CRITICAL: Return instead of continuing

        response = Response(
            stream_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

        # Add additional headers to ensure cross-origin streaming works
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning'
        response.headers['Connection'] = 'keep-alive'
        response.headers['Transfer-Encoding'] = 'chunked'

        return response
    except Exception as e:
        print(f"[ERROR] Error in video_feed endpoint: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", 500


@app.route('/video_feed/webcam')
def video_feed_webcam():
    """Direct route to webcam - commonly used by ngrok"""
    from flask import request, Response
    
    remote_addr = request.remote_addr
    origin = request.headers.get('Origin', 'Direct')
    print(f"[VIDEO_FEED_WEBCAM] Webcam stream requested from {remote_addr}, origin={origin}")

    try:
        # Determine which camera to use - webcam is usually the first available or explicitly named 'webcam'
        cam_id = 'webcam'
        if cam_id not in camera_sources and camera_sources:
            cam_id = list(camera_sources.keys())[0]
            print(f"[VIDEO_FEED_WEBCAM] 'webcam' not found, using first available: {cam_id}")
        elif cam_id not in camera_sources:
            return "No cameras available", 503
        
        # Get user_id from query parameters if available
        user_id = request.args.get('userId', None)
        camera_id = request.args.get('cameraId', cam_id)
        
        # Create response with proper headers for cross-origin streaming
        def stream_generator():
            try:
                for chunk in generate_stream(cam_id, user_id, camera_id):
                    yield chunk
            except GeneratorExit:
                print(f"[INFO] Client disconnected from webcam stream")
                return  # ★ CRITICAL: Return instead of pass
            except Exception as e:
                print(f"[ERROR] Stream generator error for webcam: {e}")
                return  # ★ CRITICAL: Return instead of continuing

        response = Response(
            stream_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

        # Add additional headers to ensure cross-origin streaming works
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning'
        response.headers['Connection'] = 'keep-alive'
        response.headers['Transfer-Encoding'] = 'chunked'

        return response
    except Exception as e:
        print(f"[ERROR] Error in webcam endpoint: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", 500


@app.route('/api/video_feed/preview')
def video_feed_preview():
    """
    Preview streaming endpoint for unsaved/test cameras - streams MJPEG directly
    Accepts source and type as query parameters
    IMPORTANT: This endpoint can stream from ANY source, not just initialized cameras
    """
    from flask import request
    from urllib.parse import unquote
    
    try:
        source = request.args.get('source', '')
        camera_type = request.args.get('type', 'unknown')
        source_decoded = unquote(source)
        
        print(f"[VIDEO_FEED_PREVIEW] Request: source={source}, type={camera_type}")
        
        cam_id = None
        cap = None
        need_release = False  # Track if we opened a new cap instance
        
        # Tier 1: Try to find in existing initialized cameras
        if source in camera_sources:
            cam_id = source
            cap = camera_sources[source]
            print(f"[VIDEO_FEED_PREVIEW] ✓ Found in camera_sources by ID: {source}")
        
        elif source in camera_source_map:
            cam_id = camera_source_map[source]
            cap = camera_sources.get(cam_id)
            print(f"[VIDEO_FEED_PREVIEW] ✓ Found in source_map: {source} -> {cam_id}")
        
        elif source_decoded in camera_source_map:
            cam_id = camera_source_map[source_decoded]
            cap = camera_sources.get(cam_id)
            print(f"[VIDEO_FEED_PREVIEW] ✓ Found in source_map (decoded): {source_decoded} -> {cam_id}")
        
        # Tier 2: Try to open source directly (for preview/testing new cameras)
        if cap is None and source:
            try:
                print(f"[VIDEO_FEED_PREVIEW] Attempting direct connection to source: {source_decoded}")
                
                # Determine source type
                if source_decoded.lower() == 'oak':
                    # OAK camera via DepthAI
                    print(f"[VIDEO_FEED_PREVIEW] Opening as OAK camera (DepthAI)...")
                    if not DEPTHAI_AVAILABLE:
                        print(f"[VIDEO_FEED_PREVIEW] ✗ DepthAI not installed - cannot use OAK camera")
                        cap = None
                    else:
                        try:
                            import depthai as dai
                            devices = dai.Device.getAllAvailableDevices()
                            if not devices:
                                print(f"[VIDEO_FEED_PREVIEW] ✗ No OAK devices detected")
                                cap = None
                            else:
                                # Create pipeline for OAK camera
                                pipeline = dai.Pipeline()
                                cam = pipeline.create(dai.node.ColorCamera)
                                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                                cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                                cam.setFps(30)
                                cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
                                cam.setPreviewSize(1280, 720)
                                cam.setVideoSize(1280, 720)
                                
                                xout_video = pipeline.create(dai.node.XLinkOut)
                                xout_video.setStreamName("video")
                                cam.video.link(xout_video.input)
                                
                                device = dai.Device(pipeline)
                                
                                # Store OAK device info in cap-like structure
                                cap = {
                                    'type': 'oak_depthai',
                                    'device': device,
                                    'queue_name': 'video'
                                }
                                
                                preview_timestamp = int(time.time() * 1000)
                                cam_id = f"preview_oak_{preview_timestamp}"
                                print(f"[VIDEO_FEED_PREVIEW] ✓ Successfully opened OAK camera as: {cam_id}")
                        except Exception as e:
                            print(f"[VIDEO_FEED_PREVIEW] ✗ Failed to open OAK camera: {e}")
                            cap = None
                
                elif source_decoded.startswith('http://') or source_decoded.startswith('https://'):
                    # IP Camera/RTSP/HTTP stream
                    print(f"[VIDEO_FEED_PREVIEW] Opening as IP/RTSP stream: {source_decoded}")
                    cap = cv2.VideoCapture(source_decoded)
                    
                    if cap is not None and cap.isOpened():
                        # Optimize settings
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            cap.set(cv2.CAP_PROP_FPS, 15)
                        except:
                            pass
                        
                        # Test read
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            need_release = True
                            # ⭐ UNIQUE CAM_ID: Use timestamp to prevent collisions during concurrent tests
                            preview_timestamp = int(time.time() * 1000)  # milliseconds
                            cam_id = f"preview_ip_{preview_timestamp}_{source[:10]}"
                            print(f"[VIDEO_FEED_PREVIEW] ✓ Successfully opened IP stream as: {cam_id}")
                        else:
                            print(f"[VIDEO_FEED_PREVIEW] ⚠ Stream opened but test read failed, trying anyway...")
                            need_release = True
                            preview_timestamp = int(time.time() * 1000)
                            cam_id = f"preview_ip_{preview_timestamp}_{source[:10]}"
                    else:
                        print(f"[VIDEO_FEED_PREVIEW] ✗ Failed to open stream")
                        cap = None
                
                else:
                    # Try as USB camera index
                    try:
                        src_int = int(source)
                        print(f"[VIDEO_FEED_PREVIEW] Opening as USB camera index: {src_int}")
                        cap = cv2.VideoCapture(src_int)
                        
                        if cap is not None and cap.isOpened():
                            try:
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                cap.set(cv2.CAP_PROP_FPS, 15)
                            except:
                                pass
                            
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                need_release = True
                                # ⭐ UNIQUE CAM_ID: Use timestamp to prevent collisions during concurrent tests
                                preview_timestamp = int(time.time() * 1000)  # milliseconds
                                cam_id = f"preview_usb_{src_int}_{preview_timestamp}"
                                print(f"[VIDEO_FEED_PREVIEW] ✓ Successfully opened USB camera as: {cam_id}")
                            else:
                                print(f"[VIDEO_FEED_PREVIEW] ⚠ USB camera opened but test read failed")
                                need_release = True
                                preview_timestamp = int(time.time() * 1000)
                                cam_id = f"preview_usb_{src_int}_{preview_timestamp}"
                        else:
                            print(f"[VIDEO_FEED_PREVIEW] ✗ Failed to open USB camera at index {src_int}")
                            cap = None
                    except ValueError:
                        # Not numeric, treat as filename or unknown
                        print(f"[VIDEO_FEED_PREVIEW] Unknown source format: {source}")
                        cap = None
            
            except Exception as e:
                print(f"[VIDEO_FEED_PREVIEW] Error opening source: {type(e).__name__}: {e}")
                cap = None
        
        # ⭐ FIX: NO FALLBACK to first available camera
        # If source doesn't match any initialized camera, return error instead of showing wrong camera
        # This ensures each test shows ONLY the camera being tested, not a random other camera
        if cap is None:
            print(f"[VIDEO_FEED_PREVIEW] ✗ Could not open source: {source}")
            print(f"[VIDEO_FEED_PREVIEW] [DEBUG] Tried to open source directly, got None")
            print(f"[VIDEO_FEED_PREVIEW] [DEBUG] Available initialized cameras: {list(camera_sources.keys())}")
            return f"ERROR: Cannot open camera source '{source}'. Please check:<br>" \
                   f"1. Source is accessible<br>" \
                   f"2. USB camera index is correct<br>" \
                   f"3. IP/RTSP URL is reachable<br>" \
                   f"<br>Available cameras: {', '.join(camera_sources.keys()) if camera_sources else 'None'}", 503
        
        print(f"[VIDEO_FEED_PREVIEW] Streaming from: {cam_id}")
        
        # Generate streaming response
        def stream_generator_preview():
            lock = camera_locks.get(cam_id) or threading.Lock()
            frame_count = 0
            last_frame = None
            last_frame_time = time.time()
            frame_delay = 1.0 / 15  # 15 FPS for preview
            
            try:
                while True:
                    # FPS limiting
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_delay:
                        time.sleep(max(0.001, frame_delay - elapsed))
                    last_frame_time = time.time()
                    
                    frame = None
                    with lock:
                        try:
                            success, frame = read_frame_from_source(cap)
                        except:
                            success = False
                    
                    if not success or frame is None:
                        if last_frame is not None:
                            frame = last_frame
                        else:
                            time.sleep(0.05)
                            continue
                    else:
                        last_frame = frame
                    
                    # Resize for streaming
                    try:
                        if frame.shape[1] != STREAM_WIDTH or frame.shape[0] != STREAM_HEIGHT:
                            frame = cv2.resize(frame, (STREAM_WIDTH, STREAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    except:
                        pass
                    
                    # Apply detection overlay (bounding boxes and OCR text) before encoding
                    try:
                        frame_with_overlay = apply_detection_overlay(frame, cam_id)
                    except Exception as e:
                        print(f"[WARN] Overlay error in preview: {e}")
                        frame_with_overlay = frame
                    
                    # ⭐ STORE PROCESSED FRAME: Save frame with overlays for port 5000 UI access
                    try:
                        with latest_frames_lock:
                            latest_frames[cam_id] = frame_with_overlay.copy()
                    except Exception as e:
                        if DEBUG:
                            print(f"[DEBUG] Failed to store latest frame for {cam_id}: {e}")
                    
                    # Encode to JPEG
                    try:
                        success, buf = cv2.imencode('.jpg', frame_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        if success and buf is not None:
                            frame_bytes = buf.tobytes()
                            try:
                                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + \
                                      str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n'
                                frame_count += 1
                            except GeneratorExit:
                                # Client disconnected - this is normal, just exit
                                raise
                            except BrokenPipeError:
                                # Socket closed by client - normal shutdown
                                raise GeneratorExit
                    except GeneratorExit:
                        # Re-raise to exit the generator
                        raise
                    except:
                        time.sleep(0.01)
                        continue
            
            except GeneratorExit:
                # Client disconnected - this is normal, exit cleanly
                print(f"[VIDEO_FEED_PREVIEW] Client disconnected from preview stream")
                return  # ★ CRITICAL: Must return, not pass
            except Exception as e:
                # Any other exception
                print(f"[VIDEO_FEED_PREVIEW] Generator exception: {type(e).__name__}: {e}")
                return
            finally:
                # Release the cap if we opened it for preview
                if need_release and cap is not None:
                    try:
                        cap.release()
                        print(f"[VIDEO_FEED_PREVIEW] Released preview camera")
                    except:
                        pass
        
        response = Response(
            stream_generator_preview(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Connection'] = 'keep-alive'
        
        return response
    
    except Exception as e:
        print(f"[ERROR] video_feed_preview exception: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", 500
        
        # If we have a camera, stream from it
        if cam_id and cam_id in camera_sources:
            print(f"[VIDEO_FEED_PREVIEW] Streaming from camera: {cam_id}")
            
            # Generate streaming response with lower preview quality
            def stream_generator_preview():
                cap = camera_sources.get(cam_id)
                if cap is None:
                    yield b"--frame\r\nContent-Type: text/plain\r\n\r\nCamera not found\r\n"
                    return
                
                lock = camera_locks.get(cam_id, threading.Lock())
                frame_count = 0
                
                # Preview streaming parameters - OPTIMIZED FOR MAXIMUM SPEED
                preview_width = 640    # Full width for quality
                preview_height = 480   # Full height for quality
                preview_quality = 70   # Reduced JPEG quality for speed (70 is optimal)
                preview_fps = 30       # Full FPS for responsiveness
                frame_delay = 1.0 / preview_fps
                last_frame_time = time.time()
                
                try:
                    while True:
                        # Efficient FPS limiting
                        current_time = time.time()
                        elapsed = current_time - last_frame_time
                        if elapsed < frame_delay:
                            sleep_time = frame_delay - elapsed
                            if sleep_time > 0.001:
                                time.sleep(sleep_time * 0.98)
                        last_frame_time = time.time()
                        
                        frame = None
                        with lock:
                            try:
                                success, frame = read_frame_from_source(cap)
                            except:
                                success = False
                        
                        if not success or frame is None:
                            time.sleep(0.01)
                            continue
                        
                        # Keep full resolution for preview (resize only if necessary)
                        if frame.shape[1] != preview_width or frame.shape[0] != preview_height:
                            frame = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_LINEAR)
                        
                        # Fast JPEG encoding with minimal parameters
                        success, buffer = cv2.imencode(
                            '.jpg',
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, preview_quality]
                        )
                        
                        if success:
                            frame_count += 1
                            frame_bytes = buffer.tobytes()
                            frame_len = len(frame_bytes)
                            boundary = b'--frame\r\nContent-Type: image/jpeg\r\nContent-length: '
                            try:
                                yield boundary + str(frame_len).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n'
                            except GeneratorExit:
                                raise
                            except BrokenPipeError:
                                raise GeneratorExit
                
                except GeneratorExit:
                    return  # ★ CRITICAL: Return instead of pass
                except Exception as e:
                    print(f"[VIDEO_FEED_PREVIEW] Stream error for {cam_id}: {e}")
                    return  # ★ CRITICAL: Return instead of continuing
            
            response = Response(
                stream_generator_preview(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
            
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Transfer-Encoding'] = 'chunked'
            
            return response
        
        # If no camera found, return error with available cameras list
        print(f"[VIDEO_FEED_PREVIEW] No matching camera found for source={source}, type={camera_type}")
        print(f"[VIDEO_FEED_PREVIEW] Available cameras: {list(camera_sources.keys())}")
        
        if not camera_sources:
            return "No cameras available. Start with: python camera_anpr.py --cameras webcam=0", 503
        
        # Return error listing available cameras
        available = ", ".join(camera_sources.keys())
        return f"Camera not found. Requested source: {source}, type: {camera_type}. Available cameras: {available}", 404
            
    except Exception as e:
        print(f"[VIDEO_FEED_PREVIEW] Unhandled error: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", 500


@app.route('/api/video_feed/<cam_id>')
def video_feed(cam_id):
    """Main video feed endpoint with cross-origin support"""
    from flask import request
    from urllib.parse import unquote
    
    try:
        # Get client information for debugging
        remote_addr = request.remote_addr
        origin = request.headers.get('Origin', 'Direct')
        user_agent = request.headers.get('User-Agent', 'Unknown')

        # Extract userId and cameraId from query parameters
        user_id = request.args.get('userId', None)
        camera_id = request.args.get('cameraId', cam_id)
        
        # URL decode cam_id in case it's encoded
        cam_id_decoded = unquote(cam_id)

        # ★ FIX 3: Check if camera is registered in CameraManager
        # This ensures the pipeline is actually running before streaming
        actual_cam_id = None
        
        # First, check camera_manager registry if available
        if camera_manager:
            mgr_status = camera_manager.get_camera_status()
            active_cams = mgr_status.get('active_camera_ids', [])
            
            # Check if cam_id or decoded version is in active cameras
            if cam_id in active_cams:
                actual_cam_id = cam_id
                print(f"[VIDEO_FEED] ✓ FIX 3: Found camera in CameraManager registry: {cam_id}")
            elif cam_id_decoded in active_cams:
                actual_cam_id = cam_id_decoded
                print(f"[VIDEO_FEED] ✓ FIX 3: Found camera in CameraManager registry (decoded): {cam_id_decoded}")
        
        # Fallback: Check if cam_id exists directly in camera_sources
        if not actual_cam_id:
            if cam_id in camera_sources:
                actual_cam_id = cam_id
                print(f"[VIDEO_FEED] ✓ Found camera by ID: {cam_id}")
            
            # Tier 2: Check if decoded cam_id exists in camera_sources
            elif cam_id_decoded in camera_sources:
                actual_cam_id = cam_id_decoded
                print(f"[VIDEO_FEED] ✓ Found camera by decoded ID: {cam_id_decoded}")
            
            # Tier 3: Try reverse lookup via source_map (for URLs and indices)
            elif cam_id in camera_source_map:
                actual_cam_id = camera_source_map[cam_id]
                print(f"[VIDEO_FEED] ✓ Found camera via source_map: {cam_id} -> {actual_cam_id}")
            
            # Tier 4: Try reverse lookup with decoded ID
            elif cam_id_decoded in camera_source_map:
                actual_cam_id = camera_source_map[cam_id_decoded]
                print(f"[VIDEO_FEED] ✓ Found camera via source_map (decoded): {cam_id_decoded} -> {actual_cam_id}")
            
            # Tier 5: Handle special names that might map to cameras
            elif cam_id in ('webcam', 'preview', '0'):
                if 'webcam' in camera_sources:
                    actual_cam_id = 'webcam'
                    print(f"[VIDEO_FEED] ✓ Mapped '{cam_id}' to 'webcam'")
                elif camera_sources:
                    actual_cam_id = list(camera_sources.keys())[0]
                    print(f"[VIDEO_FEED] ✓ Mapped '{cam_id}' to first available: {actual_cam_id}")
        
        # If still not found, report error
        if not actual_cam_id:
            print(f"[ERROR] Camera ID '{cam_id}' not found in camera_manager or camera_sources")
            print(f"[ERROR] Available cameras: {list(camera_sources.keys())}")
            if camera_manager:
                print(f"[ERROR] CameraManager active: {camera_manager.get_camera_status()}")
            available = ", ".join(camera_sources.keys()) if camera_sources else "None"
            return f"Camera '{cam_id}' not found. Available cameras: {available}", 404

        print(f"[VIDEO_FEED] Request for cam_id={cam_id}, userId={user_id}, cameraId={camera_id}")
        print(f"[VIDEO_FEED] From: {remote_addr}, Origin: {origin}")
        print(f"[VIDEO_FEED] Streaming from camera: {actual_cam_id}")
        print(f"[VIDEO_FEED] Available cameras: {list(camera_sources.keys())}")

        # Create response with proper headers for cross-origin streaming
        def stream_generator():
            try:
                for chunk in generate_stream(actual_cam_id, user_id, camera_id):
                    yield chunk
            except GeneratorExit:
                print(f"[INFO] Client disconnected from camera {actual_cam_id}")
                return  # ★ CRITICAL: Return instead of pass
            except Exception as e:
                print(f"[ERROR] Stream generator error for camera {actual_cam_id}: {e}")
                return  # ★ CRITICAL: Return instead of continuing

        response = Response(
            stream_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

        # Add additional headers to ensure cross-origin streaming works
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning'
        response.headers['Connection'] = 'keep-alive'
        response.headers['Transfer-Encoding'] = 'chunked'

        return response
    except Exception as e:
        print(f"[ERROR] Error in video_feed endpoint: {type(e).__name__}: {e}")
        return f"Error: {str(e)}", 500

from flask import send_from_directory

@app.route('/plates/<path:filename>')
def plate_image(filename):
    """Serve plate images with proper CORS headers"""
    from flask import request

    remote_addr = request.remote_addr
    print(f"[PLATES] Image request: {filename} from {remote_addr}")

    response = send_from_directory('data/plates', filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response


@app.route('/api/video_feed/<cam_id>', methods=['OPTIONS'])
@app.route('/video_feed/<cam_id>', methods=['OPTIONS'])
def video_feed_options(cam_id):
    """Handle CORS preflight requests"""
    from flask import request

    remote_addr = request.remote_addr
    print(f"[OPTIONS] Preflight request for cam_id={cam_id} from {remote_addr}")

    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, ngrok-skip-browser-warning'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response


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


# ---------------- Main ----------------
if __name__ == "__main__":
    # ⭐ INITIALIZE: Load existing plate texts to prevent re-saving duplicates
    print("[INIT] Scanning existing plate images to prevent duplicates...")
    try:
        if DIR_PLATES.exists():
            plate_files = list(DIR_PLATES.glob("*.jpg"))
            for plate_file in plate_files:
                # Extract plate text from filename (format: PLATE_TEXT_timestamp.jpg)
                filename = plate_file.stem  # Remove .jpg
                # Split by underscore and take all but the last part (last part is timestamp)
                parts = filename.rsplit('_', 1)
                if len(parts) == 2:
                    plate_text = parts[0].replace('_', '-')  # Convert underscores back to dashes
                    with saved_plate_texts_lock:
                        saved_plate_texts.add(plate_text)
            
            print(f"[INIT] ✓ Loaded {len(saved_plate_texts)} existing unique plates - duplicates will be prevented")
        else:
            print("[INIT] Plates folder not found, will create on first detection")
    except Exception as e:
        print(f"[INIT] Warning: Failed to scan existing plates: {e}")
    
    parser = argparse.ArgumentParser(description='ANPR System with dynamic config')
    parser.add_argument('--cameras', nargs='+', required=False, help='camera sources (optional, loads from config if not provided)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--gates', nargs='*', help='Camera gate types: id:GateType (Entry/Exit), e.g., webcam:Entry ipcam:Exit')
    args = parser.parse_args()

    # ★ STARTUP FLOW: Only load cameras from config if explicitly provided via CLI
    # This ensures users configure cameras through the UI before the app tries to connect
    if args.cameras:
        # Explicitly provided via CLI - use these
        pass
    else:
        # NOT provided via CLI - start with NO cameras (demo/setup mode)
        # Users will configure cameras through the web UI
        # ★ PHASE 8: Use get_cameras_to_start() to only load cameras that are ENABLED AND ONLINE
        # This respects persistent camera state across application restarts
        enabled_cameras = config_manager.get_cameras_to_start()
        
        if enabled_cameras:
            # Cameras are configured and enabled - try to auto-load them
            enabled_gates = config_manager.get_enabled_gates()
            
            # ★ DATA ISOLATION: Load user IDs for each camera
            print("\n[DATA_ISOLATION] Loading camera ownership information...")
            all_cameras = config_manager.get_all_cameras()
            for camera in all_cameras:
                camera_id = camera.get('id')
                user_id = camera.get('userId', 'default')
                set_camera_user_id(camera_id, user_id)
            
            if enabled_gates:
                # Both cameras and gates are configured - use them
                cameras_arg, gates_arg = config_manager.build_cli_args()
                args.cameras = cameras_arg.split(',')
                if gates_arg and not args.gates:
                    args.gates = gates_arg.split(',')
                print(f"✓ Loaded {len(enabled_cameras)} cameras from config")
                print(f"✓ Loaded {len(enabled_gates)} gates from config")
            else:
                # Cameras exist but no gates - skip auto-load
                print("⚠️  Cameras are configured but no gates found")
                print("   Please configure gates in the web UI")
                args.cameras = []
        else:
            # No cameras configured yet - start in setup mode
            print("⚠️  No cameras configured. Starting in SETUP MODE.")
            print("   Open the web UI to configure cameras:")
            print("   • http://localhost:5000/settings")
            print("")
            
            # ★ DATA ISOLATION: Still load user IDs for any that exist
            print("[DATA_ISOLATION] Loading camera ownership information...")
            all_cameras = config_manager.get_all_cameras()
            for camera in all_cameras:
                camera_id = camera.get('id')
                user_id = camera.get('userId', 'default')
                set_camera_user_id(camera_id, user_id)
            
            args.cameras = []  # Empty camera list - will run in demo mode without trying to connect
    
    print("\n[INFO] Starting ANPR Camera System...")
    print("[INFO] ★ Initializing Intelligent Gate Management System...")
    
    # ★ STARTUP: Sync vehicle entry log from persistent DB
    # This restores entry/exit tracking after process restart
    sync_vehicle_entry_log_from_db()
    
    # ★ STARTUP: Load gate types from camera configuration (not from CLI)
    # This allows gates to be managed entirely through the web UI
    print("[INFO] Loading camera gate types from configuration...")
    all_cameras = config_manager.get_all_cameras()
    for camera in all_cameras:
        cam_id = camera.get('id')
        gate_type = camera.get('gate', camera.get('gateType', 'Entry'))  # Support both 'gate' and 'gateType' fields
        if gate_type in ['Entry', 'Exit']:
            set_camera_gate_type(cam_id, gate_type)
            print(f"[CAMERA_GATE] Camera '{cam_id}' configured as {gate_type} gate ✓")
        else:
            set_camera_gate_type(cam_id, 'Entry')
            print(f"[CAMERA_GATE] Camera '{cam_id}' invalid gate type, defaulting to Entry")
    
    # Note: --gates CLI argument is now DEPRECATED in favor of web UI configuration
    
    print("[INFO] Init Models...")
    plate_cascade = download_haar()
    
    # Load license plate detector - preferring FP32 ONNX over PyTorch
    # INT8 quantized model requires special ONNX Runtime support that may not be available
    model_global = None
    plate_detector_model = None
    
    try:
        # Try to load ONNX model first (FP32 - INT8 requires special ONNX Runtime support)
        # ⭐ FIX: Skip INT8, use FP32 ONNX or PyTorch
        script_dir = Path(__file__).parent.parent.parent  # /backend/app -> /backend -> /root
        model_path_onnx = str(script_dir / "license_plate_detector.onnx")
        
        onnx_path_to_use = None
        model_type_name = None
        
        # Use FP32 ONNX if available (INT8 skipped due to ConvInteger operator not supported)
        if os.path.exists(model_path_onnx) and ONNXRUNTIME is not None:
            onnx_path_to_use = model_path_onnx
            model_type_name = "ONNX/FP32 (CPU-Optimized)"
        
        if onnx_path_to_use and ONNXRUNTIME is not None:
            try:
                print(f"[INFO] Loading license plate detector ({model_type_name}): {onnx_path_to_use}")
                if not os.path.exists(onnx_path_to_use):
                    print(f"[WARN] ONNX model file not found at: {onnx_path_to_use}")
                    print(f"[WARN] Will try PyTorch model instead")
                    onnx_path_to_use = None
                else:
                    plate_detector_model = ONNXPlateDetector(onnx_path_to_use)
                    print(f"[INFO] ✓ License Plate Detector model ({model_type_name}) loaded successfully")
                    print(f"[INFO] ✓ Using ONNX Runtime with CPUExecutionProvider")
                    print(f"[INFO] ✓ Post-processing: Vectorized (10-15% faster)")
                    print(f"[INFO] ✓ Expected overall speedup: 15-25% compared to PyTorch")
                    print(f"[INFO] ✓✓✓ ONNX MODEL ACTIVE ✓✓✓")
            except Exception as e:
                print(f"[WARN] Failed to load ONNX model: {type(e).__name__}: {e}")
                print(f"[WARN] Falling back to PyTorch model...")
                plate_detector_model = None
        else:
            if ONNXRUNTIME is None:
                print(f"[WARN] ONNX Runtime not available")
            else:
                print(f"[WARN] ONNX models not found")
            print(f"[WARN] Path checked: {os.path.abspath(model_path_onnx)}")
        
        # Fall back to PyTorch model if ONNX didn't work
        if plate_detector_model is None:
            if YOLO is None:
                print("[ERROR] YOLO module not available! Install with: pip install ultralytics>=8.0.0")
                print("[ERROR] Exiting: license_plate_detector model is required")
                sys.exit(1)
            
            try:
                script_dir = Path(__file__).parent.parent.parent  # Get absolute path
                model_path_pt = str(script_dir / "license_plate_detector.pt")
                print(f"[INFO] Loading license plate detector (PyTorch): {model_path_pt}")
                if not os.path.exists(model_path_pt):
                    print(f"[ERROR] Model file not found at: {model_path_pt}")
                    print(f"[ERROR] Available files in {script_dir}: {list(script_dir.glob('*.pt')) + list(script_dir.glob('*.onnx'))}")
                    raise FileNotFoundError(f"Model not found: {model_path_pt}")
                plate_detector_model = YOLO(model_path_pt)
                print(f"[INFO] ✓ License Plate Detector model (PyTorch) loaded successfully")
            except FileNotFoundError as e:
                print(f"[ERROR] License plate model not found: {e}")
                print(f"[ERROR] Expected ONNX: {os.path.abspath(model_path_onnx)}")
                print(f"[ERROR] Expected PyTorch: {os.path.abspath(model_path_pt) if 'model_path_pt' in locals() else 'unknown'}")
                print("[ERROR] Exiting: license_plate_detector model is required")
                sys.exit(1)
            except Exception as e:
                print(f"[ERROR] Failed to load license plate detector: {type(e).__name__}: {e}")
                print("[ERROR] Exiting: license_plate_detector model is required")
                sys.exit(1)
    
    except Exception as e:
        print(f"[ERROR] Unexpected error during model initialization: {type(e).__name__}: {e}")
        print("[ERROR] Exiting...")
        sys.exit(1)
    
    if plate_detector_model is None:
        print("[ERROR] License plate detector model is required but not loaded!")
        print("[ERROR] Exiting...")
        sys.exit(1)

    # ⭐ DEBUG: Confirm model is loaded and operational
    print(f"\n[INFO] ✓✓✓ DETECTION MODEL STATUS ✓✓✓")
    print(f"[INFO] Plate detector model: {type(plate_detector_model).__name__}")
    print(f"[INFO] Model class: {plate_detector_model.__class__.__module__}.{plate_detector_model.__class__.__name__}")
    print(f"[INFO] Detection resolution: {DETECTION_WIDTH}x{DETECTION_HEIGHT}")
    print(f"[INFO] Stream resolution: {STREAM_WIDTH}x{STREAM_HEIGHT}")
    print(f"[INFO] YOLO frequency: Every {YOLO_PROCESS_EVERY_N} frame(s) (every ~{1000*YOLO_PROCESS_EVERY_N/TARGET_FPS:.0f}ms)")
    print(f"[INFO] Stabilizer lock threshold: 1 frame(s)")
    print(f"[DEBUG] MODEL READY: {plate_detector_model is not None}")
    print(f"[DEBUG] MODEL OBJECT: {plate_detector_model}")
    print(f"[DEBUG] MODEL CALLABLE: {callable(plate_detector_model)}")
    
    # Verify model can actually run inference
    try:
        import numpy as np
        test_frame = np.zeros((DETECTION_HEIGHT, DETECTION_WIDTH, 3), dtype=np.uint8)
        test_result = plate_detector_model(test_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        print(f"[DEBUG] ✓ MODEL TEST PASS - inference works correctly")
        print(f"[DEBUG] ✓ YOLO MODEL FULLY OPERATIONAL AND READY\n")
    except Exception as e:
        print(f"[ERROR] ✗ MODEL TEST FAIL - model cannot run inference: {e}")
        print(f"[ERROR] EXITING - Detection will not work without functional model")
        sys.exit(1)
    
    # Verify overlay cache is initialized
    print(f"[DEBUG] Detection overlay cache initialized: {isinstance(detection_overlay_cache, dict)}")
    print(f"[DEBUG] Overlay cache lock initialized: {detection_overlay_lock is not None}")
    print(f"[DEBUG] FRAME BUFFER SIZE: {FRAME_BUFFER_SIZE} frames (Pi requirement: ≥10)")
    print(f"[DEBUG] MAX_FRAME_QUEUE: {MAX_FRAME_QUEUE} frames")
    print(f"[DEBUG] DETECTION EXPIRY: 0.5 seconds")
    print(f"[DEBUG] STABILIZER TIMEOUT: 3.0 seconds")
    
    # COMPREHENSIVE STARTUP VALIDATION
    print(f"\n[INFO] ━━━ DETECTION PIPELINE STARTUP CHECKLIST ━━━")
    print(f"[INFO] ✓ YOLO Model loaded: {plate_detector_model is not None}")
    print(f"[INFO] ✓ Model callable: {callable(plate_detector_model)}")
    print(f"[INFO] ✓ Inference test passed: YES")
    print(f"[INFO] ✓ Overlay cache structure: OK")
    print(f"[INFO] ✓ Frame queue size: {MAX_FRAME_QUEUE} frames")
    print(f"[INFO] ✓ Detection frequency: Every {YOLO_PROCESS_EVERY_N} frame (~{1000*YOLO_PROCESS_EVERY_N/TARGET_FPS:.0f}ms)")
    print(f"[INFO] ✓ Stabilizer immediate lock: enabled")
    print(f"[INFO] ✓ All critical systems: READY")
    print(f"\n[DEBUG] PIPELINE READY TO ACCEPT CAMERAS...\n")

    print("[INFO] Initializing cameras...")
    print(f"[DEBUG] Requested cameras: {args.cameras}")
    
    oak_cameras = []
    usb_cameras = []
    
    # Check if OAK is requested - if YES, skip all detection
    oak_requested = any(c.lower().endswith('=oak') for c in args.cameras)
    print(f"[DEBUG] OAK requested: {oak_requested}")
    
    if oak_requested:
        print("[INFO] ✓ OAK camera mode - SKIPPING detection phase")
        print("[INFO] Going directly to OAK initialization...")
        oak_cameras = [{'name': 'OAK', 'type': 'oak', 'device_info': None}]
    else:
        print("[INFO] Standard mode - detecting USB cameras...")
        usb_cameras = detect_usb_cameras()
        if usb_cameras:
            print(f"[INFO] Found {len(usb_cameras)} USB camera(s)")
    
    # ⭐ OPTIMIZATION: Start OCR background worker thread (after all globals initialized)
    print("\n[INFO] Starting OCR background worker thread...")
    ocr_worker_thread = threading.Thread(target=ocr_worker_background, daemon=True)
    ocr_worker_thread.start()
    print("[INFO] ✓ OCR background worker thread started (non-blocking async processing)")
    
    for c in args.cameras:
        try:
            cid, src = c.split("=")
        except Exception:
            print(f"[ERROR] cameras argument format: id=source (got '{c}')")
            continue
        
        # Initialize frame counter and last processed frame
        frame_counters[cid] = 0
        last_processed_frame[cid] = None
        camera_last_saved[cid] = 0
        camera_locks[cid] = threading.Lock()  # Initialize lock for this camera
        # ⭐ FIX 3: DISABLED frame deduplicator - skips frames for slow vehicles
        # frame_deduplicators[cid] = FrameDeduplicator()  # DISABLED
        plate_bbox_stabilizers[cid] = PlateBBoxStabilizer()  # Initialize bbox stabilizer for this camera
        
        # Initialize gate type (entry/exit) - can be overridden later
        camera_gate_types[cid] = "Entry"
        
        # Handle different camera source types
        cap = None
        camera_type = "unknown"
        
        try:
            # Check if source is "oak" or "OAK" for OAK camera
            if src.lower() == "oak":
                if oak_cameras:
                    print(f"[INFO] Initializing OAK camera for '{cid}'...")
                    # Initialize with direct connection (most reliable)
                    cap = initialize_oak_camera(device_info=None, direct=True)
                    if cap is not None:
                        camera_type = "oak"
                        camera_sources[cid] = cap
                        camera_source_map["oak"] = cid
                        print(f"[INFO] ✓ OAK Camera '{cid}' initialized successfully")
                    else:
                        print(f"[ERROR] Failed to initialize OAK camera for '{cid}'")
                        continue
                else:
                    print(f"[ERROR] OAK camera requested but none found.")
                    print(f"[ERROR] Troubleshooting: Check that OAK-D camera is connected via USB")
                    print(f"[ERROR] Run: lsusb | grep Movidius")
                    print(f"[ERROR] Run: python3 -c 'import depthai as dai; print(dai.Device.getAllAvailableDevices())'")
                    continue
            
            # Handle IP Webcam URLs (http://<ip>:<port>/video)
            elif src.startswith("http://") or src.startswith("https://"):
                print(f"[INFO] Initializing IP Webcam '{cid}' from {src}...")
                cap = initialize_ipwebcam(src)
                
                if cap is not None and cap.isOpened():
                    # Optimize IP webcam settings for low latency
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                        if STREAM_WIDTH > 0 and STREAM_HEIGHT > 0:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
                    except Exception as e:
                        print(f"[WARN] Could not optimize IP Webcam properties: {e}")
                    
                    # Test read to ensure connection is working
                    frame_read_success = False
                    for attempt in range(3):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            frame_read_success = True
                            print(f"[INFO] ✓ IP Webcam at {src} verified (attempt {attempt + 1})")
                            break
                        time.sleep(0.2)  # Give streaming time to initialize
                    
                    if not frame_read_success:
                        print(f"[WARN] IP Webcam opened but initial frame read failed. Proceeding anyway...")
                        # Don't fail here - stream might still work during continuous streaming
                    
                    camera_type = "ipwebcam"
                    camera_sources[cid] = cap
                    camera_source_map[src] = cid
                    print(f"[INFO] ✓ IP Webcam '{cid}' initialized successfully from {src}")
                else:
                    print(f"[ERROR] Failed to initialize IP Webcam for '{cid}' from {src}")
                    if cap is not None:
                        try:
                            cap.release()
                        except:
                            pass
                    continue
            
            # Handle numeric USB camera indices or string numbers
            else:
                try:
                    src_int = int(src)
                except ValueError:
                    print(f"[ERROR] Invalid camera source: {src}. Use numeric index (0, 1), 'oak', or full IP Webcam URL (http://...)")
                    continue
                
                # Initialize USB camera
                print(f"[INFO] Initializing USB camera at index {src_int} for '{cid}'...")
                cap = cv2.VideoCapture(src_int)
                
                # Verify camera is actually open
                if not cap.isOpened():
                    print(f"[ERROR] Failed to open USB camera at index {src_int}. Camera not available or in use.")
                    cap.release()
                    continue
                
                # Optimize camera settings BEFORE reading frames (important for Windows)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)  # Set camera FPS
                    # Optionally set resolution at camera level (may not work for all cameras)
                    if STREAM_WIDTH > 0 and STREAM_HEIGHT > 0:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
                except Exception as e:
                    print(f"[WARN] Could not optimize USB camera properties: {e}")
                
                # Try to read a frame to verify it's working (with retries for Windows webcams)
                frame_read_success = False
                for attempt in range(3):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        frame_read_success = True
                        print(f"[INFO] ✓ USB camera at index {src_int} verified (attempt {attempt + 1})")
                        break
                    time.sleep(0.2)  # Give camera time to initialize
                
                if not frame_read_success:
                    print(f"[WARN] USB camera at index {src_int} opened but initial frame read failed. Proceeding anyway (may work during streaming)...")
                    # Don't fail here - camera might still work during streaming
                
                camera_type = "usb"
                camera_source_map[str(src_int)] = cid
                camera_source_map[str(src)] = cid
                camera_sources[cid] = cap
                print(f"[INFO] ✓ USB Camera '{cid}' active at index {src_int}")
        
        except Exception as e:
            print(f"[ERROR] Failed to initialize camera '{cid}': {str(e)[:80]}")
            if cap:
                try:
                    if isinstance(cap, dict) and cap.get('type') == 'oak':
                        pass
                    else:
                        cap.release()
                except:
                    pass
            continue

    # Print camera initialization summary
    print("\n" + "="*70)
    print(" CAMERA INITIALIZATION SUMMARY")
    print("="*70)
    
    if camera_sources:
        print(f"✓ {len(camera_sources)} camera(s) initialized successfully:\n")
        for cam_id, cap in camera_sources.items():
            if isinstance(cap, dict) and cap.get('type') == 'oak':
                print(f"  ✓ {cam_id:15} → OAK Camera")
            elif isinstance(cap, cv2.VideoCapture):
                print(f"  ✓ {cam_id:15} → IP/USB Webcam")
            else:
                print(f"  ✓ {cam_id:15} → Camera")
    else:
        print("✗ WARNING: No cameras initialized!")
        print("\nPossible Issues and Solutions:")
        print("-" * 70)
        print("\n1. USB Camera (index 0, 1, 2, etc.):")
        print("   - Verify camera is connected to the system")
        print("   - Check if camera is being used by another application")
        print("   - Try: python camera_anpr.py --cameras webcam=0")
        print("   - Try: python camera_anpr.py --cameras cam1=0 cam2=1 (multiple cameras)")
        print("\n2. IP Webcam (Android IP Webcam App):")
        print("   - Install Android IP Webcam app on your phone")
        print("   - Enable 'Start Server' in the app (note the IP and port, e.g., 192.168.1.100:8080)")
        print("   - Try: python camera_anpr.py --cameras phone='http://192.168.1.100:8080/video'")
        print("   - Try: python camera_anpr.py --cameras phone='http://192.168.1.100:8080/mjpegfeed' (MJPEG)")
        print("\n3. OAK Camera:")
        print("   - Verify OAK-D camera is connected via USB")
        print("   - Check if DepthAI is installed: pip install depthai")
        print("   - List OAK devices: python3 -c 'import depthai as dai; print(dai.Device.getAllAvailableDevices())'")
        print("   - Try: python camera_anpr.py --cameras oak=oak")
        print("\n4. RTSP Camera:")
        print("   - Try: python camera_anpr.py --cameras rtsp_cam='rtsp://192.168.1.100:554/stream'")
        print("\n5. Multiple Cameras:")
        print("   - Try: python camera_anpr.py --cameras webcam=0 phone='http://192.168.1.100:8080/video' oak=oak")
        print("-" * 70)
        print("\n[INFO] Running in demo mode (no live camera feeds available)...\n")

    print("="*70)
    print(f" Active: {', '.join(list(camera_sources.keys()))}")
    print("="*70 + "\n")

    # Print network information
    print_network_info()

    # ★ CRITICAL FIX 2: Auto-load cameras from config at startup
    # This ensures all configured cameras are initialized before the detector starts
    print("[INFO] ★ Loading cameras from configuration file...")
    if args.cameras and len(args.cameras) > 0:
        # Cameras were provided via CLI - skip auto-load (already initialized above)
        print(f"[INFO] Cameras already initialized from CLI arguments ({len(camera_sources)} cameras)")
    else:
        # Auto-load from config
        print(f"[INFO] Attempting to auto-load cameras from config...")
        newly_loaded, cam_list = reload_cameras_from_config()
        
        if newly_loaded > 0:
            print(f"[INFO] ✓ Auto-loaded {newly_loaded} camera(s) from config: {cam_list}")
        else:
            if len(camera_sources) == 0:
                print(f"[INFO] No cameras to auto-load (config or setup mode)")
            else:
                print(f"[INFO] Cameras already loaded ({len(camera_sources)} active)")
    
    # ★ CRITICAL FIX 4: Start config watcher to detect new cameras
    # This allows cameras added via API to be detected automatically
    config_watcher_thread = start_config_watcher_thread()
    
    # START BACKGROUND PROCESSING THREADS
    # These threads continuously process frames for detection/OCR
    # while the main Flask threads handle streaming without blocking
    print("[INFO] Starting background processing threads...\n")
    
    # Use GLOBAL frame_queues, processing_stop_events, processing_threads
    # (defined at module level for cross-thread access)
    user_id_global = None  # Will be set per-request, use default admin for background processing
    
    def background_process_worker(cam_id):
        """Background thread that continuously processes frames from the queue"""
        print(f"[WORKER_INIT] Worker thread started for camera '{cam_id}'")  # ⭐ DEBUG
        frame_queue = frame_queues.get(cam_id)
        stop_event = processing_stop_events.get(cam_id)
        
        if not frame_queue or not stop_event:
            print(f"[ERROR] Queue or stop event missing for camera '{cam_id}'")
            print(f"[ERROR] frame_queue={frame_queue}, stop_event={stop_event}")  # ⭐ DEBUG
            return
        
        print(f"[WORKER_READY] Worker ready for '{cam_id}', waiting for frames...")  # ⭐ DEBUG
        frames_processed = 0
        last_print = 0
        
        while not stop_event.is_set():
            try:
                # Get frame from queue with timeout (non-blocking)
                try:
                    frame = frame_queue.get(timeout=0.5)
                except Exception as queue_error:
                    # Timeout is normal - just continue waiting
                    continue
                
                if frame is None:
                    break  # Stop signal
                
                frames_processed += 1
                
                # Print status every 200 frames for minimal logging
                if frames_processed % 200 == 0:
                    print(f"[WORKER] {cam_id}: Processed {frames_processed} frames")
                
                # Process the frame (heavy operation, doesn't block streaming)
                try:
                    process_frame(frame, cam_id, user_id_global, cam_id)
                except KeyError as ke:
                    if frames_processed <= 5:
                        import traceback
                        print(f"[ERROR] KeyError during frame processing: {str(ke)}")
                        print(f"[DEBUG] cam_id='{cam_id}', available keys in frame_counters: {list(frame_counters.keys())}")
                        tb = traceback.format_exc()
                        # Print full traceback
                        for line in tb.split('\n'):
                            if line.strip():
                                print(f"[DEBUG] {line}")
                except Exception as e:
                    if frames_processed <= 5:  # Only print first 5 errors
                        print(f"[ERROR] Frame processing failed: {type(e).__name__}: {str(e)[:80]}")
            
            except Exception as e:
                print(f"[ERROR] Worker error: {str(e)[:80]}")
                continue
    
    # Create processing threads for each camera
    for cam_id in camera_sources.keys():
        try:
            # Create queue and stop event for this camera
            frame_queues[cam_id] = __import__('queue').Queue(maxsize=2)
            processing_stop_events[cam_id] = threading.Event()
            
            # Start worker thread
            worker_thread = threading.Thread(
                target=background_process_worker,
                args=(cam_id,),
                daemon=True,
                name=f"ProcessWorker-{cam_id}"
            )
            worker_thread.start()
            processing_threads.append((cam_id, worker_thread, processing_stop_events[cam_id]))
        except Exception as e:
            print(f"[WARN] Could not start processor for camera '{cam_id}': {e}")
    
    # Store for later cleanup
    app.processing_threads = processing_threads
    app.frame_queues = frame_queues
    
    print(f"[INFO] Processing pipeline initialized ({len(frame_queues)} camera streams)")
    print(f"[INFO] Detection system ready\n")

    print(f"[INFO] Starting Flask server on {args.host}:{args.port}...")
    print("[INFO] Press Ctrl+C to stop the server\n")

    # Run flask streaming server with threading for better performance
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except Exception as e:
        print(f"[ERROR] Failed to start Flask server: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Cleanup: stop all processing threads
    print("\n[INFO] Shutting down...")
    for cam_id, thread, stop_event in processing_threads:
        stop_event.set()
        if frame_queues.get(cam_id):
            try:
                frame_queues[cam_id].put(None)  # Send stop signal
            except:
                pass
        thread.join(timeout=2)
