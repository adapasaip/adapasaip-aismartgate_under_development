"""OCR validation and reporting functions"""
import time
from typing import Dict, Any, Optional

from ..config import DEBUG

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
