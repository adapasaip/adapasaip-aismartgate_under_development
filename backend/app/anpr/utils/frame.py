"""Frame deduplication and frame reading utilities"""
import cv2
import numpy as np
import hashlib
from typing import Optional, Dict, Tuple
import threading

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

