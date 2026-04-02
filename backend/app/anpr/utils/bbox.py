"""Bounding box stabilization and IOU calculations"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

class PlateBBoxStabilizer:
    def __init__(self):
        self.locked_bbox = None
        self.last_seen = None  # Anchor point for frame-level drift detection
        self.last_seen_time = time.time()  # FIX 3: Timestamp for vehicle disappearance detection
        self.missed_frames = 0
        self.lock_threshold = 3  # CRITICAL FIX #2: Lock after 1 frame for debugging (production: 3)
        self.max_missed = 5
        self.locked = False
        self.lock_frames = 0
        self.timeout_threshold = 3.0  # FIX 2: Reset after 3.0 seconds (was 1.5) - prevents CPU spike jitter

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
            self.last_seen_time = time.time()  # FIX 3: Record first sighting
            return new_bbox

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

            if self.lock_frames >= self.lock_threshold:  # FIX 4: Use lock_threshold (now 4)
                self.locked = True
                print(f"[BBOX_LOCK] Box locked after {self.lock_frames} stable frames (4-frame mode)")

        # TRACK when locked: Ignore YOLO jitter, follow only meaningful motion
        if self.locked:
            # Update last seen time (vehicle is still being detected)
            self.last_seen_time = current_time  # FIX 3: Update timestamp on every detection
            
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

            # This allows faster response to real vehicle movement while rejecting small jitter
            if movement < 2:  # CRITICAL FIX #2: Reduced to 2px for smooth slow-moving vehicle updates
                # Small jitter - stay locked to existing position
                return self.locked_bbox

            # FOLLOW REAL MOTION with 20% response for extreme smoothness
            # 20% means we lag behind YOLO but achieve maximum stability
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
            self.last_seen_time = current_time  # FIX 3: Update timestamp
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

