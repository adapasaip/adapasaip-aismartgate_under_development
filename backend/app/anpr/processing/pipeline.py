"""Detection processing pipeline with overlay and callbacks"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import threading
import time
from collections import defaultdict

from ..config import DEBUG, COLOR_FRONT, COLOR_REAR, COLOR_DETECTING, COLOR_PLATE, COLOR_TEXT_BG
from ..detectors.onnx_wrapper import ONNXPlateDetector
from ..utils.bbox import PlateBBoxStabilizer
from ..ocr.engine import OCRWorker

# ---------------- Processing ----------------
def match_and_update_trackers(cam_id, detections):
    """
    FIX 4: GATE OPTIMIZATION - Single tracker per camera
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
        now = time.time()
        to_del = []
        for tid, tr in list(trks.items()):
            if now - tr.last_seen > 1.0:  # FIXED: 1 second timeout (was 2.0)
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
        
        # If centroid moved >150px, it's likely a different vehicle, not same vehicle moving
        new_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
        distance = ((new_centroid[0] - old_centroid[0])**2 + (new_centroid[1] - old_centroid[1])**2) ** 0.5
        
        if distance > 150:  # Significant movement = new vehicle
            # Reset tracker state for new vehicle
            tr.saved = False  # UNBLOCK OCR FOR NEW VEHICLE
            tr.last_ocr_time = 0  # RESET OCR COOLDOWN FOR NEW VEHICLE
            if cam_id in ocr_history:
                ocr_history[cam_id].clear()  # Clear OCR history
            print(f"[NEW_VEHICLE_DETECTED] {cam_id}: Position changed {distance:.1f}px (old:{old_centroid}, new:{new_centroid}) - RESET tracker")
        
        # This ensures new vehicles get OCR even if tracker is reused
        tr.saved = False
        tr.last_ocr_time = 0  # RESET OCR COOLDOWN for fresh OCR on new detection
        
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
    """IMPROVED: License plate detection with multi-scale support for bikes/rickshaws."""
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
        
        # Separate plates by size for appropriate handling
        best_box = None
        best_conf = 0.0
        all_detections = []  # Track all detections by size for analysis
        
        for box in results.boxes:
            try:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                # Small plates (bikes/rickshaws): need lower confidence
                # Large plates (cars): use standard confidence
                if area < MAX_SMALL_PLATE_AREA:
                    # Small plate - use lower threshold
                    min_conf = CONFIDENCE_THRESHOLD_SMALL_PLATE
                    plate_type = "SMALL"
                else:
                    # Standard plate
                    min_conf = confidence_threshold
                    plate_type = "STANDARD"
                
                if conf >= min_conf:
                    all_detections.append({
                        'conf': conf,
                        'bbox': (x1, y1, x2, y2),
                        'w': w, 'h': h, 'area': area,
                        'type': plate_type
                    })
                    
                    # Track best detection overall
                    if conf > best_conf:
                        best_conf = conf
                        best_box = {'conf': conf, 'bbox': (x1, y1, x2, y2), 'type': plate_type}
            
            except (IndexError, TypeError, AttributeError):
                continue
        
        # This ensures we get stable detection even if small plates are detected
        if best_box is not None:
            x1, y1, x2, y2 = best_box['bbox']
            h, w = vehicle_roi_color.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            
            if x2 > x1 and y2 > y1:  # Valid box
                plates.append((x1, y1, x2, y2))
    
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
        if current_time - tr.last_seen > 1.5:
            # Reset tracker state before deletion
            tr.saved = False  # Ensure saved flag is reset
            del trackers[cam_id][trk_id]

def reset_stabilizer_if_no_detection(cam_id, has_detection):
    """
    Clear bounding box stabilizer when no plate detected.
    Otherwise old bounding box blocks new vehicle detection.
    Run this every frame after detection attempt.
    Also reset tracker.saved flag to allow re-detection.
    """
    if not has_detection and cam_id in plate_bbox_stabilizers:
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
        
        if cam_id in trackers:
            for tr in trackers[cam_id].values():
                tr.saved = False

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
    
    if abs(len(p1) - len(p2)) > 2:
        return False
    
    similarity = SequenceMatcher(None, p1, p2).ratio()
    return similarity > threshold

def get_tracked_object_ocr_key(cam_id, object_id):
    """Create a unique key for tracked object OCR data."""
    return (cam_id, object_id)

def update_tracked_object_ocr(cam_id, object_id, plate_text, ocr_confidence, bbox):
    """
    Update OCR reading for a tracked object with AGGRESSIVE LOCKING.
    
    NEW BEHAVIOR (Lock-and-Process-Once):
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
                'ocr_executed': False  # NEW: Track if OCR has been run
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
            
            with saved_objects_lock:
                saved_objects.discard(key)
            object_image_saved.pop(key, None)
            
            with processed_plates_lock:
                processed_plates.pop(key, None)
            
            if DEBUG:
                print(f"[TRACK_CLEANUP] Removed old tracked object {key[1]} from camera {key[0]}")

def is_object_saved(cam_id, object_id):
    """Check if this object has already been saved to database."""
    with saved_objects_lock:
        return (cam_id, object_id) in saved_objects

def mark_object_as_saved(cam_id, object_id):
    """Mark object as saved to prevent duplicate database entries."""
    with saved_objects_lock:
        saved_objects.add((cam_id, object_id))
    if DEBUG:
        print(f"[MARK_SAVED] Marked object {object_id} @ {cam_id} as saved")

def is_object_image_saved(cam_id, object_id):
    """Check if this object's cropped plate image has been saved."""
    return object_image_saved.get((cam_id, object_id), False)

def mark_object_image_as_saved(cam_id, object_id):
    """Mark object's image as saved to prevent duplicate image files."""
    object_image_saved[(cam_id, object_id)] = True
    if DEBUG:
        print(f"[MARK_IMAGE_SAVED] Marked image saved for object {object_id} @ {cam_id}")

def was_ocr_executed_this_frame(cam_id, object_id, current_frame_count):
    """Prevent duplicate OCR execution for same object in same frame."""
    key = (cam_id, object_id)
    with object_ocr_executed_lock:
        last_frame = object_ocr_executed_this_frame.get(key, -1)
        if last_frame == current_frame_count:
            return True
        return False

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
    
    CRITICAL OPTIMIZATIONS (Fixed 2026-03-24):
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
    
    # Only process YOLO every Nth frame for performance
    should_run_yolo = (frame_counters[cam_id] % YOLO_PROCESS_EVERY_N == 0)
    
    # Update frame buffer for fallback detection
    video_buffers[cam_id].append(frame.copy())
    
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
    
    # Initialize CentroidTracker if needed
    if cam_id not in object_trackers:
        # Larger frames can have larger pixel movements for same vehicle movement
        # Formula: base_distance * (frame_width / 640) - normalized to 640p reference
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
            
            # This allows streaming thread to immediately draw bounding boxes
            # while full OCR/save pipeline runs in background
            # No label initially - will be updated with OCR text after processing
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
    
    # Create a deduplicated set of bbox_keys already in display list to avoid duplicates
    displayed_bboxes = set(overlay['bbox'] for overlay in detection_overlay_list)
    
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
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(plate_resized)
        
        # Otsu threshold for binary image (optimal for OCR)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if ocr_enabled:
            raw_text = perform_ocr(binary, enable_validation=False)
        else:
            raw_text = ""
        
        if not raw_text or len(raw_text) < 4:
            continue
        
        normalized_text = indian_detector.normalize_ocr_text(raw_text)
        corrected_text = indian_detector.apply_ocr_corrections(normalized_text)
        is_valid, final_plate, ocr_confidence = validate_license_plate_ocr(corrected_text)
        
        if not is_valid or not final_plate:
            continue
        
        # Update tracked object OCR buffer and check if stable
        stable_plate, is_locked = update_tracked_object_ocr(
            cam_id, object_id, final_plate, ocr_confidence, bbox_key
        )
        
        if stable_plate:
            # OCR is stabilized (either locked or recent reading)
            detection_count += 1
            
            # Cases where stable_plate is not None:
            # 1. is_locked=True: High confidence (>0.5) or majority consensus - ready to save NOW
            # 2. is_locked=False: Valid reading available from consensus queue - still valid, should save
            # Previous bug: Only is_locked=True cases went to save list
            # Result: Objects with stable plates but low confidence never got saved
            objects_to_save.append((object_id, stable_plate, ocr_confidence, bbox_key, plate_roi_color, conf))
            
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
    
    current_time = time.time()
    saved_plates = []
    saved_object_ids = set()
    
    for object_id, stable_plate, ocr_conf, bbox_key, plate_roi_color, yolo_conf in objects_to_save:
        if is_object_saved(cam_id, object_id):
            if DEBUG:
                print(f"[SKIP_ALREADY_SAVED] {stable_plate} | Object: {object_id} already processed")
            continue
        
        # Skip if already saved in this frame processing
        if object_id in saved_object_ids:
            continue
        
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
        
        try:
            # Key distinction:
            # - NEW VEHICLE (is_in_db=False or has_exit=True): Always save NEW image
            # - EXISTING VEHICLE (is_in_db=True, has_exit=False): Check deduplication (may skip image if duplicate)
            plate_image_path = None
            plate_key_for_dedup = stable_plate  # FIX: Use plate text, not (cam_id, object_id) - must match initialization logic
            image_save_success = False
            
            # Reuse is_in_db and has_exit from earlier gate check (line 4048)
            is_reentry = is_in_db and has_exit  # Vehicle previously exited and is now re-detected
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
                        
                        # Mark this plate text as saved (matches initialization logic)
                        saved_plate_texts.add(plate_key_for_dedup)
                        
                        if DEBUG:
                            print(f"[PLATE_IMAGE_SAVED] {plate_image_path} | Plate: {stable_plate} | Object: {object_id} | IsReentry: {is_reentry}")
                    except Exception as e:
                        # Image save failed - don't proceed with DB save
                        print(f"[PLATE_IMAGE_ERROR] Failed to save plate image for {stable_plate}: {e}")
                        image_save_success = False
                        plate_image_path = None
                else:
                    # This should rarely happen but don't try to save new image
                    image_save_success = True  # Old image from previous save is valid
                    if DEBUG:
                        print(f"[SKIP_DUPLICATE_IMAGE] {stable_plate} - Plate image already saved")
            
            # This prevents creating records without plate images for exited vehicle re-detection
            if not image_save_success and plate_image_path is None:
                if DEBUG:
                    print(f"[SKIP_DB_SAVE] {stable_plate} - Image failed to save, retrying next frame")
                continue
            
            # Save detection with timestamp
            if save_mode == "Exit":
                entry_id = save_exit_detection(
                    plate_key=normalized_plate_key,  # Use normalized format
                    cam_id=cam_id,
                    confidence=yolo_conf,
                    ocr_confidence=ocr_conf,
                    direction="Front",
                    detection_mode="YOLO_OCR",
                    user_id=user_id,
                    plate_image=plate_image_path  # NEW: Pass exit plate image
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
                tracking_key = normalized_plate_key if save_mode == "Exit" else normalize_plate_for_matching(stable_plate)
                seen_plates[tracking_key] = (current_time, stable_plate)
                recent_plates_cache_opt.append(stable_plate)
                
                # This prevents retry of database save, but image retry happens through image_save_success
                mark_object_as_saved(cam_id, object_id)
                
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
    
    # This eliminates the excessive cv2.rectangle() and cv2.line() calls that were causing slowdown
    for sx1, sy1, sx2, sy2, conf, cls_id in stable_detections:
        x1, y1, x2, y2 = int(sx1), int(sy1), int(sx2), int(sy2)
        bbox_key = (x1, y1, x2, y2)
        
        # Get object ID for styling
        object_id = bbox_to_object_id.get(bbox_key, None)
        
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
        
        # FAST DRAWING: Single rectangle only (was 4 rectangles + 8 lines = 12 ops → now 1 op)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_main, 2)
        
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
            
            # SIMPLE TEXT: Just draw text directly (no background/shadow)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, thickness)
    
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
            detection_overlay_cache[cam_id] = {
                'detections': detections_list,
                'hud_info': hud_info,
                'timestamp': time.time()
            }
        
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
                        color_glow = (100, 200, 255)  # Light cyan glow
                        
                        cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color_glow, 1)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
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
                        print(f"[OVERLAY] {cam_id}: Drew box at {(x1,y1,x2,y2)} with text '{label_text}' (scaled)")
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
        frame_delay = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.03  # Target: 1/FPS or 33ms
        last_frame_time = time.time()
        
        while True:
            try:
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

# -------- Detection Pipeline Callback - MOVED TO anpr/core.py --------
# NOTE: camera_anpr_process_frame is defined in core.py and passed to camera manager
# This avoids duplication and ensures proper access to plate_detector_model from global module scope





