"""MJPEG streaming and frame serving for live video feeds"""
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, Optional, Generator
import threading
import time

from ..config import DEBUG, COLOR_CAPTURED, COLOR_TEXT_BG

def generate_stream(cam_id, user_id=None, camera_id=None):
    """Generate video stream with performance optimizations and robust error handling
    
    ★ CRITICAL FIX: Enhanced with deleted camera detection to prevent FFmpeg crashes
    """
    cap = camera_sources.get(cam_id)
    if cap is None:
        print(f"[ERROR] Camera '{cam_id}' not found in sources")
        return

    frame_queue = frame_queues.get(cam_id) if frame_queues else None
    if frame_queue is None:
        print(f"[ERROR] ✗ Frame queue NOT available for '{cam_id}'!")
        print(f"[ERROR] DETECTIONS DISABLED - Background processor not connected!")
        print(f"[ERROR] This means NO vehicles will be detected on live feed")
        print(f"[DEBUG] Global frame_queues: {frame_queues}")
        print(f"[DEBUG] Available cameras in queues: {list(frame_queues.keys()) if frame_queues else 'EMPTY'}")
    else:
        print(f"[INFO] Frame queue connected for '{cam_id}' - Detections ENABLED")

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
        quality = 55  # Reduced from 70 - faster encoding, still acceptable visual quality
        
        return [
            int(cv2.IMWRITE_JPEG_QUALITY), quality,  # Lower quality for faster encoding
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 0,  # Disable optimization
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 0  # Disable progressive JPEG
        ]

    while True:
        try:
            # This gives streaming time to stop gracefully if camera is being deleted
            try:
                cam_check = camera_sources.get(cam_id)
                if cam_check is None:
                    print(f"[STREAM] {cam_id}: Camera deleted during streaming, stopping generator")
                    return  # Exit gracefully
            except RuntimeError as e:
                # Dict changed size during iteration (concurrent deletion)
                print(f"[STREAM] {cam_id}: Dictionary modified during access (camera being deleted), stopping")
                return
            
            # If set to None, camera is being removed
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

            # The detection worker needs original 1280x720 frames for accurate YOLO inference
            # Resizing to 640x360 for streaming should NOT affect detection quality
            
            # SEND TO BACKGROUND PROCESSING QUEUE (non-blocking)
            # This allows heavy YOLO/OCR processing without blocking the streaming loop
            queue_check = frame_count % frame_processing_interval
            if frame_count <= 10:
                print(f"[FRAMECOUNT_DEBUG] {cam_id}: frame={frame_count}, interval={frame_processing_interval}, modulo={queue_check}, queueing={queue_check == 0}")
            
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
                        print(f"[QUEUE_DEBUG] {cam_id}: Queued FULL-RES frame {frames_queued} ({frame.shape[1]}x{frame.shape[0]}) (frame_count={frame_count}, queue_size={frame_queue.qsize()})")
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
                queue_status = "" if frame_queue else "✗"
                print(f"[STREAM] {cam_id}: Frame {frame_count:5d} | Queue:{queue_status} | Sent:{frames_queued:3d} frames to processor")

            # Always apply the cached overlay to ensure stable display
            try:
                # apply_detection_overlay() uses the cached overlay without blocking
                # This ensures the bounding box and text remain stable between frame updates
                
                # DEBUG: Track overlay application
                if frame_count <= 5 or frame_count % 100 == 1:
                    print(f"[DEBUG] generate_stream: {cam_id} applying overlay (frame {frame_count})")
                
                frame_with_overlay = apply_detection_overlay(frame, cam_id)
                
                if frame_count <= 5 or frame_count % 100 == 1:
                    print(f"[DEBUG] generate_stream: {cam_id} overlay applied (frame {frame_count})")
            except Exception as e:
                if DEBUG:
                    print(f"[WARN] Overlay error: {e}")
                frame_with_overlay = frame
            
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            except Exception as e:
                print(f"[INIT_CAMERA] ⚠️  Could not optimize USB camera properties: {e}")
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INIT_CAMERA] USB camera verified (attempt {attempt + 1})")
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            except Exception as e:
                print(f"[INIT_CAMERA] ⚠️  Could not optimize IP Webcam properties: {e}")
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INIT_CAMERA] IP Webcam verified (attempt {attempt + 1})")
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
                        print(f"[INIT_CAMERA] OAK camera verified (frame shape: {frame.shape})")
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
                
                # These fields are set via the API and should not be overwritten
                for field in fields_to_preserve:
                    if field in existing_camera and field not in camera:
                        camera[field] = existing_camera[field]
            
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
        
        print(f"[SYNC_CAMERAS] Synced {len(enabled_cameras)} camera(s) to cameras.json (all extended fields preserved)")
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
        
        config_manager.config = config_manager._load_config()
        
        # This respects the persistent camera state set by users via the UI
        enabled_cameras = config_manager.get_cameras_to_start()
        newly_loaded = 0
        loaded_camera_ids = []
        config_camera_ids = set()
        
        # Step 1: Get all camera IDs from current config
        for camera_config in enabled_cameras:
            config_camera_ids.add(camera_config['id'])
        
        cameras_in_memory = set(camera_sources.keys())
        deleted_cameras = cameras_in_memory - config_camera_ids
        
        if deleted_cameras:
            print(f"\n[RELOAD_CAMERAS] ★ DETECTED {len(deleted_cameras)} DELETED CAMERA(S)")
            for cam_id in deleted_cameras:
                print(f"[RELOAD_CAMERAS] Stopping deleted camera: '{cam_id}'")
                try:
                    success, msg = stop_camera_pipeline_runtime(cam_id)
                    if success:
                        print(f"[RELOAD_CAMERAS] Stopped and cleaned up '{cam_id}'")
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
                print(f"[RELOAD_CAMERAS] Camera '{cam_id}' already active")
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
                    
                    # This allows detection to start immediately when source comes online
                    try:
                        # Create queue for when camera comes online
                        if cam_id not in frame_queues:
                            frame_queues[cam_id] = queue.Queue(maxsize=MAX_FRAME_QUEUE)
                            print(f"[RELOAD_CAMERAS] Frame queue created for '{cam_id}' (detection ready)")
                        
                        # Create stop event for future pipeline
                        if cam_id not in processing_stop_events:
                            processing_stop_events[cam_id] = threading.Event()
                            print(f"[RELOAD_CAMERAS] Stop event created for '{cam_id}'")
                        
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
                        print(f"[RELOAD_CAMERAS] Camera '{cam_id}' added (DISCONNECTED - detection ready when source comes online)")
                        loaded_camera_ids.append(cam_id)
                        newly_loaded += 1
                    except Exception as add_err:
                        print(f"[RELOAD_CAMERAS] ✗ Failed to add disconnected camera: {str(add_err)[:100]}")
                    continue
                
                print(f"[RELOAD_CAMERAS] Camera source initialized, starting pipeline...")
                
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
                        print(f"[RELOAD_CAMERAS] Camera '{cam_id}' added (PIPELINE_ERROR - queues ready)")
                        loaded_camera_ids.append(cam_id)
                        newly_loaded += 1
                    except Exception as add_err:
                        print(f"[RELOAD_CAMERAS] ✗ Failed to add pipeline_error camera: {str(add_err)[:100]}")
                    continue
                
                print(f"[RELOAD_CAMERAS] CAMERA '{cam_id}' FULLY LOADED ")
                loaded_camera_ids.append(cam_id)
                newly_loaded += 1
                
                # This allows the preview endpoint to find the camera by source
                camera_source_map[source] = cam_id
                if source.lower() == 'oak':
                    camera_source_map['oak'] = cam_id
                print(f"[RELOAD_CAMERAS] Source mapping added: '{source}' -> '{cam_id}'")
            
            except Exception as e:
                print(f"[RELOAD_CAMERAS] ✗ Error loading camera '{cam_id}': {str(e)[:100]}")
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
                    print(f"[RELOAD_CAMERAS] Camera '{cam_id}' added (ERROR state - queues ready)")
                    loaded_camera_ids.append(cam_id)
                    newly_loaded += 1
                except Exception as add_err:
                    print(f"[RELOAD_CAMERAS] ✗ Failed to add error camera: {str(add_err)[:100]}")
                continue
        
        print(f"[RELOAD_CAMERAS] Done! Loaded {newly_loaded} new cameras. Total active: {len(camera_sources)}")
        
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
                            print(f"[CONFIG_WATCHER] Loaded {newly_loaded} new camera(s): {cam_ids}")
                        else:
                            print(f"[CONFIG_WATCHER] Config reloaded (total active: {len(camera_sources)} cameras)")
                    except Exception as e:
                        print(f"[CONFIG_WATCHER] ✗ Error during reload: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()
                
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
                                    print(f"[CONFIG_WATCHER] Source now accessible for '{cam_id}', promoting to active...")
                                    
                                    # Start the pipeline
                                    pipeline_success, msg = start_camera_pipeline_runtime(cam_id, camera_src, camera_type)
                                    
                                    if pipeline_success:
                                        print(f"[CONFIG_WATCHER] Camera '{cam_id}' is now ACTIVE and processing!")
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
    print("[INFO] Config file watcher thread started (1-second polling for real-time updates)")
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
        # frame_deduplicators[cam_id] = FrameDeduplicator()  # DISABLED
        plate_bbox_stabilizers[cam_id] = PlateBBoxStabilizer()
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
        
        # camera_manager created the queue, now share it with the global frame_queues
        if cam_id in camera_manager.frame_queues:
            frame_queues[cam_id] = camera_manager.frame_queues[cam_id]
        
        if cam_id in camera_manager.processing_stop_events:
            processing_stop_events[cam_id] = camera_manager.processing_stop_events[cam_id]
        
        print(f"[PIPELINE_RUNTIME] Camera '{cam_id}' registered in global camera_sources")
        print(f"[PIPELINE_RUNTIME] Frame queue synced for '{cam_id}' (detections enabled)")
        print(f"[PIPELINE_RUNTIME] Video stream now available at: /video_feed/{cam_id}")
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
        
        # This prevents generate_stream() from accessing the camera after it's deleted
        # Prevents: "Invalid stream index" FFmpeg assertion error
        if cam_id in frame_queues:
            print(f"[PIPELINE_RUNTIME] [STEP 0] Signaling streams to stop for '{cam_id}'")
            try:
                # Send stop signal through queue
                frame_queues[cam_id].put(None, timeout=0.5)
            except:
                pass
            # Now safe to mark as unavailable
            frame_queues[cam_id] = None  # Signal to stop reading
            time.sleep(0.3)  # Give streaming threads time to notice and stop
        
        print(f"[PIPELINE_RUNTIME] [STEP 1] Marking camera as unavailable")
        camera_sources[cam_id] = None  # Signal unavailable before full removal
        time.sleep(0.1)  # Brief delay before removal
        
        # This ensures streams see the stop signal and exit gracefully
        if cam_id in processing_stop_events:
            processing_stop_events[cam_id].set()
            print(f"[PIPELINE_RUNTIME] [STEP 1.5] Stop event SET for '{cam_id}' - active streams will exit gracefully")
            time.sleep(0.3)  # Give streams time to see the stop signal
        
        print(f"[PIPELINE_RUNTIME] [STEP 2] Removing from global registries")
        time.sleep(0.2)  # Wait for streams to notice camera is None/stop is set
        camera_sources.pop(cam_id, None)
        
        print(f"[PIPELINE_RUNTIME] [STEP 3] Stopping pipeline with camera manager")
        try:
            success, msg = camera_manager.remove_camera(cam_id)
            if success:
                print(f"[PIPELINE_RUNTIME] [STEP 3] Pipeline stopped: {msg}")
            else:
                print(f"[PIPELINE_RUNTIME] [STEP 3] ⚠️  Pipeline stop returned: {msg}")
        except Exception as e:
            print(f"[PIPELINE_RUNTIME] [STEP 3] ⚠️  Exception during pipeline stop: {str(e)[:80]}")
        
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
        
        print(f"[PIPELINE_RUNTIME] Camera '{cam_id}' pipeline stopped and cleaned up completely")
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

# Now that helper functions are defined, initialize the camera manager
# and wire up the config API with dynamic pipeline management callbacks

camera_manager = initialize_camera_manager()
print("[INFO] Camera Manager initialized for dynamic camera pipeline control")

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
            print(f"[VIDEO_FEED_PREVIEW] Found in camera_sources by ID: {source}")
        
        elif source in camera_source_map:
            cam_id = camera_source_map[source]
            cap = camera_sources.get(cam_id)
            print(f"[VIDEO_FEED_PREVIEW] Found in source_map: {source} -> {cam_id}")
        
        elif source_decoded in camera_source_map:
            cam_id = camera_source_map[source_decoded]
            cap = camera_sources.get(cam_id)
            print(f"[VIDEO_FEED_PREVIEW] Found in source_map (decoded): {source_decoded} -> {cam_id}")
        
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
                                print(f"[VIDEO_FEED_PREVIEW] Successfully opened OAK camera as: {cam_id}")
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
                            preview_timestamp = int(time.time() * 1000)  # milliseconds
                            cam_id = f"preview_ip_{preview_timestamp}_{source[:10]}"
                            print(f"[VIDEO_FEED_PREVIEW] Successfully opened IP stream as: {cam_id}")
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
                                preview_timestamp = int(time.time() * 1000)  # milliseconds
                                cam_id = f"preview_usb_{src_int}_{preview_timestamp}"
                                print(f"[VIDEO_FEED_PREVIEW] Successfully opened USB camera as: {cam_id}")
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

        # This ensures the pipeline is actually running before streaming
        actual_cam_id = None
        
        # First, check camera_manager registry if available
        if camera_manager:
            mgr_status = camera_manager.get_camera_status()
            active_cams = mgr_status.get('active_camera_ids', [])
            
            # Check if cam_id or decoded version is in active cameras
            if cam_id in active_cams:
                actual_cam_id = cam_id
                print(f"[VIDEO_FEED] FIX 3: Found camera in CameraManager registry: {cam_id}")
            elif cam_id_decoded in active_cams:
                actual_cam_id = cam_id_decoded
                print(f"[VIDEO_FEED] FIX 3: Found camera in CameraManager registry (decoded): {cam_id_decoded}")
        
        # Fallback: Check if cam_id exists directly in camera_sources
        if not actual_cam_id:
            if cam_id in camera_sources:
                actual_cam_id = cam_id
                print(f"[VIDEO_FEED] Found camera by ID: {cam_id}")
            
            # Tier 2: Check if decoded cam_id exists in camera_sources
            elif cam_id_decoded in camera_sources:
                actual_cam_id = cam_id_decoded
                print(f"[VIDEO_FEED] Found camera by decoded ID: {cam_id_decoded}")
            
            # Tier 3: Try reverse lookup via source_map (for URLs and indices)
            elif cam_id in camera_source_map:
                actual_cam_id = camera_source_map[cam_id]
                print(f"[VIDEO_FEED] Found camera via source_map: {cam_id} -> {actual_cam_id}")
            
            # Tier 4: Try reverse lookup with decoded ID
            elif cam_id_decoded in camera_source_map:
                actual_cam_id = camera_source_map[cam_id_decoded]
                print(f"[VIDEO_FEED] Found camera via source_map (decoded): {cam_id_decoded} -> {actual_cam_id}")
            
            # Tier 5: Handle special names that might map to cameras
            elif cam_id in ('webcam', 'preview', '0'):
                if 'webcam' in camera_sources:
                    actual_cam_id = 'webcam'
                    print(f"[VIDEO_FEED] Mapped '{cam_id}' to 'webcam'")
                elif camera_sources:
                    actual_cam_id = list(camera_sources.keys())[0]
                    print(f"[VIDEO_FEED] Mapped '{cam_id}' to first available: {actual_cam_id}")
        
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



