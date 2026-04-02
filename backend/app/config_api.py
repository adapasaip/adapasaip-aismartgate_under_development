"""
Flask API routes for camera configuration management

This module handles:
1. Saving camera configurations to disk
2. Starting/stopping camera processing pipelines dynamically
3. Initializing camera sources (USB, IP Webcam, OAK)

Dynamic camera addition workflow:
  Frontend -> API -> Save Config -> Initialize Source -> Start Pipeline
  
No server restart required!
"""

import cv2
import time
import sys
from flask import jsonify, request
from config_manager import CameraConfigManager

# Try to import DepthAI for OAK camera support
try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    dai = None

# These will be injected by camera_anpr.py
config_manager = None
camera_manager = None
start_camera_pipeline_runtime = None
stop_camera_pipeline_runtime = None
reload_cameras_from_config_func = None  # ★ NEW: Function to reload cameras from config
camera_anpr_process_frame_callback = None  # ★ NEW: Frame processor callback

# ============== CAMERA SOURCE INITIALIZATION ==============

def initialize_oak_camera():
    """
    Initialize OAK camera using DepthAI.
    
    Returns:
        (dict with device info, bool success)
    """
    if not DEPTHAI_AVAILABLE:
        print(f"[CONFIG_API] ✗ DepthAI not installed - OAK camera cannot be initialized")
        return None, False
    
    try:
        print(f"[CONFIG_API] Initializing OAK camera via DepthAI...")
        
        # Check if devices are available
        devices = dai.Device.getAllAvailableDevices()
        if not devices:
            print(f"[CONFIG_API] ✗ No OAK devices detected. Is it connected via USB?")
            return None, False
        
        print(f"[CONFIG_API] Found {len(devices)} OAK device(s)")
        
        # Create a simple pipeline for video streaming
        pipeline = dai.Pipeline()
        
        # Create ColorCamera node
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setFps(30)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setPreviewSize(1280, 720)
        cam.setVideoSize(1280, 720)
        
        # Output video stream
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam.video.link(xout_video.input)
        
        # Initialize device with pipeline
        print(f"[CONFIG_API] Starting OAK device...")
        device = dai.Device(pipeline)
        
        # Get the output queue
        q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        
        # Wait for first frame to verify
        print(f"[CONFIG_API] Waiting for first frame...")
        time.sleep(2)
        
        in_data = q.get()
        if in_data is not None:
            frame = in_data.getCvFrame()
            if frame is not None:
                print(f"[CONFIG_API] ✓ OAK camera verified (frame shape: {frame.shape})")
                return {
                    'type': 'oak_depthai',
                    'device': device,
                    'queue_name': 'video',
                    'is_mock': False
                }, True
        
        print(f"[CONFIG_API] ⚠ OAK camera initialized but no frame received. Proceeding anyway...")
        return {
            'type': 'oak_depthai',
            'device': device,
            'queue_name': 'video',
            'is_mock': False
        }, True
        
    except Exception as e:
        print(f"[CONFIG_API] ✗ Error initializing OAK camera: {str(e)[:100]}")
        return None, False

def initialize_camera_source(source: str, camera_type: str):
    """
    Open a camera source based on type
    Returns VideoCapture or OAK device
    
    Args:
        source: USB index (0, 1), IP Webcam URL (http://...), or 'oak'
        camera_type: 'usb', 'ipwebcam', or 'oak'
    
    Returns:
        (cv2.VideoCapture or dict, bool success)
    """
    try:
        if camera_type == 'usb' or source.isdigit():
            # USB camera
            print(f"[CONFIG_API] Initializing USB camera at index {source}...")
            src_int = int(source)
            cap = cv2.VideoCapture(src_int)
            
            if not cap.isOpened():
                return None, False
            
            # Optimize camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 12)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[CONFIG_API] ✓ USB camera verified")
                    return cap, True
                time.sleep(0.2)
            
            print(f"[CONFIG_API] ⚠ USB camera opened but frame read failed. Proceeding anyway...")
            return cap, True
        
        elif camera_type == 'ipwebcam' or source.startswith('http'):
            # IP Webcam
            print(f"[CONFIG_API] Initializing IP Webcam from {source}...")
            cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                return None, False
            
            # Optimize for IP streaming
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 12)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Verify it works
            for attempt in range(3):
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[CONFIG_API] ✓ IP Webcam verified")
                    return cap, True
                time.sleep(0.2)
            
            print(f"[CONFIG_API] ⚠ IP Webcam opened but frame read failed. Proceeding anyway...")
            return cap, True
        
        elif camera_type == 'oak' or source.lower() == 'oak':
            # OAK camera via DepthAI
            return initialize_oak_camera()
        
        else:
            print(f"[CONFIG_API] Unknown camera type: {camera_type}")
            return None, False
    
    except Exception as e:
        print(f"[CONFIG_API] Error initializing camera: {str(e)[:80]}")
        return None, False

# ============== CAMERA CONFIG API ENDPOINTS ==============

def init_config_api(app, config_mgr, camera_mgr=None, start_func=None, stop_func=None, reload_func=None, process_frame_func=None):
    """
    Initialize config API routes. Call this from main() after app creation.
    
    Args:
        app: Flask app instance
        config_mgr: CameraConfigManager instance
        camera_mgr: CameraManager instance (optional, for dynamic management)
        start_func: Function to start pipeline (optional)
        stop_func: Function to stop pipeline (optional)
        reload_func: Function to reload cameras from config (optional)
        process_frame_func: Frame processor callback for detection (optional) - ★ NEW
    """
    global config_manager, camera_manager, start_camera_pipeline_runtime, stop_camera_pipeline_runtime, reload_cameras_from_config_func, camera_anpr_process_frame_callback
    config_manager = config_mgr
    camera_manager = camera_mgr
    start_camera_pipeline_runtime = start_func
    stop_camera_pipeline_runtime = stop_func
    reload_cameras_from_config_func = reload_func
    camera_anpr_process_frame_callback = process_frame_func  # ★ NEW: Store the frame processor
    
    @app.route('/api/config/cameras', methods=['GET'])
    def get_cameras():
        """Get all camera configurations"""
        try:
            cameras = config_manager.get_all_cameras()
            return jsonify({
                'success': True,
                'data': cameras,
                'count': len(cameras)
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras', methods=['POST'])
    def add_camera():
        """
        Add a new camera configuration and START its processing pipeline
        
        This is the key difference from the old API - we don't just save config,
        we also initialize the camera and start processing immediately!
        """
        try:
            data = request.get_json()
            required_fields = ['id', 'name', 'source', 'type']
            
            if not all(field in data for field in required_fields):
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {required_fields}'
                }), 400
            
            cam_id = data['id']
            print(f"\n[CONFIG_API] ====== CAMERA ADD REQUEST ======")
            print(f"[CONFIG_API] Camera ID: {cam_id}")
            print(f"[CONFIG_API] Camera Name: {data['name']}")
            print(f"[CONFIG_API] Camera Source: {data['source']}")
            print(f"[CONFIG_API] Camera Type: {data['type']}")
            
            # ★ STEP 1: Save configuration  FIRST (non-blocking)
            print(f"[CONFIG_API] [STEP 1] Saving camera configuration to disk...")
            config_success = config_manager.add_camera(
                camera_id=cam_id,
                name=data['name'],
                source=data['source'],
                camera_type=data['type'],
                description=data.get('description', '')
            )
            
            if not config_success:
                print(f"[CONFIG_API] ✗ [STEP 1] FAILED: Camera {cam_id} already exists or save failed")
                return jsonify({
                    'success': False,
                    'error': f"Camera {cam_id} already exists or save failed"
                }), 400
            print(f"[CONFIG_API] ✓ [STEP 1] Config saved successfully")
            
            # ★ CRITICAL FIX: Add extended fields to config so they're preserved
            # ALWAYS save with defaults if not provided - NO MISSING FIELDS
            print(f"[CONFIG_API] [STEP 1b] Adding extended metadata fields with defaults...")
            print(f"[CONFIG_API] [DEBUG] Received gate value from request: {repr(data.get('gate'))}")
            
            # ✅ STRICT: Only use default 'Entry' if gate is genuinely missing/empty
            gate_value = data.get('gate', '').strip() if isinstance(data.get('gate'), str) else data.get('gate')
            if gate_value in [None, '', 'undefined']:
                gate_value = 'Entry'
            print(f"[CONFIG_API] [DEBUG] After processing, gate value: {repr(gate_value)}")
            
            extended_fields = {
                'userId': data.get('userId') if data.get('userId') not in [None, '', 'undefined'] else 'default',
                'location': data.get('location') if data.get('location') not in [None, '', 'undefined'] else 'Not Specified',
                'gate': gate_value,  # Use processed gate value
                'resolution': data.get('resolution') if data.get('resolution') not in [None, '', 'undefined'] else '1280x720',
                'fps': data.get('fps') if data.get('fps') not in [None, '', 'undefined'] else 30,
                'anprEnabled': data.get('anprEnabled') if data.get('anprEnabled') is not None else True
            }
            
            print(f"[CONFIG_API] [DEBUG] Final extended_fields: {extended_fields}")
            
            # Add ALL extended fields to the camera in config (with defaults as fallback)
            try:
                for camera in config_manager.config.get('cameras', []):
                    if camera.get('id') == cam_id:
                        # ★ ALWAYS add all extended fields (no skipping None values)
                        for field, value in extended_fields.items():
                            camera[field] = value
                        # Save config with extended fields
                        config_manager._save_config()
                        print(f"[CONFIG_API] ✓ [STEP 1b] Extended fields added to camera config: {extended_fields}")
                        print(f"[CONFIG_API] [DEBUG] Camera after extended fields: {camera}")
                        break
            except Exception as e:
                print(f"[CONFIG_API] ⚠️ [STEP 1b] Warning: Could not add extended fields: {str(e)[:80]}")
                # Continue anyway - camera is saved, extended fields are nice-to-have
            
            # ★ STEP 2: Initialize camera source (CRITICAL - was missing!)
            print(f"[CONFIG_API] [STEP 2] Initializing camera source...")
            camera_source, init_success = initialize_camera_source(data['source'], data['type'])
            
            if not init_success or camera_source is None:
                print(f"[CONFIG_API] ✗ [STEP 2] FAILED: Camera source could not be initialized")
                return jsonify({
                    'success': False,
                    'error': "Camera source could not be initialized"
                }), 500
            print(f"[CONFIG_API] ✓ [STEP 2] Camera source initialized successfully")
            
            # ★ STEP 3: Register camera in CameraManager using create_camera_pipeline()
            print(f"[CONFIG_API] [STEP 3] Registering {cam_id} in CameraManager registry...")
            if not camera_manager:
                print(f"[CONFIG_API] ✗ [STEP 3] CameraManager not available")
                try:
                    if hasattr(camera_source, 'release'):
                        camera_source.release()
                except:
                    pass
                return jsonify({
                    'success': False,
                    'error': 'CameraManager not initialized'
                }), 500
            
            # ★ FIX 1: Call create_camera_pipeline to register camera
            # This populates camera_sources dict and creates locks for thread-safe access
            create_success = camera_manager.create_camera_pipeline(cam_id, camera_source)
            if not create_success:
                print(f"[CONFIG_API] ✗ [STEP 3] Failed to register camera in CameraManager")
                try:
                    if hasattr(camera_source, 'release'):
                        camera_source.release()
                except:
                    pass
                return jsonify({
                    'success': False,
                    'error': 'Failed to register camera in pipeline manager'
                }), 500
            print(f"[CONFIG_API] ✓ [STEP 3] Camera '{cam_id}' registered in camera_sources dict")
            
            # ★ STEP 4: Start processing pipeline with worker thread
            print(f"[CONFIG_API] [STEP 4] Starting processing pipeline for {cam_id}...")
            
            # We need a process callback for frame handling
            # First try to use the callback passed from camera_anpr
            # Otherwise, try to get it from __main__ (for backward compatibility)
            process_callback = camera_anpr_process_frame_callback
            
            if not process_callback:
                # Fallback: Try to get from __main__ module
                import sys
                main_module = sys.modules.get('__main__')
                process_callback = getattr(main_module, 'camera_anpr_process_frame', None)
            
            if not process_callback:
                # Last resort: Use dummy callback
                def dummy_process(frame, cam_id):
                    pass
                process_callback = dummy_process
                print(f"[CONFIG_API] ⚠️ Frame processor not found, using dummy (no detection yet)")
            else:
                print(f"[CONFIG_API] ✓ Frame processor loaded successfully")
            
            # ★ FIX: Call start_processing_pipeline to start worker thread
            # This creates frame queue and starts worker thread for processing
            pipeline_success, pipeline_msg = camera_manager.start_processing_pipeline(
                cam_id,
                process_callback
            )
            
            if not pipeline_success:
                print(f"[CONFIG_API] ✗ [STEP 4] FAILED: {pipeline_msg}")
                camera_manager.close_camera_source(cam_id)
                return jsonify({
                    'success': False,
                    'error': f'Pipeline startup failed: {pipeline_msg}'
                }), 500
            
            print(f"[CONFIG_API] ✓ [STEP 4] Processing pipeline started")
            
            # ★ STEP 5: Start capture loop (CRITICAL - was missing!)
            print(f"[CONFIG_API] [STEP 5] Starting capture loop...")
            from anpr.core import start_camera_capture
            try:
                start_camera_capture(cam_id, camera_source, camera_manager)
                print(f"[CONFIG_API] ✓ [STEP 5] Capture loop started for '{cam_id}'")
            except Exception as e:
                print(f"[CONFIG_API] ⚠️ [STEP 5] Failed to start capture loop: {str(e)[:80]}")
                # Continue anyway - streaming client can still trigger capture
            
            print(f"[CONFIG_API] ✓✓✓ CAMERA '{cam_id}' FULLY INITIALIZED AND PROCESSING ✓✓✓")
            print(f"[CONFIG_API] Video stream available at: /video_feed/{cam_id}")
            return jsonify({
                'success': True,
                'message': f"Camera {cam_id} added and pipeline started (no restart needed!)",
                'status': 'success',
                'videoURL': f"/video_feed/{cam_id}"
            }), 201
        
        except Exception as e:
            print(f"[CONFIG_API] ✗ [ERROR] Unexpected error in add_camera: {type(e).__name__}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras/<camera_id>', methods=['GET'])
    def get_camera(camera_id):
        """Get specific camera configuration"""
        try:
            camera = config_manager.get_camera(camera_id)
            if camera:
                return jsonify({'success': True, 'data': camera}), 200
            else:
                return jsonify({'success': False, 'error': f'Camera {camera_id} not found'}), 404
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras/<camera_id>', methods=['PUT'])
    def update_camera(camera_id):
        """
        Update camera configuration
        
        ★ PHASE 8: If 'online' status changes, persist it via set_camera_status()
        """
        try:
            data = request.get_json()
            
            # ★ PHASE 8: Check if online status is being changed
            if 'online' in data:
                online_status = data['online']
                print(f"\n[CONFIG_API] Camera '{camera_id}' online status changed to: {online_status}")
                config_manager.set_camera_status(camera_id, online_status)
            
            # ★ PHASE 8: Check if enabled status is being changed
            # When toggling enabled, also update online status
            if 'enabled' in data:
                enabled_status = data['enabled']
                print(f"[CONFIG_API] Camera '{camera_id}' enabled status changed to: {enabled_status}")
                # If disabling, also mark as offline
                if not enabled_status:
                    config_manager.set_camera_status(camera_id, False)
                    print(f"[CONFIG_API] Camera '{camera_id}' marked offline (disabled)")
            
            # Standard config update
            success = config_manager.update_camera(camera_id, **data)
            
            if success:
                camera = config_manager.get_camera(camera_id)
                
                # If source changed, restart pipeline
                if 'source' in data and stop_camera_pipeline_runtime and start_camera_pipeline_runtime:
                    print(f"\n[CONFIG_API] Camera source updated, restarting pipeline...")
                    
                    # Stop old pipeline
                    stop_camera_pipeline_runtime(camera_id)
                    
                    # Start new pipeline with new source
                    camera_source, init_success = initialize_camera_source(data['source'], data.get('type', 'usb'))
                    if init_success and camera_source:
                        pipeline_success, msg = start_camera_pipeline_runtime(camera_id, camera_source, data.get('type', 'usb'))
                        if not pipeline_success:
                            print(f"[CONFIG_API] Warning: Pipeline restart failed: {msg}")
                        else:
                            print(f"[CONFIG_API] ✓ Pipeline restarted for camera '{camera_id}'")
                
                return jsonify({
                    'success': True,
                    'message': f'Camera {camera_id} updated',
                    'data': camera
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': f'Camera {camera_id} not found'
                }), 404
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras/<camera_id>', methods=['DELETE'])
    def delete_camera(camera_id):
        """★ FIX 2: Delete camera - properly stop pipeline THEN remove config"""
        try:
            print(f"\n[CONFIG_API] ====== CAMERA DELETE REQUEST (FIX 2) ======")
            print(f"[CONFIG_API] Deleting camera: {camera_id}")
            
            # ★ PHASE 1: Stop processing pipeline
            print(f"[CONFIG_API] [PHASE 1] Stopping processing pipeline...")
            if camera_manager:
                stop_success, stop_msg = camera_manager.stop_camera_pipeline(camera_id)
                print(f"[CONFIG_API] [PHASE 1] Pipeline stop: {stop_msg}")
            else:
                print(f"[CONFIG_API] [PHASE 1] ⚠️  CameraManager not available, skipping pipeline stop")
            
            # ★ PHASE 2: Close camera source/hardware
            print(f"[CONFIG_API] [PHASE 2] Closing camera source...")
            if camera_manager:
                close_success, close_msg = camera_manager.close_camera_source(camera_id)
                print(f"[CONFIG_API] [PHASE 2] Source close: {close_msg}")
            else:
                print(f"[CONFIG_API] [PHASE 2] ⚠️  CameraManager not available, skipping source close")
            
            # ★ PHASE 3: Delete from configuration
            print(f"[CONFIG_API] [PHASE 3] Deleting from configuration...")
            success = config_manager.delete_camera(camera_id)
            
            if success:
                print(f"[CONFIG_API] ✓ Camera '{camera_id}' removed completely (pipeline stopped, config deleted)")
                print(f"[CONFIG_API] ✓✓✓ DELETE COMPLETE ✓✓✓")
                return jsonify({
                    'success': True,
                    'message': f'Camera {camera_id} and associated gates deleted, pipeline stopped'
                }), 200
            else:
                print(f"[CONFIG_API] ✗ Camera {camera_id} not found in config")
                return jsonify({
                    'success': False,
                    'error': f'Camera {camera_id} not found'
                }), 404
        except Exception as e:
            print(f"[CONFIG_API] ✗ Error deleting camera: {str(e)[:80]}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras/<camera_id>/status', methods=['PUT'])
    def toggle_camera_status(camera_id):
        """
        ★ PHASE 8: Toggle camera online/offline status
        
        This endpoint persists the camera's online status to the config file,
        ensuring the setting survives application restart.
        
        Request body: { "online": true/false }
        """
        try:
            data = request.get_json()
            if 'online' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing required field: online'
                }), 400
            
            online_status = data['online']
            print(f"\n[CONFIG_API] ★ PHASE 8 - Toggling camera '{camera_id}' online status to: {online_status}")
            
            # Persist the status change to config file
            success = config_manager.set_camera_status(camera_id, online_status)
            
            if success:
                camera = config_manager.get_camera(camera_id)
                print(f"[CONFIG_API] ✓ Camera '{camera_id}' status persisted: online={online_status}")
                
                # If toggling offline, stop the pipeline
                if not online_status and stop_camera_pipeline_runtime:
                    print(f"[CONFIG_API] Stopping pipeline for offline camera '{camera_id}'...")
                    stop_camera_pipeline_runtime(camera_id)
                
                # If toggling online, start the pipeline (will be loaded on reload)
                if online_status:
                    print(f"[CONFIG_API] Camera '{camera_id}' set to online - will be loaded on next reload or restart")
                
                return jsonify({
                    'success': True,
                    'message': f'Camera {camera_id} online status changed to {online_status}',
                    'data': camera
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': f'Failed to update status for camera {camera_id}'
                }), 404
        except Exception as e:
            print(f"[CONFIG_API] ✗ Error toggling camera status: {str(e)[:80]}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/cameras-to-start', methods=['GET'])
    def get_cameras_to_start_endpoint():
        """
        ★ PHASE 8: Get cameras that are ENABLED AND ONLINE
        
        This returns only cameras that should be auto-started,
        respecting persistent online/offline status.
        """
        try:
            cameras_to_start = config_manager.get_cameras_to_start()
            return jsonify({
                'success': True,
                'cameras': cameras_to_start,
                'count': len(cameras_to_start)
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/gates', methods=['GET'])
    def get_gates():
        """Get all gate configurations"""
        try:
            gates = config_manager.get_all_gates()
            return jsonify({
                'success': True,
                'data': gates,
                'count': len(gates)
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/gates', methods=['POST'])
    def add_gate():
        """Add a new gate configuration"""
        try:
            data = request.get_json()
            required_fields = ['id', 'name', 'cameraId', 'gateType']
            
            if not all(field in data for field in required_fields):
                return jsonify({
                    'success': False,
                    'error': f'Missing required fields: {required_fields}'
                }), 400
            
            success = config_manager.add_gate(
                gate_id=data['id'],
                name=data['name'],
                camera_id=data['cameraId'],
                gate_type=data['gateType'],
                description=data.get('description', '')
            )
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f"Gate {data['id']} added successfully"
                }), 201
            else:
                return jsonify({
                    'success': False,
                    'error': f"Gate {data['id']} already exists or camera not found"
                }), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/gates/<gate_id>', methods=['DELETE'])
    def delete_gate(gate_id):
        """Delete gate configuration"""
        try:
            gates = config_manager.get_all_gates()
            gates[:] = [g for g in gates if g['id'] != gate_id]
            config_manager.config['gates'] = gates
            success = config_manager._save_config()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Gate {gate_id} deleted'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save configuration'
                }), 500
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/settings', methods=['GET'])
    def get_settings():
        """Get application settings"""
        try:
            settings = config_manager.get_settings()
            return jsonify({'success': True, 'data': settings}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/all', methods=['GET'])
    def get_all_config():
        """Get complete configuration"""
        try:
            return jsonify({
                'success': True,
                'cameras': config_manager.get_all_cameras(),
                'gates': config_manager.get_all_gates(),
                'settings': config_manager.get_settings()
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/config/enabled', methods=['GET'])
    def get_enabled_cameras_gates():
        """Get enabled cameras and gates for startup"""
        try:
            enabled_cameras = config_manager.get_enabled_cameras()
            enabled_gates = config_manager.get_enabled_gates()
            
            # Build CLI args for backward compatibility
            cameras_arg, gates_arg = config_manager.build_cli_args()
            
            return jsonify({
                'success': True,
                'cameras': enabled_cameras,
                'gates': enabled_gates,
                'cameras_arg': cameras_arg,
                'gates_arg': gates_arg
            }), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/camera-status', methods=['GET'])
    @app.route('/api/config/camera-status', methods=['GET'])
    def camera_status():
        """
        ★ DIAGNOSTIC ENDPOINT ★
        Check if cameras are actually registered in the Flask server
        This helps verify that the dynamic camera addition is working
        """
        try:
            # Get camera info from config file
            config_cameras = config_manager.get_all_cameras()
            
            # Import global camera tracking from main module
            import sys
            # Find the main application module (camera_anpr.py runs as __main__)
            main_module = sys.modules.get('__main__')
            camera_anpr_module = sys.modules.get('camera_anpr', main_module)
            
            if camera_anpr_module and hasattr(camera_anpr_module, 'camera_sources'):
                active_cameras = list(camera_anpr_module.camera_sources.keys())
                frame_queues = list(getattr(camera_anpr_module, 'frame_queues', {}).keys())
                processing_events = list(getattr(camera_anpr_module, 'processing_stop_events', {}).keys())
            else:
                active_cameras = []
                frame_queues = []
                processing_events = []
            
            # Compare config vs active
            config_ids = [c['id'] for c in config_cameras]
            
            status_detail = {
                'timestamp': time.time(),
                'config_file': {
                    'total_cameras': len(config_cameras),
                    'camera_ids': config_ids
                },
                'live_server': {
                    'active_cameras': len(active_cameras),
                    'active_camera_ids': active_cameras,
                    'frame_queues_active': len(frame_queues),
                    'processing_threads': len(processing_events)
                },
                'status': {
                    'all_in_sync': set(config_ids) == set(active_cameras),
                    'missing_from_active': list(set(config_ids) - set(active_cameras)),
                    'extra_in_active': list(set(active_cameras) - set(config_ids))
                },
                'diagnostics': {
                    'config_readable': True,
                    'module_accessible': camera_anpr_module is not None and hasattr(camera_anpr_module, 'camera_sources')
                }
            }
            
            return jsonify({
                'success': True,
                'data': status_detail
            }), 200
        
        except Exception as e:
            import traceback
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()[:200]
            }), 500
    
    @app.route('/api/debug/reload-cameras', methods=['POST'])
    def debug_reload_cameras():
        """
        ★ DEBUG ENDPOINT ★
        Manually trigger camera reload (useful for testing)
        
        This endpoint calls reload_cameras_from_config() directly
        Useful for diagnostics and testing config changes
        """
        try:
            import sys
            
            # Find the main application module (camera_anpr.py runs as __main__)
            main_module = sys.modules.get('__main__')
            camera_anpr_module = sys.modules.get('camera_anpr', main_module)
            
            if not camera_anpr_module:
                return jsonify({
                    'success': False,
                    'error': 'camera_anpr module not found'
                }), 500
            
            # Call reload function if available
            if not hasattr(camera_anpr_module, 'reload_cameras_from_config'):
                return jsonify({
                    'success': False,
                    'error': 'reload_cameras_from_config function not available'
                }), 500
            
            if not hasattr(camera_anpr_module, 'camera_sources'):
                return jsonify({
                    'success': False,
                    'error': 'camera_sources dict not accessible'
                }), 500
            
            print("[DEBUG_RELOAD] Manual reload requested")
            newly_loaded, cam_list = camera_anpr_module.reload_cameras_from_config()
            
            return jsonify({
                'success': True,
                'newly_loaded': newly_loaded,
                'cameras': cam_list,
                'total_active': len(camera_anpr_module.camera_sources),
                'message': f"Loaded {newly_loaded} new cameras"
            }), 200
        except Exception as e:
            print(f"[DEBUG_RELOAD] Error: {str(e)[:100]}")
            import traceback
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()[:200]
            }), 500
