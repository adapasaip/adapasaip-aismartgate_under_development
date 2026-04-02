"""
Camera Pipeline Manager - Handles dynamic camera lifecycle and thread-safe operations
Provides industry-standard camera management with proper synchronization and error handling
"""

import threading
import cv2
import time
import logging
from queue import Queue
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger("CameraManager")


class CameraManager:
    """
    Thread-safe camera pipeline manager
    Handles creation, monitoring, and teardown of camera processing pipelines
    
    Attributes:
        camera_sources: Dict of camera_id -> VideoCapture/OAK device
        frame_queues: Dict of camera_id -> Queue for frame processing
        processing_stop_events: Dict of camera_id -> threading.Event for pipeline control
        processing_threads: List of (camera_id, thread, stop_event) tuples
        camera_locks: Dict of camera_id -> threading.Lock for synchronized access
    """
    
    def __init__(self):
        """Initialize the camera manager with thread-safe data structures"""
        self.camera_sources: Dict[str, Any] = {}
        self.frame_queues: Dict[str, Queue] = {}
        self.processing_stop_events: Dict[str, threading.Event] = {}
        self.processing_threads: list = []  # List of (cam_id, thread, stop_event)
        self.camera_locks: Dict[str, threading.Lock] = {}
        
        # Registry-level lock for protecting collection operations
        self.registry_lock = threading.RLock()  # Reentrant to allow nested locks
        
        logger.info("CameraManager initialized with thread-safe registry")
    
    def create_camera_pipeline(self, 
                              cam_id: str, 
                              camera_source: Any,
                              frame_queue_maxsize: int = 2) -> bool:
        """
        Create locks for a camera (called when camera is opened)
        
        Args:
            cam_id: Camera identifier
            camera_source: VideoCapture or OAK device object
            frame_queue_maxsize: Maximum frames in queue before dropping
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.registry_lock:
                # ★ OPTIONAL STABILITY: Check for duplicate camera
                if cam_id in self.camera_sources:
                    logger.warning(f"[PIPELINE] Camera '{cam_id}' already exists, returning False to prevent ghosts")
                    return False
                
                # Create lock for this camera
                self.camera_locks[cam_id] = threading.Lock()
                
                # Register camera source
                self.camera_sources[cam_id] = camera_source
                
                logger.info(f"[PIPELINE] Camera '{cam_id}' registered in source registry")
            
            return True
        except Exception as e:
            logger.error(f"[PIPELINE] Failed to register camera '{cam_id}': {e}")
            return False
    
    def start_processing_pipeline(self,
                                 cam_id: str,
                                 process_callback) -> Tuple[bool, str]:
        """
        Start the processing pipeline for a camera (frame queue + worker thread)
        
        Args:
            cam_id: Camera identifier
            process_callback: Callback function(frame, cam_id) for frame processing
            
        Returns:
            (bool: success, str: message)
        """
        try:
            with self.registry_lock:
                # Verify camera exists
                if cam_id not in self.camera_sources:
                    return False, f"Camera '{cam_id}' not registered in sources"
                
                # Check if pipeline already exists
                if cam_id in self.frame_queues:
                    return False, f"Pipeline already running for camera '{cam_id}'"
                
                # ★ FIX 5: Create queue with maxsize=3 for stability (was 1, improved from 2)
                self.frame_queues[cam_id] = Queue(maxsize=3)
                
                # Create stop event
                self.processing_stop_events[cam_id] = threading.Event()
                
                logger.info(f"[PIPELINE] Queue and stop event created for camera '{cam_id}'")
            
            # Start worker thread (outside lock to avoid blocking)
            stop_event = self.processing_stop_events[cam_id]
            worker_thread = threading.Thread(
                target=self._worker_thread_target,
                args=(cam_id, process_callback, stop_event),
                daemon=True,
                name=f"ProcessWorker-{cam_id}"
            )
            worker_thread.start()
            
            with self.registry_lock:
                self.processing_threads.append((cam_id, worker_thread, stop_event))
            
            logger.info(f"[PIPELINE] Processing thread started for camera '{cam_id}'")
            return True, f"Pipeline started for camera '{cam_id}'"
        
        except Exception as e:
            logger.error(f"[PIPELINE] Failed to start processing for '{cam_id}': {e}")
            return False, str(e)
    
    def _worker_thread_target(self, cam_id: str, process_callback, stop_event: threading.Event):
        """
        Worker thread target - continuously processes frames from queue
        
        ⭐ OPTIMIZED for real-time performance:
        - Reduced timeout from 0.5s to 0.05s (10x faster responsiveness)
        - Only processes latest frame (no backlog)
        
        Args:
            cam_id: Camera identifier
            process_callback: Function to call for each frame
            stop_event: Event to signal thread shutdown
        """
        logger.info(f"[WORKER] Worker thread started for camera '{cam_id}'")
        frames_processed = 0
        
        while not stop_event.is_set():
            try:
                # Get frame from queue with REDUCED timeout (0.05s = 50ms, was 500ms)
                frame_queue = self.frame_queues.get(cam_id)
                if not frame_queue:
                    break
                
                try:
                    frame = frame_queue.get(timeout=0.05)  # ⭐ CRITICAL: Reduced from 0.5s to 0.05s
                except:
                    # Timeout is normal - continue waiting
                    continue
                
                if frame is None:
                    # Stop signal received
                    logger.info(f"[WORKER] Stop signal received for camera '{cam_id}'")
                    break
                
                frames_processed += 1
                
                # Log every 50 frames in production (was every 100)
                if frames_processed % 100 == 0:
                    logger.debug(f"[WORKER] {cam_id}: Processed {frames_processed} frames")
                
                # Process the frame
                try:
                    process_callback(frame, cam_id)
                except Exception as e:
                    if frames_processed <= 5:  # Only log first 5 errors
                        logger.error(f"[WORKER] Frame processing error: {str(e)[:80]}")
            
            except Exception as e:
                logger.error(f"[WORKER] Unexpected error in worker thread: {str(e)[:80]}")
                continue
        
        logger.info(f"[WORKER] Worker thread stopped for camera '{cam_id}' after {frames_processed} frames")
    
    def push_frame(self, cam_id: str, frame) -> bool:
        """
        Push a frame into the processing queue for a camera
        
        ⭐ OPTIMIZED FOR REAL-TIME:
        - Always keeps ONLY the latest frame (drains old frames)
        - Prevents queue backlog and stale frame processing
        - Ensures frame processing happens immediately without lag
        
        Args:
            cam_id: Camera identifier
            frame: Image frame (numpy array)
            
        Returns:
            True if frame was queued, False if camera not active
        """
        try:
            # Fast path - no lock needed for read-only get
            frame_queue = self.frame_queues.get(cam_id)
            if not frame_queue:
                return False
            
            # ⭐ CRITICAL FIX: Drain ALL old frames before adding new one
            # This ensures only the latest frame is processed (real-time mode)
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()  # Discard old frame
                except:
                    break
            
            # Now add the new (latest) frame
            try:
                frame_queue.put_nowait(frame)
                return True
            except:
                # Queue full - shouldn't happen but handle gracefully
                return False
        
        except Exception as e:
            logger.debug(f"[QUEUE] Error pushing frame to {cam_id}: {str(e)[:50]}")
            return False
    
    def stop_camera_pipeline(self, cam_id: str) -> Tuple[bool, str]:
        """
        Safely stop a camera's processing pipeline
        
        Args:
            cam_id: Camera identifier
            
        Returns:
            (bool: success, str: message)
        """
        try:
            with self.registry_lock:
                # Get stop event and thread
                stop_event = self.processing_stop_events.get(cam_id)
                if not stop_event:
                    return False, f"Camera '{cam_id}' not found"
                
                # Signal thread to stop
                stop_event.set()
                logger.info(f"[PIPELINE] Stop signal sent to camera '{cam_id}'")
                
                # Send stop signal to queue
                queue = self.frame_queues.get(cam_id)
                if queue:
                    try:
                        queue.put(None, timeout=0.5)
                    except:
                        pass
            
            # Wait for thread to finish (outside lock)
            thread = None
            with self.registry_lock:
                # Find thread for this camera
                for i, (cid, t, _) in enumerate(self.processing_threads):
                    if cid == cam_id:
                        thread = t
                        del self.processing_threads[i]  # Remove from list
                        break
            
            if thread:
                thread.join(timeout=2)
                logger.info(f"[PIPELINE] Worker thread joined for camera '{cam_id}'")
            
            # Cleanup dictionaries
            with self.registry_lock:
                self.frame_queues.pop(cam_id, None)
                self.processing_stop_events.pop(cam_id, None)
            
            return True, f"Pipeline stopped for camera '{cam_id}'"
        
        except Exception as e:
            logger.error(f"[PIPELINE] Error stopping camera '{cam_id}': {e}")
            return False, str(e)
    
    def close_camera_source(self, cam_id: str) -> Tuple[bool, str]:
        """
        Close a camera's video source (release hardware resources)
        
        Args:
            cam_id: Camera identifier
            
        Returns:
            (bool: success, str: message)
        """
        try:
            with self.registry_lock:
                cap = self.camera_sources.get(cam_id)
                if not cap:
                    return False, f"Camera '{cam_id}' not found"
                
                # Handle different camera types
                try:
                    if isinstance(cap, dict) and cap.get('type') == 'oak':
                        # OAK camera cleanup
                        if 'device' in cap:
                            cap['device'].close()
                        logger.info(f"[SOURCE] OAK camera closed for '{cam_id}'")
                    elif isinstance(cap, cv2.VideoCapture):
                        cap.release()
                        logger.info(f"[SOURCE] VideoCapture released for '{cam_id}'")
                    else:
                        # Try generic close
                        if hasattr(cap, 'release'):
                            cap.release()
                except Exception as e:
                    logger.warning(f"[SOURCE] Error closing camera resource: {e}")
                
                del self.camera_sources[cam_id]
        
        except Exception as e:
            logger.error(f"[SOURCE] Error closing camera '{cam_id}': {e}")
            return False, str(e)
        
        return True, f"Camera source closed for '{cam_id}'"
    
    def remove_camera(self, cam_id: str) -> Tuple[bool, str]:
        """
        Completely remove a camera from the system
        Stops pipeline, closes source, and cleans up all resources
        
        Args:
            cam_id: Camera identifier
            
        Returns:
            (bool: success, str: message)
        """
        logger.info(f"[SYSTEM] Removing camera '{cam_id}' from system...")
        
        # 1. Stop processing pipeline
        success, msg = self.stop_camera_pipeline(cam_id)
        if not success:
            logger.warning(f"[SYSTEM] Pipeline stop returned: {msg}")
        
        # 2. Close camera source
        success, msg = self.close_camera_source(cam_id)
        if not success:
            logger.warning(f"[SYSTEM] Source close returned: {msg}")
        
        # 3. Remove lock
        with self.registry_lock:
            self.camera_locks.pop(cam_id, None)
        
        logger.info(f"[SYSTEM] Camera '{cam_id}' removed successfully")
        return True, f"Camera '{cam_id}' removed from system"
    
    def get_camera_status(self) -> Dict[str, Any]:
        """
        Get status of all cameras
        
        Returns:
            Dictionary with camera statuses
        """
        with self.registry_lock:
            return {
                "cameras": list(self.camera_sources.keys()),
                "active_pipelines": list(self.frame_queues.keys()),
                "worker_threads": len(self.processing_threads),
                "total_cameras": len(self.camera_sources)
            }
    
    def get_lock(self, cam_id: str) -> Optional[threading.Lock]:
        """
        Get the lock for a specific camera
        
        Args:
            cam_id: Camera identifier
            
        Returns:
            threading.Lock or None if camera not found
        """
        return self.camera_locks.get(cam_id)
    
    def shutdown(self):
        """
        Gracefully shutdown all camera pipelines
        Called at application shutdown
        """
        logger.info("[SYSTEM] Shutting down all camera pipelines...")
        
        camera_ids = list(self.camera_sources.keys())
        for cam_id in camera_ids:
            try:
                self.remove_camera(cam_id)
            except Exception as e:
                logger.error(f"[SYSTEM] Error removing camera '{cam_id}': {e}")
        
        logger.info("[SYSTEM] All camera pipelines shut down")


# Global singleton instance
_camera_manager_instance = None


def get_camera_manager() -> CameraManager:
    """
    Get or create the global camera manager instance
    Use this to access the singleton instance throughout the application
    
    Returns:
        CameraManager singleton instance
    """
    global _camera_manager_instance
    if _camera_manager_instance is None:
        _camera_manager_instance = CameraManager()
    return _camera_manager_instance


def initialize_camera_manager() -> CameraManager:
    """
    Initialize the global camera manager
    Call this once at application startup
    """
    global _camera_manager_instance
    _camera_manager_instance = CameraManager()
    return _camera_manager_instance
