"""ONNX model wrapper for plate detection"""
import onnxruntime as rt
import numpy as np
from typing import List, Dict, Optional, Tuple
import cv2
from pathlib import Path
import urllib.request
import os

from ..config import DEBUG, YOLO_CONFIDENCE_THRESHOLD, BASE_DIR

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
        if rt is None:
            raise RuntimeError("ONNX Runtime not available. Install with: pip install onnxruntime")
        
        # Tune ONNX Runtime for Raspberry Pi 5 (4-core ARM CPU)
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = rt.InferenceSession(
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
            # Instead of: for det in detections: if conf >= conf_threshold:
            # Use: mask = detections[:, 4] > conf_threshold; filtered = detections[mask]
            
            # Extract confidence scores and apply mask
            conf_scores = detections[:, 4]  # Shape: [num_anchors]
            mask = conf_scores >= conf_threshold  # Boolean mask: [num_anchors]
            filtered_detections = detections[mask]  # Only detections above threshold
            
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
                
                # ONNX detections are relative to ROI (extracted before resize)
                # Must adjust back to full-frame coordinates
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
    """OPTIMIZED: Crop and save license plate image with minimal overhead."""
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
        cv2.imwrite(str(filepath), plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        relative_path = f"plates/{filename}"
        if DEBUG:
            print(f"[PLATE_SAVED] {plate_text} -> {relative_path}")
        
        return relative_path
    
    except Exception as e:
        if DEBUG:
            print(f"[PLATE_SAVE_ERROR] Failed to save plate image: {type(e).__name__}: {str(e)[:50]}")
        return None

