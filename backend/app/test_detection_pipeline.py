#!/usr/bin/env python3.10
"""
Diagnostic script to test ONNX plate detection pipeline.
Directly tests the detection and compares with monolithic version.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the app directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from the modular code
from anpr.core import ONNXPlateDetector, ONNXResults, ONNXBoxes

print("=" * 70)
print("ANPR DETECTION PIPELINE DIAGNOSTIC")
print("=" * 70)

# Load the ONNX model
model_path = Path(__file__).parent / "license_plate_detector.onnx"
print(f"\nLoading model from: {model_path}")
print(f"Model exists: {model_path.exists()}")

if not model_path.exists():
    print(f"❌ ERROR: Model not found at {model_path}")
    sys.exit(1)

try:
    detector = ONNXPlateDetector(str(model_path))
    print(f"✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    sys.exit(1)

# Create a test image (white noise to see raw model behavior)
print("\n" + "=" * 70)
print("TEST 1: White Noise Image (should have low confidence)")
print("=" * 70)

test_image_noise = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
print(f"Test image shape: {test_image_noise.shape}")

try:
    results = detector(test_image_noise, conf=0.45)
    print(f"Model returned: {type(results)}")
    print(f"Results length: {len(results)}")
    
    if len(results) > 0:
        result = results[0]
        print(f"Result type: {type(result)}")
        print(f"Result has boxes: {hasattr(result, 'boxes')}")
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print(f"Boxes type: {type(boxes)}")
            print(f"Boxes length: {len(boxes)}")
            
            if len(boxes) == 0:
                print("⚠️  WARNING: No detections in white noise (expected)")
            else:
                for i, box in enumerate(boxes):
                    print(f"  Box {i}: {box}")
        else:
            print("❌ ERROR: Results don't have boxes attribute")
    else:
        print("❌ ERROR: Model returned empty list")
except Exception as e:
    print(f"❌ ERROR during inference: {e}")
    import traceback
    traceback.print_exc()

# Create a simple test image (license plate-like)
print("\n" + "=" * 70)
print("TEST 2: Synthetic License Plate Image")
print("=" * 70)

test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Light gray background

# Draw a plate-like shape in the middle
plate_x1, plate_y1 = 400, 300
plate_x2, plate_y2 = 880, 450  # 480x150 rect (license plate is ~500x100)

cv2.rectangle(test_image, (plate_x1, plate_y1), (plate_x2, plate_y2), (255, 255, 255), -1)  # White rect
cv2.putText(test_image, "KA-01-AB-1234", (plate_x1 + 20, plate_y1 + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

print(f"Test image shape: {test_image.shape}")

try:
    results = detector(test_image, conf=0.45)
    print(f"Model returned: {type(results)}")
    print(f"Results length: {len(results)}")
    
    if len(results) > 0:
        result = results[0]
        print(f"Result type: {type(result)}")
        print(f"Result has boxes: {hasattr(result, 'boxes')}")
        
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print(f"Boxes type: {type(boxes)}")
            print(f"Boxes length: {len(boxes)}")
            
            if len(boxes) == 0:
                print("⚠️  WARNING: No detections in synthetic plate image!")
                print("    This suggests the model may not be trained or configured correctly")
            else:
                print(f"✅ Found {len(boxes)} detection(s)")
                for i, box in enumerate(boxes):
                    print(f"  Box {i}:")
                    print(f"    Type: {type(box)}")
                    print(f"    Has xyxy: {hasattr(box, 'xyxy')}")
                    print(f"    Has conf: {hasattr(box, 'conf')}")
                    if hasattr(box, 'conf'):
                        print(f"    Conf value: {box.conf}")
                        print(f"    Conf type: {type(box.conf)}")
                    if hasattr(box, 'xyxy'):
                        print(f"    XYXY: {box.xyxy}")
        else:
            print("❌ ERROR: Results don't have boxes attribute")
    else:
        print("❌ ERROR: Model returned empty list")
except Exception as e:
    print(f"❌ ERROR during inference: {e}")
    import traceback
    traceback.print_exc()

# Check ONNX model info
print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

import onnxruntime as ort

sess = ort.InferenceSession(str(model_path))
print(f"Input: {sess.get_inputs()[0].name}, shape: {sess.get_inputs()[0].shape}")
print(f"Outputs: {[o.name for o in sess.get_outputs()]}")

# Run inference directly with ONNX to understand raw output
print("\n" + "=" * 70)
print("RAW ONNX INFERENCE (Understanding Model Output)")
print("=" * 70)

test_input = np.random.randn(1, 3, 320, 320).astype(np.float32) / 255.0
raw_output = sess.run(None, {"images": test_input})

print(f"Raw outputs: {len(raw_output)} tensors")
for i, output in enumerate(raw_output):
    print(f"  Output {i}: shape={output.shape}, dtype={output.dtype}")
    print(f"    Min={output.min():.6f}, Max={output.max():.6f}")
    print(f"    First 5 values: {output.flatten()[:5]}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

if len(results) > 0 and len(results[0].boxes) > 0:
    print("\n✅ Detection pipeline is working correctly")
else:
    print("\n❌ Detection pipeline may have issues:")
    print("   - Check if model is compatible")
    print("   - Verify confidence threshold isn't too high")
    print("   - Check ONNXBoxes parsing logic")
