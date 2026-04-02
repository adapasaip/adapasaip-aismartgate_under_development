"""
BBox Smoother - Smooths bounding box coordinates across frames
"""

import numpy as np


class BBoxSmoother:
    """Smooths bounding box coordinates using exponential moving average."""
    
    def __init__(self, alpha=0.3):
        """
        Initialize the bbox smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Default 0.3 for moderate smoothing.
        """
        self.alpha = alpha
        self.last_bbox = None
    
    def smooth(self, x1, y1, x2, y2):
        """Smooth a bounding box using exponential moving average."""
        if self.last_bbox is None:
            self.last_bbox = [x1, y1, x2, y2]
            return x1, y1, x2, y2
        
        smoothed_x1 = self.alpha * x1 + (1 - self.alpha) * self.last_bbox[0]
        smoothed_y1 = self.alpha * y1 + (1 - self.alpha) * self.last_bbox[1]
        smoothed_x2 = self.alpha * x2 + (1 - self.alpha) * self.last_bbox[2]
        smoothed_y2 = self.alpha * y2 + (1 - self.alpha) * self.last_bbox[3]
        
        self.last_bbox = [smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2]
        
        return smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2
    
    def reset(self):
        """Reset the smoother for a new object."""
        self.last_bbox = None
