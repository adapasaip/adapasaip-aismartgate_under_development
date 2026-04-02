"""Centroid-based object tracker for associating detections across frames."""

import numpy as np
from collections import defaultdict, OrderedDict


class CentroidTracker:
    """Tracks object centroids across frames using Euclidean distance."""
    
    def __init__(self, max_distance=50, max_disappeared=50):
        """
        Initialize the centroid tracker.
        
        Args:
            max_distance: Maximum distance (pixels) to associate a detection to existing object
            max_disappeared: Maximum frames an object can disappear before being removed
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
    
    def register(self, centroid):
        """Register a new object."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, rects):
        """
        Update tracker with new detections.
        
        Args:
            rects: List of (x1, y1, x2, y2, conf) detections
            
        Returns:
            Dictionary mapping bbox index to object_id
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        input_centroids = np.zeros((len(rects), 2))
        for i, (x1, y1, x2, y2, conf) in enumerate(rects):
            centroid_x = (x1 + x2) / 2.0
            centroid_y = (y1 + y2) / 2.0
            input_centroids[i] = [centroid_x, centroid_y]
        
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dist = np.linalg.norm(object_centroids[i] - input_centroids[j])
                    D[i][j] = dist
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)
            
            UsedRows = set()
            UsedCols = set()
            
            for (row, col) in zip(rows, cols):
                if row in UsedRows or col in UsedCols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                UsedRows.add(row)
                UsedCols.add(col)
            
            UnusedRows = set(range(0, D.shape[0])).difference(UsedRows)
            UnusedCols = set(range(0, D.shape[1])).difference(UsedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in UnusedRows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in UnusedCols:
                    self.register(input_centroids[col])
        
        result = {}
        for (i, (x1, y1, x2, y2, conf)) in enumerate(rects):
            centroid = input_centroids[i]
            for object_id, obj_centroid in self.objects.items():
                if np.array_equal(obj_centroid, centroid):
                    result[i] = object_id
                    break
        
        return result
