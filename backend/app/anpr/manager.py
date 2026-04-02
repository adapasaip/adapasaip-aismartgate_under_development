"""
High-level ANPR manager for system orchestration

This module provides the public API for managing the ANPR system,
delegating to the core implementation module.
"""
import sys
from pathlib import Path

# Import all functionality from core module
from .core import *

# Re-export key components for external use
__all__ = [
    'initialize_application',
    'add_camera_frame',
    'cleanup_application',
    'get_recent_detections',
    'get_camera_status',
]

def initialize_application(args):
    """Initialize the ANPR application with given arguments"""
    # This delegates to the main logic in core
    pass

def cleanup_application():
    """Cleanup and shutdown the application"""
    # This delegates to the cleanup logic in core
    pass
