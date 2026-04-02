#!/usr/bin/env python3
"""
ANPR (Automatic Number Plate Recognition) System - Entry Point

Main application entry point that starts the ANPR system with all core functionality.
All implementation is contained in the anpr.core module.

To start the system:
    python3.10 camera_anpr.py

To access the web UI:
    http://localhost:8000
    or
    http://localhost:5000 (config interface)
"""

import sys
import os
from pathlib import Path

# Ensure the app directory is in the path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    # Import core which defines the Flask app and all routes
    import anpr.core
    
    try:
        print("\n[INFO] Starting ANPR Flask server...")
        print("[INFO] Access the web UI at: http://localhost:8000")
        print("[INFO] Press Ctrl+C to stop the server\n")
        
        # Run the Flask app from core
        anpr.core.app.run(
            host="0.0.0.0",
            port=8000,
            debug=anpr.core.DEBUG,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid duplicate threads
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server shutdown initiated...")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] ANPR System stopped")
        try:
            anpr.core.cleanup_application()
        except:
            pass
