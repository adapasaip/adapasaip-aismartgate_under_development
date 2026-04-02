"""
Camera Configuration Manager
Handles loading, saving, and managing camera and gate configurations
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class CameraConfigManager:
    def __init__(self, config_path: str = None):
        """Initialize config manager with path to cameras-config.json"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../data/cameras-config.json')
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load camera configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Config file not found at {self.config_path}, creating default...")
                return self._create_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._create_default_config()
    
    def _save_config(self) -> bool:
        """Save camera configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default camera configuration"""
        return {
            "cameras": [],
            "gates": [],
            "settings": {
                "autoStartBackend": True,
                "autoStartFrontend": True,
                "backendPort": 8000,
                "frontendPort": 5000,
                "detectionsLogPath": "./data/detections.json"
            }
        }
    
    def get_enabled_cameras(self) -> List[Dict[str, Any]]:
        """Get list of enabled cameras"""
        return [cam for cam in self.config.get('cameras', []) if cam.get('enabled', False)]
    
    def get_enabled_gates(self) -> List[Dict[str, Any]]:
        """Get list of enabled gates"""
        return [gate for gate in self.config.get('gates', []) if gate.get('enabled', False)]
    
    def add_camera(self, camera_id: str, name: str, source: str, camera_type: str, 
                   description: str = "") -> bool:
        """Add a new camera configuration"""
        try:
            if any(cam['id'] == camera_id for cam in self.config['cameras']):
                print(f"⚠️ Camera {camera_id} already exists")
                return False
            
            camera = {
                "id": camera_id,
                "name": name,
                "source": source,
                "type": camera_type,
                "enabled": True,
                "description": description,
                "createdAt": datetime.now().isoformat()
            }
            self.config['cameras'].append(camera)
            return self._save_config()
        except Exception as e:
            print(f"❌ Error adding camera: {e}")
            return False
    
    def add_gate(self, gate_id: str, name: str, camera_id: str, gate_type: str,
                description: str = "") -> bool:
        """Add a new gate configuration"""
        try:
            if not any(cam['id'] == camera_id for cam in self.config['cameras']):
                print(f"❌ Camera {camera_id} not found")
                return False
            
            if any(gate['id'] == gate_id for gate in self.config['gates']):
                print(f"⚠️ Gate {gate_id} already exists")
                return False
            
            gate = {
                "id": gate_id,
                "name": name,
                "cameraId": camera_id,
                "gateType": gate_type,
                "enabled": True,
                "description": description,
                "createdAt": datetime.now().isoformat()
            }
            self.config['gates'].append(gate)
            return self._save_config()
        except Exception as e:
            print(f"❌ Error adding gate: {e}")
            return False
    
    def update_camera(self, camera_id: str, **kwargs) -> bool:
        """Update camera configuration"""
        try:
            for camera in self.config['cameras']:
                if camera['id'] == camera_id:
                    camera.update(kwargs)
                    camera['updatedAt'] = datetime.now().isoformat()
                    return self._save_config()
            print(f"❌ Camera {camera_id} not found")
            return False
        except Exception as e:
            print(f"❌ Error updating camera: {e}")
            return False
    
    def delete_camera(self, camera_id: str) -> bool:
        """Delete camera and associated gates"""
        try:
            self.config['cameras'] = [cam for cam in self.config['cameras'] if cam['id'] != camera_id]
            self.config['gates'] = [gate for gate in self.config['gates'] if gate['cameraId'] != camera_id]
            return self._save_config()
        except Exception as e:
            print(f"❌ Error deleting camera: {e}")
            return False
    
    def get_camera(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get specific camera configuration"""
        for camera in self.config['cameras']:
            if camera['id'] == camera_id:
                return camera
        return None
    
    def get_all_cameras(self) -> List[Dict[str, Any]]:
        """Get all cameras"""
        return self.config.get('cameras', [])
    
    def get_all_gates(self) -> List[Dict[str, Any]]:
        """Get all gates"""
        return self.config.get('gates', [])
    
    def get_settings(self) -> Dict[str, Any]:
        """Get application settings"""
        return self.config.get('settings', {})
    
    def set_camera_status(self, camera_id: str, online: bool) -> bool:
        """⭐ PERSISTENCE: Set camera online/offline status and save to file"""
        try:
            for camera in self.config['cameras']:
                if camera['id'] == camera_id:
                    camera['online'] = online  # ⭐ Persist online status
                    camera['statusChangedAt'] = datetime.now().isoformat()
                    return self._save_config()
            print(f"❌ Camera {camera_id} not found")
            return False
        except Exception as e:
            print(f"❌ Error setting camera status: {e}")
            return False
    
    def get_cameras_to_start(self) -> List[Dict[str, Any]]:
        """
        ⭐ STARTUP: Get all enabled AND online cameras to auto-start
        
        ★ BACKWARD COMPATIBLE: Cameras without 'online' field default to True (online)
        This ensures existing configurations work as before
        """
        return [cam for cam in self.config.get('cameras', []) 
                if cam.get('enabled', False) and cam.get('online', True)]
    
    def build_cli_args(self) -> tuple[str, str]:
        """Build CLI arguments from enabled cameras and gates for backward compatibility"""
        enabled_cams = self.get_enabled_cameras()
        enabled_gates = self.get_enabled_gates()
        
        if not enabled_cams or not enabled_gates:
            return "", ""
        
        # Build cameras arg: type=source,type=source
        cameras_arg = ",".join([f"{cam['type']}={cam['source']}" for cam in enabled_cams])
        
        # Build gates arg: type:name,type:name
        gates_arg = ",".join([gate['gateType'] for gate in enabled_gates])
        
        return cameras_arg, gates_arg


if __name__ == "__main__":
    # Example usage
    config = CameraConfigManager()
    
    print("📷 Enabled Cameras:")
    for cam in config.get_enabled_cameras():
        print(f"  - {cam['name']}: {cam['source']}")
    
    print("\n🚪 Enabled Gates:")
    for gate in config.get_enabled_gates():
        print(f"  - {gate['name']}: {gate['gateType']}")
    
    print("\n🔧 CLI Build Test:")
    cameras_arg, gates_arg = config.build_cli_args()
    print(f"  --cameras {cameras_arg}")
    print(f"  --gates {gates_arg}")
