import React, { useState, useEffect } from 'react';
import { Plus, Edit2, Trash2, Save, X, Power, AlertCircle } from 'lucide-react';

/**
 * Camera Configuration Manager Component
 * Provides UI for users to add, edit, delete cameras and gates
 * Place this in: client/src/components/camera-config.tsx
 */

interface Camera {
  id: string;
  name: string;
  source: string;
  type: 'ipcam' | 'webcam' | 'rtsp';
  enabled: boolean;
  description: string;
}

interface Gate {
  id: string;
  name: string;
  cameraId: string;
  gateType: string;
  enabled: boolean;
  description: string;
}

interface CameraConfigManagerProps {
  onConfigUpdated?: () => void;
}

export const CameraConfigManager: React.FC<CameraConfigManagerProps> = ({ onConfigUpdated }) => {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [gates, setGates] = useState<Gate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'cameras' | 'gates'>('cameras');
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);

  const [newCamera, setNewCamera] = useState<Partial<Camera>>({
    type: 'ipcam',
  });
  const [newGate, setNewGate] = useState<Partial<Gate>>({
    gateType: 'ipcam:Entry',
  });

  const API_BASE = '/api';  // Use relative URL so it works from any deployment

  // Fetch cameras and gates
  const fetchConfig = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/all`);
      const data = await response.json();

      if (data.success) {
        setCameras(data.cameras);
        setGates(data.gates);
        setError(null);
      } else {
        setError(data.error || 'Failed to load configuration');
      }
    } catch (err) {
      setError(`Error loading configuration: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfig();
  }, []);

  // Add camera
  const handleAddCamera = async () => {
    if (!newCamera.id || !newCamera.name || !newCamera.source || !newCamera.type) {
      setError('Please fill in all camera fields');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCamera),
      });

      const data = await response.json();
      if (data.success) {
        await fetchConfig();
        setNewCamera({ type: 'ipcam' });
        setIsAddingNew(false);
        setError(null);
        onConfigUpdated?.();
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(`Error adding camera: ${err}`);
    }
  };

  // Update camera
  const handleUpdateCamera = async (cameraId: string) => {
    try {
      const response = await fetch(`${API_BASE}/cameras/${cameraId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cameras.find((c) => c.id === cameraId)),
      });

      const data = await response.json();
      if (data.success) {
        await fetchConfig();
        setEditingId(null);
        setError(null);
        onConfigUpdated?.();
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(`Error updating camera: ${err}`);
    }
  };

  // Delete camera
  const handleDeleteCamera = async (cameraId: string) => {
    if (!confirm(`Delete camera "${cameraId}"? Associated gates will also be deleted.`)) return;

    try {
      const response = await fetch(`${API_BASE}/cameras/${cameraId}`, {
        method: 'DELETE',
      });

      const data = await response.json();
      if (data.success) {
        await fetchConfig();
        setError(null);
        onConfigUpdated?.();
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(`Error deleting camera: ${err}`);
    }
  };

  // Add gate
  const handleAddGate = async () => {
    if (!newGate.id || !newGate.name || !newGate.cameraId || !newGate.gateType) {
      setError('Please fill in all gate fields');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/gates`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newGate),
      });

      const data = await response.json();
      if (data.success) {
        await fetchConfig();
        setNewGate({ gateType: 'ipcam:Entry' });
        setIsAddingNew(false);
        setError(null);
        onConfigUpdated?.();
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(`Error adding gate: ${err}`);
    }
  };

  // Delete gate
  const handleDeleteGate = async (gateId: string) => {
    if (!confirm(`Delete gate "${gateId}"?`)) return;

    try {
      const response = await fetch(`${API_BASE}/gates/${gateId}`, {
        method: 'DELETE',
      });

      const data = await response.json();
      if (data.success) {
        await fetchConfig();
        setError(null);
        onConfigUpdated?.();
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError(`Error deleting gate: ${err}`);
    }
  };

  // Toggle camera enabled status
  const handleToggleCamera = (cameraId: string) => {
    const updated = cameras.map((c) =>
      c.id === cameraId ? { ...c, enabled: !c.enabled } : c
    );
    setCameras(updated);
  };

  // Toggle gate enabled status
  const handleToggleGate = (gateId: string) => {
    const updated = gates.map((g) =>
      g.id === gateId ? { ...g, enabled: !g.enabled } : g
    );
    setGates(updated);
  };

  if (loading) return <div className="p-4">Loading configuration...</div>;

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6">📷 Camera Configuration</h2>

      {/* Error Alert */}
      {error && (
        <div className="mb-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded flex items-center gap-2">
          <AlertCircle size={20} />
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 mb-6 border-b">
        <button
          onClick={() => setActiveTab('cameras')}
          className={`px-4 py-2 font-semibold border-b-2 transition ${
            activeTab === 'cameras'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          Cameras ({cameras.length})
        </button>
        <button
          onClick={() => setActiveTab('gates')}
          className={`px-4 py-2 font-semibold border-b-2 transition ${
            activeTab === 'gates'
              ? 'border-blue-500 text-blue-600'
              : 'border-transparent text-gray-600 hover:text-gray-900'
          }`}
        >
          Gates ({gates.length})
        </button>
      </div>

      {/* Cameras Tab */}
      {activeTab === 'cameras' && (
        <div>
          <div className="mb-6">
            <button
              onClick={() => setIsAddingNew(!isAddingNew)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <Plus size={20} />
              Add Camera
            </button>
          </div>

          {/* Add new camera form */}
          {isAddingNew && activeTab === 'cameras' && (
            <div className="mb-6 p-4 border border-blue-300 rounded bg-blue-50">
              <h3 className="font-semibold mb-4">New Camera</h3>
              <div className="grid grid-cols-1 gap-3">
                <input
                  type="text"
                  placeholder="Camera ID (e.g., entry-cam)"
                  value={newCamera.id || ''}
                  onChange={(e) => setNewCamera({ ...newCamera, id: e.target.value })}
                  className="border p-2 rounded"
                />
                <input
                  type="text"
                  placeholder="Camera Name"
                  value={newCamera.name || ''}
                  onChange={(e) => setNewCamera({ ...newCamera, name: e.target.value })}
                  className="border p-2 rounded"
                />
                <select
                  value={newCamera.type || ''}
                  onChange={(e) => setNewCamera({ ...newCamera, type: e.target.value as Camera['type'] })}
                  className="border p-2 rounded"
                >
                  <option value="ipcam">IP Camera</option>
                  <option value="webcam">USB Webcam</option>
                  <option value="rtsp">RTSP Stream</option>
                </select>
                <input
                  type="text"
                  placeholder="Source (URL for IP/RTSP, device ID for webcam)"
                  value={newCamera.source || ''}
                  onChange={(e) => setNewCamera({ ...newCamera, source: e.target.value })}
                  className="border p-2 rounded font-mono text-sm"
                />
                <input
                  type="text"
                  placeholder="Description"
                  value={newCamera.description || ''}
                  onChange={(e) => setNewCamera({ ...newCamera, description: e.target.value })}
                  className="border p-2 rounded"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleAddCamera}
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Save Camera
                  </button>
                  <button
                    onClick={() => setIsAddingNew(false)}
                    className="flex-1 px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Camera list */}
          <div className="space-y-3">
            {cameras.map((camera) => (
              <div
                key={camera.id}
                className={`p-4 border rounded flex items-center justify-between ${
                  camera.enabled ? 'border-green-300 bg-green-50' : 'border-gray-300 bg-gray-50'
                }`}
              >
                <div className="flex-1">
                  <h4 className="font-semibold">{camera.name}</h4>
                  <p className="text-sm text-gray-600">{camera.description}</p>
                  <p className="text-xs text-gray-500 font-mono mt-1">
                    {camera.type} → {camera.source}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleToggleCamera(camera.id)}
                    className={`p-2 rounded ${
                      camera.enabled
                        ? 'bg-green-500 text-white hover:bg-green-600'
                        : 'bg-gray-400 text-white hover:bg-gray-500'
                    }`}
                  >
                    <Power size={18} />
                  </button>
                  <button
                    onClick={() => handleDeleteCamera(camera.id)}
                    className="p-2 rounded bg-red-500 text-white hover:bg-red-600"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gates Tab */}
      {activeTab === 'gates' && (
        <div>
          <div className="mb-6">
            <button
              onClick={() => setIsAddingNew(!isAddingNew)}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <Plus size={20} />
              Add Gate
            </button>
          </div>

          {/* Add new gate form */}
          {isAddingNew && activeTab === 'gates' && (
            <div className="mb-6 p-4 border border-blue-300 rounded bg-blue-50">
              <h3 className="font-semibold mb-4">New Gate</h3>
              <div className="grid grid-cols-1 gap-3">
                <input
                  type="text"
                  placeholder="Gate ID (e.g., entry-gate)"
                  value={newGate.id || ''}
                  onChange={(e) => setNewGate({ ...newGate, id: e.target.value })}
                  className="border p-2 rounded"
                />
                <input
                  type="text"
                  placeholder="Gate Name"
                  value={newGate.name || ''}
                  onChange={(e) => setNewGate({ ...newGate, name: e.target.value })}
                  className="border p-2 rounded"
                />
                <select
                  value={newGate.cameraId || ''}
                  onChange={(e) => setNewGate({ ...newGate, cameraId: e.target.value })}
                  className="border p-2 rounded"
                >
                  <option value="">Select Camera</option>
                  {cameras.map((cam) => (
                    <option key={cam.id} value={cam.id}>
                      {cam.name} ({cam.id})
                    </option>
                  ))}
                </select>
                <select
                  value={newGate.gateType || ''}
                  onChange={(e) => setNewGate({ ...newGate, gateType: e.target.value })}
                  className="border p-2 rounded"
                >
                  <option value="ipcam:Entry">IP Camera - Entry</option>
                  <option value="ipcam:Exit">IP Camera - Exit</option>
                  <option value="webcam:Entry">Webcam - Entry</option>
                  <option value="webcam:Exit">Webcam - Exit</option>
                  <option value="rtsp:Entry">RTSP - Entry</option>
                  <option value="rtsp:Exit">RTSP - Exit</option>
                </select>
                <input
                  type="text"
                  placeholder="Description"
                  value={newGate.description || ''}
                  onChange={(e) => setNewGate({ ...newGate, description: e.target.value })}
                  className="border p-2 rounded"
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleAddGate}
                    className="flex-1 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                  >
                    Save Gate
                  </button>
                  <button
                    onClick={() => setIsAddingNew(false)}
                    className="flex-1 px-4 py-2 bg-gray-400 text-white rounded hover:bg-gray-500"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Gate list */}
          <div className="space-y-3">
            {gates.map((gate) => {
              const camera = cameras.find((c) => c.id === gate.cameraId);
              return (
                <div
                  key={gate.id}
                  className={`p-4 border rounded flex items-center justify-between ${
                    gate.enabled ? 'border-green-300 bg-green-50' : 'border-gray-300 bg-gray-50'
                  }`}
                >
                  <div className="flex-1">
                    <h4 className="font-semibold">{gate.name}</h4>
                    <p className="text-sm text-gray-600">{gate.description}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Camera: {camera?.name || 'Not found'} | Type: {gate.gateType}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleToggleGate(gate.id)}
                      className={`p-2 rounded ${
                        gate.enabled
                          ? 'bg-green-500 text-white hover:bg-green-600'
                          : 'bg-gray-400 text-white hover:bg-gray-500'
                      }`}
                    >
                      <Power size={18} />
                    </button>
                    <button
                      onClick={() => handleDeleteGate(gate.id)}
                      className="p-2 rounded bg-red-500 text-white hover:bg-red-600"
                    >
                      <Trash2 size={18} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Info box */}
      <div className="mt-6 p-4 bg-blue-100 border border-blue-300 rounded text-sm text-blue-800">
        <p className="font-semibold mb-2">💡 How to use:</p>
        <ul className="list-disc list-inside space-y-1">
          <li>Add your cameras (IP camera, webcam, or RTSP stream)</li>
          <li>Create gates and link them to cameras</li>
          <li>Enable/disable cameras and gates with the power button</li>
          <li>Only enabled cameras and gates will be used when the system starts</li>
          <li>Restart the application for changes to take effect</li>
        </ul>
      </div>
    </div>
  );
};

export default CameraConfigManager;
