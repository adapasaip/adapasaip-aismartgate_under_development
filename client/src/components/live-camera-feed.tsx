import { useState, useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Camera, AnprDetection } from "@shared/schema";
import { getCameraConnectionInfo } from "@/lib/camera-utils";
import { BACKEND_URL } from "@/lib/config";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest } from "@/lib/queryClient";
import { MjpegStreamPlayer } from "@/lib/mjpeg-stream";

export default function LiveCameraFeed() {
  const [selectedCamera, setSelectedCamera] = useState<string>("");
  const [secondsAgo, setSecondsAgo] = useState<number>(0);
  const [isStreamConnected, setIsStreamConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const { user, subUser } = useAuth();
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  const { data: cameras = [] } = useQuery<Camera[]>({
    queryKey: ["/api/cameras", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/cameras?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      return Array.isArray(data) ? data : [];
    },
    enabled: userIdParam !== "default",
  });

  const { data: detections = [] } = useQuery<AnprDetection[]>({
    queryKey: ["/api/detections/recent", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/detections/recent?userId=${encodeURIComponent(userIdParam)}`);
      if (!response.ok) throw new Error("Failed to fetch detections");
      const data = await response.json();
      // Ensure data is an array
      return Array.isArray(data) ? data : [];
    },
    refetchInterval: 2000, // Real-time updates every 2 seconds
    enabled: userIdParam !== "default",
    staleTime: 500, // Data stale after 500ms
    gcTime: 1000, // Keep in cache for 1 second
  });

  const lastDetection = detections.length > 0 ? detections[0] : null;
  const activeCameras = Array.isArray(cameras) ? cameras.filter((camera) => camera.status === "Online") : [];

  // Update seconds ago dynamically
  useEffect(() => {
    if (!lastDetection) return;

    const interval = setInterval(() => {
      const detectionTime = new Date(lastDetection.detectedAt).getTime();
      const now = new Date().getTime();
      const seconds = Math.floor((now - detectionTime) / 1000);
      setSecondsAgo(seconds);
    }, 1000);

    return () => clearInterval(interval);
  }, [lastDetection]);

  // Reset state when camera changes
  useEffect(() => {
    setIsLoading(true);
    setIsStreamConnected(false);
  }, [selectedCamera]);

  const selectedCameraObj = Array.isArray(cameras) ? cameras.find(c => c.id === selectedCamera) : undefined;
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamPlayerRef = useRef<MjpegStreamPlayer | null>(null);

  // Construct MJPEG stream URL directly from Python backend
  const getStreamUrl = () => {
    if (!selectedCameraObj) return "";
    
    // ⭐ CRITICAL FIX: Use camera-specific endpoint to enable overlays
    // Flask backend applies bounding boxes and OCR text via persistent_overlays cache
    // The camera ID is REQUIRED for Flask to find and apply camera-specific overlays
    // Preview endpoint is generic and does NOT include overlays
    const pythonBackendUrl = process.env.REACT_APP_PYTHON_BACKEND_URL || "http://localhost:8000";
    const cameraId = selectedCameraObj.id;
    
    // Use camera-specific endpoint instead of generic preview
    // This enables overlay rendering: /api/video_feed/{cam_id}
    return `${pythonBackendUrl}/api/video_feed/${cameraId}`;
  };

  const streamUrl = getStreamUrl();

  // Handle camera stream playback
  useEffect(() => {
    if (!selectedCamera || !streamUrl || !canvasRef.current) {
      setIsStreamConnected(false);
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setIsStreamConnected(false);

    // Cleanup previous stream player
    if (streamPlayerRef.current) {
      streamPlayerRef.current.stop();
      streamPlayerRef.current = null;
    }

    // Create new MJPEG stream player with optimized settings
    streamPlayerRef.current = new MjpegStreamPlayer(
      canvasRef.current,
      streamUrl,
      () => {
        setIsLoading(false);
        setIsStreamConnected(true);
      },
      () => {
        setIsLoading(false);
        setIsStreamConnected(false);
      }
    );

    streamPlayerRef.current.start();

    return () => {
      if (streamPlayerRef.current) {
        streamPlayerRef.current.stop();
        streamPlayerRef.current = null;
      }
    };
  }, [selectedCamera, streamUrl]);

  return (
    <div className="lg:col-span-2 bg-white rounded-2xl shadow-lg border-2 border-blue-100 w-full overflow-hidden hover:border-blue-300 hover:shadow-xl transition-all duration-300">
      {/* Header with gradient background */}
      <div className="p-6 border-b-2 border-blue-100 bg-gradient-to-r from-blue-50 via-blue-25 to-white relative overflow-hidden">
        {/* Decorative accent line */}
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-300 via-blue-100 to-transparent"></div>
        
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 relative z-10">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-br from-blue-100 to-blue-50 rounded-lg border border-blue-200 shadow-sm">
              <i className="fas fa-video text-blue-600 text-lg"></i>
            </div>
            <div>
              <h2 className="text-lg sm:text-2xl font-bold text-blue-900">
                Live Camera Feed
              </h2>
              <p className="text-xs text-slate-500 mt-1"><i className="fas fa-play-circle text-green-500 mr-1.5"></i>Real-time MJPEG stream</p>
            </div>
          </div>

          <Select value={selectedCamera} onValueChange={setSelectedCamera}>
            <SelectTrigger className="w-full sm:w-[260px] border-2 border-blue-300 rounded-xl focus:border-blue-500 bg-gradient-to-r from-blue-50 to-white font-medium shadow-sm hover:shadow-md transition-all" data-testid="select-camera">
              <SelectValue placeholder="Select camera" />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              {activeCameras.map((camera) => (
                <SelectItem key={camera.id} value={camera.id} className="cursor-pointer">
                  <i className="fas fa-camera mr-2 text-blue-600"></i>
                  <span className="font-medium">{camera.name}</span>
                </SelectItem>
              ))}
              {activeCameras.length === 0 && (
                <SelectItem value="none" disabled>
                  <span className="text-slate-500">No active cameras</span>
                </SelectItem>
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Feed Area */}
      <div className="p-4 sm:p-6 bg-gradient-to-b from-slate-50 to-white">
        <div className="bg-black rounded-xl aspect-video flex items-center justify-center relative overflow-hidden w-full border-4 border-slate-900 shadow-2xl group">
          {selectedCamera && selectedCameraObj ? (
            <>
              {/* Canvas for MJPEG Stream Player */}
              {streamUrl && (
                <canvas
                  ref={canvasRef}
                  className="w-full h-full"
                  style={{
                    aspectRatio: '16/9',
                    display: isStreamConnected || isLoading ? 'block' : 'none',
                  }}
                />
              )}

              {/* Error State */}
              {!isStreamConnected && !isLoading && (
                <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-slate-800 to-slate-900 text-white p-4">
                  <i className="fas fa-exclamation-triangle text-5xl mb-3 opacity-40 text-red-400"></i>
                  <h3 className="text-lg font-semibold mb-2 text-red-300">Stream Unavailable</h3>
                  <p className="text-sm text-slate-400 text-center mb-4 max-w-md">
                    Unable to connect to camera stream
                  </p>
                  <div className="bg-slate-700/50 backdrop-blur-sm border border-slate-600 p-3 rounded-lg text-xs text-slate-300 max-w-md mt-2 font-mono">
                    <p className="mb-2 text-slate-200"><i className="fas fa-camera mr-2 text-blue-400"></i><strong>Camera:</strong> {selectedCameraObj?.name}</p>
                    <p className="mb-2 text-slate-200"><i className="fas fa-video mr-2 text-blue-400"></i><strong>Type:</strong> {selectedCameraObj ? getCameraConnectionInfo(selectedCameraObj, BACKEND_URL).displayName : 'Unknown'}</p>
                    <p className="text-slate-400 text-[10px] mt-3 leading-relaxed border-t border-slate-600 pt-2">
                      <i className="fas fa-lightbulb text-yellow-400 mr-1"></i>Ensure Python backend is running on port 8000
                    </p>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {isLoading && !isStreamConnected && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-800 bg-opacity-75">
                  <div className="text-white text-center">
                    <div className="animate-spin mb-3">
                      <svg className="w-12 h-12 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </div>
                    <p className="text-sm font-medium">Connecting to camera...</p>
                  </div>
                </div>
              )}

              {/* LIVE Indicator - Glass Design - Compact */}
              {isStreamConnected && (
                <div className="absolute z-50 top-3 right-3 flex items-center gap-1.5 bg-gradient-to-r from-green-500/50 via-green-500/45 to-emerald-500/50 text-white px-3 py-1.5 rounded-full text-[10px] sm:text-xs font-bold shadow-lg hover:shadow-green-500/50 transition-all duration-300 border border-green-400/60 backdrop-blur-md">
                  <span className="w-2 h-2 bg-white rounded-full animate-pulse shadow-md"></span>
                  <span className="tracking-wide">LIVE</span>
                </div>
              )}

              {/* Detection Overlay - Glass Design with Entry/Exit Status - Compact */}
              {isStreamConnected && (
                <div className="absolute z-50 bottom-2 sm:bottom-3 left-2 sm:left-3 bg-gradient-to-br from-slate-900/20 via-slate-900/15 to-slate-950/20 text-white px-3 py-2 rounded-xl text-[10px] sm:text-xs backdrop-blur-lg border border-slate-400/15 shadow-lg min-w-[140px] hover:border-slate-400/25 transition-all duration-200">
                  {lastDetection ? (
                    <>
                      {/* Compact Header */}
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <div className="flex items-center gap-1 min-w-0">
                          <i className="fas fa-check-circle text-green-400 text-sm flex-shrink-0"></i>
                          <span className="text-slate-100 font-semibold truncate">Detection</span>
                        </div>
                        {/* Entry/Exit Status Badge - Compact */}
                        <span className={`px-1.5 py-0.5 rounded-full text-[8px] sm:text-[9px] font-bold uppercase tracking-tight backdrop-blur-sm border flex-shrink-0 ${
                          lastDetection.gate === 'Entry' 
                            ? 'bg-blue-500/50 text-blue-100 border-blue-400/60' 
                            : 'bg-orange-500/50 text-orange-100 border-orange-400/60'
                        }`}>
                          <i className={`fas ${lastDetection.gate === 'Entry' ? 'fa-arrow-right' : 'fa-arrow-left'} mr-0.5`}></i>
                          {lastDetection.gate ? lastDetection.gate.slice(0, 3) : 'N/A'}
                        </span>
                      </div>
                      
                      {/* License Plate - Compact */}
                      <div className="font-mono text-green-300 font-bold text-sm sm:text-base tracking-wider mb-1 pb-1 border-b border-slate-500/15">
                        {lastDetection.licensePlate}
                      </div>
                      
                      {/* Timestamp - Compact */}
                      <div className="text-slate-300 text-[9px] sm:text-[10px] flex items-center gap-1">
                        <i className="fas fa-clock text-slate-400 flex-shrink-0"></i>
                        <span className="truncate">{secondsAgo === 0 ? "now" : `${secondsAgo}s ago`}</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center gap-1.5 mb-1">
                        <i className="fas fa-spinner text-blue-400 animate-spin text-sm flex-shrink-0"></i>
                        <span className="text-slate-100 font-semibold text-[9px] truncate">Scanning...</span>
                      </div>
                      <div className="text-slate-400 text-[8px] sm:text-[9px]">
                        <i className="fas fa-scan text-slate-500 mr-0.5"></i>Active
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* Camera Info - Glass Design - Compact */}
              {isStreamConnected && selectedCameraObj && (
                <div className="absolute z-50 top-3 left-3 bg-gradient-to-r from-blue-500/50 via-purple-500/45 to-blue-500/50 text-white px-2.5 py-1.5 rounded-full text-[10px] sm:text-xs font-semibold shadow-lg hover:shadow-blue-500/50 transition-all duration-300 border border-blue-400/60 backdrop-blur-md flex items-center gap-1.5">
                  <i className="fas fa-video text-blue-100 text-sm flex-shrink-0"></i>
                  <span className="tracking-wide truncate max-w-[120px] sm:max-w-[160px]">{getCameraConnectionInfo(selectedCameraObj, BACKEND_URL).displayName}</span>
                </div>
              )}
            </>
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-slate-50 via-blue-25 to-slate-50 px-4">
              <div className="text-center">
                <div className="p-4 bg-blue-100/50 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                  <i className="fas fa-camera text-4xl text-blue-600"></i>
                </div>
                <p className="text-base font-bold text-slate-800 mb-2">No camera selected</p>
                <p className="text-sm text-slate-600 max-w-xs">Select a camera from the dropdown above to view live feed</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
