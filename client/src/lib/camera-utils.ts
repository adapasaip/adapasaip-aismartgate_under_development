/**
 * Camera URL utilities for handling different camera types
 * Supports: IP cameras, CCTV streams, NGROK tunnels, local webcams, RTSP streams
 */

import type { Camera } from "@shared/schema";

export type CameraUrlType = 
  | "desktop"
  | "usb"
  | "ip-http"
  | "rtsp"
  | "mobile-app"
  | "ngrok"
  | "oak"
  | "direct-url";

/**
 * Determines the camera type based on the source string
 */
export function detectCameraType(source: string, type?: string): CameraUrlType {
  // If type is provided in camera schema, use that
  if (type) {
    if (type.includes("Desktop")) return "desktop";
    if (type.includes("USB")) return "usb";
    if (type.includes("IP")) return "ip-http";
    if (type.includes("RTSP")) return "rtsp";
    if (type.includes("OAK")) return "oak";
    if (type.includes("Mobile") && type.includes("NGROK")) return "ngrok";
    if (type.includes("Mobile") && !type.includes("NGROK")) return "mobile-app";
  }

  // Detect from source string
  if (source === "0") {
    return "desktop"; // Source 0 = Desktop/Default Webcam -> http://localhost:8000/api/video_feed/webcam
  }
  if (source === "1") {
    return "usb"; // Source 1 = USB Connected Camera
  }
  if (!isNaN(Number(source))) {
    return "usb"; // Numeric sources >= 2 also map to USB cameras
  }

  if (source.startsWith("rtsp://")) return "rtsp";
  if (source.startsWith("http://") || source.startsWith("https://")) {
    if (source.includes("ngrok")) return "ngrok";
    return "ip-http";
  }

  // If it looks like an IP with a port, assume it's an IP camera
  if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/.test(source)) {
    return "ip-http";
  }

  // OAK cameras typically have a UUID-like source
  if (source && /^[a-zA-Z0-9]{16,}$/.test(source)) {
    return "oak";
  }

  return "direct-url";
}

/**
 * Constructs the proper video feed URL based on camera type and source
 * Handles different camera streaming protocols
 * All camera feeds are proxied through the backend for security and CORS handling
 * Authentication is handled via HTTP-only cookies (no tokens in URL)
 */
export function getCameraFeedUrl(camera: Camera, backendUrl: string = "", userId: string = "default"): string {
  const { source, type, id } = camera;
  const cameraType = detectCameraType(source, type);

  // Determine the base URL
  // If backendUrl is provided, use it. Otherwise use relative paths (same origin)
  const baseUrl = backendUrl || "";

  console.log('[getCameraFeedUrl] Building URL for camera:', {
    cameraId: id,
    source: source,
    type: cameraType,
    userId: userId
  });

  // For preview (unsaved camera), pass source and type as query params
  if (id === "preview") {
    console.log('[getCameraFeedUrl] USING_DETECTED_TYPE - Detected camera type:', cameraType, '| Original form type:', type);
    const params = new URLSearchParams({
      source: source || "0",
      type: cameraType,  // Pass the detected camera type (e.g., "desktop", "rtsp") instead of full description
      userId: userId // Include userId for proper data isolation
    });
    const previewUrl = `${baseUrl}/api/video_feed/preview?${params.toString()}`;
    console.log('[getCameraFeedUrl] BUILT_PARAMS - Preview URL:', previewUrl, '| Actual type value in URL:', cameraType);
    return previewUrl;
  }

  // All camera types go through the backend proxy
  // This ensures proper CORS handling, security, and consistent streaming
  // The backend will handle:
  // - HTTP/HTTPS: Direct proxy streaming
  // - RTSP: Requires transcoding (backend returns error with instructions)
  // - Local (0, 1): Requires camera service (backend returns error with instructions)
  // Authentication is handled via HTTP-only cookie automatically
  const params = new URLSearchParams({
    userId: userId // Include userId for proper data isolation and detection tagging
  });
  const url = `${baseUrl}/api/video_feed/${camera.id}?${params.toString()}`;
  console.log('[getCameraFeedUrl] Final URL (auth via cookie):', url);
  return url;
}

/**
 * Checks if a camera type requires special handling
 */
export function isStreamingRequired(camera: Camera): boolean {
  const { source, type } = camera;
  const cameraType = detectCameraType(source, type);
  
  return cameraType === "rtsp";
}

/**
 * Gets camera connection info for display/debugging
 */
export function getCameraConnectionInfo(camera: Camera, backendUrl: string = "", userId: string = "default"): {
  type: CameraUrlType;
  displayName: string;
  url: string;
  requiresAuth?: boolean;
  note?: string;
} {
  const type = detectCameraType(camera.source, camera.type);
  const url = getCameraFeedUrl(camera, backendUrl, userId);

  const info: Record<CameraUrlType, any> = {
    "desktop": {
      displayName: "Desktop Webcam",
      requiresAuth: false,
      note: "Local system camera"
    },
    "usb": {
      displayName: "USB Camera",
      requiresAuth: false,
      note: "USB connected camera"
    },
    "oak": {
      displayName: "OAK Camera",
      requiresAuth: false,
      note: "OAK-D 3D camera via DepthAI"
    },
    "ip-http": {
      displayName: "IP Camera (HTTP)",
      requiresAuth: true,
      note: "Network IP camera with HTTP stream"
    },
    "rtsp": {
      displayName: "RTSP Stream",
      requiresAuth: true,
      note: "Real Time Streaming Protocol - may require server-side transcoding"
    },
    "mobile-app": {
      displayName: "Mobile App (IP Webcam)",
      requiresAuth: false,
      note: "IP Webcam app or similar mobile streaming"
    },
    "ngrok": {
      displayName: "NGROK Tunnel",
      requiresAuth: false,
      note: "Secure tunnel via NGROK service"
    },
    "direct-url": {
      displayName: "Direct URL",
      requiresAuth: false,
      note: "Custom streaming URL"
    }
  };

  return {
    type,
    url,
    ...info[type]
  };
}

/**
 * Validates if a camera source URL is accessible
 * Returns true if likely valid, false if obviously invalid
 */
export function isValidCameraSource(source: string): boolean {
  // Check for empty string
  if (!source || source.trim() === "") return false;

  // Valid local camera indices
  if (source === "0" || source === "1") return true;

  // Valid URLs
  if (source.startsWith("http://") || source.startsWith("https://") || source.startsWith("rtsp://")) {
    try {
      new URL(source);
      return true;
    } catch {
      return false;
    }
  }

  // Could be a numeric index
  if (/^\d+$/.test(source)) return true;

  return false;
}
