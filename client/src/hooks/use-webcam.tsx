import { useEffect, useRef, useState } from "react";

export function useWebcam(deviceIndex: string = "0") {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function startWebcam() {
      try {
        setIsLoading(true);
        setError(null);

        // Get available video devices
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        if (videoDevices.length === 0) {
          throw new Error("No camera devices found");
        }

        // Parse device index (0, 1, 2, etc.)
        const index = parseInt(deviceIndex) || 0;
        const selectedDevice = videoDevices[index] || videoDevices[0];

        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: selectedDevice.deviceId,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });

        if (!mounted) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setIsLoading(false);
      } catch (err) {
        if (!mounted) return;
        console.error("Webcam error:", err);
        setError(err instanceof Error ? err.message : "Failed to access camera");
        setIsLoading(false);
      }
    }

    startWebcam();

    return () => {
      mounted = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [deviceIndex]);

  return { videoRef, error, isLoading };
}
