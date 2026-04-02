import { useEffect, useRef, useState } from "react";
import { MjpegStreamPlayer } from "@/lib/mjpeg-stream";

interface MjpegPlayerProps {
  streamUrl: string;
  className?: string;
  onConnect?: () => void;
  onError?: () => void;
}

export function MjpegPlayer({
  streamUrl,
  className = "w-full h-full",
  onConnect,
  onError,
}: MjpegPlayerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playerRef = useRef<MjpegStreamPlayer | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!streamUrl || !canvasRef.current) return;

    // Cleanup previous player
    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }

    setIsConnected(false);

    // Create and start new player
    playerRef.current = new MjpegStreamPlayer(
      canvasRef.current,
      streamUrl,
      () => {
        setIsConnected(true);
        onConnect?.();
      },
      () => {
        setIsConnected(false);
        onError?.();
      }
    );

    playerRef.current.start();

    return () => {
      if (playerRef.current) {
        playerRef.current.stop();
        playerRef.current = null;
      }
    };
  }, [streamUrl, onConnect, onError]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        aspectRatio: "16/9",
        backgroundColor: "#1e293b",
      }}
    />
  );
}
