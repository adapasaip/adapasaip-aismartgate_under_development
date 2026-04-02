/**
 * MJPEG Stream Player - Handles continuous MJPEG streaming to canvas
 * Optimized for low-latency playback with frame dropping and efficient rendering
 */

export class MjpegStreamPlayer {
  private canvas: HTMLCanvasElement;
  private streamUrl: string;
  private onConnect: () => void;
  private onError: () => void;
  private isRunning: boolean = false;
  private abortController: AbortController | null = null;
  private pendingFrame: Uint8Array | null = null;
  private isDrawing: boolean = false;
  private frameCount: number = 0;

  constructor(
    canvas: HTMLCanvasElement,
    streamUrl: string,
    onConnect: () => void,
    onError: () => void
  ) {
    this.canvas = canvas;
    this.streamUrl = streamUrl;
    this.onConnect = onConnect;
    this.onError = onError;
  }

  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.abortController = new AbortController();
    this.playStream();
  }

  stop(): void {
    this.isRunning = false;
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }

  private async playStream(): Promise<void> {
    try {
      const response = await fetch(this.streamUrl, {
        signal: this.abortController?.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const contentType = response.headers.get("content-type") || "";
      if (!contentType.includes("multipart")) {
        throw new Error("Not a valid MJPEG stream");
      }

      // Signal successful connection
      this.onConnect();

      const reader = response.body.getReader();
      let buffer = new Uint8Array();
      
      // Start frame render loop
      this.startRenderLoop();

      while (this.isRunning) {
        const { done, value } = await reader.read();
        if (done) break;

        // Append new data to buffer
        buffer = this.appendBuffer(buffer, value);

        // Use optimized JPEG marker search instead of boundary parsing
        const jpegStartMarker = new Uint8Array([0xff, 0xd8]); // JPEG SOI
        const jpegEndMarker = new Uint8Array([0xff, 0xd9]);   // JPEG EOI
        
        let jpegStartIndex = this.findPatternOptimized(buffer, jpegStartMarker);
        
        while (jpegStartIndex !== -1) {
          // Find the JPEG end marker
          const jpegEndIndex = this.findPatternOptimized(
            buffer,
            jpegEndMarker,
            jpegStartIndex + 2
          );

          if (jpegEndIndex !== -1) {
            // Extract complete JPEG frame
            const jpegData = buffer.slice(jpegStartIndex, jpegEndIndex + 2);
            
            // Store frame for rendering (drop old pending frame if not rendered yet)
            this.pendingFrame = jpegData;
            this.frameCount++;

            // Remove processed data from buffer
            buffer = buffer.slice(jpegEndIndex + 2);
            
            // Look for next frame
            jpegStartIndex = this.findPatternOptimized(buffer, jpegStartMarker);
          } else {
            // Incomplete frame, wait for more data
            break;
          }
        }

        // Keep only recent data (max 300KB to prevent memory bloat)
        if (buffer.length > 300000) {
          // Find the last JPEG start marker and keep from there
          let lastJpegStart = -1;
          for (let i = buffer.length - 1000; i >= 0; i--) {
            if (buffer[i] === 0xff && buffer[i + 1] === 0xd8) {
              lastJpegStart = i;
              break;
            }
          }
          if (lastJpegStart > 0) {
            buffer = buffer.slice(lastJpegStart);
          } else {
            buffer = new Uint8Array(); // Clear and start fresh
          }
        }
      }

      reader.releaseLock();
    } catch (error) {
      if (error instanceof Error && error.name !== "AbortError") {
        console.error("Stream error:", error);
        this.onError();
      }
    }
  }

  private startRenderLoop(): void {
    const renderFrame = () => {
      if (!this.isRunning) return;

      // Check if we have a pending frame to render
      if (this.pendingFrame && !this.isDrawing) {
        this.isDrawing = true;
        
        // Render the frame directly without async overhead
        this.renderFrameSync(this.pendingFrame).then(() => {
          this.isDrawing = false;
        });
      }

      requestAnimationFrame(renderFrame);
    };

    requestAnimationFrame(renderFrame);
  }

  private renderFrameSync(jpegData: Uint8Array): Promise<void> {
    return new Promise((resolve) => {
      const blob = new Blob([jpegData], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);

      const img = new Image();
      img.onload = () => {
        // Directly render without nested requestAnimationFrame
        const ctx = this.canvas.getContext("2d", { alpha: false });
        if (ctx) {
          // Update canvas size only if needed
          if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
            this.canvas.width = img.width;
            this.canvas.height = img.height;
          }
          ctx.drawImage(img, 0, 0);
        }
        URL.revokeObjectURL(url);
        resolve();
      };
      
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve();
      };
      
      img.src = url;
    });
  }

  private findPatternOptimized(
    buffer: Uint8Array,
    pattern: Uint8Array,
    startIndex: number = 0
  ): number {
    // Fast path using indexOf for first byte, then check pattern
    const firstByte = pattern[0];
    const patternLen = pattern.length;

    for (let i = startIndex; i <= buffer.length - patternLen; i++) {
      if (buffer[i] === firstByte) {
        // Quick check for remaining pattern bytes
        let match = true;
        for (let j = 1; j < patternLen; j++) {
          if (buffer[i + j] !== pattern[j]) {
            match = false;
            break;
          }
        }
        if (match) return i;
      }
    }
    return -1;
  }

  private appendBuffer(
    buffer: Uint8Array,
    newData: Uint8Array
  ): Uint8Array {
    const combined = new Uint8Array(buffer.length + newData.length);
    combined.set(buffer);
    combined.set(newData, buffer.length);
    return combined;
  }
}
