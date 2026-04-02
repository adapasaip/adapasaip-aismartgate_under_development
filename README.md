# AI Smart Gate - ANPR & OCR System

A comprehensive Automatic Number Plate Recognition (ANPR) and Optical Character Recognition (OCR) system with a full-stack web application for real-time vehicle detection, identification, and management at gates and entry points.

## Overview

AI Smart Gate is an intelligent gate access control system that uses advanced computer vision (YOLO object detection) and OCR technology to automatically detect and recognize vehicle license plates. It provides real-time monitoring with a React dashboard, multi-camera support, user management, and detailed activity logging.

## Features

### Core Functionality
- Real-time license plate detection using YOLOv8 neural networks
- Automatic license plate character recognition via Tesseract OCR
- Multi-camera support (IP cameras, MJPEG streams, USB webcams)
- Live MJPEG video streaming to web interface
- Vehicle database with detection history
- Confidence scoring and detection accuracy metrics

### System Features
- User authentication with role-based access control
- Sub-user management and permission delegation
- Comprehensive activity logging and audit trails
- Dashboard with real-time statistics and analytics
- Recent detection history with image capture
- Performance monitoring and system diagnostics
- Responsive web interface with dark/light modes

### Technical Capabilities
- Multi-threaded camera processing
- Hardware-optimized detection (GPU/CUDA support)
- Optimized inference for resource-constrained systems (Raspberry Pi)
- CORS-enabled REST API
- Secure cookie-based authentication
- Drizzle ORM database abstraction

## Prerequisites

### System Requirements
- Linux operating system (tested on Ubuntu 20.04+, Raspberry Pi OS)
- Python 3.10+
- Node.js 18+
- npm or yarn package manager
- 2GB+ RAM minimum (4GB+ recommended)

### Hardware (Optional)
- IP camera with HTTP video stream support
- USB webcam
- NVIDIA GPU for accelerated inference (optional, uses CPU fallback)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd aismartgate-modularized
```

### 2. Backend Setup

#### Create Python Virtual Environment

```bash
cd backend
python3.10 -m venv anpr_venv
source anpr_venv/bin/activate  # On Windows: anpr_venv\Scripts\activate
```

#### Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Download Pre-trained Models

The system uses YOLOv8 for detection. Models are downloaded automatically on first run or can be pre-downloaded:

```bash
python -m ultralytics.yolo detect predict model=yolov8n.pt source=data/test.jpg
```

### 3. Frontend Setup

From the root directory:

```bash
npm install
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# Server Configuration
NODE_ENV=production
PORT=3000

# Database (optional - uses JSON files by default)
DATABASE_URL=your_database_connection_string

# Authentication
JWT_SECRET=your_secret_key_here

# API Configuration
API_BASE_URL=http://localhost:3000/api
VITE_API_BASE_URL=http://localhost:3000/api
```

## Configuration

### Camera Configuration

Cameras are configured via command-line arguments when starting the ANPR backend:

#### Webcam
```bash
cd backend
source anpr_venv/bin/activate
python app/camera_anpr.py --cameras webcam=0
```

#### IP Camera (MJPEG Stream)
```bash
python app/camera_anpr.py --cameras ipcam=http://192.168.0.219:8080/video
```

#### Multiple Cameras
```bash
python app/camera_anpr.py \
  --cameras webcam=0 ipcam=http://192.168.1.100:8080/video \
  --gates webcam:Entry ipcam:Driveway
```

#### Available Camera Sources
- `webcam=<device_index>`: USB webcam (0 for default)
- `ipcam=<stream_url>`: IP camera MJPEG stream URL
- `rtsp=<rtsp_url>`: RTSP stream (requires OpenCV RTSP support)

### Performance Optimization

#### YOLOv8 Model Selection
- `yolov8n.pt` - Nano (fastest, lowest accuracy)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (slower, higher accuracy)
- `yolov8l.pt` - Large (slowest, highest accuracy)

Configured in [backend/app/anpr/core.py](backend/app/anpr/core.py):
```python
MODEL_TYPE = "yolov8n"  # Change as needed
```

#### OCR Optimization
Configure Tesseract path and language in [backend/app/anpr/ocr/](backend/app/anpr/ocr/):
```python
TESSERACT_PATH = "/usr/bin/tesseract"
LANGUAGE = "eng"  # Supported: eng, deu, fra, etc.
```

## Usage

### Starting the Application

#### Development Mode
```bash
npm run dev
```

Starts the frontend development server and backend API.

#### Production Mode
```bash
npm run build
npm run start
```

### Starting Only the Backend (ANPR Service)
```bash
bash backend/start-backend.sh
```

Or with custom camera configuration:
```bash
source backend/anpr_venv/bin/activate
cd backend/app
python camera_anpr.py --cameras ipcam=http://192.168.0.219:8080/video --gates ipcam:Entry
```

### System Service (systemd)

Install as a system service for automatic startup:

```bash
sudo cp aismartgate.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable aismartgate.service
sudo systemctl start aismartgate.service
```

Monitor service status:
```bash
sudo systemctl status aismartgate.service
sudo journalctl -u aismartgate.service -f
```

## API Endpoints

The REST API is available at `http://localhost:3000/api`

### Authentication
- `POST /auth/login` - User login
- `POST /auth/logout` - Logout
- `POST /auth/register` - User registration

### Cameras
- `GET /cameras` - List all cameras
- `POST /cameras` - Add new camera
- `GET /cameras/:id` - Get camera details
- `PATCH /cameras/:id` - Update camera
- `DELETE /cameras/:id` - Remove camera

### Detections
- `GET /detections` - List recent detections
- `GET /detections/license/:plate` - Search by plate number
- `GET /detections/:id` - Get detection details

### Vehicles
- `GET /vehicles` - List registered vehicles
- `POST /vehicles` - Add vehicle
- `PATCH /vehicles/:id` - Update vehicle info

### Activity Logs
- `GET /logs` - Get activity history
- `GET /logs/user/:userId` - Get user-specific logs

### Sub-users
- `GET /subusers` - List sub-users
- `POST /subusers` - Create sub-user
- `PATCH /subusers/:id` - Update sub-user
- `DELETE /subusers/:id` - Remove sub-user

## Project Structure

```
aismartgate-modularized/
├── backend/
│   ├── app/
│   │   ├── camera_anpr.py              # Main ANPR processing entry point
│   │   ├── camera_anpr_original_monolithic.py  # Legacy monolithic version
│   │   ├── camera_manager.py           # Camera manager with multi-threading
│   │   ├── config_api.py               # Configuration API endpoints
│   │   ├── config_manager.py           # Configuration management
│   │   ├── test_detection_pipeline.py  # Detection pipeline testing utility
│   │   ├── static/                     # Static files for Flask
│   │   ├── anpr/                       # Modularized ANPR system
│   │   │   ├── __init__.py
│   │   │   ├── app.py                  # ANPR app initialization
│   │   │   ├── config.py               # Configuration module
│   │   │   ├── core.py                 # Core ANPR logic
│   │   │   ├── manager.py              # Manager orchestration
│   │   │   ├── camera/                 # Camera interface modules
│   │   │   ├── detectors/              # YOLO detection modules
│   │   │   ├── ocr/                    # Tesseract OCR modules
│   │   │   ├── processing/             # Image processing utilities
│   │   │   ├── storage/                # Data storage handlers
│   │   │   ├── streaming/              # MJPEG streaming modules
│   │   │   ├── tracking/               # Vehicle tracking modules
│   │   │   └── utils/                  # Utility functions
│   │   └── __pycache__/
│   ├── server/
│   │   ├── index.ts                    # Express server entry point
│   │   ├── routes.ts                   # API route definitions
│   │   ├── auth.ts                     # Authentication handlers
│   │   ├── storage.ts                  # Database operations
│   │   └── vite.ts                     # Vite integration
│   ├── shared/
│   │   └── schema.ts                   # Shared type definitions
│   ├── anpr_venv/                      # Python virtual environment
│   ├── requirements.txt                # Python dependencies
│   ├── setup_venv.sh                   # Virtual environment setup script
│   ├── start-backend.sh                # Backend startup script (Unix)
│   └── start-backend.ps1               # Backend startup script (Windows)
├── client/
│   ├── index.html                      # HTML entry point
│   ├── public/                         # Static assets
│   ├── src/
│   │   ├── main.tsx                    # React entry point
│   │   ├── App.tsx                     # Root component
│   │   ├── index.css                   # Global styles
│   │   ├── components/
│   │   │   ├── active-cameras-grid.tsx
│   │   │   ├── camera-config.tsx
│   │   │   ├── image-zoom-modal.tsx
│   │   │   ├── live-camera-feed.tsx
│   │   │   ├── mjpeg-player.tsx
│   │   │   ├── recent-detections.tsx
│   │   │   ├── sub-user-management.tsx
│   │   │   └── ui/                     # Shadcn UI components
│   │   ├── pages/
│   │   │   ├── dashboard.tsx
│   │   │   ├── cameras.tsx
│   │   │   ├── vehicles.tsx
│   │   │   ├── login.tsx
│   │   │   ├── profile.tsx
│   │   │   ├── sub-user-profile.tsx
│   │   │   └── not-found.tsx
│   │   ├── hooks/
│   │   │   ├── use-auth.tsx
│   │   │   ├── use-mobile.tsx
│   │   │   ├── use-subusers.tsx
│   │   │   └── use-webcam.tsx
│   │   └── lib/
│   │       ├── config.ts
│   │       ├── utils.ts
│   │       ├── mjpeg-stream.ts
│   │       ├── camera-utils.ts
│   │       └── queryClient.ts
├── data/
│   ├── users.json
│   ├── cameras.json
│   ├── detections.json
│   ├── vehicles.json
│   ├── subusers.json
│   ├── activity-logs.json
│   ├── haarcascade_plate.xml           # Cascade classifier for license plates
│   ├── plates/                         # Detected plate images
│   ├── vehicles/                       # Vehicle images
│   └── videos/                         # Recording directory
├── vite/                               # Vite build output/config
├── package.json                        # Frontend & backend dependencies
├── tsconfig.json                       # TypeScript configuration
├── vite.config.ts                      # Vite build configuration
├── tailwind.config.ts                  # Tailwind CSS configuration
├── postcss.config.js                   # PostCSS configuration
├── drizzle.config.ts                   # Database ORM configuration
├── aismartgate.service                 # systemd service file
├── aismartgate-anpr.desktop            # Desktop shortcut file
├── QUICK_REFERENCE.sh                  # Quick reference commands
├── start-app.sh                        # Main application starter
├── license_plate_detector.onnx         # ONNX model (full precision)
├── license_plate_detector_int8.onnx    # ONNX model (INT8 quantized)
└── README.md                           # This file
```

## Technology Stack

### Frontend
- React 18+ with TypeScript
- Vite for fast build and development
- Tailwind CSS for styling
- Shadcn UI for component library
- TanStack React Query for data fetching
- React Hook Form for form management
- Zod for schema validation

### Backend
- Express.js for REST API
- Flask for camera streaming
- FastAPI framework integration
- Drizzle ORM for database operations
- Node.js runtime environment

### Computer Vision & ML
- YOLOv8 from Ultralytics for object detection
- PyTorch for deep learning inference
- OpenCV for image processing
- Tesseract for Optical Character Recognition

### DevOps & Build
- TypeScript for type safety
- esbuild for server bundling
- Cross-env for environment management
- systemd for service management

## Architecture & Performance Notes

### C++ Dependencies and Performance Benefits

While the codebase is 100% Python and TypeScript, the system leverages compiled C++ libraries through Python bindings for critical performance-sensitive operations. This hybrid approach provides:

**Why C++ is Used (Indirectly)**

The following dependencies contain optimized C++ implementations:

- **PyTorch** - Deep learning inference compiled in C++ with CUDA/CPU optimization
- **OpenCV** - Real-time image processing with highly optimized algorithms
- **NumPy** - Vectorized mathematical operations for rapid data processing
- **Tesseract** - Advanced OCR engine with C++ core
- **Pillow** - Image encoding/decoding with C extensions

**Performance Benefits**

- **10-100x faster execution** compared to pure Python equivalents for vision and ML tasks
- **Raspberry Pi capability** - Optimized for resource-constrained ARM64 systems
- **Real-time processing** - License plate detection completes in milliseconds, not seconds
- **Minimal memory footprint** - Efficient C++ implementations reduce RAM consumption
- **GPU acceleration** - CUDA support for NVIDIA GPUs when available, CPU fallback for compatibility

**No Additional Dependencies Required**

These C++ libraries are pre-compiled into binary wheels (`.whl` files) that `pip` installs automatically. You do not need a C++ compiler installed on your system—the binaries are already built for your architecture (ARM64 for Raspberry Pi, x86_64 for Linux).

**Trade-offs**

Pure Python alternatives exist but are impractical for this use case:
- EasyOCR instead of Tesseract would be similar speed
- Pure Python ML would be 50-100x slower (unsuitable for real-time detection)
- scikit-image instead of OpenCV would be 3-5x slower

The current architecture balances Python's development speed with C++'s execution performance.

## Database Schema

The system stores data in JSON files with the following structure:

### Users
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "password": "hashed_password",
  "name": "User Name",
  "role": "admin|user",
  "createdAt": "2024-01-01T00:00:00Z"
}
```

### Cameras
```json
{
  "id": "uuid",
  "name": "Entry Gate",
  "source": "http://192.168.1.100:8080/video",
  "type": "ipcam|webcam|rtsp",
  "location": "Main Gate",
  "active": true
}
```

### Detections
```json
{
  "id": "uuid",
  "cameraId": "uuid",
  "licensePlate": "ABC123",
  "confidence": 0.95,
  "timestamp": "2024-01-01T12:30:00Z",
  "imageUrl": "/data/plates/abc123_timestamp.jpg",
  "vehicleInfo": {}
}
```

## Performance Monitoring

The system includes built-in performance monitoring. Check metrics via:

```bash
curl http://localhost:3000/api/metrics
```

Returns CPU usage, memory consumption, detection latency, and camera stream health.

## Troubleshooting

### Camera Connection Issues
- Verify IP camera URL is accessible: `curl http://192.168.1.100:8080/video`
- Check network connectivity between host and camera
- Ensure camera credentials are correct if required
- Try disabling CORS restrictions in browser developer tools

### Low Detection Accuracy
- Ensure adequate lighting for plate recognition
- Adjust camera angle for better plate visibility
- Use a larger YOLO model (e.g., yolov8m instead of yolov8n)
- Check that plates are not obscured or damaged

### OCR Errors
- Verify Tesseract installation: `tesseract --version`
- Check that language data is installed: `/usr/share/tesseract-ocr/tessdata/`
- Try preprocessing with different image thresholds in [backend/app/anpr/ocr/](backend/app/anpr/ocr/)

### Performance Issues
- Monitor system resources: `top`, `htop`, or `nvidia-smi` for GPU
- Reduce camera resolution or FPS
- Use a smaller YOLO model (yolov8n)
- Increase inference batch size if processing multiple cameras
- Check logs: `sudo journalctl -u aismartgate.service`

### Python Virtual Environment Issues
- Recreate venv: `rm -rf backend/anpr_venv && python3.10 -m venv backend/anpr_venv`
- Activate environment: `source backend/anpr_venv/bin/activate`
- Reinstall dependencies: `pip install -r backend/requirements.txt`

## Development

### Building Frontend Only
```bash
npm run build
```

### Building Backend Only
```bash
npm run build:server
```

### Building Everything
```bash
npm run build:all
```

### Type Checking
```bash
npm run check
```

### Database Migrations
```bash
npm run db:push
```

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows the existing style
- Comments describe complex logic
- Changes are tested
- Database migrations are included if needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support & Issues

For bugs, feature requests, or questions:
- Check existing GitHub Issues
- Create a new issue with detailed information
- Include system logs: `cat build_output.txt`
- Include diagnostic info: `bash diagnostics-pi.sh`

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## Disclaimer

This system is designed for authorized access control and vehicle management. Ensure compliance with local privacy laws and regulations regarding video surveillance and license plate recognition. Users are responsible for proper camera placement, data protection, and legal compliance.
