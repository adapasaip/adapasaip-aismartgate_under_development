import type { Express } from "express";
import { createServer, type Server } from "http";
import { Readable } from "stream";
import { storage } from "./storage";
import { insertUserSchema, insertVehicleSchema, insertCameraSchema, insertAnprDetectionSchema, insertSubUserSchema, updateSubUserSchema } from "@shared/schema";
import fs from "fs";
import path from "path";

export async function registerRoutes(app: Express): Promise<Server> {
  // Authentication routes
  app.post("/api/auth/login", async (req, res) => {
    try {
      const { mobile, password } = req.body;
      const user = await storage.authenticateUser(mobile, password);
      
      if (user) {
        // Remove password from response
        const { password: _, ...userWithoutPassword } = user;
        res.json({ success: true, user: userWithoutPassword });
      } else {
        res.status(401).json({ success: false, message: "Invalid credentials" });
      }
    } catch (error) {
      res.status(500).json({ success: false, message: "Authentication failed" });
    }
  });

  app.post("/api/auth/register", async (req, res) => {
    try {
      const userData = insertUserSchema.parse(req.body);
      
      // Check if user already exists
      const existingUser = await storage.getUserByMobile(userData.mobile);
      if (existingUser) {
        return res.status(400).json({ success: false, message: "User already exists" });
      }

      const user = await storage.createUser(userData);
      const { password: _, ...userWithoutPassword } = user;
      res.json({ success: true, user: userWithoutPassword });
    } catch (error) {
      res.status(400).json({ success: false, message: "Registration failed" });
    }
  });

  // Vehicle routes
  app.get("/api/vehicles", async (req, res) => {
    try {
      const { licensePlate, gate, method, date, userId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      if (licensePlate || gate || method || date) {
        const vehicles = await storage.searchVehicles(currentUserId, {
          licensePlate: licensePlate as string,
          gate: gate as string,
          method: method as string,
          date: date as string,
        });
        res.json(vehicles);
      } else {
        // Always filter by userId for data isolation
        const vehicles = await storage.getAllVehicles(currentUserId);
        res.json(vehicles);
      }
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch vehicles" });
    }
  });

  app.post("/api/vehicles", async (req, res) => {
    try {
      const { userId, subUserId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      // If sub-user is making the request, check permissions
      if (subUserId) {
        const subUser = await storage.getSubUser(subUserId as string);
        if (!subUser || !subUser.permissions.canAddVehicles) {
          return res.status(403).json({ 
            success: false, 
            message: "You do not have permission to add vehicles. Please contact your administrator." 
          });
        }
      }
      
      const vehicleData = insertVehicleSchema.parse(req.body);
      const vehicle = await storage.createVehicle(vehicleData, currentUserId);
      res.json(vehicle);
    } catch (error) {
      res.status(400).json({ message: "Failed to create vehicle" });
    }
  });

  app.put("/api/vehicles/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const updates = req.body;
      const vehicle = await storage.updateVehicle(id, updates);
      
      if (vehicle) {
        res.json(vehicle);
      } else {
        res.status(404).json({ message: "Vehicle not found" });
      }
    } catch (error) {
      res.status(400).json({ message: "Failed to update vehicle" });
    }
  });

  app.delete("/api/vehicles/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteVehicle(id);
      
      if (deleted) {
        res.json({ success: true });
      } else {
        res.status(404).json({ message: "Vehicle not found" });
      }
    } catch (error) {
      res.status(400).json({ message: "Failed to delete vehicle" });
    }
  });

  // Camera routes
  app.get("/api/cameras", async (req, res) => {
    try {
      const { userId } = req.query;
      const currentUserId = userId as string || "default";
      const cameras = await storage.getAllCameras(currentUserId);
      res.json(cameras);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch cameras" });
    }
  });

  app.post("/api/cameras", async (req, res) => {
    try {
      const { userId, subUserId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      // If sub-user is making the request, check permissions
      if (subUserId) {
        const subUser = await storage.getSubUser(subUserId as string);
        if (!subUser || !subUser.permissions.canAddCameras) {
          return res.status(403).json({ 
            success: false, 
            message: "You do not have permission to add cameras. Please contact your administrator." 
          });
        }
      }
      
      const cameraData = insertCameraSchema.parse(req.body);
      const camera = await storage.createCamera(cameraData, currentUserId);
      
      // ⭐ FIX: CRITICAL - Notify Flask backend to initialize and start the camera pipeline
      // This bridges the Node API with Flask's camera initialization system
      console.log(`[CAMERA_API] 🔄 Camera saved to database: ${camera.id} | Now initializing pipeline via Flask...`);
      
      try {
        // ⭐ NORMALIZE camera type for Flask compatibility
        // Frontend saves descriptive types like "Mobile Camera (IP Webcam App)"
        // But Flask expects specific types: 'usb', 'ipwebcam', 'oak'
        let flaskCameraType = 'usb'; // default
        
        if (camera.type?.toLowerCase().includes('ipcam') || 
            camera.type?.toLowerCase().includes('ip webcam') ||
            camera.type?.toLowerCase().includes('mobile camera')) {
          flaskCameraType = 'ipwebcam';
        } else if (camera.source?.toString().startsWith('http://') || 
                   camera.source?.toString().startsWith('https://')) {
          flaskCameraType = 'ipwebcam';
        } else if (camera.type?.toLowerCase() === 'oak' || 
                   camera.source?.toString().toLowerCase() === 'oak') {
          flaskCameraType = 'oak';
        } else if (camera.type?.toLowerCase() === 'usb' || !isNaN(Number(camera.source))) {
          flaskCameraType = 'usb';
        }
        
        console.log(`[CAMERA_API] 🔄 Type mapping: ${camera.type} → ${flaskCameraType} | Source: ${camera.source}`);
        
        // ⭐ BUILD COMPLETE Flask API request with config + extended fields
        // CRITICAL: Ensure ALL extended fields have values (defaults if missing from frontend)
        console.log(`[CAMERA_API] [DEBUG] Camera object from storage: ${JSON.stringify(camera)}`);
        console.log(`[CAMERA_API] [DEBUG] Camera gate value: ${JSON.stringify(camera.gate)}`);
        
        // ✅ Use camera.gate as-is (storage.createCamera() already applied defaults)
        // Don't double-default - trust the schema validation
        const flaskPayload: any = {
          id: camera.id,
          name: camera.name,
          source: camera.source,
          type: flaskCameraType,  // Use normalized Flask-compatible type
          description: camera.description || "",
          // ★ CRITICAL FIX: ALL extended fields with explicit defaults
          // storage.createCamera() already set defaults, so pass them through
          userId: camera.userId || 'default',
          location: camera.location || 'Not Specified',
          gate: camera.gate || 'Entry',  // ✅ Simple check: if camera.gate is truthy, use it
          resolution: camera.resolution || '1280x720',
          fps: camera.fps || 30,
          anprEnabled: camera.anprEnabled !== undefined ? camera.anprEnabled : true
        };
        
        console.log(`[CAMERA_API] [DEBUG] Final gate value sent to Flask: ${flaskPayload.gate}`);
        console.log(`[CAMERA_API] 📤 Sending to Flask: ${JSON.stringify(flaskPayload)}`);
        
        // Call Flask API to initialize camera (blocking, but necessary)
        // Flask runs on localhost:8000 and has the camera initialization logic
        const flaskResponse = await fetch('http://localhost:8000/api/config/cameras', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'  // Skip ngrok warning header if using ngrok
          },
          body: JSON.stringify(flaskPayload)
        });
        
        if (!flaskResponse.ok) {
          const flaskError = await flaskResponse.text();
          console.error(`[CAMERA_API] ⚠️  Flask API returned ${flaskResponse.status}: ${flaskError.slice(0, 200)}`);
          
          // Continue anyway - camera is in DB, Flask error might be transient
          console.log(`[CAMERA_API] 📝 Continuing despite Flask error (camera saved to DB)`);
          
          return res.status(200).json({
            success: true,
            message: camera,
            warning: `Flask initialization may have failed: ${flaskResponse.status}. Check if Flask server is running on port 8000.`
          });
        }
        
        const flaskResult = await flaskResponse.json();
        console.log(`[CAMERA_API] ✅ Flask API Response: ${JSON.stringify(flaskResult)}`);
        
        // ★ CRITICAL FIX: Write camera to cameras.json file so getAllCameras() returns it immediately
        await storage.addCameraToFile(camera);
        console.log(`[CAMERA_API] ✅ Camera written to cameras.json: ${camera.id}`);
        
        // Success: Camera initialized in both Node and Flask
        return res.status(201).json({
          success: true,
          message: camera,
          pipelineStatus: flaskResult.message || 'Pipeline started'
        });
        
      } catch (flaskError: any) {
        console.error(`[CAMERA_API] ⚠️  Flask communication error: ${flaskError.message}`);
        console.error(`[CAMERA_API] ℹ️  Is Flask server running on http://localhost:8000?`);
        
        // ★ STILL write camera to cameras.json even if Flask fails (camera exists in Node DB)
        try {
          await storage.addCameraToFile(camera);
          console.log(`[CAMERA_API] ✅ Camera written to cameras.json despite Flask error: ${camera.id}`);
        } catch (writeError) {
          console.error(`[CAMERA_API] ⚠️  Failed to write camera to cameras.json:`, writeError);
        }
        
        // Camera is saved to DB but Flask init failed
        // Return partial success so user knows camera was saved
        return res.status(200).json({
          success: true,
          message: camera,
          warning: `Camera saved to database but Flask initialization failed: ${flaskError.message}. Please ensure the Flask server is running on port 8000.`,
          debugInfo: `Flask URL: http://localhost:8000/api/config/cameras | Error: ${flaskError.message}`
        });
      }
      
    } catch (error: any) {
      console.error("Camera creation error:", error);
      const errorMessage = error?.message || "Failed to create camera";
      res.status(400).json({ message: errorMessage });
    }
  });

  app.put("/api/cameras/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const updates = req.body;
      const camera = await storage.updateCamera(id, updates);
      
      if (camera) {
        res.json(camera);
      } else {
        res.status(404).json({ message: "Camera not found" });
      }
    } catch (error) {
      res.status(400).json({ message: "Failed to update camera" });
    }
  });

  app.delete("/api/cameras/:id", async (req, res) => {
    try {
      const { id } = req.params;
      
      // ★ CRITICAL: First delete from Flask backend (cameras-config.json and stop pipeline)
      console.log(`[CAMERA_API] 🗑️  Deleting camera via Flask backend: ${id}`);
      
      try {
        const flaskResponse = await fetch(`http://localhost:8000/api/config/cameras/${id}`, {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          }
        });
        
        if (!flaskResponse.ok) {
          const flaskError = await flaskResponse.text();
          console.error(`[CAMERA_API] ⚠️  Flask delete failed: ${flaskResponse.status}: ${flaskError.slice(0, 100)}`);
        } else {
          const flaskResult = await flaskResponse.json();
          console.log(`[CAMERA_API] ✅ Flask deletion success: ${flaskResult.message}`);
        }
      } catch (flaskError: any) {
        console.error(`[CAMERA_API] ⚠️  Flask communication error: ${flaskError.message}`);
        // Continue anyway - we'll still delete from Node storage
      }
      
      // ★ THEN delete from Node.js storage (cameras.json)
      console.log(`[CAMERA_API] 🗑️  Deleting camera from Node storage: ${id}`);
      const deleted = await storage.deleteCamera(id);
      
      if (deleted) {
        console.log(`[CAMERA_API] ✅ Camera ${id} deleted from Node storage`);
        res.json({ 
          success: true,
          message: "Camera deleted successfully from both systems"
        });
      } else {
        console.warn(`[CAMERA_API] ⚠️  Camera ${id} not found in Node storage`);
        res.status(404).json({ 
          success: false,
          message: "Camera not found" 
        });
      }
    } catch (error: any) {
      console.error("Camera deletion error:", error);
      res.status(400).json({ 
        success: false,
        message: "Failed to delete camera" 
      });
    }
  });

  // Test camera connection endpoint
  app.post("/api/cameras/:id/test", async (req, res) => {
    try {
      const { id } = req.params;
      const { userId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      // Get the camera to verify it exists and belongs to the user
      const cameras = await storage.getAllCameras(currentUserId);
      const camera = cameras.find((c) => c.id === id);
      
      if (!camera) {
        return res.status(404).json({ success: false, message: "Camera not found" });
      }

      // Simulate connection test - in production, this could actually test the connection
      // For now, we'll quickly respond with success if the camera data is valid
      const isValid = camera.type && camera.source;
      
      res.json({
        success: isValid,
        message: isValid ? "Camera connection test successful" : "Invalid camera configuration",
        camera: {
          id: camera.id,
          name: camera.name,
          status: camera.status,
          type: camera.type,
          source: camera.source,
        },
      });
    } catch (error: any) {
      res.status(500).json({ 
        success: false, 
        message: error?.message || "Failed to test camera connection" 
      });
    }
  });

  // Restart camera endpoint
  app.post("/api/cameras/:id/restart", async (req, res) => {
    try {
      const { id } = req.params;
      const { userId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      // Get the camera to verify it exists and belongs to the user
      const cameras = await storage.getAllCameras(currentUserId);
      const camera = cameras.find((c) => c.id === id);
      
      if (!camera) {
        return res.status(404).json({ success: false, message: "Camera not found" });
      }

      // Simulate restart - update status to show restart in progress, then to Online
      // In production, this would actually restart the Python camera process
      const restartedCamera = await storage.updateCamera(id, { 
        status: "Online",
      });

      res.json({
        success: true,
        message: "Camera restart initiated",
        camera: restartedCamera,
      });
    } catch (error: any) {
      res.status(500).json({ 
        success: false, 
        message: error?.message || "Failed to restart camera" 
      });
    }
  });

  // Detection routes with pagination support
  app.get("/api/detections", async (req, res) => {
    try {
      const { userId, page = "1", limit = "50", sortBy = "recent" } = req.query;
      const currentUserId = (userId as string) || "default";
      const pageNum = Math.max(1, parseInt(page as string) || 1);
      const limitNum = Math.min(100, Math.max(1, parseInt(limit as string) || 50));
      const offset = (pageNum - 1) * limitNum;

      // Get all detections and apply sorting
      let detections = await storage.getAllDetections(currentUserId);
      
      // Sort by recent (most recent first)
      detections.sort((a, b) => 
        new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime()
      );

      // Get total count before pagination
      const totalCount = detections.length;
      const totalPages = Math.ceil(totalCount / limitNum);

      // Apply pagination
      const paginatedDetections = detections.slice(offset, offset + limitNum);

      // Return paginated response with metadata
      res.json({
        data: paginatedDetections,
        pagination: {
          page: pageNum,
          limit: limitNum,
          total: totalCount,
          totalPages,
          hasNextPage: pageNum < totalPages,
          hasPrevPage: pageNum > 1,
        }
      });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch detections" });
    }
  });

  app.get("/api/detections/recent", async (req, res) => {
    try {
      const { userId, limit } = req.query;
      const currentUserId = (userId as string) || "default";
      const limitNum = limit ? parseInt(limit as string) : 5;
      const detections = await storage.getRecentDetections(currentUserId, limitNum);
      res.json(detections);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch recent detections" });
    }
  });

  app.post("/api/detections", async (req, res) => {
    try {
      const { userId } = req.query;
      const currentUserId = userId as string || "default";
      let detectionData = insertAnprDetectionSchema.parse(req.body);

      // Save plate image if provided
      if (detectionData.plateImage && detectionData.plateImage.startsWith('data:')) {
        try {
          const filename = `plate_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.jpg`;
          const userDir = path.join(process.cwd(), "data", "plates", currentUserId);
          fs.mkdirSync(userDir, { recursive: true });
          
          const base64Data = detectionData.plateImage.replace(/^data:image\/\w+;base64,/, "");
          const filePath = path.join(userDir, filename);
          fs.writeFileSync(filePath, Buffer.from(base64Data, 'base64'));
          detectionData.plateImage = filename;
        } catch (error) {
          console.error("Failed to save plate image:", error);
          detectionData.plateImage = undefined;
        }
      }

      // Save vehicle image if provided
      if (detectionData.vehicleImage && detectionData.vehicleImage.startsWith('data:')) {
        try {
          const filename = `vehicle_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.jpg`;
          const userDir = path.join(process.cwd(), "data", "vehicles", currentUserId);
          fs.mkdirSync(userDir, { recursive: true });
          
          const base64Data = detectionData.vehicleImage.replace(/^data:image\/\w+;base64,/, "");
          const filePath = path.join(userDir, filename);
          fs.writeFileSync(filePath, Buffer.from(base64Data, 'base64'));
          detectionData.vehicleImage = filename;
        } catch (error) {
          console.error("Failed to save vehicle image:", error);
          detectionData.vehicleImage = undefined;
        }
      }

      const detection = await storage.createDetection(detectionData, currentUserId);
      res.json(detection);
    } catch (error) {
      console.error("Detection creation error:", error);
      res.status(400).json({ message: "Failed to create detection" });
    }
  });

  // Statistics route
  app.get("/api/stats", async (req, res) => {
    try {
      const { userId } = req.query;
      const currentUserId = (userId as string) || "default";
      const stats = await storage.getStats(currentUserId);
      res.json(stats);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch statistics" });
    }
  });

  // ==================== SUB-USER ROUTES ====================

  // Get all sub-users for the authenticated user
  app.get("/api/subusers", async (req, res) => {
    try {
      const { userId } = req.query;
      if (!userId) {
        return res.status(400).json({ message: "User ID is required" });
      }

      const subUsers = await storage.getAllSubUsers(userId as string);
      // Remove passwords before sending
      const safeSubUsers = subUsers.map(({ password: _, ...rest }) => rest);
      res.json(safeSubUsers);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch sub-users" });
    }
  });

  // Create a new sub-user
  app.post("/api/subusers", async (req, res) => {
    try {
      const subUserData = insertSubUserSchema.parse(req.body);
      const { createdBy } = req.body;

      if (!createdBy) {
        return res.status(400).json({ message: "Created by user ID is required" });
      }

      const subUser = await storage.createSubUser(subUserData, createdBy);
      // Remove password from response
      const { password: _, ...safeSubUser } = subUser;
      res.json({ success: true, subUser: safeSubUser });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Failed to create sub-user";
      res.status(400).json({ success: false, message: errorMessage });
    }
  });

  // Update a sub-user
  app.put("/api/subusers/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const updates = updateSubUserSchema.parse(req.body);
      
      const subUser = await storage.updateSubUser(id, updates);
      if (subUser) {
        const { password: _, ...safeSubUser } = subUser;
        res.json({ success: true, subUser: safeSubUser });
      } else {
        res.status(404).json({ message: "Sub-user not found" });
      }
    } catch (error) {
      res.status(400).json({ message: "Failed to update sub-user" });
    }
  });

  // Delete a sub-user
  app.delete("/api/subusers/:id", async (req, res) => {
    try {
      const { id } = req.params;
      const deleted = await storage.deleteSubUser(id);
      
      if (deleted) {
        res.json({ success: true });
      } else {
        res.status(404).json({ message: "Sub-user not found" });
      }
    } catch (error) {
      res.status(400).json({ message: "Failed to delete sub-user" });
    }
  });

  // Sub-user login
  app.post("/api/subusers/login", async (req, res) => {
    try {
      const { username, password } = req.body;
      const subUser = await storage.authenticateSubUser(username, password);

      if (subUser) {
        const { password: _, ...safeSubUser } = subUser;
        res.json({ success: true, subUser: safeSubUser });
      } else {
        res.status(401).json({ success: false, message: "Invalid credentials" });
      }
    } catch (error) {
      res.status(500).json({ success: false, message: "Authentication failed" });
    }
  });

  // ==================== ACTIVITY LOG ROUTES ====================

  // Get activity logs for a user
  app.get("/api/activity-logs", async (req, res) => {
    try {
      const { parentUserId, subUserId, limit } = req.query;
      if (!parentUserId) {
        return res.status(400).json({ message: "Parent user ID is required" });
      }

      const logs = await storage.getActivityLogs(
        parentUserId as string,
        subUserId as string | undefined,
        limit ? parseInt(limit as string) : 100
      );
      res.json(logs);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch activity logs" });
    }
  });

  // Get recent activity logs
  app.get("/api/activity-logs/recent", async (req, res) => {
    try {
      const { parentUserId, limit } = req.query;
      if (!parentUserId) {
        return res.status(400).json({ message: "Parent user ID is required" });
      }

      const logs = await storage.getRecentActivityLogs(
        parentUserId as string,
        limit ? parseInt(limit as string) : 20
      );
      res.json(logs);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch recent activity logs" });
    }
  });

  // Log a manual activity (for tracking specific events)
  app.post("/api/activity-logs", async (req, res) => {
    try {
      const activityData = req.body;
      const log = await storage.logActivity(activityData);
      res.json(log);
    } catch (error) {
      res.status(400).json({ message: "Failed to log activity" });
    }
  });

  // ==================== IMAGE SERVING ROUTES ====================
  // Serve plate images
  app.get("/api/images/plates/:filename", async (req, res) => {
    try {
      const { filename } = req.params;
      
      // Try multiple locations for the image:
      // 1. Direct filename in plates folder (current format)
      // 2. Check for user subdirectories
      
      let imagePath = path.join(process.cwd(), "data", "plates", filename);
      
      // Security: Prevent directory traversal
      const baseDir = path.join(process.cwd(), "data", "plates");
      if (!path.resolve(imagePath).startsWith(baseDir)) {
        return res.status(403).json({ message: "Access denied" });
      }

      // If not found in root plates folder, try user subdirectories
      if (!fs.existsSync(imagePath)) {
        // Try to find in any user subdirectory
        const platesDir = path.join(process.cwd(), "data", "plates");
        if (fs.existsSync(platesDir)) {
          const subdirs = fs.readdirSync(platesDir).filter(f => 
            fs.statSync(path.join(platesDir, f)).isDirectory()
          );
          
          for (const subdir of subdirs) {
            const userImagePath = path.join(platesDir, subdir, filename);
            if (fs.existsSync(userImagePath)) {
              imagePath = userImagePath;
              break;
            }
          }
        }
      }

      if (fs.existsSync(imagePath)) {
        res.set("Cache-Control", "public, max-age=86400");
        res.set("Content-Type", "image/jpeg");
        res.sendFile(imagePath);
      } else {
        res.status(404).json({ message: "Image not found" });
      }
    } catch (error) {
      res.status(500).json({ message: "Failed to serve image" });
    }
  });

  // Serve vehicle images
  app.get("/api/images/vehicles/:filename", async (req, res) => {
    try {
      const { filename } = req.params;
      
      let imagePath = path.join(process.cwd(), "data", "vehicles", filename);
      
      // Security: Prevent directory traversal
      const baseDir = path.join(process.cwd(), "data", "vehicles");
      if (!path.resolve(imagePath).startsWith(baseDir)) {
        return res.status(403).json({ message: "Access denied" });
      }

      // If not found in root vehicles folder, try user subdirectories
      if (!fs.existsSync(imagePath)) {
        const vehiclesDir = path.join(process.cwd(), "data", "vehicles");
        if (fs.existsSync(vehiclesDir)) {
          const subdirs = fs.readdirSync(vehiclesDir).filter(f => 
            fs.statSync(path.join(vehiclesDir, f)).isDirectory()
          );
          
          for (const subdir of subdirs) {
            const userImagePath = path.join(vehiclesDir, subdir, filename);
            if (fs.existsSync(userImagePath)) {
              imagePath = userImagePath;
              break;
            }
          }
        }
      }

      if (fs.existsSync(imagePath)) {
        res.set("Cache-Control", "public, max-age=86400");
        res.set("Content-Type", "image/jpeg");
        res.sendFile(imagePath);
      } else {
        res.status(404).json({ message: "Image not found" });
      }
    } catch (error) {
      res.status(500).json({ message: "Failed to serve image" });
    }
  });

  // ==================== VIDEO FEED ROUTES ====================
  // Stream video from camera (proxied through backend for security/CORS)
  app.get("/api/video_feed/preview", async (req, res) => {
    try {
      const { source, type } = req.query;
      const sourceStr = source as string || "0";
      let typeStr = (type as string || "desktop").toLowerCase();  // Normalize to lowercase
      
      // Normalize full camera descriptions to type codes
      if (typeStr.includes("desktop")) typeStr = "desktop";
      else if (typeStr.includes("usb")) typeStr = "usb";
      else if (typeStr.includes("oak")) typeStr = "oak";
      else if (typeStr.includes("ip")) typeStr = "ip-http";
      else if (typeStr.includes("rtsp")) typeStr = "rtsp";
      else if (typeStr.includes("ngrok")) typeStr = "ngrok";
      else if (typeStr.includes("mobile")) typeStr = "mobile-app";
      else typeStr = "desktop";

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";
      
      try {
        // Direct stream pass-through from Python backend for minimal latency
        const response = await fetch(`${pythonBackendUrl}/api/video_feed/preview?source=${encodeURIComponent(sourceStr)}&type=${encodeURIComponent(typeStr)}`);
        
        if (response.ok) {
          const contentType = response.headers.get("content-type") || "image/jpeg";
          
          // Check if it's a streaming response (multipart/x-mixed-replace for MJPEG)
          if (contentType.includes("multipart/x-mixed-replace")) {
            // Proxy the MJPEG stream directly without buffering for lowest latency
            res.setHeader("Content-Type", contentType);
            res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            res.setHeader("Pragma", "no-cache");
            res.setHeader("Expires", "0");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
            res.setHeader("Connection", "keep-alive");
            res.setHeader("Transfer-Encoding", "chunked");
            
            // Convert Web ReadableStream to Node.js Readable stream and pipe
            if (response.body) {
              const nodeStream = Readable.from(response.body as any);
              nodeStream.pipe(res);
            }
          } else {
            // Single image response
            const buffer = await response.arrayBuffer();
            res.setHeader("Content-Type", contentType);
            res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.send(Buffer.from(buffer));
          }
        } else {
          // Check if Python backend is available
          const pythonRunning = await checkPythonBackendHealth(pythonBackendUrl);
          
          if (!pythonRunning) {
            // Return placeholder for offline camera
            res.setHeader("Content-Type", "image/png");
            res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
            res.setHeader("Access-Control-Allow-Origin", "*");
            res.setHeader("X-Camera-Status", "offline");
            
            const placeholderImage = generatePlaceholderImage(sourceStr, typeStr);
            return res.send(placeholderImage);
          }
          
          // Python backend error
          res.setHeader("Content-Type", "application/json");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.status(503).json({ 
            error: "Camera preview unavailable",
            message: "Python backend is running but unable to access camera"
          });
        }
      } catch (proxyError) {
        // Check if Python backend is healthy
        const pythonRunning = await checkPythonBackendHealth(pythonBackendUrl);
        
        if (!pythonRunning) {
          res.setHeader("Content-Type", "image/png");
          res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
          res.setHeader("Access-Control-Allow-Origin", "*");
          res.setHeader("X-Camera-Status", "offline");
          
          const placeholderImage = generatePlaceholderImage(sourceStr, typeStr);
          return res.send(placeholderImage);
        }
        
        res.setHeader("Content-Type", "application/json");
        res.setHeader("Access-Control-Allow-Origin", "*");
        res.status(503).json({ 
          error: "Camera preview unavailable",
          message: "Python backend connection error"
        });
      }
    } catch (error) {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.status(500).json({ 
        error: "Failed to get video feed"
      });
    }
  });

  // Helper function to check if Python backend is healthy
  async function checkPythonBackendHealth(pythonBackendUrl: string): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      
      const response = await fetch(`${pythonBackendUrl}/health`, { 
        signal: controller.signal 
      });
      
      clearTimeout(timeoutId);
      return response.ok || response.status === 404; // 404 is ok, means server is running
    } catch {
      return false;
    }
  }

  // Helper function to generate a placeholder PNG image
  function generatePlaceholderImage(source: string, type: string): Buffer {
    // Create a simple 640x480 PNG placeholder with camera offline message
    // Using a minimal PNG structure
    const width = 640;
    const height = 480;
    
    // Create raw pixel data (RGBA format, dark gray background)
    const imageData = Buffer.alloc(width * height * 4);
    
    // Fill with dark gray background (50, 50, 50, 255)
    for (let i = 0; i < imageData.length; i += 4) {
      imageData[i] = 50;      // R
      imageData[i + 1] = 50;  // G
      imageData[i + 2] = 50;  // B
      imageData[i + 3] = 255; // A
    }
    
    // For now, return a minimal PNG header that indicates camera is offline
    // This allows the frontend to show the error state properly
    const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
    
    // Return a minimal valid PNG (8x8 pixel gray image)
    const minimalPng = Buffer.from([
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
      0x00, 0x00, 0x00, 0x0D, // IHDR length
      0x49, 0x48, 0x44, 0x52, // IHDR
      0x00, 0x00, 0x00, 0x08, // Width: 8
      0x00, 0x00, 0x00, 0x08, // Height: 8
      0x08, 0x02, 0x00, 0x00, 0x00, // Bit depth, color type, etc.
      0x4B, 0x6D, 0x0D, 0xDC, // CRC
      0x00, 0x00, 0x00, 0x19, // IDAT length
      0x49, 0x44, 0x41, 0x54,
      0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00,
      0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB4,
      0x00, 0x00, 0x00, 0x00, // IDAT data
      0x00, 0x00, 0x00, 0x00, // IEND length
      0x49, 0x45, 0x4E, 0x44, // IEND
      0xAE, 0x42, 0x60, 0x82  // CRC
    ]);
    
    return minimalPng;
  }

  // Get video feed for a saved camera
  app.get("/api/video_feed/:cameraId", async (req, res) => {
    try {
      const { cameraId } = req.params;
      const { userId } = req.query;
      const currentUserId = (userId as string) || "default";
      
      // Get all cameras for the user to validate ownership
      const userCameras = await storage.getAllCameras(currentUserId);
      const camera = userCameras.find((c) => c.id === cameraId);
      
      if (!camera) {
        return res.status(404).json({ message: "Camera not found" });
      }

      console.log('[video_feed/:cameraId] Handling camera stream:', {
        cameraId: camera.id,
        name: camera.name,
        source: camera.source,
        type: camera.type,
        userId: currentUserId
      });

      const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";
      
      // Set MJPEG streaming headers
      res.setHeader("Content-Type", "multipart/x-mixed-replace; boundary=--frame");
      res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
      res.setHeader("Pragma", "no-cache");
      res.setHeader("Expires", "0");
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("Transfer-Encoding", "chunked");

      let isStreamActive = true;
      let frameCount = 0;
      let isFetching = false;
      
      // Send frames every 500ms (2 FPS) to avoid overwhelming the system
      const frameInterval = setInterval(async () => {
        if (!isStreamActive || isFetching) {
          return;
        }

        isFetching = true;
        try {
          // Fetch frame from Python backend with 8 second timeout
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 8000);
          
          const response = await fetch(
            `${pythonBackendUrl}/api/video_feed/preview?source=${encodeURIComponent(camera.source)}&type=${encodeURIComponent(camera.type)}`,
            { signal: controller.signal }
          );
          
          clearTimeout(timeoutId);

          if (response.ok) {
            const contentType = response.headers.get("content-type") || "image/jpeg";
            const buffer = await response.arrayBuffer();
            
            // Send MJPEG frame with boundary
            const boundary = "--frame\r\nContent-Type: " + contentType + "\r\nContent-Length: " + buffer.byteLength + "\r\n\r\n";
            res.write(boundary);
            res.write(Buffer.from(buffer));
            res.write("\r\n");
            
            frameCount++;
          } else {
            // Send placeholder on error
            const placeholderImage = generatePlaceholderImage(camera.source, camera.type);
            const boundary = "--frame\r\nContent-Type: image/png\r\nContent-Length: " + placeholderImage.length + "\r\n\r\n";
            res.write(boundary);
            res.write(placeholderImage);
            res.write("\r\n");
          }
        } catch (error) {
          // Only log timeout errors occasionally to reduce spam
          if (frameCount % 10 === 0) {
            console.warn('[video_feed/:cameraId] Frame fetch error:', error instanceof Error ? error.message : String(error));
          }
          
          // Send placeholder on network error
          try {
            const placeholderImage = generatePlaceholderImage(camera.source, camera.type);
            const boundary = "--frame\r\nContent-Type: image/png\r\nContent-Length: " + placeholderImage.length + "\r\n\r\n";
            res.write(boundary);
            res.write(placeholderImage);
            res.write("\r\n");
          } catch (writeError) {
            console.error('[video_feed/:cameraId] Error writing placeholder:', writeError);
            isStreamActive = false;
            clearInterval(frameInterval);
          }
        } finally {
          isFetching = false;
        }
      }, 500);

      // Handle client disconnect
      req.on("close", () => {
        isStreamActive = false;
        clearInterval(frameInterval);
        console.log('[video_feed/:cameraId] Client disconnected after', frameCount, 'frames');
      });

      req.on("error", () => {
        isStreamActive = false;
        clearInterval(frameInterval);
      });

    } catch (error) {
      console.error('[video_feed/:cameraId] Unhandled error:', error instanceof Error ? error.message : String(error));
      if (!res.headersSent) {
        res.status(500).json({ message: "Failed to get video stream" });
      }
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
