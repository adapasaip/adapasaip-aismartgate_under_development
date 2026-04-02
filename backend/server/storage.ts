import { 
  type User, type InsertUser, 
  type Vehicle, type InsertVehicle, 
  type Camera, type InsertCamera, 
  type AnprDetection, type InsertAnprDetection, 
  type Stats,
  type SubUser, type InsertSubUser, type UpdateSubUser,
  type ActivityLog, type InsertActivityLog
} from "@shared/schema";
import { randomUUID } from "crypto";
import fs from "fs/promises";
import path from "path";
// @ts-ignore
import bcrypt from "bcryptjs";

// Database sync disabled - using JSON file storage only
// import { databaseSync } from "./database-sync";

export interface IStorage {
  // User methods
  getUser(id: string): Promise<User | undefined>;
  getUserByMobile(mobile: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  authenticateUser(mobile: string, password: string): Promise<User | null>;
  updateUser(id: string, updates: Partial<User>): Promise<User | undefined>;

  // Vehicle methods - with userId filtering
  getAllVehicles(userId: string): Promise<Vehicle[]>;
  getVehicle(id: string): Promise<Vehicle | undefined>;
  createVehicle(vehicle: InsertVehicle, userId: string): Promise<Vehicle>;
  updateVehicle(id: string, vehicle: Partial<Vehicle>): Promise<Vehicle | undefined>;
  deleteVehicle(id: string): Promise<boolean>;
  searchVehicles(userId: string, filters: { licensePlate?: string; gate?: string; method?: string; date?: string }): Promise<Vehicle[]>;

  // Camera methods - with userId filtering
  getAllCameras(userId: string): Promise<Camera[]>;
  getCamera(id: string): Promise<Camera | undefined>;
  createCamera(camera: InsertCamera, userId: string): Promise<Camera>;
  updateCamera(id: string, camera: Partial<Camera>): Promise<Camera | undefined>;
  deleteCamera(id: string): Promise<boolean>;

  // ANPR Detection methods - with userId filtering
  getAllDetections(userId: string): Promise<AnprDetection[]>;
  getRecentDetections(userId: string, limit?: number): Promise<AnprDetection[]>;
  createDetection(detection: InsertAnprDetection, userId: string): Promise<AnprDetection>;

  // Sub-user methods
  getAllSubUsers(parentUserId: string): Promise<SubUser[]>;
  getSubUser(id: string): Promise<SubUser | undefined>;
  getSubUserByUsername(username: string, parentUserId: string): Promise<SubUser | undefined>;
  createSubUser(subUser: InsertSubUser, createdBy: string): Promise<SubUser>;
  updateSubUser(id: string, updates: UpdateSubUser): Promise<SubUser | undefined>;
  deleteSubUser(id: string): Promise<boolean>;
  authenticateSubUser(username: string, password: string): Promise<SubUser | null>;

  // Activity log methods
  logActivity(activity: InsertActivityLog): Promise<ActivityLog>;
  getActivityLogs(parentUserId: string, subUserId?: string, limit?: number): Promise<ActivityLog[]>;
  getRecentActivityLogs(parentUserId: string, limit?: number): Promise<ActivityLog[]>;

  // Statistics - per user
  getStats(userId: string): Promise<Stats>;
}

export class JsonStorage implements IStorage {
  private dataDir = path.join(process.cwd(), 'data');
  private usersFile = path.join(this.dataDir, 'users.json');
  private camerasFile = path.join(this.dataDir, 'cameras.json');
  private detectionsFile = path.join(this.dataDir, 'detections.json');
  private subUsersFile = path.join(this.dataDir, 'subusers.json');
  private activityLogsFile = path.join(this.dataDir, 'activity-logs.json');

  // In-memory caches with TTL for reducing file I/O
  private fileCache: Map<string, { data: any[]; timestamp: number }> = new Map();
  private cacheTTL = 500; // Cache for 500ms to batch operations and reduce disk reads
  private writeQueues: Map<string, { data: any[]; timeout: NodeJS.Timeout }> = new Map();

  constructor() {
    this.initializeData();
  }

  private async initializeData() {
    try {
      await fs.mkdir(this.dataDir, { recursive: true });
      
      // Initialize users with default admin if no users exist
      try {
        await fs.access(this.usersFile);
      } catch {
        const hashedPassword = await bcrypt.hash("admin123", 10);
        const defaultUser: User = {
          id: randomUUID(),
          mobile: "1234567890",
          password: hashedPassword,
          fullName: "Admin User",
          email: "admin@aismartgate.com",
          role: "admin",
          isActive: true,
          createdAt: new Date().toISOString(),
        };
        await this.writeFile(this.usersFile, [defaultUser]);
      }

      // Initialize other files with empty arrays
      const filesToInit = [
        this.camerasFile,
        this.detectionsFile,
        this.subUsersFile,
        this.activityLogsFile
      ];

      for (const file of filesToInit) {
        try {
          await fs.access(file);
        } catch {
          await this.writeFile(file, []);
        }
      }
    } catch (error) {
      console.error('Failed to initialize data directory:', error);
    }
  }

  private isCacheValid(filepath: string): boolean {
    const cached = this.fileCache.get(filepath);
    if (!cached) return false;
    
    // ⭐ CRITICAL: For detections.json, ALWAYS re-read since Python writes directly to file
    // Without this, cache becomes stale and new detections from Python backend won't appear
    if (filepath.includes('detections.json')) {
      return false; // Never use cache for detections - always read fresh from disk
    }
    
    return Date.now() - cached.timestamp < this.cacheTTL;
  }

  private async readFile<T>(filepath: string): Promise<T[]> {
    try {
      // Check cache first
      if (this.isCacheValid(filepath)) {
        return this.fileCache.get(filepath)!.data;
      }

      const data = await fs.readFile(filepath, 'utf-8');
      let parsed: T[];

      // ⭐ JSON LINES SUPPORT: Detect format and parse accordingly
      // detections.json uses JSON Lines format (Python backend), other files use regular JSON array
      if (filepath.includes('detections.json')) {
        // JSON Lines format: one JSON object per line
        parsed = [];
        const lines = data.trim().split('\n');
        for (const line of lines) {
          if (line.trim()) {
            try {
              parsed.push(JSON.parse(line));
            } catch {
              // Skip malformed lines
              console.error(`[STORAGE] Skipped malformed line in ${filepath}: ${line.slice(0, 50)}`);
            }
          }
        }
      } else {
        // Regular JSON array format for other files
        parsed = JSON.parse(data);
      }

      // Cache the result
      this.fileCache.set(filepath, { data: parsed, timestamp: Date.now() });
      
      return parsed;
    } catch {
      return [];
    }
  }

  private async writeFile<T>(filepath: string, data: T[]): Promise<void> {
    // Invalidate cache immediately
    this.fileCache.delete(filepath);

    // Cancel pending write for this file if it exists
    if (this.writeQueues.has(filepath)) {
      clearTimeout(this.writeQueues.get(filepath)!.timeout);
    }

    // Return a promise that resolves when the write is complete
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(async () => {
        try {
          // ⭐ JSON LINES SUPPORT: Write detections.json in JSON Lines format
          // Other files use regular JSON array format
          let fileContent: string;
          
          if (filepath.includes('detections.json')) {
            // JSON Lines format: one JSON object per line, compact without indentation
            fileContent = data.map(item => JSON.stringify(item)).join('\n');
          } else {
            // Regular JSON array format for other files
            fileContent = JSON.stringify(data, null, 2);
          }
          
          await fs.writeFile(filepath, fileContent);
          this.writeQueues.delete(filepath);
          resolve(); // Resolve promise when write is done
        } catch (error) {
          console.error(`[STORAGE] Error writing to ${filepath}:`, error);
          reject(error);
        }
      }, 10); // Reduced from 50ms to 10ms for faster writes (still batches operations)

      // Update the queued data in case of batching
      this.writeQueues.set(filepath, { data, timeout });
    });
  }

  // ==================== USER METHODS ====================
  
  async getUser(id: string): Promise<User | undefined> {
    const users = await this.readFile<User>(this.usersFile);
    return users.find(user => user.id === id);
  }

  async getUserByMobile(mobile: string): Promise<User | undefined> {
    const users = await this.readFile<User>(this.usersFile);
    return users.find(user => user.mobile === mobile);
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const users = await this.readFile<User>(this.usersFile);
    
    // Hash password before storing
    const hashedPassword = await bcrypt.hash(insertUser.password, 10);
    
    const user: User = {
      ...insertUser,
      password: hashedPassword,
      id: randomUUID(),
      role: "user",
      isActive: true,
      createdAt: new Date().toISOString(),
    };
    
    users.push(user);
    await this.writeFile(this.usersFile, users);
    return user;
  }

  async authenticateUser(mobile: string, password: string): Promise<User | null> {
    const user = await this.getUserByMobile(mobile);
    if (!user || !user.isActive) {
      return null;
    }
    
    // Compare password with hashed password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return null;
    }

    // Update last login
    await this.updateUser(user.id, { lastLogin: new Date().toISOString() });
    
    return user;
  }

  async updateUser(id: string, updates: Partial<User>): Promise<User | undefined> {
    const users = await this.readFile<User>(this.usersFile);
    const index = users.findIndex(user => user.id === id);
    if (index === -1) return undefined;

    users[index] = { ...users[index], ...updates };
    await this.writeFile(this.usersFile, users);
    return users[index];
  }

  async deleteUserData(userId: string): Promise<boolean> {
    try {
      // Delete user
      const users = await this.readFile<User>(this.usersFile);
      const userIndex = users.findIndex(user => user.id === userId);
      if (userIndex === -1) return false;

      users.splice(userIndex, 1);
      await this.writeFile(this.usersFile, users);

      // Delete all user's vehicles (from detections.json)
      const detections = await this.readFile<AnprDetection>(this.detectionsFile);
      const userDetections = detections.filter(d => d.userId === userId);
      const filteredDetections = detections.filter(d => d.userId !== userId);
      await this.writeFile(this.detectionsFile, filteredDetections);

      // Delete all user's cameras
      const cameras = await this.readFile<Camera>(this.camerasFile);
      const userCameras = cameras.filter(c => c.userId === userId);
      const filteredCameras = cameras.filter(c => c.userId !== userId);
      await this.writeFile(this.camerasFile, filteredCameras);

      // Note: All user's detections (including vehicles) were already deleted above

      return true;
    } catch (error) {
      console.error('Failed to delete user data:', error);
      return false;
    }
  }

  // ==================== VEHICLE METHODS ====================
  // ★ CONSOLIDATED: Vehicles are now read from detections.json (single source of truth)
  
  async getAllVehicles(userId: string): Promise<Vehicle[]> {
    // Single source of truth: detections.json
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);

    // Filter by userId - all detections must have userId
    const filtered = detections.filter(d => {
      if (userId === "default") {
        return !d.userId || d.userId === "default";
      }
      return d.userId === userId || d.userId === "default" || !d.userId;
    });

    // Convert detections to vehicle format
    const vehicles: Vehicle[] = filtered.map(d => ({
      id: d.id,
      userId: d.userId,
      licensePlate: d.licensePlate,
      vehicleType: d.vehicleType || "Unknown",
      driverName: d.driverName || "", // Now available in detection
      driverMobile: d.driverMobile || "", // Now available in detection
      entryTime: d.entryTime || d.detectedAt,
      exitTime: d.exitTime || undefined,
      gate: d.gate || "Entry",
      detectionMethod: (d.detectionMethod || "ANPR") as "ANPR" | "Manual",
      status: d.status as "Registered" | "Unregistered",
      plateImage: d.plateImage,
      vehicleImage: d.vehicleImage,
      notes: d.notes || `Detection: ${d.captureReason || 'Vehicle detected'}`,
      confidence: d.confidence,
      ocrConfidence: d.ocrConfidence,
      cameraId: d.cameraId,
    }));
    
    // Sort by entry time descending (newest first)
    vehicles.sort((a, b) => new Date(b.entryTime).getTime() - new Date(a.entryTime).getTime());
    
    return vehicles;
  }

  async getVehicle(id: string): Promise<Vehicle | undefined> {
    // Single source of truth: detections.json
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const detection = detections.find(d => d.id === id);
    
    if (detection) {
      return {
        id: detection.id,
        userId: detection.userId,
        licensePlate: detection.licensePlate,
        vehicleType: detection.vehicleType || "Unknown",
        driverName: detection.driverName || "",
        driverMobile: detection.driverMobile || "",
        entryTime: detection.entryTime || detection.detectedAt,
        exitTime: detection.exitTime || undefined,
        gate: detection.gate || "Entry",
        detectionMethod: (detection.detectionMethod || "ANPR") as "ANPR" | "Manual",
        status: detection.status as "Registered" | "Unregistered",
        plateImage: detection.plateImage,
        vehicleImage: detection.vehicleImage,
        notes: detection.notes || `Detection: ${detection.captureReason || 'Vehicle detected'}`,
        confidence: detection.confidence,
        ocrConfidence: detection.ocrConfidence,
        cameraId: detection.cameraId,
      };
    }

    return undefined;
  }

  async createVehicle(insertVehicle: InsertVehicle, userId: string): Promise<Vehicle> {
    // Write to detections.json (single source of truth)
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    
    // Convert Vehicle to AnprDetection format
    const detection: AnprDetection = {
      id: randomUUID(),
      userId,
      licensePlate: insertVehicle.licensePlate,
      vehicleType: insertVehicle.vehicleType,
      driverName: insertVehicle.driverName,
      driverMobile: insertVehicle.driverMobile,
      entryTime: insertVehicle.entryTime,
      exitTime: insertVehicle.exitTime,
      gate: insertVehicle.gate as "Entry" | "Exit",
      detectionMethod: insertVehicle.detectionMethod,
      status: insertVehicle.status as "Registered" | "Unregistered",
      plateImage: insertVehicle.plateImage,
      vehicleImage: insertVehicle.vehicleImage,
      notes: insertVehicle.notes,
      confidence: 1.0, // Manually added vehicles have high confidence
      ocrConfidence: 1.0,
      cameraId: insertVehicle.cameraId || "",
      detectedAt: insertVehicle.entryTime,
      captureReason: "Manual entry",
    };
    
    detections.push(detection);
    await this.writeFile(this.detectionsFile, detections);

    // Return as Vehicle format
    return {
      id: detection.id,
      userId,
      licensePlate: insertVehicle.licensePlate,
      vehicleType: insertVehicle.vehicleType,
      driverName: insertVehicle.driverName || "",
      driverMobile: insertVehicle.driverMobile || "",
      entryTime: insertVehicle.entryTime,
      exitTime: insertVehicle.exitTime,
      gate: insertVehicle.gate as "Entry" | "Exit",
      detectionMethod: insertVehicle.detectionMethod as "ANPR" | "Manual",
      status: insertVehicle.status as "Registered" | "Unregistered",
      plateImage: insertVehicle.plateImage,
      vehicleImage: insertVehicle.vehicleImage,
      notes: insertVehicle.notes,
      confidence: 1.0,
      ocrConfidence: 1.0,
      cameraId: insertVehicle.cameraId || "",
    };
  }

  async updateVehicle(id: string, updates: Partial<Vehicle>): Promise<Vehicle | undefined> {
    // Single source of truth: detections.json
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const detectionIndex = detections.findIndex(d => d.id === id);
    
    if (detectionIndex === -1) return undefined;

    const detection = detections[detectionIndex];
    
    // Map vehicle updates to detection fields
    if (updates.status) detection.status = updates.status;
    if (updates.driverName !== undefined) detection.driverName = updates.driverName;
    if (updates.driverMobile !== undefined) detection.driverMobile = updates.driverMobile;
    if (updates.notes !== undefined) detection.notes = updates.notes;
    if (updates.vehicleType) detection.vehicleType = updates.vehicleType;
    if (updates.gate) detection.gate = updates.gate as "Entry" | "Exit";
    if (updates.exitTime !== undefined) detection.exitTime = updates.exitTime;
    
    detections[detectionIndex] = detection;
    await this.writeFile(this.detectionsFile, detections);

    // Return the updated vehicle view
    return {
      id: detection.id,
      userId: detection.userId,
      licensePlate: detection.licensePlate,
      vehicleType: detection.vehicleType || "Unknown",
      driverName: detection.driverName || "",
      driverMobile: detection.driverMobile || "",
      entryTime: detection.entryTime || detection.detectedAt,
      exitTime: detection.exitTime || undefined,
      gate: detection.gate || "Entry",
      detectionMethod: (detection.detectionMethod || "ANPR") as "ANPR" | "Manual",
      status: detection.status as "Registered" | "Unregistered",
      plateImage: detection.plateImage,
      vehicleImage: detection.vehicleImage,
      notes: detection.notes || `Detection: ${detection.captureReason || 'Vehicle detected'}`,
      confidence: detection.confidence,
      ocrConfidence: detection.ocrConfidence,
      cameraId: detection.cameraId,
    };
  }

  async deleteVehicle(id: string): Promise<boolean> {
    // Single source of truth: detections.json
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const detectionIndex = detections.findIndex(d => d.id === id);
    
    if (detectionIndex === -1) return false;

    detections.splice(detectionIndex, 1);
    await this.writeFile(this.detectionsFile, detections);

    return true;
  }

  async searchVehicles(userId: string, filters: { licensePlate?: string; gate?: string; method?: string; date?: string }): Promise<Vehicle[]> {
    // Get user's vehicles first (properly filtered by ownership)
    const userVehicles = await this.getAllVehicles(userId);

    // Apply search filters
    return userVehicles.filter(vehicle => {
      if (filters.licensePlate && !vehicle.licensePlate.toLowerCase().includes(filters.licensePlate.toLowerCase())) {
        return false;
      }
      if (filters.gate && filters.gate !== "All" && vehicle.gate !== filters.gate) {
        return false;
      }
      if (filters.method && filters.method !== "All" && vehicle.detectionMethod !== filters.method) {
        return false;
      }
      if (filters.date && !vehicle.entryTime.startsWith(filters.date)) {
        return false;
      }
      return true;
    });
  }

  // ==================== CAMERA METHODS ====================
  
  async getAllCameras(userId: string): Promise<Camera[]> {
    // ★ CRITICAL FIX: Read from cameras.json which is synced by Python backend
    // Python backend syncs cameras-config.json → cameras.json
    // The cameras.json is the SOURCE OF TRUTH from Flask backend
    // 
    // IMPORTANT: Flask OVERWRITES cameras.json regularly, removing the userId field
    // that Node.js added. This is by design - forEach backend owns data for cameras.
    // So we ALWAYS return ALL cameras (no userId filtering) because:
    // 1. Cameras are synced from Flask backend which doesn't use userId
    // 2. userId filtering happens at the API layer (routes.ts), not storage layer
    // 3. Multiple users see all cameras (they're global system resources)
    
    try {
      const cameras = await this.readFile<Camera>(this.camerasFile);
      return cameras;
    } catch (error) {
      console.error('[STORAGE] Error reading cameras from cameras.json:', error);
      return [];
    }
  }

  async getCamera(id: string): Promise<Camera | undefined> {
    const cameras = await this.readFile<Camera>(this.camerasFile);
    return cameras.find(camera => camera.id === id);
  }

  async addCameraToFile(camera: Camera): Promise<void> {
    // ★ FIX: Explicitly write camera to cameras.json after creation
    // This ensures getAllCameras() can immediately return the new camera
    const cameras = await this.readFile<Camera>(this.camerasFile);
    
    // ★ CRITICAL FIX: Ensure all extended fields are present with defaults
    // ✅ IMPORTANT: Check for undefined/null/empty, preserve user values
    const completeCamera: Camera = {
      ...camera,
      // Ensure all extended fields have values (use defaults if not provided)
      status: camera.status && camera.status.trim() ? camera.status : 'Online',
      userId: camera.userId && camera.userId !== 'undefined' ? camera.userId : 'default',
      location: camera.location && camera.location.trim() ? camera.location : 'Not Specified',
      gate: camera.gate && camera.gate.trim() ? camera.gate : 'Entry',
      resolution: camera.resolution && camera.resolution.trim() ? camera.resolution : '1280x720',
      fps: camera.fps || 30,
      anprEnabled: camera.anprEnabled !== undefined ? camera.anprEnabled : true
    };
    
    // Check if camera already exists (by ID)
    const existingIndex = cameras.findIndex(c => c.id === camera.id);
    
    if (existingIndex >= 0) {
      // Update existing - merge with existing data to preserve any fields
      cameras[existingIndex] = { ...cameras[existingIndex], ...completeCamera };
    } else {
      // Add new
      cameras.push(completeCamera);
    }
    
    await this.writeFile(this.camerasFile, cameras);
    
    // ★ DO NOT sync to cameras-config.json here!
    // Flask API handles cameras-config.json.
    // Python watcher reads cameras-config.json and syncs to cameras.json.
    // Node.js only manages cameras.json.
    // ★ Syncing here causes race conditions and prevents watcher from detecting changes.
  }

  async createCamera(insertCamera: InsertCamera, userId: string): Promise<Camera> {
    // ★ CRITICAL FIX: Ensure all extended fields have default values
    // Frontend might not always send all fields, so we set sensible defaults
    // ✅ IMPORTANT: Check for undefined/null, NOT empty string!
    // This preserves user values while only using defaults when fields are truly missing
    const camera: Camera = {
      ...insertCamera,
      id: randomUUID(),
      userId: userId || 'default',
      createdAt: new Date().toISOString(),
      // ★ CRITICAL: Set defaults for extended fields ONLY if truly missing (undefined/null)
      // Do NOT default empty strings - user may have intentionally left fields empty
      status: insertCamera.status !== undefined && insertCamera.status !== null ? insertCamera.status : 'Online',
      location: insertCamera.location !== undefined && insertCamera.location !== null && insertCamera.location.trim() !== '' 
        ? insertCamera.location 
        : 'Not Specified',
      gate: insertCamera.gate !== undefined && insertCamera.gate !== null && insertCamera.gate.trim() !== '' 
        ? insertCamera.gate 
        : 'Entry',
      resolution: insertCamera.resolution !== undefined && insertCamera.resolution !== null ? insertCamera.resolution : '1280x720',
      fps: insertCamera.fps !== undefined && insertCamera.fps !== null ? insertCamera.fps : 30,
      anprEnabled: insertCamera.anprEnabled !== undefined && insertCamera.anprEnabled !== null ? insertCamera.anprEnabled : true
    };
    
    // DON'T add to cameras array yet - let the Flask route handle the full sync
    // Just return the camera object with ALL fields populated
    // The routes.ts file will handle writing to cameras.json after Flask confirms

    return camera;
  }

  async updateCamera(id: string, updates: Partial<Camera>): Promise<Camera | undefined> {
    const cameras = await this.readFile<Camera>(this.camerasFile);
    const index = cameras.findIndex(camera => camera.id === id);
    if (index === -1) return undefined;

    cameras[index] = { ...cameras[index], ...updates };
    await this.writeFile(this.camerasFile, cameras);
    
    // ★ NOTE: Do NOT sync to cameras-config.json
    // Flask API handles cameras-config.json updates.
    // Python watcher syncs cameras-config.json → cameras.json.

    return cameras[index];
  }

  async deleteCamera(id: string): Promise<boolean> {
    const cameras = await this.readFile<Camera>(this.camerasFile);
    const index = cameras.findIndex(camera => camera.id === id);
    if (index === -1) return false;

    cameras.splice(index, 1);
    await this.writeFile(this.camerasFile, cameras);
    
    // ★ NOTE: Do NOT remove from cameras-config.json
    // Flask API handles cameras-config.json deletions.
    // Python watcher detects removal and cleans up.

    return true;
  }

  // ==================== ANPR DETECTION METHODS ====================
  
  async getAllDetections(userId: string): Promise<AnprDetection[]> {
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const cameras = await this.getAllCameras(userId);
    const cameraIds = cameras.map(c => c.id);

    // Filter detections to show only user's data
    // Handle both new detections (with userId) and old detections (without userId)
    // Also handle backward compatibility with "default" userId
    const filtered = detections.filter(d => {
      // ⭐ CRITICAL FIX: For "default" user (not logged in), show ALL detections from any user
      // This allows testing and viewing detections even when not authenticated
      if (userId === "default") {
        return true;
      }

      // For authenticated users:
      // 1. Show detections with matching userId
      // 2. Show detections with userId "default" (system-captured vehicles)
      // 3. Show detections from cameras they own
      if (d.userId === userId) return true;
      if (d.userId === "default" || !d.userId) return true;
      if (cameraIds.includes(d.cameraId)) return true;

      // Reject all others to prevent cross-user data leakage
      return false;
    });

    return filtered;
  }

  async getRecentDetections(userId: string, limit = 5): Promise<AnprDetection[]> {
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const vehicles = await this.getAllVehicles(userId);
    const cameras = await this.getAllCameras(userId);
    const cameraIds = cameras.map(c => c.id);

    // Filter detections to show only user's data
    // Handle both new detections (with userId) and old detections (without userId)
    // Also handle backward compatibility with "default" userId
    const userDetections = detections.filter(d => {
      // ⭐ CRITICAL FIX: For "default" user (not logged in), show ALL detections from any user
      // This allows testing and viewing detections even when not authenticated
      // Production: Remove this or change to more restrictive logic
      if (userId === "default") {
        return true; // Show all detections for default user (testing mode)
      }

      // For authenticated users:
      // 1. Show detections with matching userId
      // 2. Show detections with userId "default" (system-captured vehicles)
      // 3. Show detections from cameras they own
      if (d.userId === userId) return true;
      if (d.userId === "default" || !d.userId) return true;
      if (cameraIds.includes(d.cameraId)) return true;

      // Reject all others to prevent cross-user data leakage
      return false;
    });

    // ★ ENHANCED: Sort by actual event time - use exitTime for exits, detectedAt for entries
    // This ensures vehicles that EXIT appear most recent (based on exit time)
    const sortedDetections = userDetections.sort((a, b) => {
      // Get the actual event time for each detection
      const aEventTime = a.exitTime 
        ? new Date(a.exitTime).getTime() 
        : new Date(a.detectedAt).getTime();
      const bEventTime = b.exitTime 
        ? new Date(b.exitTime).getTime() 
        : new Date(b.detectedAt).getTime();
      
      // Sort in DESCENDING order (most recent first)
      return bEventTime - aEventTime;
    });

    // Deduplicate: Group by license plate and keep only most recent event per plate
    const seenPlates = new Set<string>();
    const uniqueDetections = sortedDetections
      .filter(detection => {
        if (seenPlates.has(detection.licensePlate)) {
          return false;
        }
        seenPlates.add(detection.licensePlate);
        return true;
      })
      .slice(0, limit); // Limit to specified count (default 5)

    // Enrich with vehicle status
    return uniqueDetections.map(detection => {
      const vehicle = vehicles.find(v => v.licensePlate === detection.licensePlate);
      return {
        ...detection,
        status: vehicle?.status || detection.status,
      };
    });
  }

  async createDetection(insertDetection: InsertAnprDetection, userId: string): Promise<AnprDetection> {
    const detections = await this.readFile<AnprDetection>(this.detectionsFile);
    const detection: AnprDetection = {
      ...insertDetection,
      id: randomUUID(),
      userId,
    };
    detections.push(detection);
    await this.writeFile(this.detectionsFile, detections);

    return detection;
  }

  // ==================== STATISTICS ====================
  
  async getStats(userId: string): Promise<Stats> {
    const vehicles = await this.getAllVehicles(userId);
    const cameras = await this.getAllCameras(userId);
    const detections = await this.getAllDetections(userId);
    
    const today = new Date().toISOString().split('T')[0];
    const todayEntries = vehicles.filter(v => v.entryTime.startsWith(today)).length;
    const todayExits = vehicles.filter(v => v.exitTime && v.exitTime.startsWith(today)).length;
    const activeCameras = cameras.filter(c => c.status === "Online").length;
    const anprEnabled = cameras.filter(c => c.anprEnabled).length;

    return {
      totalVehicles: vehicles.length,
      todayEntries,
      todayExits,
      activeCameras,
      anprDetections: detections.length,
      totalCameras: cameras.length,
      anprEnabled,
      framesCaptured: `${Math.floor(detections.length * 30)}K`,
    };
  }

  // ==================== SUB-USER METHODS ====================

  async getAllSubUsers(parentUserId: string): Promise<SubUser[]> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    return subUsers.filter(su => su.parentUserId === parentUserId && su.isActive);
  }

  async getSubUser(id: string): Promise<SubUser | undefined> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    return subUsers.find(su => su.id === id);
  }

  async getSubUserByUsername(username: string, parentUserId: string): Promise<SubUser | undefined> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    return subUsers.find(su => su.username === username && su.parentUserId === parentUserId && su.isActive);
  }

  async createSubUser(insertSubUser: InsertSubUser, createdBy: string): Promise<SubUser> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    
    // Check limit: max 5 sub-users per parent
    const existingCount = subUsers.filter(su => su.parentUserId === insertSubUser.parentUserId && su.isActive).length;
    if (existingCount >= 5) {
      throw new Error("Maximum 5 sub-users allowed per account");
    }

    // Check if username already exists for this parent
    const existingSubUser = await this.getSubUserByUsername(insertSubUser.username, insertSubUser.parentUserId);
    if (existingSubUser) {
      throw new Error("Username already exists for this account");
    }

    const hashedPassword = await bcrypt.hash(insertSubUser.password, 10);

    const subUser: SubUser = {
      ...insertSubUser,
      password: hashedPassword,
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      createdBy,
    };

    subUsers.push(subUser);
    await this.writeFile(this.subUsersFile, subUsers);

    // Log the activity
    await this.logActivity({
      parentUserId: insertSubUser.parentUserId,
      subUserId: "",
      action: "ADD_SUBUSER",
      details: { username: subUser.username, email: subUser.email, fullName: subUser.fullName, permissions: subUser.permissions },
      timestamp: new Date().toISOString(),
      status: "SUCCESS",
    });

    return subUser;
  }

  async updateSubUser(id: string, updates: UpdateSubUser): Promise<SubUser | undefined> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    const index = subUsers.findIndex(su => su.id === id);
    if (index === -1) return undefined;

    const updated: SubUser = {
      ...subUsers[index],
      ...updates,
      updatedAt: new Date().toISOString(),
    };

    subUsers[index] = updated;
    await this.writeFile(this.subUsersFile, subUsers);

    // Log the activity
    await this.logActivity({
      parentUserId: subUsers[index].parentUserId,
      subUserId: "",
      action: "UPDATE_SUBUSER",
      details: { username: updated.username, email: updated.email, fullName: updated.fullName, updates: Object.keys(updates) },
      timestamp: new Date().toISOString(),
      status: "SUCCESS",
    });

    return updated;
  }

  async deleteSubUser(id: string): Promise<boolean> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    const subUser = subUsers.find(su => su.id === id);
    if (!subUser) return false;

    // Soft delete - mark as inactive
    const index = subUsers.findIndex(su => su.id === id);
    subUsers[index].isActive = false;
    await this.writeFile(this.subUsersFile, subUsers);

    // Log the activity
    await this.logActivity({
      parentUserId: subUser.parentUserId,
      subUserId: "",
      action: "DELETE_SUBUSER",
      details: { username: subUser.username, email: subUser.email },
      timestamp: new Date().toISOString(),
      status: "SUCCESS",
    });

    return true;
  }

  async authenticateSubUser(username: string, password: string): Promise<SubUser | null> {
    const subUsers = await this.readFile<SubUser>(this.subUsersFile);
    const subUser = subUsers.find(su => su.username === username && su.isActive);

    if (!subUser) {
      return null;
    }

    const isPasswordValid = await bcrypt.compare(password, subUser.password);
    if (!isPasswordValid) {
      return null;
    }

    // Update last login
    const index = subUsers.findIndex(su => su.id === subUser.id);
    subUsers[index].lastLogin = new Date().toISOString();
    await this.writeFile(this.subUsersFile, subUsers);

    // Log login activity
    await this.logActivity({
      parentUserId: subUser.parentUserId,
      subUserId: subUser.id,
      action: "LOGIN",
      timestamp: new Date().toISOString(),
      status: "SUCCESS",
    });

    return subUser;
  }

  // ==================== ACTIVITY LOG METHODS ====================

  async logActivity(activity: InsertActivityLog): Promise<ActivityLog> {
    const logs = await this.readFile<ActivityLog>(this.activityLogsFile);

    const log: ActivityLog = {
      ...activity,
      id: randomUUID(),
    };

    logs.push(log);
    
    // Keep only last 10,000 logs for performance (sliding window)
    if (logs.length > 10000) {
      logs.splice(0, logs.length - 10000);
    }

    await this.writeFile(this.activityLogsFile, logs);
    return log;
  }

  async getActivityLogs(parentUserId: string, subUserId?: string, limit = 100): Promise<ActivityLog[]> {
    const logs = await this.readFile<ActivityLog>(this.activityLogsFile);

    let filtered = logs.filter(log => log.parentUserId === parentUserId);

    if (subUserId) {
      filtered = filtered.filter(log => log.subUserId === subUserId);
    }

    // Sort by timestamp descending and apply limit
    return filtered
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  async getRecentActivityLogs(parentUserId: string, limit = 20): Promise<ActivityLog[]> {
    const logs = await this.readFile<ActivityLog>(this.activityLogsFile);

    return logs
      .filter(log => log.parentUserId === parentUserId)
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }
}

export const storage = new JsonStorage();
