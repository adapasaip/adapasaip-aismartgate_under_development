import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// User schema
export const userSchema = z.object({
  id: z.string(),
  mobile: z.string()
    .min(10, "Mobile number must be at least 10 digits")
    .max(15, "Mobile number must not exceed 15 characters")
    .regex(/^\d+$/, "Mobile number must contain only digits"),
  password: z.string()
    .min(6, "Password must be at least 6 characters")
    .max(100, "Password must not exceed 100 characters"),
  fullName: z.string()
    .min(2, "Full name must be at least 2 characters")
    .max(100, "Full name must not exceed 100 characters")
    .optional(),
  email: z.string()
    .email("Invalid email address")
    .optional()
    .or(z.literal("")),
  avatar: z.string().optional(), // URL or base64 string
  role: z.enum(["user", "admin"]).default("user"),
  lastLogin: z.string().optional(),
  isActive: z.boolean().default(true),
  createdAt: z.string(),
});

export const insertUserSchema = userSchema.omit({ id: true, createdAt: true, lastLogin: true });
export const updateUserSchema = userSchema.omit({ id: true, createdAt: true, password: true });
export type User = z.infer<typeof userSchema>;
export type InsertUser = z.infer<typeof insertUserSchema>;
export type UpdateUser = z.infer<typeof updateUserSchema>;

// Vehicle schema
export const vehicleSchema = z.object({
  id: z.string(),
  userId: z.string(),
  licensePlate: z.string()
    .min(1, "License plate is required")
    .max(20, "License plate must not exceed 20 characters")
    .regex(/^[A-Z0-9\s-]+$/i, "License plate contains invalid characters"),
  vehicleType: z.enum(["Car", "Truck", "Motorcycle"]),
  driverName: z.string()
    .min(0)
    .max(100, "Driver name must not exceed 100 characters")
    .optional(),
  driverMobile: z.string()
    .min(0)
    .max(15, "Mobile number must not exceed 15 characters")
    .regex(/^[\d\s\-\+]*$/, "Mobile number contains invalid characters")
    .optional()
    .or(z.literal("")),
  entryTime: z.string()
    .datetime("Invalid entry time format"),
  exitTime: z.string()
    .datetime("Invalid exit time format")
    .optional(),
  gate: z.enum(["Entry", "Exit"]),
  detectionMethod: z.enum(["ANPR", "Manual"]),
  status: z.enum(["Registered", "Unregistered"]),
  plateImage: z.string().optional(),
  vehicleImage: z.string().optional(),
  notes: z.string()
    .max(500, "Notes must not exceed 500 characters")
    .optional(),
  confidence: z.number().optional(),
  ocrConfidence: z.number().optional(),
  cameraId: z.string().optional(),
});

export const insertVehicleSchema = vehicleSchema.omit({ id: true, userId: true });
export type Vehicle = z.infer<typeof vehicleSchema>;
export type InsertVehicle = z.infer<typeof insertVehicleSchema>;

// Camera schema
const cameraBaseSchema = z.object({
  id: z.string(),
  userId: z.string(),
  name: z.string()
    .min(1, "Camera name is required")
    .max(100, "Camera name must not exceed 100 characters"),
  type: z.enum([
    "Desktop Camera (0)",
    "USB Camera (1)",
    "OAK Camera (USB)",
    "IP Camera (http://...)",
    "RTSP Camera (rtsp://...)",
    "Mobile Camera (IP Webcam App)",
    "Mobile Camera via NGROK"
  ]),
  source: z.string()
    .max(500, "Camera source must not exceed 500 characters"),
  description: z.string()
    .max(500, "Description must not exceed 500 characters")
    .optional()
    .or(z.literal("")),
  location: z.string()
    .min(1, "Location is required")
    .max(200, "Location must not exceed 200 characters"),
  gate: z.enum(["Entry", "Exit"]),
  resolution: z.enum(["640x480", "1280x720", "1920x1080"]),
  fps: z.number()
    .min(1, "FPS must be at least 1")
    .max(60, "FPS must not exceed 60")
    .default(30),
  anprEnabled: z.boolean().default(true),
  status: z.enum(["Online", "Offline"]).default("Offline"),
  createdAt: z.string(),
});

// Apply refinement to ensure source is required for non-OAK cameras
export const cameraSchema = cameraBaseSchema.refine(
  (data) => {
    // OAK cameras can have empty source, others must have source
    if (data.type.includes("OAK")) {
      return true; // OAK cameras allow empty source for auto-detection
    }
    return data.source && data.source.trim().length > 0;
  },
  {
    message: "Camera source is required",
    path: ["source"]
  }
);

export const insertCameraSchema = cameraBaseSchema
  .omit({ id: true, userId: true, createdAt: true })
  .refine(
    (data) => {
      // OAK cameras can have empty source, others must have source
      if (data.type.includes("OAK")) {
        return true; // OAK cameras allow empty source for auto-detection
      }
      return data.source && data.source.trim().length > 0;
    },
    {
      message: "Camera source is required",
      path: ["source"]
    }
  );

export type Camera = z.infer<typeof cameraSchema>;
export type InsertCamera = z.infer<typeof insertCameraSchema>;

// ANPR Detection schema
export const anprDetectionSchema = z.object({
  id: z.string(),
  userId: z.string(),
  licensePlate: z.string()
    .min(1, "License plate is required")
    .max(20, "License plate must not exceed 20 characters"),
  confidence: z.number()
    .min(0, "Confidence must be between 0 and 1")
    .max(1, "Confidence must be between 0 and 1"),
  cameraId: z.string(),
  detectedAt: z.string()
    .datetime("Invalid detection timestamp"),
  status: z.enum(["Registered", "Unregistered"]),
  gate: z.enum(["Entry", "Exit"]).optional(), // NEW: Track if this is entry or exit detection
  vehicleType: z.string().optional(),
  plateImage: z.string().optional(),
  vehicleImage: z.string().optional(),
  ocrConfidence: z.number().optional(),
  detectionMethod: z.string().optional(),
  direction: z.string().optional(),
  captureReason: z.string().optional(),
  captured: z.boolean().optional(),
  exitTime: z.string().datetime("Invalid exit timestamp").optional(), // NEW: Track exit time
  // NEW: Driver information fields for manual editing
  driverName: z.string()
    .min(0)
    .max(100, "Driver name must not exceed 100 characters")
    .optional(),
  driverMobile: z.string()
    .min(0)
    .max(15, "Mobile number must not exceed 15 characters")
    .regex(/^[\d\s\-\+]*$/, "Mobile number contains invalid characters")
    .optional()
    .or(z.literal("")),
  notes: z.string().optional(), // NEW: Allow notes field for detections
  entryTime: z.string().optional(), // NEW: Add entryTime as alias for detectedAt
});

export const insertAnprDetectionSchema = anprDetectionSchema.omit({ id: true, userId: true });
export type AnprDetection = z.infer<typeof anprDetectionSchema>;
export type InsertAnprDetection = z.infer<typeof insertAnprDetectionSchema>;

// Sub-User Permissions schema
export const permissionsSchema = z.object({
  canAddVehicles: z.boolean().default(true),
  canDeleteVehicles: z.boolean().default(false),
  canUpdateVehicles: z.boolean().default(false),
  canViewVehicles: z.boolean().default(true),
  canAddCameras: z.boolean().default(false),
  canDeleteCameras: z.boolean().default(false),
  canUpdateCameras: z.boolean().default(false),
  canViewCameras: z.boolean().default(true),
  canViewDetections: z.boolean().default(true),
  canManageSubUsers: z.boolean().default(false),
  canExportData: z.boolean().default(false),
});

export type Permissions = z.infer<typeof permissionsSchema>;

// Sub-User schema
export const subUserSchema = z.object({
  id: z.string(),
  parentUserId: z.string(), // Main account owner
  username: z.string(),
  email: z.string().email(),
  mobile: z.string().optional(),
  fullName: z.string(),
  password: z.string(),
  permissions: permissionsSchema,
  isActive: z.boolean().default(true),
  lastLogin: z.string().optional(),
  createdAt: z.string(),
  createdBy: z.string(), // ID of admin who created this sub-user
  updatedAt: z.string().optional(),
});

export const insertSubUserSchema = subUserSchema.omit({ 
  id: true, 
  createdAt: true, 
  lastLogin: true, 
  updatedAt: true,
  createdBy: true 
});

export const updateSubUserSchema = subUserSchema.omit({ 
  id: true, 
  createdAt: true, 
  password: true,
  parentUserId: true,
  createdBy: true,
  lastLogin: true,
  updatedAt: true
});

export type SubUser = z.infer<typeof subUserSchema>;
export type InsertSubUser = z.infer<typeof insertSubUserSchema>;
export type UpdateSubUser = z.infer<typeof updateSubUserSchema>;

// Activity Log schema for tracking sub-user actions
export const activityLogSchema = z.object({
  id: z.string(),
  parentUserId: z.string(),
  subUserId: z.string(),
  action: z.enum([
    "LOGIN",
    "ADD_VEHICLE",
    "UPDATE_VEHICLE",
    "DELETE_VEHICLE",
    "VIEW_VEHICLE",
    "ADD_CAMERA",
    "UPDATE_CAMERA",
    "DELETE_CAMERA",
    "ADD_SUBUSER",
    "UPDATE_SUBUSER",
    "DELETE_SUBUSER",
    "EXPORT_DATA",
    "UPDATE_PROFILE",
    "LOGOUT"
  ]),
  details: z.record(z.any()).optional(), // JSON object with additional info
  timestamp: z.string(),
  ipAddress: z.string().optional(),
  status: z.enum(["SUCCESS", "FAILED"]).default("SUCCESS"),
});

export const insertActivityLogSchema = activityLogSchema.omit({ id: true });
export type ActivityLog = z.infer<typeof activityLogSchema>;
export type InsertActivityLog = z.infer<typeof insertActivityLogSchema>;

// Statistics schema
export const statsSchema = z.object({
  totalVehicles: z.number(),
  todayEntries: z.number(),
  todayExits: z.number(),
  activeCameras: z.number(),
  anprDetections: z.number(),
  totalCameras: z.number(),
  anprEnabled: z.number(),
  framesCaptured: z.string(),
});

export type Stats = z.infer<typeof statsSchema>;
