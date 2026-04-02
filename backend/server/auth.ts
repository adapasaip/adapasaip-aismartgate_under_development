import jwt from "jsonwebtoken";
import type { User } from "@shared/schema";
import type { Request, Response, NextFunction } from "express";

// JWT secret - in production, use environment variable
const JWT_SECRET = process.env.JWT_SECRET || "your-secret-key-change-in-production";
const JWT_EXPIRES_IN = "7d"; // Token expires in 7 days

export interface JWTPayload {
  userId: string;
  mobile: string;
  role: string;
}

/**
 * Generate JWT token for a user
 */
export function generateToken(user: User): string {
  const payload: JWTPayload = {
    userId: user.id,
    mobile: user.mobile,
    role: user.role,
  };

  return jwt.sign(payload, JWT_SECRET, {
    expiresIn: JWT_EXPIRES_IN,
  });
}

/**
 * Verify JWT token
 */
export function verifyToken(token: string): JWTPayload | null {
  try {
    return jwt.verify(token, JWT_SECRET) as JWTPayload;
  } catch (error) {
    return null;
  }
}

/**
 * Express middleware to authenticate requests
 * Extracts token from Authorization header or x-auth-token header
 */
export function authenticateToken(
  req: Request,
  res: Response,
  next: NextFunction
) {
  // Get token from header
  const authHeader = req.headers["authorization"];
  const token = authHeader?.startsWith("Bearer ")
    ? authHeader.substring(7)
    : req.headers["x-auth-token"] as string;

  if (!token) {
    return res.status(401).json({ message: "Access token required" });
  }

  const payload = verifyToken(token);

  if (!payload) {
    return res.status(401).json({ message: "Invalid or expired token" });
  }

  // Attach user info to request
  (req as any).user = payload;
  next();
}

/**
 * Optional authentication - continues even if token is invalid
 * Used for endpoints that work differently based on auth status
 */
export function optionalAuth(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const authHeader = req.headers["authorization"];
  const token = authHeader?.startsWith("Bearer ")
    ? authHeader.substring(7)
    : req.headers["x-auth-token"] as string;

  if (token) {
    const payload = verifyToken(token);
    if (payload) {
      (req as any).user = payload;
    }
  }

  next();
}
