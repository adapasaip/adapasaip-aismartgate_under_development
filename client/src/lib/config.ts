/**
 * Application configuration
 * Environment variables and constants
 */

// Get the backend URL from environment variables
// In development, this is typically empty (same origin)
// In production or when frontend/backend are on different ports, set VITE_BACKEND_URL
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "";

// You can also configure it manually if not using environment variables:
// For example, if your backend is on http://localhost:5000 and frontend on different port:
// export const BACKEND_URL = "http://localhost:5000";

export const API_BASE_URL = BACKEND_URL;
