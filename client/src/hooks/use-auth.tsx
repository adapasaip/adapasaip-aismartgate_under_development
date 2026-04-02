import { createContext, useContext, useState, useEffect } from "react";
import { useToast } from "./use-toast";
import { apiRequest } from "@/lib/queryClient";

interface User {
  id: string;
  mobile: string;
  fullName?: string;
  email?: string;
  avatar?: string;
  role: string;
  createdAt?: string;
}

interface SubUser {
  id: string;
  parentUserId: string;
  username: string;
  email: string;
  mobile?: string;
  fullName: string;
  permissions: Record<string, boolean>;
  isActive: boolean;
  lastLogin?: string;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  subUser: SubUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isSubUserAuthenticated: boolean;
  login: (mobile: string, password: string) => Promise<void>;
  subUserLogin: (username: string, password: string) => Promise<void>;
  register: (fullName: string, mobile: string, password: string, email?: string) => Promise<void>;
  logout: () => void;
  updateProfile: (data: Partial<User>) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  deleteAccount: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const changePassword = async (currentPassword: string, newPassword: string) => {
      try {
        if (!token) throw new Error("Not authenticated");
        const response = await apiRequest("PUT", "/api/auth/password", { currentPassword, newPassword });
        const data = await response.json();
        if (response.ok) {
          toast({ title: "Success", description: "Password changed successfully" });
        } else {
          throw new Error(data.message || "Failed to change password");
        }
      } catch (error) {
        toast({
          title: "Error",
          description: error instanceof Error ? error.message : "Failed to change password",
          variant: "destructive",
        });
        throw error;
      }
    };
  const [user, setUser] = useState<User | null>(null);
  const [subUser, setSubUser] = useState<SubUser | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const { toast } = useToast();

  const isAuthenticated = !!user && !!token;
  const isSubUserAuthenticated = !!subUser;

  // Load user and token from localStorage on mount
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    const storedToken = localStorage.getItem("token");
    const storedSubUser = localStorage.getItem("subUser");

    if (storedUser && storedToken) {
      try {
        setUser(JSON.parse(storedUser));
        setToken(storedToken);
      } catch (error) {
        localStorage.removeItem("user");
        localStorage.removeItem("token");
      }
    }

    if (storedSubUser) {
      try {
        setSubUser(JSON.parse(storedSubUser));
      } catch (error) {
        localStorage.removeItem("subUser");
        localStorage.removeItem("isSubUser");
      }
    }
  }, []);

  const login = async (mobile: string, password: string) => {
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mobile, password }),
      });

      const data = await response.json();

      if (data.success && data.user) {
        setUser(data.user);
        // Generate a simple token for localStorage (not used for auth, just for session tracking)
        const token = `user_${data.user.id}_${Date.now()}`;
        setToken(token);
        localStorage.setItem("user", JSON.stringify(data.user));
        localStorage.setItem("token", token);

        toast({
          title: "Success",
          description: "Logged in successfully",
        });
      } else {
        throw new Error(data.message || "Login failed");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Login failed",
        variant: "destructive",
      });
      throw error;
    }
  };

  const subUserLogin = async (username: string, password: string) => {
    try {
      const response = await fetch("/api/subusers/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      const data = await response.json();

      if (data.success && data.subUser) {
        setSubUser(data.subUser);
        localStorage.setItem("subUser", JSON.stringify(data.subUser));
        localStorage.setItem("isSubUser", "true");

        toast({
          title: "Success",
          description: "Sub-user logged in successfully",
        });
      } else {
        throw new Error(data.message || "Sub-user login failed");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Sub-user login failed",
        variant: "destructive",
      });
      throw error;
    }
  };

  const register = async (fullName: string, mobile: string, password: string, email?: string) => {
    try {
      const response = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fullName, mobile, password, email }),
      });

      const data = await response.json();

      if (data.success && data.user) {
        setUser(data.user);
        // Generate a simple token for localStorage (not used for auth, just for session tracking)
        const token = `user_${data.user.id}_${Date.now()}`;
        setToken(token);
        localStorage.setItem("user", JSON.stringify(data.user));
        localStorage.setItem("token", token);

        toast({
          title: "Success",
          description: "Registration successful",
        });
      } else {
        throw new Error(data.message || "Registration failed");
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Registration failed",
        variant: "destructive",
      });
      throw error;
    }
  };

  const logout = () => {
    setUser(null);
    setSubUser(null);
    setToken(null);
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    localStorage.removeItem("subUser");
    localStorage.removeItem("isSubUser");
    toast({
      title: "Success",
      description: "Logged out successfully",
    });
  };

  const updateProfile = async (data: Partial<User>) => {
    try {
      if (!token) throw new Error("Not authenticated");

      const response = await apiRequest("PUT", "/api/auth/profile", data);
      const updatedUser = await response.json();

      setUser(updatedUser);
      localStorage.setItem("user", JSON.stringify(updatedUser));

      toast({
        title: "Success",
        description: "Profile updated successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to update profile",
        variant: "destructive",
      });
      throw error;
    }
  };

  const deleteAccount = async () => {
    try {
      if (!token) throw new Error("Not authenticated");

      await apiRequest("DELETE", "/api/auth/profile");

      setUser(null);
      setToken(null);
      localStorage.removeItem("user");
      localStorage.removeItem("token");

      toast({
        title: "Success",
        description: "Account deleted successfully",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to delete account",
        variant: "destructive",
      });
      throw error;
    }
  };

  return (
    <AuthContext.Provider value={{ user, subUser, token, isAuthenticated, isSubUserAuthenticated, login, subUserLogin, register, logout, updateProfile, changePassword, deleteAccount }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
