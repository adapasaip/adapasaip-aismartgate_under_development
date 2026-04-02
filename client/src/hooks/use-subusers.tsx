import { useState, useEffect, useCallback } from "react";
import { useAuth } from "./use-auth";
import { useToast } from "./use-toast";
import type { SubUser, UpdateSubUser, Permissions, ActivityLog } from "@shared/schema";

export interface SubUserWithoutPassword extends Omit<SubUser, 'password'> {}

interface SubUserManagement {
  subUsers: SubUserWithoutPassword[];
  activityLogs: ActivityLog[];
  loading: boolean;
  error: string | null;
  
  // Sub-user operations
  createSubUser: (data: Omit<SubUser, 'id' | 'createdAt' | 'createdBy' | 'lastLogin' | 'updatedAt'>) => Promise<SubUserWithoutPassword>;
  updateSubUser: (id: string, data: UpdateSubUser) => Promise<void>;
  deleteSubUser: (id: string) => Promise<void>;
  
  // Activity log operations
  fetchActivityLogs: (subUserId?: string, limit?: number) => Promise<void>;
  recentActivityLogs: ActivityLog[];
}

const apiRequest = async (
  method: string,
  endpoint: string,
  body?: unknown,
) => {
  const response = await fetch(endpoint, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.message || "Request failed");
  }

  return response.json();
};

export function useSubUsers(): SubUserManagement {
  const { user } = useAuth();
  const { toast } = useToast();
  const [subUsers, setSubUsers] = useState<SubUserWithoutPassword[]>([]);
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);
  const [recentActivityLogs, setRecentActivityLogs] = useState<ActivityLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch sub-users
  const fetchSubUsers = useCallback(async () => {
    if (!user?.id) return;

    setLoading(true);
    setError(null);
    try {
      const data = await apiRequest("GET", `/api/subusers?userId=${user.id}`);
      setSubUsers(data);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to fetch sub-users";
      setError(errorMsg);
      console.error(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  // Fetch activity logs
  const fetchActivityLogs = useCallback(async (subUserId?: string, limit = 100) => {
    if (!user?.id) return;

    setLoading(true);
    try {
      const url = new URL("/api/activity-logs", window.location.origin);
      url.searchParams.append("parentUserId", user.id);
      if (subUserId) url.searchParams.append("subUserId", subUserId);
      url.searchParams.append("limit", limit.toString());

      const data = await apiRequest("GET", url.toString());
      setActivityLogs(data);
    } catch (err) {
      console.error("Failed to fetch activity logs:", err);
    } finally {
      setLoading(false);
    }
  }, [user?.id]);

  // Fetch recent activity logs
  const fetchRecentActivityLogs = useCallback(async (limit = 20) => {
    if (!user?.id) return;

    try {
      const data = await apiRequest(
        "GET",
        `/api/activity-logs/recent?parentUserId=${user.id}&limit=${limit}`
      );
      setRecentActivityLogs(data);
    } catch (err) {
      console.error("Failed to fetch recent activity logs:", err);
    }
  }, [user?.id]);

  // Create sub-user
  const createSubUser = async (
    data: Omit<SubUser, 'id' | 'createdAt' | 'createdBy' | 'lastLogin' | 'updatedAt'>
  ): Promise<SubUserWithoutPassword> => {
    if (!user?.id) throw new Error("User not authenticated");

    try {
      const response = await apiRequest("POST", "/api/subusers", {
        ...data,
        createdBy: user.id,
      });

      if (response.success) {
        setSubUsers((prev) => [...prev, response.subUser]);
        toast({
          title: "Success",
          description: `Sub-user "${data.username}" created successfully`,
        });
        await fetchRecentActivityLogs();
        return response.subUser;
      }
      throw new Error(response.message || "Failed to create sub-user");
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to create sub-user";
      toast({
        title: "Error",
        description: errorMsg,
        variant: "destructive",
      });
      throw err;
    }
  };

  // Update sub-user
  const updateSubUser = async (id: string, data: UpdateSubUser) => {
    try {
      const response = await apiRequest("PUT", `/api/subusers/${id}`, data);

      if (response.success) {
        setSubUsers((prev) =>
          prev.map((su) => (su.id === id ? response.subUser : su))
        );
        toast({
          title: "Success",
          description: "Sub-user updated successfully",
        });
        await fetchRecentActivityLogs();
      } else {
        throw new Error(response.message || "Failed to update sub-user");
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to update sub-user";
      toast({
        title: "Error",
        description: errorMsg,
        variant: "destructive",
      });
      throw err;
    }
  };

  // Delete sub-user
  const deleteSubUser = async (id: string) => {
    try {
      const response = await apiRequest("DELETE", `/api/subusers/${id}`);

      if (response.success) {
        setSubUsers((prev) => prev.filter((su) => su.id !== id));
        toast({
          title: "Success",
          description: "Sub-user deleted successfully",
        });
        await fetchRecentActivityLogs();
      } else {
        throw new Error(response.message || "Failed to delete sub-user");
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to delete sub-user";
      toast({
        title: "Error",
        description: errorMsg,
        variant: "destructive",
      });
      throw err;
    }
  };

  // Auto-fetch data on mount
  useEffect(() => {
    if (user?.id) {
      fetchSubUsers();
      fetchRecentActivityLogs();
    }
  }, [user?.id, fetchSubUsers, fetchRecentActivityLogs]);

  return {
    subUsers,
    activityLogs,
    recentActivityLogs,
    loading,
    error,
    createSubUser,
    updateSubUser,
    deleteSubUser,
    fetchActivityLogs,
  };
}
