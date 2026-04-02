import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import StatsCard from "@/components/stats-card";
import { MjpegPlayer } from "@/components/mjpeg-player";
import type { Camera, InsertCamera, Stats } from "@shared/schema";
import { insertCameraSchema } from "@shared/schema";
import { getCameraConnectionInfo, getCameraFeedUrl } from "@/lib/camera-utils";
import { BACKEND_URL } from "@/lib/config";

export default function Cameras() {
  const [isPreviewActive, setIsPreviewActive] = useState(false);
  const [isLocalCamera, setIsLocalCamera] = useState(false);
  const { toast } = useToast();
  const { user, subUser } = useAuth();

  // Check if sub-user has permission to add cameras
  const canAddCameras = !subUser || subUser.permissions?.canAddCameras;
  
  // Build userId for queries
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  // Helper to get placeholder text based on camera type
  const getSourcePlaceholder = (type: string) => {
    if (!type) return "Enter camera source...";
    if (type.includes("Desktop")) return "0";
    if (type.includes("USB")) return "1";
    if (type.includes("OAK")) return "Auto-detect (leave empty)";
    if (type.includes("IP")) return "http://192.168.1.100:80/stream";
    if (type.includes("RTSP")) return "rtsp://192.168.1.100:554/stream";
    if (type.includes("Mobile") && !type.includes("NGROK")) return "http://192.168.1.100:8080/video";
    if (type.includes("NGROK")) return "https://xxxxx-ngrok-free.dev";
    return "Enter camera source...";
  };

  const form = useForm<InsertCamera>({
    resolver: zodResolver(insertCameraSchema),
    defaultValues: {
      name: "",
      type: "Desktop Camera (0)",
      source: "",
      location: "Main Entrance",  // ✅ Default to meaningful value, not empty string
      gate: "Entry",  // ✅ Explicitly set Entry as default
      resolution: "1280x720",
      fps: 30,
      anprEnabled: true,
      status: "Offline",
    },
  });

  const { data: cameras = [], isLoading, refetch: refetchCameras } = useQuery<Camera[]>({
    queryKey: ["/api/cameras", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/cameras?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      // Ensure data is an array
      return Array.isArray(data) ? data : [];
    },
    staleTime: 0, // Mark as stale immediately to ensure fresh data on invalidation
    refetchOnMount: true, // ⭐ CRITICAL FIX: Refetch when component mounts or remounts
    refetchOnWindowFocus: true, // ⭐ Refetch when user returns to window/tab
  });

  const { data: stats, refetch: refetchStats } = useQuery<Stats>({
    queryKey: ["/api/stats", userIdParam],
    queryFn: async () => {
      const response = await fetch(`/api/stats?userId=${encodeURIComponent(userIdParam)}`);
      return response.json();
    },
    staleTime: 0, // Mark as stale immediately to ensure fresh data on invalidation
    refetchOnMount: true, // ⭐ CRITICAL FIX: Refetch when component mounts or remounts
    refetchOnWindowFocus: true, // ⭐ Refetch when user returns to window/tab
  });

  // ⭐ ADDITIONAL FIX: Add visibility change listener to refetch when page becomes visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        console.log('[CAMERA_PAGE] Page became visible, refetching cameras & stats...');
        refetchCameras();
        refetchStats();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, [refetchCameras, refetchStats]);

  const addCameraMutation = useMutation({
    mutationFn: async (data: InsertCamera) => {
      const queryParams = new URLSearchParams();
      // Sub-users should use their parent's userId for data ownership
      const effectiveUserId = subUser ? subUser.parentUserId : user?.id;
      if (effectiveUserId) queryParams.append("userId", effectiveUserId);
      if (subUser) queryParams.append("subUserId", subUser.id);
      const response = await apiRequest("POST", `/api/cameras?${queryParams.toString()}`, data);
      if (!response.ok) {
        throw new Error(`Failed to add camera: ${response.statusText}`);
      }
      return response.json();
    },
    onSuccess: async () => {
      // ✅ Immediately invalidate cache
      queryClient.invalidateQueries({ queryKey: ["/api/cameras", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      
      // ✅ Wait for refetch to complete before continuing
      await Promise.all([
        refetchCameras(),
        refetchStats()
      ]);
      
      form.reset();
      setIsPreviewActive(false);
      
      // ✅ Show toast after refetch completes
      toast({ title: "Success", description: "Camera added successfully" });
    },
    onError: (error: any) => {
      const message = error?.response?.message || "Failed to add camera";
      console.error("[MUTATION] Error adding camera:", error);
      toast({
        title: "Error",
        description: message,
        variant: "destructive",
      });
    },
  });

  const deleteCameraMutation = useMutation({
    mutationFn: async (id: string) => {
      const queryParams = new URLSearchParams();
      const effectiveUserId = subUser ? subUser.parentUserId : user?.id;
      if (effectiveUserId) queryParams.append("userId", effectiveUserId);
      const response = await apiRequest("DELETE", `/api/cameras/${id}?${queryParams.toString()}`);
      if (!response.ok) {
        throw new Error(`Failed to delete camera: ${response.statusText}`);
      }
      return response.json();
    },
    onSuccess: async () => {
      // ✅ Immediately invalidate cache
      queryClient.invalidateQueries({ queryKey: ["/api/cameras", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      
      // ✅ Wait for refetch to complete before showing toast
      await Promise.all([
        refetchCameras(),
        refetchStats()
      ]);
      
      // ✅ Show toast after refetch completes
      toast({ title: "Success", description: "Camera deleted successfully" });
    },
    onError: (error) => {
      console.error("[MUTATION] Error deleting camera:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to delete camera",
        variant: "destructive",
      });
    },
  });

  const updateCameraMutation = useMutation({
    mutationFn: async ({ id, status }: { id: string; status: string }) => {
      const response = await apiRequest("PUT", `/api/cameras/${id}`, { status });
      if (!response.ok) {
        throw new Error(`Failed to update camera: ${response.statusText}`);
      }
      const data = await response.json();
      if (!data || !data.id) {
        throw new Error("Invalid response from server - camera data missing");
      }
      return data;
    },
    onSuccess: async (data) => {
      console.log("[MUTATION] Camera updated successfully:", data);
      // ✅ Immediately invalidate cache
      queryClient.invalidateQueries({ queryKey: ["/api/cameras", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      
      // ✅ Wait for refetch to complete before showing toast
      await Promise.all([
        refetchCameras(),
        refetchStats()
      ]);
      
      // ✅ Show toast after refetch completes
      toast({
        title: "Success",
        description: `Camera status updated to ${data.status}`,
      });
    },
    onError: (error) => {
      console.error("[MUTATION] Error updating camera:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to update camera status",
        variant: "destructive",
      });
    },
  });

  const handleTestCamera = async () => {
    const formData = form.getValues();
    if (!formData.name || !formData.source) {
      toast({
        title: "Error",
        description: "Please fill in camera name and source",
        variant: "destructive",
      });
      return;
    }

    // All cameras (including local ones) now use backend preview
    setIsLocalCamera(false);
    setIsPreviewActive(true);
    toast({
      title: "Camera Test",
      description: "Testing camera connection via backend..."
    });
  };

  const handleStopPreview = async () => {
    setIsPreviewActive(false);
    setIsLocalCamera(false);
    toast({ title: "Preview Stopped", description: "Camera preview stopped" });
  };


  const handleConnectionCheck = (camera: Camera) => {
    if (updateCameraMutation.isPending) {
      toast({
        title: "Already updating",
        description: "Please wait for the previous update to complete",
        variant: "destructive",
      });
      return;
    }
    
    const newStatus = camera.status === "Online" ? "Offline" : "Online";
    updateCameraMutation.mutate({ id: camera.id, status: newStatus });
  };

  const onSubmit = (data: InsertCamera) => {
    if (!canAddCameras) {
      toast({
        title: "Permission Denied",
        description: "You do not have permission to add cameras. Please contact your administrator.",
        variant: "destructive",
      });
      return;
    }
    addCameraMutation.mutate(data);
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 bg-white min-h-screen">
      {/* Decorative header background */}
      <div className="fixed inset-0 -z-10 h-64 bg-gradient-to-br from-blue-50 via-blue-25 to-transparent pointer-events-none"></div>
      
      {/* Header Section */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-2">
          <div>
            <h1 className="text-5xl font-bold text-blue-900 mb-2 flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl text-white shadow-lg">
                <i className="fas fa-camera text-2xl"></i>
              </div>
              Camera Management
            </h1>
            <p className="text-slate-600 text-base font-medium max-w-2xl">
              <i className="fas fa-check-circle text-green-500 mr-2"></i>
              Real-time monitoring and configuration of all your security cameras
            </p>
          </div>
        </div>
      </div>

      {/* Stats - Enhanced with blue & white theme */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-10">
        <div className="bg-white rounded-2xl border-2 border-blue-100 p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:border-blue-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Total Cameras</p>
              <p className="text-4xl font-bold text-blue-900 mt-2">{stats?.totalCameras || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-blue-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-blue-200 hover:to-blue-100">
              <i className="fas fa-camera text-3xl text-blue-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-blue-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Total Cameras
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-blue-100">
            <span className="text-xs text-slate-500"><i className="fas fa-arrow-up text-green-500 mr-1"></i>All systems operational</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-green-100 p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:border-green-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Active Cameras</p>
              <p className="text-4xl font-bold text-green-900 mt-2">{stats?.activeCameras || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-green-100 to-green-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-green-200 hover:to-green-100">
              <i className="fas fa-check-circle text-3xl text-green-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-green-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Active Cameras
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-green-100">
            <span className="text-xs text-slate-500"><i className="fas fa-pulse mr-1"></i>Live & monitoring</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-blue-100 p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:border-blue-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">ANPR Enabled</p>
              <p className="text-4xl font-bold text-blue-900 mt-2">{stats?.anprEnabled || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-blue-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-blue-200 hover:to-blue-100">
              <i className="fas fa-eye text-3xl text-blue-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-blue-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                ANPR Enabled
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-blue-100">
            <span className="text-xs text-slate-500"><i className="fas fa-barcode mr-1"></i>Plate detection active</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-blue-100 p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:border-blue-300">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Frames Captured</p>
              <p className="text-4xl font-bold text-blue-900 mt-2">{stats?.framesCaptured || "0K"}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-blue-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-blue-200 hover:to-blue-100">
              <i className="fas fa-chart-bar text-3xl text-blue-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-blue-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Frames Captured
              </div>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-blue-100">
            <span className="text-xs text-slate-500"><i className="fas fa-zap mr-1"></i>Processing streaming</span>
          </div>
        </div>
      </div>

      {/* Form + Preview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        {/* Form - Left side */}
        <div className="bg-gradient-to-br from-blue-50 via-white to-blue-50 rounded-2xl shadow-md border-2 border-blue-100 p-8 hover:border-blue-300 transition-all">
          <div className="mb-8 pb-6 border-b-2 border-blue-200">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl text-white shadow-md">
                <i className="fas fa-plus text-xl"></i>
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Add New Camera</h2>
            </div>
            <p className="text-slate-600 text-sm ml-12"><i className="fas fa-lightbulb mr-2 text-yellow-500"></i>Configure a new camera for real-time monitoring</p>
          </div>

          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">

              {/* Responsive Inputs */}
              <div className="grid grid-cols-1 gap-5">

                {/* Camera Name */}
                <FormField
                  control={form.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-tag text-blue-600 text-xs"></i>
                        </div>
                        Name
                      </FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="Main Gate Entry" 
                          {...field}
                          className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg transition-colors bg-white/80 hover:bg-white"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                {/* Gate */}
                <FormField
                  control={form.control}
                  name="gate"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-sign-in-alt text-blue-600 text-xs"></i>
                        </div>
                        Gate Type
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg bg-white/80 hover:bg-white transition-colors">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="Entry">
                            <i className="fas fa-sign-in-alt mr-2"></i>Entry Gate (Creates new records)
                          </SelectItem>
                          <SelectItem value="Exit">
                            <i className="fas fa-sign-out-alt mr-2"></i>Exit Gate (Updates existing records)
                          </SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* Camera Type */}
                <FormField
                  control={form.control}
                  name="type"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-camera text-blue-600 text-xs"></i>
                        </div>
                        Type
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg bg-white/80 hover:bg-white transition-colors">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="Desktop Camera (0)">
                            <i className="fas fa-laptop mr-2"></i>Desktop Camera (0)
                          </SelectItem>
                          <SelectItem value="USB Camera (1)">
                            <i className="fas fa-usb mr-2"></i>USB Camera (1)
                          </SelectItem>
                          <SelectItem value="OAK Camera (USB)">
                            <i className="fas fa-microchip mr-2"></i>OAK Camera (USB)
                          </SelectItem>
                          <SelectItem value="IP Camera (http://...)">
                            <i className="fas fa-globe mr-2"></i>IP Camera (http://…)
                          </SelectItem>
                          <SelectItem value="RTSP Camera (rtsp://...)">
                            <i className="fas fa-stream mr-2"></i>RTSP Camera (rtsp://…)
                          </SelectItem>
                          <SelectItem value="Mobile Camera (IP Webcam App)">
                            <i className="fas fa-mobile-alt mr-2"></i>Mobile Camera (IP Webcam App)
                          </SelectItem>
                          <SelectItem value="Mobile Camera via NGROK">
                            <i className="fas fa-tunnel mr-2"></i>Mobile Camera via NGROK
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />

                {/* Source */}
                <FormField
                  control={form.control}
                  name="source"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-link text-blue-600 text-xs"></i>
                        </div>
                        Source
                      </FormLabel>
                      <FormControl>
                        <Input
                          placeholder={getSourcePlaceholder(form.watch("type"))}
                          {...field}
                          className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg transition-colors bg-white/80 hover:bg-white"
                        />
                      </FormControl>
                      <p className="text-xs text-slate-500 mt-1 ml-2 flex items-center gap-1">
                        <i className="fas fa-lightbulb text-yellow-500"></i>
                        {form.watch("type")?.includes("Desktop")
                          ? "Use 0 for default camera"
                          : form.watch("type")?.includes("USB")
                          ? "Use 1 for USB camera, 2+ for additional cameras"
                          : form.watch("type")?.includes("OAK")
                          ? "Leave empty to auto-detect OAK camera, or enter device ID"
                          : form.watch("type")?.includes("IP")
                          ? "Format: http://IP:PORT or IP:PORT"
                          : form.watch("type")?.includes("RTSP")
                          ? "Format: rtsp://IP:PORT/stream"
                          : form.watch("type")?.includes("Mobile")
                          ? form.watch("type")?.includes("NGROK")
                            ? "Your NGROK tunnel URL"
                            : "IP Webcam app URL (usually port 8080)"
                          : "Enter the camera source URL or index"}
                      </p>
                    </FormItem>
                  )}
                />

                {/* Location */}
                <FormField
                  control={form.control}
                  name="location"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-map-pin text-blue-600 text-xs"></i>
                        </div>
                        Location / Gate Name
                      </FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="e.g., Main Entrance, Parking Lot 2, Exit Gate A" 
                          {...field}
                          className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg transition-colors bg-white/80 hover:bg-white"
                        />
                      </FormControl>
                      <p className="text-xs text-slate-500 mt-1 ml-2">The name/location of this camera gate (e.g., "Main Entrance", "Parking Exit")</p>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                {/* Resolution */}
                <FormField
                  control={form.control}
                  name="resolution"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2 mb-2">
                        <div className="w-5 h-5 rounded bg-blue-100 flex items-center justify-center">
                          <i className="fas fa-expand text-blue-600 text-xs"></i>
                        </div>
                        Resolution
                      </FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger className="border-2 border-blue-100 focus:border-blue-500 focus:ring-0 rounded-lg bg-white/80 hover:bg-white transition-colors">
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="640x480">640x480</SelectItem>
                          <SelectItem value="1280x720">1280x720</SelectItem>
                          <SelectItem value="1920x1080">1920x1080</SelectItem>
                        </SelectContent>
                      </Select>
                    </FormItem>
                  )}
                />
              </div>

              {/* Buttons - Enhanced styling */}
              <div className="flex flex-col sm:flex-row flex-wrap gap-3 pt-6 border-t-2 border-blue-200">
                <Button 
                  type="button" 
                  variant="outline" 
                  onClick={handleTestCamera}
                  className="flex-1 border-2 border-blue-400 text-blue-600 hover:bg-blue-50 hover:border-blue-600 hover:text-blue-700 rounded-lg transition-all font-semibold"
                >
                  <i className="fas fa-play mr-2"></i>Test Camera
                </Button>

                <Button 
                  type="button" 
                  variant="outline" 
                  onClick={handleStopPreview}
                  className="flex-1 border-2 border-slate-300 text-slate-700 hover:bg-slate-100 hover:border-slate-400 rounded-lg transition-all font-semibold"
                >
                  <i className="fas fa-stop mr-2"></i>Stop Preview
                </Button>

                <Button 
                  type="submit" 
                  disabled={addCameraMutation.isPending || !canAddCameras}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg transition-all shadow-md hover:shadow-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <i className="fas fa-save mr-2"></i>
                  {addCameraMutation.isPending ? "Saving..." : "Save Camera"}
                </Button>

                <Button 
                  type="button" 
                  variant="secondary" 
                  onClick={() => form.reset()}
                  className="flex-1 bg-slate-200 text-slate-700 hover:bg-slate-300 rounded-lg transition-all font-semibold"
                >
                  <i className="fas fa-times mr-2"></i>Cancel
                </Button>
              </div>
            </form>
          </Form>
        </div>

        {/* Camera Preview - Right side (balanced width) */}
        <div className="bg-gradient-to-br from-blue-50 via-white to-blue-50 rounded-2xl shadow-md border-2 border-blue-100 p-8 hover:border-blue-300 transition-all flex flex-col">
          <div className="mb-6 pb-4 border-b-2 border-blue-200">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl text-white shadow-md">
                <i className="fas fa-eye text-xl"></i>
              </div>
              <h3 className="text-xl font-bold text-slate-900">Live Preview</h3>
            </div>
          </div>

          <div className="bg-slate-900 rounded-xl aspect-video flex items-center justify-center relative overflow-hidden border-2 border-slate-800">
            {isPreviewActive && form.watch("source") ? (
              <>
                {/* Live MJPEG Stream using canvas-based player for continuous video */}
                <MjpegPlayer
                  streamUrl={(() => {
                    const source = form.watch("source") || "0";
                    const type = form.watch("type") || "Desktop Camera (0)";
                    const pythonBackendUrl = process.env.REACT_APP_PYTHON_BACKEND_URL || "http://localhost:8000";
                    return `${pythonBackendUrl}/api/video_feed/preview?source=${encodeURIComponent(source)}&type=${encodeURIComponent(type)}`;
                  })()}
                  className="w-full h-full"
                  onConnect={() => {
                    console.log("[CAMERA_PREVIEW] Stream connected successfully");
                  }}
                  onError={() => {
                    const source = form.watch("source");
                    const type = form.watch("type");

                    console.error("[CAMERA_PREVIEW] Stream failed:", {
                      source,
                      type,
                    });

                    toast({
                      title: "Camera Preview Failed",
                      description: `Unable to connect to camera. Check if Python backend is running on port 8000.`,
                      variant: "destructive",
                    });
                  }}
                />
              </>
            ) : (
              <div className="text-white text-center p-4">
                <i className="fas fa-camera text-5xl mb-3 opacity-60"></i>
                <p className="opacity-70 text-sm font-bold">
                  {isPreviewActive ? "No source provided" : "Camera Preview"}
                </p>
                <p className="opacity-50 text-xs mt-2">
                  {isPreviewActive ? "Enter camera source to preview" : 'Click "Test Camera" to start'}
                </p>
              </div>
            )}

            {/* Loading Indicator */}
            {isPreviewActive && (
              <div className="absolute top-3 right-3 flex items-center gap-2 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-4 py-2 rounded-full text-xs font-bold shadow-lg">
                <span className="w-2.5 h-2.5 bg-white rounded-full animate-pulse"></span>
                TESTING
              </div>
            )}
          </div>

          {/* Preview status */}
          <div className="mt-6 flex items-center justify-between text-sm p-4 bg-gradient-to-r from-blue-50 to-slate-50 rounded-lg border-2 border-blue-200">
            <span className="text-slate-700 font-semibold flex items-center gap-2"><i className={`fas mr-2 ${ isPreviewActive ? 'fa-circle text-green-500 animate-pulse' : 'fa-circle text-slate-400' }`}></i>Status:</span>
            <Badge
              variant={isPreviewActive ? "secondary" : "destructive"}
              className={isPreviewActive ? "bg-green-100 text-green-700 border border-green-300 font-bold" : "bg-slate-200 text-slate-700 border border-slate-300 font-bold"}
            >
              <i className={`fas mr-1 ${isPreviewActive ? 'fa-check-circle text-green-600' : 'fa-circle text-slate-500'}`}></i>
              {isPreviewActive ? "Testing" : "Idle"}
            </Badge>
          </div>

          {/* Help text for camera setup */}
          {isPreviewActive && (
            <div className="mt-5 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-lg text-xs text-slate-700">
              <div className="flex items-start gap-3">
                <i className="fas fa-info-circle text-blue-600 mt-0.5 flex-shrink-0 text-base"></i>
                <div>
                  <strong className="text-blue-900 block mb-2">Testing Camera Connection...</strong>
                  <p className="text-slate-600 mb-2 text-xs">
                    The system is attempting to connect to your camera:
                  </p>
                  <ul className="list-disc list-inside space-y-1 text-slate-600 text-xs">
                    <li>Ensure the camera is powered on and connected</li>
                    <li>Verify the source/URL is correct</li>
                    <li>Check the camera isn't in use by another application</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Camera List Container - Enhanced */}
      <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-100 overflow-hidden hover:border-blue-200 transition-colors mb-10">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-8 border-b-2 border-blue-100 bg-gradient-to-r from-blue-50 to-slate-50">
          <h2 className="text-2xl font-bold text-blue-900 flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <i className="fas fa-list text-blue-600"></i>
            </div>
            Camera Network
          </h2>
          <div className="flex items-center gap-2">
            {isLoading ? (
              <div className="text-slate-600 font-semibold flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                Loading...
              </div>
            ) : (
              <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-bold text-sm">
                {cameras.length} {cameras.length === 1 ? "camera" : "cameras"}
              </span>
            )}
          </div>
        </div>

        {/* Mobile & Tablet Card View (below lg) */}
        <div className="block lg:hidden">
          {cameras.length === 0 ? (
            <div className="text-center text-slate-500 py-16 p-4">
              <i className="fas fa-camera text-5xl mb-4 opacity-20 block"></i>
              <p className="text-slate-600 font-semibold text-lg">No cameras configured</p>
              <p className="text-slate-500 text-sm mt-2">Add a camera or adjust your filters</p>
            </div>
          ) : isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 sm:p-6">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="bg-white rounded-2xl border-2 border-blue-100 p-6 animate-pulse">
                  <div className="space-y-4">
                    <div className="h-8 bg-slate-200 rounded-lg"></div>
                    <div className="h-4 bg-slate-100 rounded"></div>
                    <div className="h-4 bg-slate-100 rounded w-3/4"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 p-4 sm:p-6">
              {cameras.map((camera) => (
                <div 
                  key={camera.id} 
                  className="bg-white rounded-xl border-2 border-blue-100 p-4 shadow-sm hover:shadow-md hover:border-blue-300 transition-all"
                >
                  {/* Header with status */}
                  <div className="flex items-start justify-between mb-4 pb-3 border-b-2 border-blue-100">
                    <div className="flex-1">
                      <h3 className="text-base font-bold text-blue-900 truncate">{camera.name}</h3>
                      <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                        <i className="fas fa-map-pin text-blue-600"></i>
                        {camera.location}
                      </p>
                    </div>
                    <Badge 
                      className={`whitespace-nowrap ml-2 font-bold border text-xs ${
                        camera.status === "Online" 
                          ? "bg-green-100 text-green-700 border-green-300" 
                          : "bg-slate-100 text-slate-700 border-slate-300"
                      }`}
                    >
                      <i className={`fas mr-1 ${camera.status === "Online" ? 'fa-check-circle' : 'fa-times-circle'}`}></i>
                      {camera.status}
                    </Badge>
                  </div>

                  {/* Camera details grid */}
                  <div className="space-y-2 mb-4 text-xs">
                    <div className="bg-blue-50 rounded-lg p-3">
                      <div className="text-slate-600 font-semibold flex items-center gap-1 mb-1">
                        <i className="fas fa-video text-blue-600"></i>Type
                      </div>
                      <div className="text-slate-900 font-medium">{camera.type.split("(")[0].trim()}</div>
                    </div>
                    <div className="bg-blue-50 rounded-lg p-3">
                      <div className="text-slate-600 font-semibold flex items-center gap-1 mb-1">
                        <i className="fas fa-dooropen text-blue-600"></i>Gate
                      </div>
                      <div className={`font-bold ${camera.gate === "Entry" ? 'text-blue-700' : 'text-orange-700'}`}>
                        {camera.gate}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-50 rounded-lg p-2">
                        <div className="text-slate-600 font-semibold text-xs flex items-center gap-1 mb-1">
                          <i className="fas fa-film text-blue-600"></i>FPS
                        </div>
                        <div className="text-slate-900 font-medium">{camera.fps}</div>
                      </div>
                      <div className="bg-slate-50 rounded-lg p-2">
                        <div className="text-slate-600 font-semibold text-xs flex items-center gap-1 mb-1">
                          <i className="fas fa-barcode text-blue-600"></i>ANPR
                        </div>
                        <div className={`font-bold text-xs ${camera.anprEnabled ? 'text-green-700' : 'text-slate-600'}`}>
                          {camera.anprEnabled ? "On" : "Off"}
                        </div>
                      </div>
                    </div>
                    <div className="text-slate-500 py-2 px-3 bg-blue-50 rounded-lg font-mono truncate text-xs">
                      <i className="fas fa-plug text-blue-600 mr-1"></i>{camera.source}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 pt-3 border-t border-blue-100">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1 text-blue-600 hover:text-blue-700 hover:bg-blue-50 border border-blue-200 rounded-lg transition-all font-semibold text-xs"
                      onClick={() => {
                        form.setValue("name", camera.name);
                        form.setValue("source", camera.source);
                        form.setValue("type", camera.type);
                        form.setValue("location", camera.location);
                        form.setValue("gate", camera.gate);
                        form.setValue("fps", camera.fps);
                        form.setValue("anprEnabled", camera.anprEnabled);
                        handleTestCamera();
                      }}
                      title="Test Camera Connection"
                    >
                      <i className="fas fa-play mr-1"></i>
                      <span className="hidden sm:inline">Test</span>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1 text-slate-600 hover:text-blue-600 hover:bg-blue-50 border border-slate-200 rounded-lg transition-all font-semibold text-xs"
                      onClick={() => handleConnectionCheck(camera)}
                      disabled={updateCameraMutation.isPending}
                      title="Check Connection Status"
                    >
                      <i className="fas fa-signal mr-1"></i>
                      <span className="hidden sm:inline">Check</span>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1 text-red-600 hover:text-red-700 hover:bg-red-50 border border-red-200 rounded-lg transition-all font-semibold text-xs"
                      onClick={() => deleteCameraMutation.mutate(camera.id)}
                      disabled={deleteCameraMutation.isPending}
                      title="Delete Camera"
                    >
                      <i className="fas fa-trash mr-1"></i>
                      <span className="hidden sm:inline">Remove</span>
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Desktop Table View (lg and above) */}
        <div className="hidden lg:block overflow-auto">
          <table className="w-full">
            <thead className="bg-gradient-to-r from-blue-600 via-blue-500 to-blue-600 border-b-4 border-blue-700 text-white sticky top-0 z-10 shadow-md">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-tag text-blue-200 mr-2"></i>Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-map-marker-alt text-blue-200 mr-2"></i>Location
                </th>
                <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-video text-blue-200 mr-2"></i>Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-dooropen text-blue-200 mr-2"></i>Gate
                </th>
                <th className="px-4 py-3 text-center text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-film text-blue-200 mr-2"></i>FPS
                </th>
                <th className="px-4 py-3 text-center text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-barcode text-blue-200 mr-2"></i>ANPR
                </th>
                <th className="px-4 py-3 text-center text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-circle-check text-blue-200 mr-2"></i>Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-bold uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-plug text-blue-200 mr-2"></i>Source
                </th>
                <th className="px-4 py-3 text-center text-xs font-bold uppercase tracking-wider">
                  <i className="fas fa-cog text-blue-200 mr-2"></i>Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-200 bg-white">
              {isLoading ? (
                <tr>
                  <td colSpan={9} className="px-4 py-8 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                      <span className="text-blue-600 font-semibold">Loading cameras...</span>
                    </div>
                  </td>
                </tr>
              ) : cameras.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-4 py-12 text-center">
                    <i className="fas fa-camera text-5xl mb-3 opacity-20 block"></i>
                    <p className="text-slate-600 font-semibold">No cameras configured</p>
                  </td>
                </tr>
              ) : (
                cameras.map((camera, index) => (
                  <tr
                    key={camera.id}
                    className={`transition-all duration-200 border-l-4 border-l-transparent hover:border-l-blue-500 hover:shadow-md ${
                      index % 2 === 0 ? "bg-white" : "bg-blue-50/40"
                    } hover:bg-blue-100/30`}
                    data-testid={`camera-row-${camera.id}`}
                  >
                    {/* Name Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-bold text-blue-900">
                      {camera.name}
                    </td>

                    {/* Location Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-700">
                      <div className="flex items-center gap-1">
                        <i className="fas fa-map-pin text-blue-600 text-xs"></i>
                        {camera.location}
                      </div>
                    </td>

                    {/* Type Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-sm">
                      <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded-lg text-xs font-semibold border border-blue-300">
                        {camera.type.split("(")[0].trim()}
                      </span>
                    </td>

                    {/* Gate Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-sm font-semibold">
                      <span className={`px-2 py-1 rounded-lg text-xs font-bold ${
                        camera.gate === "Entry"
                          ? "bg-blue-100 text-blue-700"
                          : "bg-orange-100 text-orange-700"
                      }`}>
                        {camera.gate}
                      </span>
                    </td>

                    {/* FPS Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-center">
                      <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded-lg text-xs font-semibold">
                        {camera.fps}
                      </span>
                    </td>

                    {/* ANPR Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-center">
                      <button
                        onClick={() => {}}
                        className={`px-2 py-1 text-xs font-bold rounded-lg cursor-pointer transition-all border ${
                          camera.anprEnabled
                            ? "bg-green-100 text-green-700 border-green-300 hover:bg-green-200 hover:border-green-400"
                            : "bg-slate-100 text-slate-700 border-slate-300 hover:bg-slate-200 hover:border-slate-400"
                        }`}
                        data-testid={`anpr-toggle-${camera.id}`}
                        title="ANPR Status"
                      >
                        <i className={`fas ${camera.anprEnabled ? "fa-check-circle" : "fa-times-circle"} mr-1`}></i>
                        {camera.anprEnabled ? "On" : "Off"}
                      </button>
                    </td>

                    {/* Status Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-center">
                      <button
                        onClick={() => handleConnectionCheck(camera)}
                        disabled={updateCameraMutation.isPending}
                        className={`px-2 py-1 text-xs font-bold rounded-lg transition-all border ${
                          updateCameraMutation.isPending
                            ? "bg-slate-100 text-slate-400 border-slate-300 cursor-not-allowed opacity-50"
                            : camera.status === "Online"
                              ? "bg-green-100 text-green-700 border-green-300 hover:bg-green-200 hover:border-green-400 cursor-pointer"
                              : "bg-slate-100 text-slate-700 border-slate-300 hover:bg-slate-200 hover:border-slate-400 cursor-pointer"
                        }`}
                        data-testid={`status-toggle-${camera.id}`}
                        title={updateCameraMutation.isPending ? "Updating camera status..." : "Click to toggle connection check"}
                      >
                        {updateCameraMutation.isPending ? (
                          <>
                            <i className="fas fa-spinner fa-spin mr-1"></i>
                            Updating...
                          </>
                        ) : (
                          <>
                            <i className={`fas ${camera.status === "Online" ? "fa-check-circle" : "fa-times-circle"} mr-1`}></i>
                            {camera.status === "Online" ? "Online" : "Offline"}
                          </>
                        )}
                      </button>
                    </td>

                    {/* Source Column */}
                    <td className="px-4 py-3 text-sm text-slate-700">
                      <span className="font-mono text-xs text-slate-600 truncate max-w-xs flex items-center gap-1">
                        <i className="fas fa-plug text-blue-600 text-xs flex-shrink-0"></i>
                        {camera.source}
                      </span>
                    </td>

                    {/* Actions Column */}
                    <td className="px-4 py-3 whitespace-nowrap text-sm">
                      <div className="flex gap-1 justify-center">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-blue-600 hover:text-blue-700 hover:bg-blue-100 border border-blue-200 rounded-lg transition-all text-xs"
                          onClick={() => {
                            form.setValue("name", camera.name);
                            form.setValue("source", camera.source);
                            form.setValue("type", camera.type);
                            form.setValue("location", camera.location);
                            form.setValue("gate", camera.gate);
                            form.setValue("fps", camera.fps);
                            form.setValue("anprEnabled", camera.anprEnabled);
                            handleTestCamera();
                          }}
                          data-testid={`button-test-${camera.id}`}
                          title="Test Camera Connection"
                        >
                          <i className="fas fa-play"></i>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-slate-600 hover:text-blue-600 hover:bg-blue-100 border border-slate-200 rounded-lg transition-all text-xs"
                          onClick={() => handleConnectionCheck(camera)}
                          disabled={updateCameraMutation.isPending}
                          title="Check Connection Status"
                        >
                          <i className="fas fa-signal"></i>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 hover:text-red-700 hover:bg-red-100 border border-red-200 rounded-lg transition-all text-xs"
                          onClick={() => deleteCameraMutation.mutate(camera.id)}
                          disabled={deleteCameraMutation.isPending}
                          title="Delete Camera"
                        >
                          <i className="fas fa-trash"></i>
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Camera Connection Guide - Enhanced styling */}
      <div className="bg-gradient-to-br from-blue-50 to-white border-2 border-blue-200 rounded-2xl p-8 mt-10">
        <h3 className="text-2xl font-bold text-blue-900 mb-2 flex items-center gap-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <i className="fas fa-book text-blue-600"></i>
          </div>
          Camera Source Guide
        </h3>
        <p className="text-slate-600 mb-8">Learn how to connect different types of cameras to your ANPR system</p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Desktop Camera */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-laptop text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">Desktop/Webcam</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">Local camera on your system</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-terminal text-blue-600 mr-2"></i>0
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>Use 0 for default, 1, 2+ for others</p>
          </div>

          {/* IP Camera */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-globe text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">IP Camera</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">Network-connected HTTP camera</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-globe text-blue-600 mr-2"></i>http://192.168.1.100:80
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>Include port number if needed</p>
          </div>

          {/* RTSP Camera */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-stream text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">RTSP Stream</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">Real-time streaming protocol</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-signal text-blue-600 mr-2"></i>rtsp://192.168.1.100:554
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>Typical port is 554</p>
          </div>

          {/* IP Webcam App */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-mobile-alt text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">IP Webcam App</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">Mobile phone streaming app</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-phone text-blue-600 mr-2"></i>http://192.168.1.50:8080
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>Usually runs on port 8080</p>
          </div>

          {/* NGROK Tunnel */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-shield text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">NGROK Tunnel</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">Secure external tunnel access</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-link text-blue-600 mr-2"></i>https://abc-ngrok-free.dev
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>NGROK tunnel URL</p>
          </div>

          {/* CCTV DVR */}
          <div className="bg-white rounded-xl p-6 border-2 border-blue-100 hover:border-blue-300 hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-100 rounded-lg">
                <i className="fas fa-cube text-blue-600 text-xl"></i>
              </div>
              <h4 className="font-bold text-slate-900">CCTV/DVR</h4>
            </div>
            <p className="text-sm text-slate-600 mb-4">DVR HTTP streaming</p>
            <div className="bg-slate-100 p-3 rounded-lg mb-3 font-mono text-xs text-slate-800 break-all">
              <i className="fas fa-server text-blue-600 mr-2"></i>http://192.168.1.100:8000
            </div>
            <p className="text-xs text-slate-500"><i className="fas fa-arrow-right text-blue-500 mr-1"></i>Check DVR manual for URL</p>
          </div>
        </div>

        {/* Helpful tips footer */}
        <div className="mt-8 p-6 bg-blue-100 border-2 border-blue-300 rounded-xl">
          <h4 className="font-bold text-blue-900 mb-3 flex items-center gap-2">
            <i className="fas fa-lightbulb text-yellow-500"></i>Pro Tips
          </h4>
          <ul className="space-y-2 text-sm text-slate-700">
            <li><i className="fas fa-check text-green-600 mr-2"></i><strong>Test First:</strong> Always test camera connection before saving</li>
            <li><i className="fas fa-check text-green-600 mr-2"></i><strong>ANPR Enabled:</strong> Ensure ANPR is enabled for license plate detection</li>
            <li><i className="fas fa-check text-green-600 mr-2"></i><strong>Network Stable:</strong> Maintain stable network connection for reliable monitoring</li>
            <li><i className="fas fa-check text-green-600 mr-2"></i><strong>FPS Setting:</strong> Higher FPS = better detection but more CPU usage</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
