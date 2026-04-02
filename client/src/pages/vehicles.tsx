import { useState } from "react";
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
} from "@/components/ui/form";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest, queryClient } from "@/lib/queryClient";
import type { Vehicle, InsertVehicle } from "@shared/schema";
import { insertVehicleSchema } from "@shared/schema";
import { ImageZoomModal } from "@/components/image-zoom-modal";

const RECORDS_PER_PAGE = 10;

interface SearchFilters {
  licensePlate?: string;
  gate?: string;
  method?: string;
  date?: string;
}

export default function Vehicles() {
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({});
  const [editingVehicle, setEditingVehicle] = useState<Vehicle | null>(null);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [zoomImageSrc, setZoomImageSrc] = useState<string>("");
  const [isZoomModalOpen, setIsZoomModalOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const { toast } = useToast();
  const { user, subUser } = useAuth();

  // Check if sub-user has permission to add vehicles
  const canAddVehicles = !subUser || subUser.permissions?.canAddVehicles;
  
  // Build userId for queries
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const editForm = useForm<{
    driverName: string;
    driverMobile: string;
  }>({
    resolver: zodResolver(
      insertVehicleSchema.pick({
        driverName: true,
        driverMobile: true,
      })
    ),
    defaultValues: {
      driverName: "",
      driverMobile: "",
    },
  });

  const form = useForm<InsertVehicle>({
    resolver: zodResolver(insertVehicleSchema),
    defaultValues: {
      licensePlate: "",
      vehicleType: "Car",
      driverName: "",
      driverMobile: "",
      gate: "Entry",
      detectionMethod: "Manual",
      status: "Registered",
      notes: "",
      entryTime: new Date().toISOString(),
    },
  });

  // Query for vehicles - use standard query pattern with userId
  const { data: allVehicles = [], isLoading, refetch: refetchVehicles } = useQuery<Vehicle[]>({
    queryKey: ["/api/vehicles", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/vehicles?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      // Ensure data is an array
      return Array.isArray(data) ? data : [];
    },
    staleTime: 0, // Mark as stale immediately to ensure fresh data on invalidation
  });

  // Get matching plates for search suggestions (after allVehicles is defined)
  const matchingPlates = allVehicles
    .filter((v) =>
      v.licensePlate.toLowerCase().includes(searchInput.toLowerCase())
    )
    .map((v) => v.licensePlate)
    .filter((plate, index, self) => self.indexOf(plate) === index)
    .slice(0, 5);

  const handleSearchSuggestion = (plate: string) => {
    setSearchFilters({ ...searchFilters, licensePlate: plate });
    setSearchInput("");
    setShowSuggestions(false);
    setCurrentPage(1);
  };

  // Filter vehicles client-side based on search filters
  const filteredVehicles = allVehicles.filter((vehicle) => {
    if (
      searchFilters.licensePlate &&
      !vehicle.licensePlate
        .toLowerCase()
        .includes(searchFilters.licensePlate.toLowerCase())
    ) {
      return false;
    }
    if (
      searchFilters.gate &&
      searchFilters.gate !== "All" &&
      vehicle.gate !== searchFilters.gate
    ) {
      return false;
    }
    if (
      searchFilters.method &&
      searchFilters.method !== "All" &&
      vehicle.detectionMethod !== searchFilters.method
    ) {
      return false;
    }
    if (
      searchFilters.date &&
      !vehicle.entryTime.startsWith(searchFilters.date)
    ) {
      return false;
    }
    return true;
  });

  // Pagination logic
  const totalPages = Math.ceil(filteredVehicles.length / RECORDS_PER_PAGE);
  const vehicles = filteredVehicles.slice(
    (currentPage - 1) * RECORDS_PER_PAGE,
    currentPage * RECORDS_PER_PAGE
  );

  const addVehicleMutation = useMutation({
    mutationFn: async (data: InsertVehicle) => {
      const queryParams = new URLSearchParams();
      // Sub-users should use their parent's userId for data ownership
      const effectiveUserId = subUser ? subUser.parentUserId : user?.id;
      if (effectiveUserId) queryParams.append("userId", effectiveUserId);
      if (subUser) queryParams.append("subUserId", subUser.id);
      const response = await apiRequest("POST", `/api/vehicles?${queryParams.toString()}`, data);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vehicles", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      refetchVehicles();
      setCurrentPage(1);
      form.reset();
      toast({
        title: "Success",
        description: "Vehicle entry added successfully",
      });
    },
    onError: (error: any) => {
      const message = error?.response?.message || "Failed to add vehicle entry";
      toast({
        title: "Error",
        description: message,
        variant: "destructive",
      });
    },
  });

  const deleteVehicleMutation = useMutation({
    mutationFn: async (id: string) => {
      const queryParams = new URLSearchParams();
      const effectiveUserId = subUser ? subUser.parentUserId : user?.id;
      if (effectiveUserId) queryParams.append("userId", effectiveUserId);
      const response = await apiRequest("DELETE", `/api/vehicles/${id}?${queryParams.toString()}`);
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/vehicles", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      refetchVehicles();
      toast({
        title: "Success",
        description: "Vehicle entry deleted successfully",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to delete vehicle entry",
        variant: "destructive",
      });
    },
  });

  const updateVehicleMutation = useMutation({
    mutationFn: async (payload: Partial<Vehicle> & { id: string }) => {
      const response = await apiRequest(
        "PUT",
        `/api/vehicles/${payload.id}`,
        payload
      );
      return response.json();
    },
    onSuccess: () => {
      // Use the correct query key format with userId
      queryClient.invalidateQueries({ queryKey: ["/api/vehicles", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/detections", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/detections/recent", userIdParam] });
      queryClient.invalidateQueries({ queryKey: ["/api/stats", userIdParam] });
      refetchVehicles();
      toast({
        title: "Updated",
        description: "Vehicle updated successfully",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to update vehicle",
        variant: "destructive",
      });
    },
  });

  const handleSearch = () => {
    setCurrentPage(1);
    queryClient.invalidateQueries({ queryKey: ["/api/vehicles", userIdParam] });
  };

  const handleClearFilters = () => {
    setSearchFilters({});
    setCurrentPage(1);
    queryClient.invalidateQueries({ queryKey: ["/api/vehicles", userIdParam] });
  };

  const handleEditDriver = (vehicle: Vehicle) => {
    setEditingVehicle(vehicle);
    editForm.reset({
      driverName: vehicle.driverName || "",
      driverMobile: vehicle.driverMobile || "",
    });
    setIsEditDialogOpen(true);
  };

  const handleCloseEditDialog = () => {
    setIsEditDialogOpen(false);
    setEditingVehicle(null);
    editForm.reset();
  };

  const onEditSubmit = (data: { driverName: string; driverMobile: string }) => {
    if (!editingVehicle) return;
    updateVehicleMutation.mutate({
      id: editingVehicle.id,
      driverName: data.driverName,
      driverMobile: data.driverMobile,
    });
    handleCloseEditDialog();
  };

  const onSubmit = (data: InsertVehicle) => {
    if (!canAddVehicles) {
      toast({
        title: "Permission Denied",
        description: "You do not have permission to add vehicles. Please contact your administrator.",
        variant: "destructive",
      });
      return;
    }
    addVehicleMutation.mutate(data);
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 bg-white min-h-screen">
      {/* Decorative header background */}
      <div className="fixed inset-0 -z-10 h-64 bg-gradient-to-br from-blue-50 via-blue-25 to-transparent pointer-events-none"></div>

      {/* Header Section */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-2">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 mb-2 flex items-center gap-3">
              <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                <i className="fas fa-car text-2xl"></i>
              </div>
              Vehicle Records
            </h1>
            <p className="text-slate-600 text-base font-medium max-w-2xl">
              <i className="fas fa-check-circle text-[#3175F1] mr-2"></i>
              Manage and track all vehicle entries and exits in real-time
            </p>
          </div>
        </div>
      </div>

      {/* Search Filters - Professional */}
      <div className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl shadow-lg border border-blue-100 p-6 sm:p-8 mb-10 hover:shadow-xl hover:border-blue-200 transition-all duration-300">
        {/* Section Header with Icon */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8 pb-6 border-b border-blue-100">
          <div className="flex items-center gap-4 mb-4 sm:mb-0">
            <div className="p-3 backdrop-blur-md bg-[#3175F1]/20 border border-blue-200 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/30 transition-all">
              <i className="fas fa-sliders-h text-lg"></i>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Search & Filter</h2>
              <p className="text-sm text-slate-500 mt-1">Refine your vehicle records</p>
            </div>
          </div>
          {(searchFilters.licensePlate || searchFilters.gate || searchFilters.method) && (
            <div className="flex items-center gap-2 bg-blue-50 px-3 py-1 rounded-full">
              <span className="text-xs font-medium text-blue-700">
                <i className="fas fa-filter mr-1"></i>Active Filters: {(searchFilters.licensePlate ? 1 : 0) + (searchFilters.gate ? 1 : 0) + (searchFilters.method ? 1 : 0)}
              </span>
            </div>
          )}
        </div>

        {/* Filter Inputs Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
          {/* License Plate Filter with Smart Search */}
          <div className="group">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
              <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                <i className="fas fa-hashtag text-xs"></i>
              </span>
              License Plate
            </label>
            <div className="relative">
              <Input
                placeholder="Enter plate number..."
                value={searchInput || searchFilters.licensePlate || ""}
                onChange={(e) => {
                  setSearchInput(e.target.value);
                  setShowSuggestions(true);
                }}
                onFocus={() => setShowSuggestions(true)}
                data-testid="search-license-plate"
                className={`border-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 placeholder:text-slate-400 ${
                  searchFilters.licensePlate || searchInput
                    ? "border-blue-400 bg-blue-50/50 shadow-sm"
                    : "border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500 focus:bg-white"
                }`}
              />
              {(searchFilters.licensePlate || searchInput) && (
                <button
                  onClick={() => {
                    setSearchFilters({ ...searchFilters, licensePlate: undefined });
                    setSearchInput("");
                    setShowSuggestions(false);
                    setCurrentPage(1);
                  }}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-blue-600 transition-colors"
                >
                  <i className="fas fa-times-circle"></i>
                </button>
              )}
              
              {/* Search Suggestions Dropdown */}
              {showSuggestions && matchingPlates.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border-2 border-blue-300 rounded-xl shadow-lg z-50 max-h-56 overflow-y-auto">
                  {matchingPlates.map((plate) => (
                    <button
                      key={plate}
                      onClick={() => handleSearchSuggestion(plate)}
                      className="w-full px-4 py-2.5 text-left hover:bg-blue-100 border-b border-blue-50 last:border-b-0 transition-colors flex items-center gap-2 group/suggestion"
                    >
                      <i className="fas fa-check-circle text-blue-600 opacity-0 group-hover/suggestion:opacity-100 transition-opacity"></i>
                      <span className="font-semibold text-slate-900">{plate}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Gate Direction Filter */}
          <div className="group">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
              <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                <i className="fas fa-arrow-right-arrow-left text-xs"></i>
              </span>
              Gate Direction
            </label>
            <Select
              value={searchFilters.gate ? searchFilters.gate : "all"}
              onValueChange={(value) =>
                setSearchFilters({
                  ...searchFilters,
                  gate: value === "all" ? undefined : value,
                })
              }
            >
              <SelectTrigger className={`border-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                searchFilters.gate
                  ? "border-blue-400 bg-blue-50/50 shadow-sm"
                  : "border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500"
              }`}>
                <SelectValue placeholder="Select gate..." />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="all">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-door-open text-slate-500"></i>All Gates
                  </span>
                </SelectItem>
                <SelectItem value="Entry">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-arrow-right-to-bracket text-green-600"></i>Entry
                  </span>
                </SelectItem>
                <SelectItem value="Exit">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-arrow-right-from-bracket text-red-600"></i>Exit
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Detection Method Filter */}
          <div className="group">
            <label className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
              <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                <i className="fas fa-scanner text-xs"></i>
              </span>
              Detection Method
            </label>
            <Select
              value={searchFilters.method ? searchFilters.method : "all"}
              onValueChange={(value) =>
                setSearchFilters({
                  ...searchFilters,
                  method: value === "all" ? undefined : value,
                })
              }
            >
              <SelectTrigger className={`border-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                searchFilters.method
                  ? "border-blue-400 bg-blue-50/50 shadow-sm"
                  : "border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500"
              }`}>
                <SelectValue placeholder="Select method..." />
              </SelectTrigger>
              <SelectContent className="rounded-xl">
                <SelectItem value="all">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-list text-slate-500"></i>All Methods
                  </span>
                </SelectItem>
                <SelectItem value="ANPR">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-camera text-blue-600"></i>ANPR
                  </span>
                </SelectItem>
                <SelectItem value="Manual">
                  <span className="flex items-center gap-2">
                    <i className="fas fa-keyboard text-purple-600"></i>Manual
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
          <Button
            onClick={() => {
              if (searchInput) {
                setSearchFilters({ ...searchFilters, licensePlate: searchInput });
                setSearchInput("");
              }
              setCurrentPage(1);
              setShowSuggestions(false);
            }}
            disabled={!searchInput && !searchFilters.licensePlate && !searchFilters.gate && !searchFilters.method}
            className="bg-gradient-to-r from-[#3175F1] to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-200 py-2.5 px-6 flex-1 sm:flex-none disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="button-search"
          >
            <i className="fas fa-search mr-2.5"></i>Search Records
          </Button>
          {(searchFilters.licensePlate || searchFilters.gate || searchFilters.method) && (
            <Button
              onClick={handleClearFilters}
              variant="outline"
              className="border-2 border-slate-300 text-slate-700 hover:bg-slate-100 hover:border-slate-400 rounded-xl font-semibold transition-all duration-200 py-2.5 px-6"
              data-testid="button-clear-filters"
            >
              <i className="fas fa-rotate-left mr-2.5"></i>Clear Filters
            </Button>
          )}
        </div>
      </div>

      {/* Add Vehicle Form - Professional */}
      <div className="bg-gradient-to-br from-white to-blue-50/30 rounded-2xl shadow-lg border border-blue-100 p-6 sm:p-8 mb-10 hover:shadow-xl hover:border-blue-200 transition-all duration-300">
        {/* Form Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8 pb-6 border-b border-blue-100">
          <div className="flex items-center gap-4">
            <div className="p-3 backdrop-blur-md bg-[#3175F1]/20 border border-blue-200 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/30 transition-all">
              <i className="fas fa-plus-circle text-lg"></i>
            </div>
            <div>
              <h2 className="text-2xl font-bold text-slate-900">Add New Vehicle</h2>
              <p className="text-sm text-slate-500 mt-1">Register a new vehicle entry to the system</p>
            </div>
          </div>
        </div>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            {/* Form Fields Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <FormField
                control={form.control}
                name="licensePlate"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-hashtag text-xs"></i>
                      </span>
                      License Plate
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="ABC1234"
                        {...field}
                        data-testid="input-license-plate"
                        className="border-2 rounded-xl px-4 py-2.5 text-sm font-medium uppercase text-center font-mono transition-all duration-200 placeholder:text-slate-400 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500 focus:bg-white"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="vehicleType"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-car text-xs"></i>
                      </span>
                      Vehicle Type
                    </FormLabel>
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger data-testid="select-vehicle-type" className="border-2 rounded-xl text-sm font-medium transition-all duration-200 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent className="rounded-xl">
                        <SelectItem value="Car">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-car text-blue-600"></i>Car
                          </span>
                        </SelectItem>
                        <SelectItem value="Truck">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-truck text-purple-600"></i>Truck
                          </span>
                        </SelectItem>
                        <SelectItem value="Motorcycle">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-motorcycle text-orange-600"></i>Motorcycle
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="driverName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-user text-xs"></i>
                      </span>
                      Driver Name
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="John Doe"
                        {...field}
                        data-testid="input-driver-name"
                        className="border-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 placeholder:text-slate-400 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500 focus:bg-white"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="driverMobile"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-phone text-xs"></i>
                      </span>
                      Driver Mobile
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="+1 234 567 8900"
                        {...field}
                        data-testid="input-driver-mobile"
                        className="border-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 placeholder:text-slate-400 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500 focus:bg-white"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </div>

            {/* Additional Details Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <FormField
                control={form.control}
                name="gate"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-arrow-right-arrow-left text-xs"></i>
                      </span>
                      Gate Direction
                    </FormLabel>
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger data-testid="select-gate" className="border-2 rounded-xl text-sm font-medium transition-all duration-200 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent className="rounded-xl">
                        <SelectItem value="Entry">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-arrow-right-to-bracket text-green-600"></i>Entry
                          </span>
                        </SelectItem>
                        <SelectItem value="Exit">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-arrow-right-from-bracket text-red-600"></i>Exit
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="detectionMethod"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-scanner text-xs"></i>
                      </span>
                      Detection Method
                    </FormLabel>
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger className="border-2 rounded-xl text-sm font-medium transition-all duration-200 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent className="rounded-xl">
                        <SelectItem value="ANPR">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-camera text-blue-600"></i>ANPR
                          </span>
                        </SelectItem>
                        <SelectItem value="Manual">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-keyboard text-purple-600"></i>Manual
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="status"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-circle-check text-xs"></i>
                      </span>
                      Status
                    </FormLabel>
                    <Select
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                    >
                      <FormControl>
                        <SelectTrigger className="border-2 rounded-xl text-sm font-medium transition-all duration-200 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500">
                          <SelectValue />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent className="rounded-xl">
                        <SelectItem value="Registered">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-check-circle text-green-600"></i>Registered
                          </span>
                        </SelectItem>
                        <SelectItem value="Unregistered">
                          <span className="flex items-center gap-2">
                            <i className="fas fa-times-circle text-red-600"></i>Unregistered
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="notes"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="block text-xs font-bold text-slate-700 uppercase tracking-widest mb-2 flex items-center gap-2">
                      <span className="p-1.5 bg-blue-100 rounded-md text-blue-600">
                        <i className="fas fa-note-sticky text-xs"></i>
                      </span>
                      Additional Notes
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Optional notes or remarks..."
                        {...field}
                        data-testid="input-notes"
                        className="border-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 placeholder:text-slate-400 border-slate-200 bg-white hover:border-blue-200 focus:border-blue-500 focus:bg-white"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />
            </div>

            {/* Form Actions */}
            <div className="flex flex-col sm:flex-row gap-3 pt-6 border-t border-blue-100">
              <Button
                type="submit"
                disabled={addVehicleMutation.isPending || !canAddVehicles}
                data-testid="button-add-vehicle"
                className="bg-gradient-to-r from-[#3175F1] to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-xl font-semibold shadow-lg hover:shadow-xl transition-all duration-200 py-2.5 px-6 flex-1 sm:flex-none"
              >
                {addVehicleMutation.isPending ? (
                  <>
                    <i className="fas fa-spinner animate-spin mr-2.5"></i>
                    Processing...
                  </>
                ) : (
                  <>
                    <i className="fas fa-check-circle mr-2.5"></i>
                    Register Vehicle
                  </>
                )}
              </Button>
              {!canAddVehicles && (
                <div className="flex items-center gap-2 text-red-600 text-sm font-semibold">
                  <i className="fas fa-lock"></i>No permission to add vehicles
                </div>
              )}
            </div>
          </form>
        </Form>
      </div>

      {/* Vehicle Records Container - Enhanced */}
      <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-100 overflow-hidden hover:border-blue-200 transition-colors">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-8 border-b-2 border-blue-100 bg-gradient-to-r from-blue-50 to-slate-50">
          <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <i className="fas fa-list text-blue-600"></i>
            </div>
            Vehicle Records
          </h2>
          <div className="flex items-center gap-3 flex-wrap">
            {isLoading ? (
              <div className="text-slate-600 font-semibold flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                Loading...
              </div>
            ) : (
              <>
                <span className="bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-bold text-sm">
                  {filteredVehicles.length} Total
                </span>
                <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full font-bold text-sm">
                  Page {currentPage} of {Math.max(1, totalPages)}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Mobile Card View */}
        <div className="block md:hidden">
          {vehicles.length === 0 ? (
            <div className="text-center text-slate-500 py-16 p-4">
              <i className="fas fa-car text-5xl mb-4 opacity-20 block"></i>
              <p className="text-slate-600 font-semibold text-lg">No vehicles found</p>
              <p className="text-slate-500 text-sm mt-2">Add a vehicle or adjust your filters</p>
            </div>
          ) : (
            <div className="space-y-4 p-6">
              {vehicles.map((vehicle) => (
                <div
                  key={vehicle.id}
                  className="bg-white rounded-xl border-2 border-blue-100 p-5 shadow-sm hover:shadow-md hover:border-blue-300 transition-all"
                >
                  {/* Plate Image - Fixed Size */}
                  {vehicle.plateImage && (
                    <div className="mb-4 flex justify-center">
                      <div className="relative group">
                        <div className="w-40 h-20 bg-gradient-to-br from-slate-100 to-slate-50 rounded-lg border-2 border-blue-300 flex items-center justify-center overflow-hidden">
                          <img
                            src={`/api/images/plates/${vehicle.plateImage.split('/').pop()}`}
                            alt={`Plate ${vehicle.licensePlate}`}
                            className="max-w-full max-h-full object-contain cursor-pointer transition-all duration-200 group-hover:scale-105"
                            onClick={() => {
                              setZoomImageSrc(`/api/images/plates/${vehicle.plateImage.split('/').pop()}`);
                              setIsZoomModalOpen(true);
                            }}
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = "";
                            }}
                            title="Click to zoom"
                          />
                        </div>
                        <div className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center bg-black/10 pointer-events-none">
                          <i className="fas fa-search-plus text-white text-lg drop-shadow"></i>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Vehicle Info */}
                  <div className="flex items-start justify-between mb-4 pb-4 border-b border-blue-100">
                    <div>
                      <div className="font-mono font-bold text-lg text-blue-900">{vehicle.licensePlate}</div>
                      <div className="text-xs text-slate-600 mt-1 flex items-center gap-2">
                        <i className="fas fa-car-side text-blue-600"></i>
                        <span>{vehicle.vehicleType}</span>
                      </div>
                    </div>
                    <div
                      className={`px-3 py-1 text-xs font-bold rounded-full border cursor-pointer transition-all ${
                        vehicle.status === "Registered"
                          ? "bg-green-50 text-green-700 border-green-300 hover:bg-green-100"
                          : "bg-red-50 text-red-700 border-red-300 hover:bg-red-100"
                      }`}
                      onClick={() =>
                        updateVehicleMutation.mutate({
                          id: vehicle.id,
                          status:
                            vehicle.status === "Registered"
                              ? "Unregistered"
                              : "Registered",
                        })
                      }
                    >
                      {vehicle.status}
                    </div>
                  </div>

                  {/* Details Grid */}
                  <div className="space-y-2 text-xs mb-4">
                    <div className="flex justify-between">
                      <span className="text-slate-600 font-semibold flex items-center gap-1">
                        <i className="fas fa-sign-in-alt text-blue-600"></i>Entry
                      </span>
                      <span className="text-slate-900 font-medium">{new Date(vehicle.entryTime).toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600 font-semibold flex items-center gap-1">
                        <i className="fas fa-sign-out-alt text-blue-600"></i>Exit
                      </span>
                      <span className="text-slate-900 font-medium">{vehicle.exitTime ? new Date(vehicle.exitTime).toLocaleString() : <span className="text-slate-400">—</span>}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600 font-semibold flex items-center gap-1">
                        <i className="fas fa-dooropen text-blue-600"></i>Gate
                      </span>
                      <span className="text-blue-700 font-medium">{vehicle.gate}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-600 font-semibold flex items-center gap-1">
                        <i className="fas fa-barcode text-blue-600"></i>Method
                      </span>
                      <span className="text-slate-900 font-medium">{vehicle.detectionMethod}</span>
                    </div>
                    {vehicle.driverName && (
                      <div className="flex justify-between">
                        <span className="text-slate-600 font-semibold flex items-center gap-1">
                          <i className="fas fa-user text-blue-600"></i>Driver
                        </span>
                        <span className="text-slate-900 font-medium">{vehicle.driverName}</span>
                      </div>
                    )}
                    {vehicle.driverMobile && (
                      <div className="flex justify-between">
                        <span className="text-slate-600 font-semibold flex items-center gap-1">
                          <i className="fas fa-phone text-blue-600"></i>Mobile
                        </span>
                        <span className="text-slate-900 font-medium">{vehicle.driverMobile}</span>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex gap-2 pt-4 border-t border-blue-100">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg font-semibold"
                      onClick={() => handleEditDriver(vehicle)}
                    >
                      <i className="fas fa-edit mr-2"></i>Edit
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="flex-1 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg font-semibold"
                      onClick={() => deleteVehicleMutation.mutate(vehicle.id)}
                    >
                      <i className="fas fa-trash-alt mr-2"></i>Delete
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Desktop Table View */}
        <div className="hidden md:block overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gradient-to-r from-blue-100 to-slate-100 border-b-2 border-blue-200">
              <tr>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-image text-blue-600 mr-2"></i>Plate Image
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-barcode text-blue-600 mr-2"></i>License
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-sign-in-alt text-blue-600 mr-2"></i>Entry
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-sign-out-alt text-blue-600 mr-2"></i>Exit
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-camera text-blue-600 mr-2"></i>Method
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-check-circle text-blue-600 mr-2"></i>Status
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-user text-blue-600 mr-2"></i>Driver
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider whitespace-nowrap">
                  <i className="fas fa-phone text-blue-600 mr-2"></i>Mobile
                </th>
                <th className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider">
                  <i className="fas fa-cog text-blue-600 mr-2"></i>Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-blue-100">
              {isLoading ? (
                <tr>
                  <td colSpan={9} className="px-6 py-8 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                      <span className="text-blue-600 font-semibold">Loading vehicles...</span>
                    </div>
                  </td>
                </tr>
              ) : vehicles.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-6 py-12 text-center">
                    <i className="fas fa-car text-5xl mb-3 opacity-20 block"></i>
                    <p className="text-slate-600 font-semibold">No vehicles found</p>
                  </td>
                </tr>
              ) : (
                vehicles.map((vehicle) => (
                  <tr
                    key={vehicle.id}
                    className="hover:bg-blue-50 transition-colors"
                    data-testid={`vehicle-row-${vehicle.id}`}
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {vehicle.plateImage ? (
                        <div className="relative group">
                          <div className="w-32 h-16 bg-gradient-to-br from-slate-100 to-slate-50 rounded-lg border-2 border-blue-300 flex items-center justify-center overflow-hidden">
                            <img
                              src={`/api/images/plates/${vehicle.plateImage.split('/').pop()}`}
                              alt={`Plate ${vehicle.licensePlate}`}
                              className="max-w-full max-h-full object-contain cursor-pointer transition-all duration-200 group-hover:scale-105"
                              onClick={() => {
                                setZoomImageSrc(`/api/images/plates/${vehicle.plateImage.split('/').pop()}`);
                                setIsZoomModalOpen(true);
                              }}
                              onError={(e) => {
                                (e.target as HTMLImageElement).src = "";
                              }}
                              title="Click to zoom"
                            />
                          </div>
                          <div className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center justify-center bg-black/10 pointer-events-none">
                            <i className="fas fa-search-plus text-white text-sm drop-shadow"></i>
                          </div>
                        </div>
                      ) : (
                        <span className="text-slate-400 text-xs">No image</span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono font-bold text-blue-900">
                      {vehicle.licensePlate}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                      {new Date(vehicle.entryTime).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                      {vehicle.exitTime ? new Date(vehicle.exitTime).toLocaleString() : <span className="text-slate-400">—</span>}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                      {vehicle.detectionMethod}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`px-3 py-1 text-xs font-bold rounded-full cursor-pointer transition-all border ${
                          vehicle.status === "Registered"
                            ? "bg-green-50 text-green-700 border-green-300 hover:bg-green-100"
                            : "bg-red-50 text-red-700 border-red-300 hover:bg-red-100"
                        }`}
                        onClick={() =>
                          updateVehicleMutation.mutate({
                            id: vehicle.id,
                            status: vehicle.status === "Registered" ? "Unregistered" : "Registered",
                          })
                        }
                        data-testid={`status-toggle-${vehicle.id}`}
                      >
                        {vehicle.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                      {vehicle.driverName || <span className="text-slate-400">Unknown</span>}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">
                      {vehicle.driverMobile || <span className="text-slate-400">—</span>}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                          onClick={() => handleEditDriver(vehicle)}
                          data-testid={`button-edit-${vehicle.id}`}
                        >
                          <i className="fas fa-edit"></i>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                          onClick={() => deleteVehicleMutation.mutate(vehicle.id)}
                          data-testid={`button-delete-${vehicle.id}`}
                        >
                          <i className="fas fa-trash-alt"></i>
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination - Enhanced */}
        {totalPages > 1 && (
          <div className="bg-gradient-to-r from-blue-50 to-slate-50 border-t-2 border-blue-100 px-6 sm:px-8 py-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-sm text-slate-600 font-medium">
              Showing <span className="font-bold text-blue-900">{(currentPage - 1) * RECORDS_PER_PAGE + 1}</span> to{" "}
              <span className="font-bold text-blue-900">{Math.min(currentPage * RECORDS_PER_PAGE, filteredVehicles.length)}</span> of{" "}
              <span className="font-bold text-blue-900">{filteredVehicles.length}</span> records
            </div>
            <div className="flex gap-2 flex-wrap justify-center sm:justify-end">
              <Button
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
                variant="outline"
                className="border-2 border-blue-200 text-blue-600 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-all"
              >
                <i className="fas fa-chevron-left mr-2"></i>Previous
              </Button>
              <div className="flex gap-1 items-center">
                {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                  <Button
                    key={page}
                    onClick={() => handlePageChange(page)}
                    className={`w-10 h-10 rounded-lg font-bold transition-all ${
                      currentPage === page
                        ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg"
                        : "border-2 border-blue-200 text-blue-600 hover:bg-blue-50"
                    }`}
                  >
                    {page}
                  </Button>
                ))}
              </div>
              <Button
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage === totalPages}
                variant="outline"
                className="border-2 border-blue-200 text-blue-600 hover:bg-blue-50 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-all"
              >
                Next <i className="fas fa-chevron-right ml-2"></i>
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Edit Driver Dialog - Enhanced */}
      <Dialog open={isEditDialogOpen} onOpenChange={setIsEditDialogOpen}>
        <DialogContent className="sm:max-w-[500px] border-2 border-blue-200 rounded-2xl">
          <DialogHeader className="border-b-2 border-blue-100 pb-4 mb-4">
            <DialogTitle className="text-blue-900 text-2xl font-bold flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <i className="fas fa-edit text-blue-600"></i>
              </div>
              Edit Driver Information
            </DialogTitle>
            <DialogDescription className="text-slate-700 mt-2">
              Update driver details for vehicle{" "}
              <span className="font-mono font-bold text-blue-900 text-base">{editingVehicle?.licensePlate}</span>
            </DialogDescription>
          </DialogHeader>
          <Form {...editForm}>
            <form
              onSubmit={editForm.handleSubmit(onEditSubmit)}
              className="space-y-5"
            >
              <FormField
                control={editForm.control}
                name="driverName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-slate-700 font-semibold flex items-center gap-2 mb-2">
                      <i className="fas fa-user text-blue-600"></i>Driver Name
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="Enter driver name"
                        {...field}
                        data-testid="edit-driver-name"
                        className="border-2 border-blue-100 focus:border-blue-500 rounded-lg transition-colors"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <FormField
                control={editForm.control}
                name="driverMobile"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-slate-700 font-semibold flex items-center gap-2 mb-2">
                      <i className="fas fa-phone text-blue-600"></i>Driver Mobile
                    </FormLabel>
                    <FormControl>
                      <Input
                        placeholder="+1 234 567 8900"
                        {...field}
                        data-testid="edit-driver-mobile"
                        className="border-2 border-blue-100 focus:border-blue-500 rounded-lg transition-colors"
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              <DialogFooter className="flex gap-3 pt-4 border-t-2 border-blue-100 mt-6">
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleCloseEditDialog}
                  className="border-2 border-blue-200 text-blue-600 hover:bg-blue-50 rounded-lg font-semibold"
                >
                  <i className="fas fa-times mr-2"></i>Cancel
                </Button>
                <Button
                  type="submit"
                  disabled={editForm.formState.isSubmitting}
                  className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg font-semibold shadow-md transition-all"
                  data-testid="button-save-driver"
                >
                  {editForm.formState.isSubmitting ? (
                    <>
                      <i className="fas fa-spinner animate-spin mr-2"></i>
                      Saving...
                    </>
                  ) : (
                    <>
                      <i className="fas fa-check mr-2"></i>
                      Save Changes
                    </>
                  )}
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      {/* Image Zoom Modal */}
      <ImageZoomModal
        isOpen={isZoomModalOpen}
        onOpenChange={setIsZoomModalOpen}
        imageSrc={zoomImageSrc}
        altText="Vehicle License Plate"
        title="License Plate Image Viewer"
      />
    </div>
  );
}
