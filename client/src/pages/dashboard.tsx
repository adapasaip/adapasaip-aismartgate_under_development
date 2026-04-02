import { useQuery } from "@tanstack/react-query";
import StatsCard from "@/components/stats-card";
import LiveCameraFeed from "@/components/live-camera-feed";
import RecentDetections from "@/components/recent-detections";
import type { Stats, Vehicle } from "@shared/schema";
import { Button } from "@/components/ui/button";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { Card, CardContent } from "@/components/ui/card";
import { apiRequest } from "@/lib/queryClient";

export default function Dashboard() {
  const [, setLocation] = useLocation();
  const { user, subUser, isAuthenticated, isSubUserAuthenticated } = useAuth();

  // Redirect if neither main user nor sub-user is authenticated
  if (!isAuthenticated && !isSubUserAuthenticated) {
    setLocation("/login");
    return null;
  }

  const currentUser = user || subUser;

  // Build query params for user-specific data
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery<Stats>({
    queryKey: ["/api/stats", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/stats?userId=${encodeURIComponent(userIdParam)}`);
      return response.json();
    },
    refetchInterval: 5000, // Reduced from 30s to 5s for real-time updates
    staleTime: 2000, // Data considered stale after 2 seconds
  });

  const { data: vehicles = [], refetch: refetchVehicles } = useQuery<Vehicle[]>({
    queryKey: ["/api/vehicles", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/vehicles?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      // Ensure data is an array
      return Array.isArray(data) ? data : [];
    },
    select: (data) => {
      // Sort by entry time (newest first) and take top 10
      return Array.isArray(data) ? data
        .sort((a, b) => new Date(b.entryTime).getTime() - new Date(a.entryTime).getTime())
        .slice(0, 10) : [];
    },
    refetchInterval: 5000, // Reduced from 30s to 5s for real-time updates
    staleTime: 2000,
  });

  const { data: detections = [], refetch: refetchDetections } = useQuery({
    queryKey: ["/api/detections", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/detections?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      // Handle both array and wrapped response
      if (Array.isArray(data)) return data;
      if (data && data.data && Array.isArray(data.data)) return data.data;
      return [];
    },
    refetchInterval: 3000, // Reduced from 30s to 3s for real-time detection updates
    staleTime: 1000,
  });

  // 🔥 Merge detections + vehicles by licensePlate
  const mergedVehicles = Array.isArray(vehicles) ? vehicles.map((v) => {
    const match = Array.isArray(detections) ? detections
      .filter((d: any) => d.licensePlate === v.licensePlate)
      .sort(
        (a: any, b: any) =>
          new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime()
      )[0] : undefined;

    return {
      ...v,
      plateImage: v.plateImage || match?.plateImage || null,
      vehicleImage: v.vehicleImage || match?.vehicleImage || null,
      detectedAt: match?.detectedAt || null,
    };
  }) : [];

  if (statsLoading) {
    return (
      <div className="p-4 sm:p-6">
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-800 mb-2">
            Dashboard
          </h1>
          <p className="text-slate-600 text-sm sm:text-base">
            Real-time monitoring and analytics
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 sm:gap-6">
          {[...Array(5)].map((_, i) => (
            <div
              key={i}
              className="bg-white rounded-xl shadow-sm border border-slate-200 p-6 animate-pulse"
            >
              <div className="h-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 bg-white min-h-screen">
      {/* Decorative header background */}
      <div className="fixed inset-0 -z-10 h-80 bg-gradient-to-br from-blue-50 via-blue-25 to-white pointer-events-none"></div>

      {/* Header */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 mb-2 flex items-center gap-3">
              <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                <i className="fas fa-gauge-high text-2xl"></i>
              </div>
              Dashboard
            </h1>
            <p className="text-slate-600 text-base font-medium">
              <i className="fas fa-chart-line text-[#3175F1] mr-2"></i>
              Real-time ANPR monitoring and analytics
            </p>
          </div>
          {currentUser && (
            <div className="bg-white rounded-2xl border-2 border-blue-100 shadow-sm p-5 hover:shadow-md transition-all">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-14 h-14 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full text-white shadow-md">
                  <i className="fas fa-user text-lg"></i>
                </div>
                <div>
                  <p className="font-bold text-blue-900">{currentUser.fullName || "User"}</p>
                  <p className="text-xs text-slate-500">{subUser ? `Sub-User • ${subUser.username}` : currentUser.mobile}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Stats Cards - Enhanced Professional Design */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 sm:gap-6 mb-10">
        <div className="bg-white rounded-2xl border-2 border-blue-100 p-6 shadow-sm hover:shadow-md hover:border-blue-300 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Total Vehicles</p>
              <p className="text-4xl font-bold text-blue-900 mt-3">{stats?.totalVehicles || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-blue-100 to-blue-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-blue-200 hover:to-blue-100">
              <i className="fas fa-car text-3xl text-blue-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-blue-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Total Vehicles
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-blue-100">
            <span className="text-xs text-slate-500"><i className="fas fa-arrow-up text-green-500 mr-1"></i>+12% from last month</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-green-100 p-6 shadow-sm hover:shadow-md hover:border-green-300 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Today's Entries</p>
              <p className="text-4xl font-bold text-green-900 mt-3">{stats?.todayEntries || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-green-100 to-green-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-green-200 hover:to-green-100">
              <i className="fas fa-sign-in-alt text-3xl text-green-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-green-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Today's Entries
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-green-100">
            <span className="text-xs text-slate-500"><i className="fas fa-arrow-up text-green-500 mr-1"></i>+5% from yesterday</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-yellow-100 p-6 shadow-sm hover:shadow-md hover:border-yellow-300 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Today's Exits</p>
              <p className="text-4xl font-bold text-yellow-900 mt-3">{stats?.todayExits || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-yellow-100 to-yellow-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-yellow-200 hover:to-yellow-100">
              <i className="fas fa-sign-out-alt text-3xl text-yellow-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-yellow-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Today's Exits
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-yellow-100">
            <span className="text-xs text-slate-500"><i className="fas fa-arrow-up text-green-500 mr-1"></i>+3% from yesterday</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-purple-100 p-6 shadow-sm hover:shadow-md hover:border-purple-300 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">Active Cameras</p>
              <p className="text-4xl font-bold text-purple-900 mt-3">{stats?.activeCameras || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-purple-100 to-purple-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-purple-200 hover:to-purple-100">
              <i className="fas fa-camera text-3xl text-purple-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-purple-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                Active Cameras
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-purple-100">
            <span className="text-xs text-slate-500"><i className="fas fa-check-circle text-green-500 mr-1"></i>{stats?.totalCameras || 0} total cameras</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl border-2 border-orange-100 p-6 shadow-sm hover:shadow-md hover:border-orange-300 transition-all duration-300">
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className="text-sm font-semibold text-slate-600 uppercase tracking-wide">ANPR Detections</p>
              <p className="text-4xl font-bold text-orange-900 mt-3">{stats?.anprDetections || 0}</p>
            </div>
            <div className="w-16 h-16 bg-gradient-to-br from-orange-100 to-orange-50 rounded-2xl flex items-center justify-center cursor-help group relative transition-all duration-200 hover:from-orange-200 hover:to-orange-100">
              <i className="fas fa-eye text-3xl text-orange-600 transition-transform group-hover:scale-110 duration-200"></i>
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 px-3 py-2 bg-orange-600/90 text-white text-xs rounded-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none shadow-lg backdrop-blur-sm font-semibold">
                ANPR Detections
              </div>
            </div>
          </div>
          <div className="pt-4 border-t border-orange-100">
            <span className="text-xs text-slate-500"><i className="fas fa-award text-green-500 mr-1"></i>94% accuracy rate</span>
          </div>
        </div>
      </div>

      {/* Live Camera + Recent Detections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <LiveCameraFeed />
        <RecentDetections />
      </div>

      {/* Recent Vehicles Table - Enhanced styling */}
      <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-100 overflow-hidden">
        <div className="p-6 border-b-2 border-blue-100 bg-gradient-to-r from-blue-50 via-white to-slate-50">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <h2 className="text-2xl font-bold text-blue-900 mb-1 flex items-center gap-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <i className="fas fa-car text-blue-600"></i>
                </div>
                Recent Vehicle Entries
              </h2>
              <p className="text-sm text-slate-600 ml-11"><i className="fas fa-info-circle text-blue-500 mr-1"></i>Latest 10 detected vehicles</p>
            </div>
            <Button
              onClick={() => setLocation("/vehicles")}
              data-testid="button-view-all-vehicles"
              className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-lg shadow-md transition-all"
            >
              <i className="fas fa-arrow-right mr-2"></i>View All Vehicles
            </Button>
          </div>
        </div>

        <div className="overflow-x-auto">
          {/* Responsive Table: Card on mobile, table on md+ */}
          <div className="block md:hidden">
            {mergedVehicles.length === 0 ? (
              <div className="text-center text-slate-500 py-8">
                <i className="fas fa-car text-4xl mb-2 opacity-30"></i>
                <p className="text-slate-600">No vehicle entries found</p>
              </div>
            ) : (
              <div className="space-y-3 p-4">
                {mergedVehicles.map((vehicle) => (
                  <div key={vehicle.id} className="bg-gradient-to-br from-blue-50 to-white rounded-lg border-2 border-blue-200 p-4 shadow-sm hover:shadow-md transition-shadow">
                    {/* Image and Status Row */}
                    <div className="flex items-start gap-2 mb-3">
                      {vehicle.plateImage ? (
                        <img src={`/api/images/plates/${vehicle.plateImage.split('/').pop()}`} alt={vehicle.licensePlate} className="h-14 w-28 object-contain rounded-lg border border-blue-200 bg-blue-50 flex-shrink-0" onError={(e) => {(e.target as HTMLImageElement).style.display = "none"}} />
                      ) : (
                        <div className="h-14 w-28 rounded-lg border border-blue-200 bg-blue-50 flex items-center justify-center text-xs text-slate-400 flex-shrink-0">No Image</div>
                      )}
                      <div className="flex-1 min-w-0">
                        {/* License Plate - Full width, no truncate */}
                        <div className="font-mono font-bold text-sm text-blue-900 break-words leading-tight">{vehicle.licensePlate}</div>
                        <div className="text-xs text-slate-600 mt-1 flex items-center gap-1">
                          <i className="fas fa-car-side text-blue-600"></i><span className="truncate">{vehicle.vehicleType}</span>
                        </div>
                        <span className={`inline-block mt-2 px-2 py-1 text-xs font-semibold rounded-full border ${vehicle.status === "Registered" ? "bg-green-100 text-green-700 border-green-300" : "bg-red-100 text-red-700 border-red-300"}`}>
                          {vehicle.status}
                        </span>
                      </div>
                    </div>
                    {/* Entry/Exit Times Row */}
                    <div className="space-y-2 text-xs border-t border-blue-100 pt-3">
                      <div className="flex justify-between items-start gap-2">
                        <span className="text-slate-600 font-semibold flex items-center gap-1 flex-shrink-0"><i className="fas fa-sign-in-alt text-blue-600"></i>Entry:</span>
                        <span className="font-medium text-slate-900 text-right">{new Date(vehicle.entryTime).toLocaleDateString([], { month: 'short', day: 'numeric' })} {new Date(vehicle.entryTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                      </div>
                      <div className="flex justify-between items-start gap-2">
                        <span className="text-slate-600 font-semibold flex items-center gap-1 flex-shrink-0"><i className="fas fa-sign-out-alt text-blue-600"></i>Exit:</span>
                        <span className="font-medium text-slate-900 text-right">{vehicle.exitTime ? `${new Date(vehicle.exitTime).toLocaleDateString([], { month: 'short', day: 'numeric' })} ${new Date(vehicle.exitTime).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}` : <span className="text-slate-400">—</span>}</span>
                      </div>
                      <div className="flex justify-between items-start gap-2">
                        <span className="text-slate-600 font-semibold flex items-center gap-1 flex-shrink-0"><i className="fas fa-gate text-blue-600"></i>Gate:</span>
                        <span className="font-medium text-blue-700">{vehicle.gate}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <table className="min-w-full hidden md:table">
            <thead className="bg-gradient-to-r from-blue-100 to-slate-100 border-b-2 border-blue-300">
              <tr>
                {[
                  "Image",
                  "License Plate",
                  "Vehicle Type",
                  "Entry Time",
                  "Exit Time",
                  "Gate",
                  "Status",
                ].map((header) => (
                  <th
                    key={header}
                    className="px-6 py-4 text-left text-xs font-bold text-blue-900 uppercase tracking-wider"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-blue-200">
              {mergedVehicles.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-slate-500">
                    <i className="fas fa-car text-4xl mb-2 opacity-30"></i>
                    <p className="text-slate-600">No vehicle entries found</p>
                  </td>
                </tr>
              ) : (
                mergedVehicles.map((vehicle) => (
                  <tr key={vehicle.id} className="hover:bg-blue-50 transition-colors" data-testid={`vehicle-row-${vehicle.id}`}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      {vehicle.plateImage ? (
                        <img src={`/api/images/plates/${vehicle.plateImage.split('/').pop()}`} alt={vehicle.licensePlate} className="h-14 w-24 object-contain rounded-lg border border-blue-200 bg-blue-50 transition-all duration-200 hover:scale-150 hover:z-50 hover:shadow-xl" onError={(e) => {(e.target as HTMLImageElement).style.display = "none"}} />
                      ) : (
                        <div className="h-14 w-24 rounded-lg border border-blue-200 bg-blue-50 flex items-center justify-center text-xs text-slate-400">No Image</div>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-mono font-bold text-blue-900">{vehicle.licensePlate}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700"><i className="fas fa-car-side text-blue-600 mr-2"></i>{vehicle.vehicleType}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{new Date(vehicle.entryTime).toLocaleString()}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-700">{vehicle.exitTime ? new Date(vehicle.exitTime).toLocaleString() : <span className="text-slate-400">—</span>}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-700">{vehicle.gate}</td>
                    <td className="px-6 py-4 whitespace-nowrap"><span className={`px-3 py-1 text-xs font-bold rounded-full border ${vehicle.status === "Registered" ? "bg-green-50 text-green-700 border-green-300" : "bg-red-50 text-red-700 border-red-300"}`}>{vehicle.status}</span></td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
