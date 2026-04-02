import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest } from "@/lib/queryClient";
import type { Camera } from "@shared/schema";

interface ActiveCamerasGridProps {
  compact?: boolean;
}

export default function ActiveCamerasGrid({ compact = false }: ActiveCamerasGridProps) {
  const { user, subUser } = useAuth();
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  const { data: cameras = [], isLoading } = useQuery<Camera[]>({
    queryKey: ["/api/cameras", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/cameras?userId=${encodeURIComponent(userIdParam)}`);
      const data = await response.json();
      return Array.isArray(data) ? data : [];
    },
    refetchInterval: 5000,
  });

  const activeCameras = cameras.filter(c => c.status === "Online");

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="bg-white rounded-xl border-2 border-blue-100 p-5 animate-pulse">
            <div className="h-20 bg-slate-200 rounded-lg mb-3"></div>
            <div className="h-4 bg-slate-100 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div>
      <h3 className="text-lg font-bold text-blue-900 mb-4 flex items-center gap-2">
        <i className="fas fa-video text-blue-600 text-xl"></i>
        Active Cameras ({activeCameras.length}/{cameras.length})
      </h3>
      
      {cameras.length === 0 ? (
        <div className="bg-white rounded-xl border-2 border-blue-100 p-8 text-center">
          <i className="fas fa-camera text-4xl text-slate-300 mb-3 block"></i>
          <p className="text-slate-600">No cameras configured</p>
        </div>
      ) : (
        <div className={`grid ${compact ? "grid-cols-1 sm:grid-cols-2 gap-3" : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"}`}>
          {cameras.map((camera) => (
            <div
              key={camera.id}
              className={`bg-white rounded-xl border-2 ${
                camera.status === "Online" ? "border-green-200 hover:border-green-400" : "border-slate-200 opacity-60"
              } hover:shadow-lg transition-all duration-300 overflow-hidden group`}
            >
              {/* Status bar */}
              <div className={`h-1 ${camera.status === "Online" ? "bg-gradient-to-r from-green-500 to-green-400" : "bg-slate-300"}`}></div>

              {/* Content */}
              <div className="p-4">
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h4 className="font-bold text-slate-900 text-sm truncate">{camera.name}</h4>
                    <p className="text-xs text-slate-500 mt-1 flex items-center gap-1">
                      <i className="fas fa-map-pin text-blue-500"></i>
                      {camera.location}
                    </p>
                  </div>
                  <Badge
                    className={`ml-2 whitespace-nowrap font-bold text-xs ${
                      camera.status === "Online"
                        ? "bg-green-100 text-green-700 border border-green-300"
                        : "bg-slate-100 text-slate-600 border border-slate-300"
                    }`}
                  >
                    <i className={`fas mr-1 ${camera.status === "Online" ? "fa-circle-check text-green-600" : "fa-circle-xmark"}`}></i>
                    {camera.status}
                  </Badge>
                </div>

                {/* Camera info grid */}
                <div className="space-y-2 mb-3">
                  <div className="flex items-center justify-between text-xs bg-blue-50 px-2 py-1.5 rounded">
                    <span className="text-slate-600 font-semibold"><i className="fas fa-video text-blue-500 mr-1.5"></i>Type</span>
                    <span className="text-slate-800 text-right truncate max-w-24">{camera.type.split("(")[0].trim()}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs bg-slate-50 px-2 py-1.5 rounded">
                    <span className="text-slate-600 font-semibold"><i className="fas fa-door-open text-blue-500 mr-1.5"></i>Gate</span>
                    <span className={`font-bold ${camera.gate === "Entry" ? "text-blue-600" : "text-orange-600"}`}>{camera.gate}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs bg-blue-50 px-2 py-1.5 rounded">
                    <span className="text-slate-600 font-semibold"><i className="fas fa-film text-blue-500 mr-1.5"></i>FPS</span>
                    <span className="text-slate-800 font-medium">{camera.fps}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs bg-slate-50 px-2 py-1.5 rounded">
                    <span className="text-slate-600 font-semibold"><i className="fas fa-barcode text-blue-500 mr-1.5"></i>ANPR</span>
                    <Badge
                      className={`text-xs ${
                        camera.anprEnabled
                          ? "bg-green-100 text-green-700 border border-green-300"
                          : "bg-slate-100 text-slate-600 border border-slate-300"
                      }`}
                    >
                      {camera.anprEnabled ? "Active" : "Off"}
                    </Badge>
                  </div>
                </div>

                {/* Quick action button */}
                {camera.status === "Online" && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-full text-xs bg-blue-50 hover:bg-blue-100 text-blue-600 border border-blue-200 rounded-lg transition-all"
                  >
                    <i className="fas fa-eye-slash mr-1"></i>View Feed
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
