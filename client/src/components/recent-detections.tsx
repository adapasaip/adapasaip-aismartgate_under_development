import { useQuery } from "@tanstack/react-query";
import type { AnprDetection } from "@shared/schema";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/hooks/use-auth";
import { apiRequest } from "@/lib/queryClient";

export default function RecentDetections() {
  const { user, subUser } = useAuth();
  const userIdParam = user ? user.id : (subUser?.parentUserId || "default");

  const { data: detections = [], isLoading, refetch } = useQuery<AnprDetection[]>({
    queryKey: ["/api/detections/recent", userIdParam],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/detections/recent?userId=${encodeURIComponent(userIdParam)}`);
      if (!response.ok) throw new Error("Failed to fetch detections");
      const data = await response.json();
      // Ensure data is an array
      const arr = Array.isArray(data) ? data : [];
      
      // ★ Sort by most recent detection time (entry or exit) - don't prioritize by type
      const sorted = arr.sort((a, b) => {
        // Use exitTime if available (exit detection), otherwise use detectedAt
        const aTime = a.exitTime ? new Date(a.exitTime).getTime() : new Date(a.detectedAt).getTime();
        const bTime = b.exitTime ? new Date(b.exitTime).getTime() : new Date(b.detectedAt).getTime();
        return bTime - aTime; // Most recent first
      });
      
      // ★ Limit to last 5 detections
      return sorted.slice(0, 5);
    },
    refetchInterval: 2000, // Real-time updates every 2 seconds
    staleTime: 500, // Data stale after 500ms
    gcTime: 1000, // Keep in cache for 1 second
    enabled: !!userIdParam, // Only fetch if we have a userId
  });

  return (
    <div className="bg-white rounded-2xl shadow-sm border-2 border-blue-100 w-full overflow-hidden hover:border-blue-200 transition-colors">
      {/* Header */}
      <div className="p-6 border-b-2 border-blue-100 bg-gradient-to-r from-blue-50 via-white to-slate-50">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-100 rounded-lg">
            <i className="fas fa-eye text-orange-600"></i>
          </div>
          <div>
            <h2 className="text-base sm:text-lg font-bold text-blue-900">
              Recent ANPR Detections
            </h2>
            <p className="text-xs text-slate-500 mt-1"><i className="fas fa-barcode text-orange-500 mr-1"></i>Latest plate recognition results</p>
          </div>
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoading ? (
        <div className="p-4 sm:p-6 space-y-3">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="flex items-center gap-3 p-4 bg-slate-50 border border-slate-200 rounded-lg animate-pulse hover:bg-blue-50 transition-colors"
            >
              <div className="w-2.5 h-2.5 bg-slate-300 rounded-full flex-shrink-0"></div>
              <div className="flex-1 min-w-0">
                <div className="h-5 bg-slate-300 rounded w-32 mb-1.5"></div>
                <div className="h-3 bg-slate-200 rounded w-40"></div>
              </div>
              <div className="h-6 bg-slate-300 rounded-full w-20 flex-shrink-0"></div>
            </div>
          ))}
        </div>
      ) : (
        <div className="p-4 sm:p-6">
          <div className="space-y-2">
            {detections.length === 0 ? (
              <div className="text-center py-12 px-4 bg-gradient-to-b from-blue-50 to-white rounded-lg border-2 border-dashed border-slate-200">
                <i className="fas fa-eye text-5xl mb-4 opacity-20 text-slate-400"></i>
                <p className="text-slate-700 font-semibold">No recent detections</p>
                <p className="text-slate-500 text-sm mt-1">Detection results will appear here</p>
              </div>
            ) : (
              detections.map((detection) => (
                <div
                  key={detection.id}
                  className="
                    flex flex-col gap-3 p-3 sm:p-4 
                    bg-gradient-to-r from-slate-50 to-white border-l-4
                    rounded-lg w-full hover:shadow-md transition-all duration-200 cursor-pointer group
                    animate-in duration-300 ease-out slide-in-from-top-2
                  "
                  style={{
                    borderLeftColor: detection.status === 'Registered' ? '#10b981' : '#ef4444'
                  }}
                  data-testid={`detection-${detection.id}`}
                >
                  {/* Top Row: Status Dot + Plate + Timestamp */}
                  <div className="flex items-start gap-2 sm:gap-3 min-w-0">
                    {/* Status Dot */}
                    <div
                      className={`w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full flex-shrink-0 mt-0.5 ${
                        detection.status === 'Registered'
                          ? 'bg-green-500'
                          : 'bg-red-500'
                      }`}
                    ></div>

                    {/* Plate + Timestamp Container */}
                    <div className="flex flex-col gap-1 flex-1 min-w-0">
                      {/* License Plate */}
                      <div className="font-mono text-xs sm:text-sm font-bold text-slate-900 whitespace-nowrap group-hover:text-blue-700 transition-colors truncate">
                        {detection.licensePlate}
                      </div>

                      {/* Timestamp - Show exit time if available (actual event time), otherwise entry time */}
                      <div className="text-xs text-slate-500 flex items-center gap-1 whitespace-nowrap">
                        <i className={`fas ${detection.exitTime ? 'fa-sign-out-alt' : 'fa-clock'} text-slate-400 flex-shrink-0`}></i>
                        <span className="truncate">
                          {(() => {
                            const displayTime = detection.exitTime ? new Date(detection.exitTime) : new Date(detection.detectedAt);
                            return `${displayTime.toLocaleDateString([], { month: 'short', day: 'numeric' })}, ${displayTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
                          })()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Bottom Row: Status & Gate Badges — Full Width on Mobile, Right-Aligned on Desktop */}
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    {/* Left Side Empty Spacer on Mobile to align with content above */}
                    <div className="w-2.5 sm:w-3 flex-shrink-0"></div>
                    
                    {/* Badges Container */}
                    <div className="flex items-center justify-end gap-1.5 sm:gap-2 flex-wrap">
                      {/* Gate Badge (Entry/Exit) */}
                      {detection.gate && (
                        <Badge
                          className={`
                            px-2 py-1 rounded-full text-xs font-semibold
                            border-2 whitespace-nowrap flex-shrink-0 inline-flex items-center gap-1
                            ${
                              detection.gate === "Entry"
                                ? "bg-blue-100 text-blue-700 border-blue-300"
                                : "bg-orange-100 text-orange-700 border-orange-300"
                            }
                          `}
                        >
                          <i className={`fas ${
                            detection.gate === "Entry"
                              ? "fa-arrow-down"
                              : "fa-arrow-up"
                          }`}></i>
                          <span>{detection.gate}</span>
                        </Badge>
                      )}
                      
                      {/* Status Badge */}
                      <Badge
                        className={`
                          px-2 py-1 rounded-full text-xs font-semibold
                          border-2 whitespace-nowrap flex-shrink-0 inline-flex items-center gap-1
                          ${
                            detection.status === "Registered"
                              ? "bg-green-100 text-green-700 border-green-300"
                              : "bg-red-100 text-red-700 border-red-300"
                          }
                        `}
                      >
                        <i className={`fas ${
                          detection.status === "Registered"
                            ? "fa-check-circle"
                            : "fa-exclamation-circle"
                        }`}></i>
                        <span>{detection.status}</span>
                      </Badge>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
