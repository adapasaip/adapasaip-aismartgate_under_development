import { useState, useEffect } from "react";
import { useSubUsers } from "@/hooks/use-subusers";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ActivityLog } from "@shared/schema";

interface SubUserActivityLogProps {
  parentUserId: string;
  subUsers: any[];
}

const ACTION_ICONS: Record<string, string> = {
  LOGIN: "fas fa-sign-in-alt text-blue-600",
  ADD_VEHICLE: "fas fa-plus-circle text-green-600",
  UPDATE_VEHICLE: "fas fa-edit text-yellow-600",
  DELETE_VEHICLE: "fas fa-trash text-red-600",
  VIEW_VEHICLE: "fas fa-eye text-gray-600",
  ADD_CAMERA: "fas fa-camera-plus text-green-600",
  UPDATE_CAMERA: "fas fa-edit text-yellow-600",
  DELETE_CAMERA: "fas fa-trash text-red-600",
  EXPORT_DATA: "fas fa-download text-purple-600",
  UPDATE_PROFILE: "fas fa-user-edit text-blue-600",
  LOGOUT: "fas fa-sign-out-alt text-red-600",
};

const ACTION_LABELS: Record<string, string> = {
  LOGIN: "Login",
  ADD_VEHICLE: "Added Vehicle",
  UPDATE_VEHICLE: "Updated Vehicle",
  DELETE_VEHICLE: "Deleted Vehicle",
  VIEW_VEHICLE: "Viewed Vehicle",
  ADD_CAMERA: "Added Camera",
  UPDATE_CAMERA: "Updated Camera",
  DELETE_CAMERA: "Deleted Camera",
  EXPORT_DATA: "Exported Data",
  UPDATE_PROFILE: "Updated Profile",
  LOGOUT: "Logout",
};

export default function SubUserActivityLog({
  parentUserId,
  subUsers,
}: SubUserActivityLogProps) {
  const { recentActivityLogs, fetchActivityLogs, loading } = useSubUsers();
  const [selectedSubUser, setSelectedSubUser] = useState<string>("all");
  const [filteredLogs, setFilteredLogs] = useState<ActivityLog[]>([]);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    // Filter logs based on selected sub-user and search term
    let filtered = recentActivityLogs;

    if (selectedSubUser !== "all") {
      filtered = filtered.filter((log) => log.subUserId === selectedSubUser);
    }

    if (searchTerm) {
      filtered = filtered.filter(
        (log) =>
          ACTION_LABELS[log.action]
            .toLowerCase()
            .includes(searchTerm.toLowerCase()) ||
          (log.details && JSON.stringify(log.details).toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    setFilteredLogs(filtered);
  }, [recentActivityLogs, selectedSubUser, searchTerm]);

  const handleRefresh = async () => {
    await fetchActivityLogs(selectedSubUser !== "all" ? selectedSubUser : undefined);
  };

  const getSubUserName = (subUserId: string) => {
    const subUser = subUsers.find((su) => su.id === subUserId);
    return subUser ? subUser.username : "Admin";
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString();
  };

  return (
    <Card className="mt-6">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Activity Log</CardTitle>
          <CardDescription>Track all sub-user activities and actions</CardDescription>
        </div>
        <Button onClick={handleRefresh} variant="outline" size="sm">
          <i className="fas fa-sync-alt mr-2"></i>
          Refresh
        </Button>
      </CardHeader>

      <CardContent>
        {/* Filters */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="text-sm font-medium text-slate-700 mb-2 block">
              Filter by Sub-User
            </label>
            <Select value={selectedSubUser} onValueChange={setSelectedSubUser}>
              <SelectTrigger>
                <SelectValue placeholder="Select sub-user" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Activities</SelectItem>
                {subUsers.map((subUser) => (
                  <SelectItem key={subUser.id} value={subUser.id}>
                    {subUser.username}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <label className="text-sm font-medium text-slate-700 mb-2 block">
              Search Activities
            </label>
            <Input
              placeholder="Search by action..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Activity Table */}
        {loading ? (
          <div className="text-center py-12 text-slate-400">
            <i className="fas fa-spinner fa-spin text-4xl mb-3 block"></i>
            <p className="text-sm font-medium">Loading activity logs...</p>
          </div>
        ) : filteredLogs.length === 0 ? (
          <div className="text-center py-12 text-slate-500 bg-slate-50 rounded-lg border border-slate-200 p-6">
            <i className="fas fa-inbox text-4xl mb-3 block opacity-50"></i>
            <p className="font-medium">No activities found</p>
            <p className="text-xs text-slate-500 mt-2">Activities will appear here when sub-users take action</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gradient-to-r from-blue-100 to-slate-100 border-b-2 border-blue-200">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                    <i className="fas fa-clock text-blue-600 mr-2"></i>Time
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                    <i className="fas fa-user-tie text-blue-600 mr-2"></i>Sub-User
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                    <i className="fas fa-tasks text-blue-600 mr-2"></i>Action
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                    <i className="fas fa-flag text-blue-600 mr-2"></i>Status
                  </th>
                  <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                    <i className="fas fa-info-circle text-blue-600 mr-2"></i>Details
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-blue-100">
                {filteredLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-blue-50 transition-colors">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm text-slate-600 font-medium">
                        {formatTime(log.timestamp)}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-semibold text-slate-900">
                      <div className="flex items-center">
                        <div className="w-2 h-2 rounded-full bg-blue-600 mr-2"></div>
                        {getSubUserName(log.subUserId)}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-700">
                      <div className="flex items-center">
                        <i
                          className={`${ACTION_ICONS[log.action] || 'fas fa-circle text-gray-400'} mr-2 text-blue-600`}
                        ></i>
                        <span>{ACTION_LABELS[log.action] || log.action}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold ${
                          log.status === "SUCCESS"
                            ? "bg-green-100 text-green-800"
                            : "bg-red-100 text-red-800"
                        }`}
                      >
                        {log.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-700">
                      {log.details ? (
                        <span className="text-sm max-w-xs truncate block font-medium text-slate-600">
                          {JSON.stringify(log.details)}
                        </span>
                      ) : (
                        <span className="text-slate-400">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Summary Statistics */}
        {filteredLogs.length > 0 && (
          <div className="mt-8 pt-6 border-t-2 border-blue-100 grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-slate-50 rounded-lg border border-blue-100">
              <div className="text-3xl font-bold text-blue-600">
                {filteredLogs.filter((l) => l.action === "LOGIN").length}
              </div>
              <div className="text-xs text-slate-600 mt-2 font-semibold uppercase tracking-wider">Logins</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-slate-50 rounded-lg border border-green-100">
              <div className="text-3xl font-bold text-green-600">
                {filteredLogs.filter((l) => l.action.startsWith("ADD_")).length}
              </div>
              <div className="text-xs text-slate-600 mt-2 font-semibold uppercase tracking-wider">Items Added</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-yellow-50 to-slate-50 rounded-lg border border-yellow-100">
              <div className="text-3xl font-bold text-yellow-600">
                {filteredLogs.filter((l) => l.action.startsWith("UPDATE_")).length}
              </div>
              <div className="text-xs text-slate-600 mt-2 font-semibold uppercase tracking-wider">Items Updated</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-slate-50 to-blue-50 rounded-lg border border-slate-200">
              <div className="text-3xl font-bold text-slate-700">
                {filteredLogs.length}
              </div>
              <div className="text-xs text-slate-600 mt-2 font-semibold uppercase tracking-wider">Total Events</div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
