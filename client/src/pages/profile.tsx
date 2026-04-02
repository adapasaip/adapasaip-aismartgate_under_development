import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import SubUserManagement from "@/components/sub-user-management";
import SubUserActivityLog from "@/components/sub-user-activity-log";
import { useSubUsers } from "@/hooks/use-subusers";
import { apiRequest } from "@/lib/queryClient";
import type { Vehicle } from "@shared/schema";

export default function ProfilePage() {
  const { user, updateProfile, deleteAccount, logout } = useAuth();
  const { subUsers } = useSubUsers();
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const { data: vehicles = [] } = useQuery<Vehicle[]>({
    queryKey: ["/api/vehicles", user?.id],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/vehicles?userId=${encodeURIComponent(user?.id || "")}`);
      const data = await response.json();
      return Array.isArray(data) ? data : [];
    },
    enabled: !!user?.id,
  });

  const [isEditing, setIsEditing] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [showPasswordDialog, setShowPasswordDialog] = useState(false);
  const [activeTab, setActiveTab] = useState<"account" | "security" | "subusers">("account");

  const [formData, setFormData] = useState({
    fullName: user?.fullName || "",
    email: user?.email || "",
    mobile: user?.mobile || "",
    avatar: user?.avatar || "",
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: "",
    newPassword: "",
    confirmNewPassword: "",
  });
  const [passwordError, setPasswordError] = useState<string | null>(null);

  const handlePasswordChange = async () => {
    setPasswordError(null);
    if (!passwordData.newPassword || passwordData.newPassword.length < 6) {
      setPasswordError("New password must be at least 6 characters.");
      return;
    }
    if (passwordData.newPassword !== passwordData.confirmNewPassword) {
      setPasswordError("New passwords do not match.");
      return;
    }
    try {
      await fetch("/api/auth/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          currentPassword: passwordData.currentPassword,
          newPassword: passwordData.newPassword,
        }),
      });
      setShowPasswordDialog(false);
      setPasswordData({ currentPassword: "", newPassword: "", confirmNewPassword: "" });
      toast({ title: "Success", description: "Password changed successfully" });
    } catch (error) {
      setPasswordError("Failed to change password. Please check your current password.");
    }
  };

  const handleSave = async () => {
    try {
      await updateProfile(formData);
      setIsEditing(false);
      toast({ title: "Success", description: "Profile updated successfully" });
    } catch (error) {
      toast({ title: "Error", description: "Failed to update profile", variant: "destructive" });
    }
  };

  const handleDelete = async () => {
    try {
      await deleteAccount();
      setShowDeleteDialog(false);
      setLocation("/");
    } catch (error) {
      toast({ title: "Error", description: "Failed to delete account", variant: "destructive" });
    }
  };

  if (!user) {
    return null;
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 bg-white min-h-screen">
      {/* Decorative header background */}
      <div className="fixed inset-0 -z-10 h-80 bg-gradient-to-br from-blue-50 via-purple-25 to-white pointer-events-none"></div>

      {/* Header */}
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-800 mb-2 flex items-center gap-3">
              <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                <i className="fas fa-user-circle text-2xl"></i>
              </div>
              My Profile
            </h1>
            <p className="text-slate-600 text-base font-medium">
              <i className="fas fa-shield-alt text-[#3175F1] mr-2"></i>
              Manage your account, security, and sub-users
            </p>
          </div>
        </div>
      </div>

      {/* Tab Navigation - Premium Design */}
      <div className="mb-8 overflow-x-auto">
        <div className="flex gap-2 sm:gap-3 border-b-2 border-[#3175F1]/20 bg-gradient-to-r from-[#3175F1]/5 to-transparent rounded-t-2xl p-1 sm:p-2 min-w-min">
          {["account", "security", "subusers"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as typeof activeTab)}
              className={`px-3 sm:px-5 py-2 sm:py-2.5 text-sm sm:text-base font-semibold transition-all duration-300 rounded-lg flex items-center gap-1.5 sm:gap-2 whitespace-nowrap ${
                activeTab === tab
                  ? "bg-[#3175F1] text-white shadow-lg"
                  : "text-slate-600 hover:text-[#3175F1] hover:bg-white/50"
              }`}
            >
              {tab === "account" && <i className="fas fa-address-card text-sm sm:text-base"></i>}
              {tab === "security" && <i className="fas fa-lock text-sm sm:text-base"></i>}
              {tab === "subusers" && <i className="fas fa-users text-sm sm:text-base"></i>}
              <span>{tab === "account" ? "Account" : tab === "security" ? "Security" : "Sub-Users"}</span>
            </button>
          ))}
        </div>
      </div>

        {/* Account Tab */}
        {activeTab === "account" && (
          <div className="space-y-6 sm:space-y-8">
            {/* Profile Card */}
            <div>
              <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="border-b-2 border-[#3175F1]/10 pb-4 sm:pb-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div className="flex items-start sm:items-center gap-3">
                      <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                        <i className="fas fa-user text-xl"></i>
                      </div>
                      <div>
                        <CardTitle className="text-lg sm:text-xl text-slate-800">Account Information</CardTitle>
                        <CardDescription className="text-xs sm:text-sm text-slate-600">Update your personal details and profile</CardDescription>
                      </div>
                    </div>
                    {!isEditing && (
                      <Button
                        onClick={() => setIsEditing(true)}
                        className="flex items-center justify-center gap-2 bg-gradient-to-r from-[#3175F1] to-[#3175F1] hover:from-[#2563E0] hover:to-[#2563E0] text-white shadow-lg font-semibold px-4 sm:px-6 text-sm sm:text-base w-full sm:w-auto"
                      >
                        <i className="fas fa-pen text-sm sm:text-base"></i><span className="hidden sm:inline">Edit Profile</span><span className="sm:hidden">Edit</span>
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="pt-6 sm:pt-8">
                  <div className="space-y-6 sm:space-y-8">
                    {/* Avatar Section */}
                    <div className="flex flex-col sm:flex-row items-center gap-6 sm:gap-8 p-4 sm:p-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-2xl border-2 border-[#3175F1]/20">
                      <div className="relative">
                        <div className="w-28 h-28 backdrop-blur-md bg-[#3175F1]/40 border-4 border-white/50 rounded-full flex items-center justify-center text-white text-5xl shadow-lg">
                          {user.avatar ? (
                            <img
                              src={user.avatar}
                              alt="Avatar"
                              className="w-28 h-28 rounded-full object-cover"
                            />
                          ) : (
                            <i className="fas fa-user"></i>
                          )}
                        </div>
                        {isEditing && (
                          <label className="absolute bottom-0 right-0 cursor-pointer">
                            <input
                              type="file"
                              accept="image/*"
                              style={{ display: "none" }}
                              onChange={(e) => {
                                const file = e.target.files?.[0];
                                if (file) {
                                  const reader = new FileReader();
                                  reader.onloadend = () => {
                                    setFormData((prev) => ({ ...prev, avatar: reader.result as string }));
                                  };
                                  reader.readAsDataURL(file);
                                }
                              }}
                            />
                            <div className="backdrop-blur-md bg-[#3175F1]/40 border-2 border-white/50 rounded-full p-3 shadow-lg text-white hover:bg-[#3175F1]/60 transition-all flex items-center justify-center w-10 h-10">
                              <i className="fas fa-camera text-lg"></i>
                            </div>
                          </label>
                        )}
                      </div>
                      <div className="flex-1 text-center sm:text-left">
                        <h4 className="text-xl sm:text-2xl font-bold text-blue-900">{user.fullName}</h4>
                        <p className="text-slate-600 text-sm sm:text-base mb-3">{user.email}</p>
                        <Badge className="bg-[#3175F1]/10 text-[#3175F1] border-2 border-[#3175F1]/30 px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-semibold inline-block">
                          <i className="fas fa-crown mr-2"></i>Administrator
                        </Badge>
                      </div>
                    </div>

                    {/* Form Fields */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
                      <div>
                        <Label htmlFor="fullName" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                          <i className="fas fa-user text-[#3175F1]"></i>Full Name
                        </Label>
                        <Input
                          id="fullName"
                          value={formData.fullName}
                          onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                          disabled={!isEditing}
                          className={`mt-2 sm:mt-3 bg-white border-2 text-sm sm:text-base text-slate-800 placeholder-slate-400 rounded-lg sm:rounded-xl transition-all ${!isEditing ? "bg-slate-50 border-slate-200" : "border-[#3175F1] focus:border-[#3175F1]"}`}
                          placeholder="Enter your full name"
                        />
                      </div>
                      <div>
                        <Label htmlFor="email" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                          <i className="fas fa-envelope text-[#3175F1]"></i>Email
                        </Label>
                        <Input
                          id="email"
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                          disabled={!isEditing}
                          className={`mt-2 sm:mt-3 bg-white border-2 text-sm sm:text-base text-slate-800 placeholder-slate-400 rounded-lg sm:rounded-xl transition-all ${!isEditing ? "bg-slate-50 border-slate-200" : "border-[#3175F1] focus:border-[#3175F1]"}`}
                          placeholder="your@email.com"
                        />
                      </div>
                      <div>
                        <Label htmlFor="mobile" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                          <i className="fas fa-phone text-[#3175F1]"></i>Mobile Number
                        </Label>
                        <Input
                          id="mobile"
                          type="tel"
                          value={formData.mobile}
                          onChange={(e) => setFormData({ ...formData, mobile: e.target.value })}
                          disabled={!isEditing}
                          className={`mt-2 sm:mt-3 bg-white border-2 text-sm sm:text-base text-slate-800 placeholder-slate-400 rounded-lg sm:rounded-xl transition-all ${!isEditing ? "bg-slate-50 border-slate-200" : "border-[#3175F1] focus:border-[#3175F1]"}`}
                          placeholder="Enter your mobile number"
                        />
                      </div>
                      <div>
                        <Label htmlFor="role" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                          <i className="fas fa-shield-alt text-[#3175F1]"></i>Role
                        </Label>
                        <Input
                          id="role"
                          value="Administrator"
                          disabled
                          className="mt-2 sm:mt-3 bg-slate-50 border-2 border-slate-200 text-sm sm:text-base text-slate-800 rounded-lg sm:rounded-xl"
                        />
                      </div>
                    </div>

                    {/* User ID */}
                    <div className="bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-xl sm:rounded-2xl p-4 sm:p-6 border-2 border-[#3175F1]/20">
                      <p className="text-xs text-slate-600 font-semibold mb-2 uppercase tracking-wide flex items-center gap-2">
                        <i className="fas fa-fingerprint text-[#3175F1]"></i>Unique ID
                      </p>
                      <p className="font-mono text-xs sm:text-sm text-[#3175F1] bg-white px-3 sm:px-4 py-2 sm:py-3 rounded-lg border border-[#3175F1]/20 break-all">{user.id}</p>
                    </div>

                    {/* Action Buttons */}
                    {isEditing && (
                      <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 pt-6 border-t-2 border-slate-200">
                        <Button
                          onClick={handleSave}
                        className="bg-[#3175F1] hover:bg-[#2563E0] text-white shadow-lg flex-1 font-semibold text-sm sm:text-base py-2 sm:py-3"
                        >
                          <i className="fas fa-check-circle mr-2"></i><span className="hidden sm:inline">Save Changes</span><span className="sm:hidden">Save</span>
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            setIsEditing(false);
                            setFormData({
                              fullName: user?.fullName || "",
                              email: user?.email || "",
                              mobile: user?.mobile || "",
                              avatar: user?.avatar || "",
                            });
                          }}
                          className="border-2 border-slate-300 text-slate-700 hover:bg-slate-100 flex-1 text-sm sm:text-base py-2 sm:py-3"
                        >
                          <i className="fas fa-times-circle mr-2"></i>Cancel
                        </Button>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Organization Info */}
              <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="border-b-2 border-[#3175F1]/10 pb-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5">
                  <div className="flex items-center gap-3">
                      <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                        <i className="fas fa-building text-xl"></i>
                      </div>
                      <div>
                        <CardTitle className="text-base sm:text-lg text-slate-800">Organization & Status</CardTitle>
                      <CardDescription className="text-xs sm:text-sm text-slate-600">Your organization information and account status</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-6 sm:pt-8">
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
                    <div className="bg-gradient-to-br from-[#3175F1]/10 to-[#3175F1]/5 rounded-xl sm:rounded-2xl p-4 sm:p-6 border-2 border-[#3175F1]/20">
                      <p className="text-xs text-slate-600 font-semibold mb-2 uppercase tracking-wide flex items-center gap-2">
                        <i className="fas fa-calendar text-[#3175F1]"></i>Member Since
                      </p>
                      <p className="text-lg sm:text-2xl font-bold text-[#3175F1]">{new Date(user.createdAt || Date.now()).toLocaleDateString()}</p>
                    </div>
                    <div className="bg-gradient-to-br from-[#3175F1]/10 to-[#3175F1]/5 rounded-xl sm:rounded-2xl p-4 sm:p-6 border-2 border-[#3175F1]/20">
                      <p className="text-xs text-slate-600 font-semibold mb-2 uppercase tracking-wide flex items-center gap-2">
                        <i className="fas fa-users text-[#3175F1]"></i>Sub-Users
                      </p>
                      <p className="text-lg sm:text-2xl font-bold text-[#3175F1]">{subUsers.length}<span className="text-sm sm:text-lg text-[#3175F1] ml-1">/5</span></p>
                    </div>
                    <div className="bg-gradient-to-br from-[#3175F1]/10 to-[#3175F1]/5 rounded-xl sm:rounded-2xl p-4 sm:p-6 border-2 border-[#3175F1]/20">
                      <p className="text-xs text-slate-600 font-semibold mb-2 uppercase tracking-wide flex items-center gap-2">
                        <i className="fas fa-shield-check text-[#3175F1]"></i>Status
                      </p>
                      <Badge className="bg-[#3175F1] text-white text-xs sm:text-sm py-1.5 sm:py-2"><i className="fas fa-check-circle mr-1"></i>Active</Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Security Tab */}
        {activeTab === "security" && (
          <div className="space-y-6 sm:space-y-8">
            {/* Password Section */}
            <div>
              <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="border-b-2 border-[#3175F1]/10 pb-4 sm:pb-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5">
                  <div className="flex items-start sm:items-center gap-3">
                    <div className="p-3 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl text-[#3175F1] shadow-lg flex items-center justify-center w-12 h-12 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                      <i className="fas fa-lock text-xl"></i>
                    </div>
                    <div>
                      <CardTitle className="text-base sm:text-lg text-slate-800">Security & Password</CardTitle>
                      <CardDescription className="text-xs sm:text-sm text-slate-600">Manage your password and security settings</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-6 sm:pt-8">
                  <div className="space-y-4 sm:space-y-6">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 sm:p-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-lg sm:rounded-2xl border-2 border-[#3175F1]/20">
                      <div className="flex items-start sm:items-center gap-3 sm:gap-4">
                        <div className="w-12 sm:w-14 h-12 sm:h-14 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg sm:rounded-xl flex items-center justify-center text-[#3175F1] shadow-lg hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                          <i className="fas fa-key text-lg sm:text-xl"></i>
                        </div>
                        <div>
                          <h4 className="font-bold text-slate-800 text-sm sm:text-base">Password</h4>
                          <p className="text-xs sm:text-sm text-slate-600">Update your account password</p>
                        </div>
                      </div>
                      <Button
                        onClick={() => setShowPasswordDialog(true)}
                        className="bg-[#3175F1] hover:bg-[#2563E0] text-white shadow-lg font-semibold text-sm sm:text-base w-full sm:w-auto py-2 sm:py-2.5"
                      >
                        <i className="fas fa-edit mr-2"></i>Change
                      </Button>
                    </div>

                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 sm:p-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-lg sm:rounded-2xl border-2 border-[#3175F1]/20">
                      <div className="flex items-start sm:items-center gap-3 sm:gap-4">
                        <div className="w-12 sm:w-14 h-12 sm:h-14 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg sm:rounded-xl flex items-center justify-center text-[#3175F1] shadow-lg hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                          <i className="fas fa-shield-alt text-lg sm:text-xl"></i>
                        </div>
                        <div>
                          <h4 className="font-bold text-slate-800 text-sm sm:text-base">Two-Factor Auth</h4>
                          <p className="text-xs sm:text-sm text-slate-600">Add extra security to your account</p>
                        </div>
                      </div>
                      <Badge className="bg-[#3175F1]/20 text-[#3175F1] border-2 border-[#3175F1]/30 px-3 sm:px-4 py-1.5 sm:py-2 font-semibold text-xs sm:text-sm inline-block">
                        <i className="fas fa-star mr-1"></i>Coming Soon
                      </Badge>
                    </div>

                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 sm:p-6 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-lg sm:rounded-2xl border-2 border-[#3175F1]/20">
                      <div className="flex items-start sm:items-center gap-3 sm:gap-4">
                        <div className="w-12 sm:w-14 h-12 sm:h-14 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg sm:rounded-xl flex items-center justify-center text-[#3175F1] shadow-lg hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                          <i className="fas fa-history text-lg sm:text-xl"></i>
                        </div>
                        <div>
                          <h4 className="font-bold text-slate-800 text-sm sm:text-base">Login History</h4>
                          <p className="text-xs sm:text-sm text-slate-600">Monitor your recent activity</p>
                        </div>
                      </div>
                      <Button
                        variant="outline"
                        className="border-2 border-[#3175F1]/30 text-[#3175F1] hover:bg-[#3175F1]/10 text-sm sm:text-base w-full sm:w-auto py-2 sm:py-2.5"
                        disabled
                      >
                        <i className="fas fa-eye mr-2"></i>View
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Danger Zone */}
              <Card className="bg-white border-2 border-red-200/50 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="border-b-2 border-red-200/30 pb-4 sm:pb-6 bg-gradient-to-r from-red-50/50 to-red-50/25">
                  <div className="flex items-start sm:items-center gap-3">
                    <div className="p-3 backdrop-blur-md bg-red-500/40 border border-white/50 rounded-xl text-red-600 shadow-lg flex items-center justify-center w-12 h-12 hover:bg-red-500/70 hover:border-white/80 transition-all flex-shrink-0">
                      <i className="fas fa-exclamation-triangle text-xl"></i>
                    </div>
                    <div>
                      <CardTitle className="text-base sm:text-lg text-red-700">Danger Zone</CardTitle>
                      <CardDescription className="text-xs sm:text-sm text-slate-600">Irreversible actions - proceed with caution</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="pt-6 sm:pt-8">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 sm:p-6 bg-gradient-to-r from-red-50 to-red-50/50 rounded-lg sm:rounded-2xl border-2 border-red-200/50">
                    <div className="flex items-start sm:items-center gap-3 sm:gap-4">
                        <div className="w-12 sm:w-14 h-12 sm:h-14 backdrop-blur-md bg-red-500/40 border border-white/50 rounded-lg sm:rounded-xl flex items-center justify-center text-red-600 shadow-lg hover:bg-red-500/70 hover:border-white/80 transition-all flex-shrink-0">
                        <i className="fas fa-trash-alt text-lg sm:text-xl"></i>
                      </div>
                      <div>
                        <h4 className="font-bold text-red-900 text-sm sm:text-base">Delete Account</h4>
                        <p className="text-xs sm:text-sm text-slate-600">Permanently delete your account and all data</p>
                      </div>
                    </div>
                    <Button
                      onClick={() => setShowDeleteDialog(true)}
                      className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-lg text-sm sm:text-base w-full sm:w-auto py-2 sm:py-2.5 font-semibold"
                    >
                      <i className="fas fa-times-circle mr-2"></i><span className="hidden sm:inline">Delete</span><span className="sm:hidden">Delete Account</span>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Sub-Users Tab */}
        {activeTab === "subusers" && (
          <div className="space-y-6 sm:space-y-8">
            <SubUserManagement parentUserId={user.id} />
            <SubUserActivityLog parentUserId={user.id} subUsers={subUsers} />
          </div>
        )}

      {/* Password Change Dialog */}
      <Dialog open={showPasswordDialog} onOpenChange={setShowPasswordDialog}>
        <DialogContent className="bg-white border-2 border-[#3175F1]/30 w-[95%] sm:w-full rounded-lg sm:rounded-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center text-base sm:text-lg text-slate-800">
              <div className="p-2 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg text-[#3175F1] mr-2 sm:mr-3 flex items-center justify-center w-10 h-10 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all flex-shrink-0">
                <i className="fas fa-lock-open text-base sm:text-lg"></i>
              </div>
              Change Password
            </DialogTitle>
            <DialogDescription className="text-xs sm:text-sm text-slate-600">
              Enter your current password and new password to secure your account.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 sm:space-y-5">
            <div>
              <Label htmlFor="currentPassword" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-key text-[#3175F1]"></i>Current Password
              </Label>
              <Input
                id="currentPassword"
                type="password"
                value={passwordData.currentPassword}
                onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
                placeholder="Enter current password"
                className="mt-2 bg-white border-2 border-[#3175F1]/30 text-sm text-slate-800 placeholder-slate-400 rounded-lg focus:border-[#3175F1]"
              />
            </div>
            <div>
              <Label htmlFor="newPassword" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-lock text-[#3175F1]"></i>New Password
              </Label>
              <Input
                id="newPassword"
                type="password"
                value={passwordData.newPassword}
                onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
                placeholder="Enter new password"
                className="mt-2 bg-white border-2 border-[#3175F1]/30 text-sm text-slate-800 placeholder-slate-400 rounded-lg focus:border-[#3175F1]"
              />
            </div>
            <div>
              <Label htmlFor="confirmNewPassword" className="text-xs sm:text-sm text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-check text-[#3175F1]"></i>Confirm Password
              </Label>
              <Input
                id="confirmNewPassword"
                type="password"
                value={passwordData.confirmNewPassword}
                onChange={(e) => setPasswordData({ ...passwordData, confirmNewPassword: e.target.value })}
                placeholder="Confirm new password"
                className="mt-2 bg-white border-2 border-slate-300 text-sm text-slate-800 placeholder-slate-400 rounded-lg"
              />
            </div>
            {passwordError && (
              <div className="bg-red-50 border-2 border-red-200 rounded-lg p-3 sm:p-4">
                <p className="text-red-700 text-xs sm:text-sm font-medium flex items-center">
                  <i className="fas fa-exclamation-circle mr-2 text-base sm:text-lg"></i>
                  {passwordError}
                </p>
              </div>
            )}
          </div>
          <DialogFooter className="flex-col sm:flex-row gap-3">
            <Button 
              variant="outline" 
              onClick={() => setShowPasswordDialog(false)}
              className="border-2 border-slate-300 text-slate-700 hover:bg-slate-100 text-sm sm:text-base w-full sm:w-auto"
            >
              <i className="fas fa-times mr-2"></i>Cancel
            </Button>
            <Button onClick={handlePasswordChange} className="bg-[#3175F1] hover:bg-[#2563E0] text-white shadow-lg font-semibold text-sm sm:text-base w-full sm:w-auto">
              <i className="fas fa-check-circle mr-2"></i>Update Password
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="bg-white border-2 border-red-200/50 w-[95%] sm:w-full rounded-lg sm:rounded-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center text-base sm:text-lg text-red-700">
              <div className="p-2 backdrop-blur-md bg-red-500/40 border border-white/50 rounded-lg text-red-600 mr-2 sm:mr-3 flex items-center justify-center w-10 h-10 hover:bg-red-500/70 hover:border-white/80 transition-all flex-shrink-0">
                <i className="fas fa-exclamation-triangle text-base sm:text-lg"></i>
              </div>
              Delete Account
            </DialogTitle>
            <DialogDescription className="text-xs sm:text-sm text-slate-600">
              This action is permanent and cannot be undone. All your data will be irreversibly deleted.
            </DialogDescription>
          </DialogHeader>
          <div className="bg-red-50 border-2 border-red-200 rounded-lg p-3 sm:p-5 space-y-2 sm:space-y-3">
            <p className="text-xs sm:text-sm text-red-800 font-bold flex items-center">
              <i className="fas fa-warning text-sm sm:text-lg mr-2"></i>This will permanently delete:
            </p>
            <ul className="text-xs sm:text-sm text-red-700 space-y-1.5 sm:space-y-2 list-none">
              <li className="flex items-center"><i className="fas fa-trash mr-2 text-red-600"></i>Your account and profile</li>
              <li className="flex items-center"><i className="fas fa-trash mr-2 text-red-600"></i>All vehicles ({vehicles?.length || 0})</li>
              <li className="flex items-center"><i className="fas fa-trash mr-2 text-red-600"></i>All cameras</li>
              <li className="flex items-center"><i className="fas fa-trash mr-2 text-red-600"></i>All detections</li>
              <li className="flex items-center"><i className="fas fa-trash mr-2 text-red-600"></i>{subUsers.length} sub-users</li>
            </ul>
          </div>
          <DialogFooter className="flex-col sm:flex-row gap-3">
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
              className="border-2 border-slate-300 text-slate-700 hover:bg-slate-100 text-sm sm:text-base w-full sm:w-auto"
            >
              <i className="fas fa-times mr-2"></i>Cancel
            </Button>
            <Button
              onClick={handleDelete}
              className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white shadow-lg text-sm sm:text-base w-full sm:w-auto font-semibold"
            >
              <i className="fas fa-trash-alt mr-2"></i>Delete Forever
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
