import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function SubUserProfilePage() {
  const { subUser, logout } = useAuth();
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const [isEditing, setIsEditing] = useState(false);
  const [showPasswordDialog, setShowPasswordDialog] = useState(false);
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);

  const [formData, setFormData] = useState({
    fullName: subUser?.fullName || "",
    email: subUser?.email || "",
    mobile: subUser?.mobile || "",
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
      const response = await fetch("/api/subusers/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          currentPassword: passwordData.currentPassword,
          newPassword: passwordData.newPassword,
        }),
      });
      
      if (!response.ok) {
        throw new Error("Failed to change password");
      }
      
      setShowPasswordDialog(false);
      setPasswordData({ currentPassword: "", newPassword: "", confirmNewPassword: "" });
      toast({ title: "Success", description: "Password changed successfully" });
    } catch (error) {
      setPasswordError("Failed to change password. Please check your current password.");
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch(`/api/subusers/${subUser?.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to update profile");
      }

      setIsEditing(false);
      toast({ title: "Success", description: "Profile updated successfully" });
    } catch (error) {
      toast({ title: "Error", description: "Failed to update profile", variant: "destructive" });
    }
  };

  const handleLogout = () => {
    logout();
    setLocation("/login");
  };

  if (!subUser) {
    return null;
  }

  const getPermissionIcon = (allowed: boolean) => {
    return allowed ? (
      <i className="fas fa-check-circle text-green-500"></i>
    ) : (
      <i className="fas fa-times-circle text-red-500"></i>
    );
  };

  const permissionGroups = {
    "Vehicle Management": [
      { key: "canViewVehicles", label: "View Vehicles" },
      { key: "canAddVehicles", label: "Add Vehicles" },
      { key: "canUpdateVehicles", label: "Update Vehicles" },
      { key: "canDeleteVehicles", label: "Delete Vehicles" },
    ],
    "Camera Management": [
      { key: "canViewCameras", label: "View Cameras" },
      { key: "canAddCameras", label: "Add Cameras" },
      { key: "canUpdateCameras", label: "Update Cameras" },
      { key: "canDeleteCameras", label: "Delete Cameras" },
    ],
    "System Access": [
      { key: "canViewDetections", label: "View Detections" },
      { key: "canManageSubUsers", label: "Manage Sub-Users" },
      { key: "canExportData", label: "Export Data" },
    ],
  };

  return (
    <div className="p-4 sm:p-6 lg:p-8 bg-white min-h-screen">
      {/* Decorative header background */}
      <div className="fixed inset-0 -z-10 h-80 bg-gradient-to-br from-[#3175F1]/10 via-purple-50/20 to-white pointer-events-none"></div>

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
              Manage your account and permissions
            </p>
          </div>
        </div>
      </div>

      <Tabs defaultValue="account" className="space-y-8">
        <TabsList className="grid w-full grid-cols-3 bg-gradient-to-r from-[#3175F1]/5 to-transparent border-2 border-[#3175F1]/20 rounded-2xl p-2">
          <TabsTrigger value="account" className="gap-2 font-bold text-base data-[state=active]:bg-[#3175F1] data-[state=active]:text-white data-[state=active]:shadow-lg rounded-xl transition-all">
            <i className="fas fa-address-card text-lg"></i>
            Account
          </TabsTrigger>
          <TabsTrigger value="permissions" className="gap-2 font-bold text-base data-[state=active]:bg-[#3175F1] data-[state=active]:text-white data-[state=active]:shadow-lg rounded-xl transition-all">
            <i className="fas fa-user-shield text-lg"></i>
            Permissions
          </TabsTrigger>
          <TabsTrigger value="security" className="gap-2 font-bold text-base data-[state=active]:bg-[#3175F1] data-[state=active]:text-white data-[state=active]:shadow-lg rounded-xl transition-all">
            <i className="fas fa-lock text-lg"></i>
            Security
          </TabsTrigger>
        </TabsList>

        {/* Account Tab */}
        <TabsContent value="account">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Profile Card */}
            <div className="lg:col-span-2">
              <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 border-b-2 border-[#3175F1]/10">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-4">
                      <div className="w-20 h-20 rounded-full backdrop-blur-md bg-[#3175F1]/40 border-4 border-white/50 flex items-center justify-center flex-shrink-0 text-[#3175F1] text-3xl shadow-lg">
                        <i className="fas fa-user"></i>
                      </div>
                      <div>
                        <CardTitle className="text-xl text-slate-800">{subUser.fullName}</CardTitle>
                        <CardDescription className="text-slate-600 mt-1">
                          <Badge className="bg-[#3175F1]/10 text-[#3175F1] border-2 border-[#3175F1]/30">
                            <i className="fas fa-shield-alt mr-1"></i>
                            Sub-User Account
                          </Badge>
                        </CardDescription>
                      </div>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="pt-6 space-y-6">
                  {isEditing ? (
                    <div className="space-y-5">
                      <div>
                        <Label htmlFor="fullName" className="text-slate-700 font-semibold text-base flex items-center gap-2">
                          <i className="fas fa-user text-[#3175F1]"></i>Full Name
                        </Label>
                        <Input
                          id="fullName"
                          value={formData.fullName}
                          onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                          className="mt-3 bg-white border-2 border-[#3175F1] text-slate-800 placeholder-slate-400 rounded-xl"
                        />
                      </div>
                      <div>
                        <Label htmlFor="email" className="text-slate-700 font-semibold text-base flex items-center gap-2">
                          <i className="fas fa-envelope text-[#3175F1]"></i>Email Address
                        </Label>
                        <Input
                          id="email"
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                          className="mt-3 bg-white border-2 border-[#3175F1] text-slate-800 placeholder-slate-400 rounded-xl"
                        />
                      </div>
                      <div>
                        <Label htmlFor="mobile" className="text-slate-700 font-semibold text-base flex items-center gap-2">
                          <i className="fas fa-phone text-[#3175F1]"></i>Mobile Number
                        </Label>
                        <Input
                          id="mobile"
                          value={formData.mobile}
                          onChange={(e) => setFormData({ ...formData, mobile: e.target.value })}
                          className="mt-3 bg-white border-2 border-[#3175F1] text-slate-800 placeholder-slate-400 rounded-xl"
                        />
                      </div>
                      <div className="flex gap-3 pt-4">
                        <Button onClick={handleSave} className="bg-[#3175F1] hover:bg-[#2563E0] text-white gap-2 flex-1 font-semibold shadow-lg">
                          <i className="fas fa-save"></i>
                          Save Changes
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            setIsEditing(false);
                            setFormData({
                              fullName: subUser.fullName,
                              email: subUser.email,
                              mobile: subUser.mobile || "",
                            });
                          }}
                          className="border-2 border-[#3175F1]/30 text-[#3175F1] hover:bg-[#3175F1]/10"
                        >
                          <i className="fas fa-times mr-1"></i>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="text-sm font-medium text-slate-600">Full Name</h4>
                          <p className="text-lg text-slate-800 mt-1">{subUser.fullName}</p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setIsEditing(true)}
                          className="gap-2 border-2 border-[#3175F1]/30 text-[#3175F1] hover:bg-[#3175F1]/10"
                        >
                          <i className="fas fa-edit"></i>
                          Edit
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-4 pt-4 border-t-2 border-[#3175F1]/10">
                        <div>
                          <h4 className="text-sm font-medium text-slate-600">Email</h4>
                          <p className="text-slate-800 mt-1 break-all">{subUser.email}</p>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-slate-600">Mobile</h4>
                          <p className="text-slate-800 mt-1">{subUser.mobile || "Not provided"}</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 pt-4 border-t-2 border-[#3175F1]/10">
                        <div>
                          <h4 className="text-sm font-medium text-slate-600">Account Status</h4>
                          <Badge className="mt-2 bg-[#3175F1] text-white">
                            <i className="fas fa-check-circle mr-1"></i>
                            Active
                          </Badge>
                        </div>
                        <div>
                          <h4 className="text-sm font-medium text-slate-600">Member Since</h4>
                          <p className="text-slate-800 mt-1">
                            {new Date(subUser.createdAt).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Info Card */}
            <div>
              <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 border-b-2 border-[#3175F1]/10">
                  <div className="flex items-center gap-3">
                    <div className="p-2 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg text-[#3175F1] shadow-lg flex items-center justify-center w-10 h-10 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                      <i className="fas fa-info-circle text-base"></i>
                    </div>
                    <CardTitle className="text-lg text-slate-800">
                      Account Info
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="pt-6 space-y-4">
                  <div className="bg-[#3175F1]/10 p-4 rounded-lg border-2 border-[#3175F1]/20">
                    <h4 className="text-xs font-medium text-[#3175F1] uppercase tracking-wider">User ID</h4>
                    <p className="text-sm text-slate-800 font-mono mt-2 break-all">{subUser.id}</p>
                  </div>
                  <div className="bg-slate-50 p-4 rounded-lg border-2 border-slate-200">
                    <h4 className="text-xs font-medium text-slate-700 uppercase tracking-wider">Last Login</h4>
                    <p className="text-slate-800 mt-2">
                      {subUser.lastLogin
                        ? new Date(subUser.lastLogin).toLocaleString()
                        : "First login"}
                    </p>
                  </div>
                  <div className="bg-[#3175F1]/10 border-2 border-[#3175F1]/30 p-4 rounded-lg">
                    <p className="text-sm text-[#3175F1] font-medium">
                      <i className="fas fa-lightbulb mr-2"></i>
                      Your account is managed by an administrator.
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        {/* Permissions Tab */}
        <TabsContent value="permissions">
          <div className="space-y-6">
            {Object.entries(permissionGroups).map(([group, perms]) => (
              <Card key={group} className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
                <CardHeader className="border-b-2 border-[#3175F1]/10 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5">
                  <CardTitle className="text-lg text-slate-800 flex items-center gap-2">
                    <div className="p-2 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg text-[#3175F1] shadow-lg flex items-center justify-center w-10 h-10 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                      {group === "Vehicle Management" && <i className="fas fa-car text-base"></i>}
                      {group === "Camera Management" && <i className="fas fa-video text-base"></i>}
                      {group === "System Access" && <i className="fas fa-lock text-base"></i>}
                    </div>
                    {group}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {perms.map((perm) => (
                      <div key={perm.key} className="flex items-center justify-between p-4 bg-gradient-to-br from-[#3175F1]/10 to-[#3175F1]/5 rounded-lg border-2 border-[#3175F1]/20">
                        <span className="text-slate-800 font-semibold flex items-center gap-2">
                          {(subUser.permissions as any)[perm.key] ? (
                            <i className="fas fa-check-circle text-[#3175F1] text-lg"></i>
                          ) : (
                            <i className="fas fa-times-circle text-red-500 text-lg"></i>
                          )}
                          {perm.label}
                        </span>
                        <span className="text-sm font-semibold">
                          {getPermissionIcon((subUser.permissions as any)[perm.key])}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security">
          <Card className="bg-white border-2 border-[#3175F1]/20 shadow-lg hover:shadow-xl transition-all">
            <CardHeader className="border-b-2 border-[#3175F1]/10 bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5">
              <CardTitle className="text-lg text-slate-800 flex items-center gap-2">
                <div className="p-2 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg text-[#3175F1] shadow-lg flex items-center justify-center w-10 h-10 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                  <i className="fas fa-shield-alt text-base"></i>
                </div>
                Security Settings
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6 space-y-6">
              <div className="bg-gradient-to-r from-[#3175F1]/10 to-[#3175F1]/5 rounded-2xl p-6 border-2 border-[#3175F1]/20">
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-xl flex items-center justify-center text-[#3175F1] shadow-lg hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                    <i className="fas fa-key text-lg"></i>
                  </div>
                  <div className="flex-1">
                    <h4 className="font-bold text-slate-800 text-base">Password</h4>
                    <p className="text-sm text-slate-600">Update your account password</p>
                  </div>
                  <Button
                    onClick={() => setShowPasswordDialog(true)}
                    className="bg-[#3175F1] hover:bg-[#2563E0] text-white shadow-lg font-semibold"
                  >
                    <i className="fas fa-edit mr-2"></i>Change
                  </Button>
                </div>
              </div>

              <div className="border-t-2 border-[#3175F1]/10 pt-6">
                <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center gap-2">
                  <div className="w-5 h-5 flex items-center justify-center">
                    <i className="fas fa-sign-out-alt text-red-500 text-sm"></i>
                  </div>
                  Logout
                </h4>
                <Button
                  onClick={() => setShowLogoutDialog(true)}
                  className="bg-red-600 hover:bg-red-700 text-white w-full gap-2 shadow-lg font-semibold"
                >
                  <i className="fas fa-sign-out-alt"></i>
                  Logout from This Session
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Password Change Dialog */}
      <Dialog open={showPasswordDialog} onOpenChange={setShowPasswordDialog}>
        <DialogContent className="bg-white border-2 border-[#3175F1]/30">
          <DialogHeader>
            <DialogTitle className="text-lg text-slate-800 flex items-center">
              <div className="p-2 backdrop-blur-md bg-[#3175F1]/30 border border-white/50 rounded-lg text-[#3175F1] mr-3 flex items-center justify-center w-10 h-10 hover:bg-[#3175F1]/60 hover:border-white/80 transition-all">
                <i className="fas fa-lock-open text-lg"></i>
              </div>
              Change Password
            </DialogTitle>
            <DialogDescription className="text-slate-600">
              Enter your current password and set a new password
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label htmlFor="currentPassword" className="text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-key text-[#3175F1]"></i>Current Password
              </Label>
              <Input
                id="currentPassword"
                type="password"
                value={passwordData.currentPassword}
                onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
                className="mt-2 bg-white border-2 border-[#3175F1]/30 text-slate-800 placeholder-slate-400 rounded-lg focus:border-[#3175F1]"
              />
            </div>
            <div>
              <Label htmlFor="newPassword" className="text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-lock text-[#3175F1]"></i>New Password
              </Label>
              <Input
                id="newPassword"
                type="password"
                value={passwordData.newPassword}
                onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
                className="mt-2 bg-white border-2 border-[#3175F1]/30 text-slate-800 placeholder-slate-400 rounded-lg focus:border-[#3175F1]"
              />
            </div>
            <div>
              <Label htmlFor="confirmPassword" className="text-slate-700 font-semibold flex items-center gap-2">
                <i className="fas fa-check text-[#3175F1]"></i>Confirm Password
              </Label>
              <Input
                id="confirmPassword"
                type="password"
                value={passwordData.confirmNewPassword}
                onChange={(e) => setPasswordData({ ...passwordData, confirmNewPassword: e.target.value })}
                className="mt-2 bg-white border-2 border-[#3175F1]/30 text-slate-800 placeholder-slate-400 rounded-lg focus:border-[#3175F1]"
              />
            </div>
            {passwordError && (
              <div className="p-3 bg-red-50 border-2 border-red-200 rounded text-red-700 text-sm font-medium">
                <i className="fas fa-exclamation-circle mr-2"></i>
                {passwordError}
              </div>
            )}
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowPasswordDialog(false);
                setPasswordError(null);
                setPasswordData({ currentPassword: "", newPassword: "", confirmNewPassword: "" });
              }}
              className="border-2 border-[#3175F1]/30 text-[#3175F1] hover:bg-[#3175F1]/10"
            >
              Cancel
            </Button>
            <Button onClick={handlePasswordChange} className="bg-[#3175F1] hover:bg-[#2563E0] text-white shadow-lg font-semibold">
              <i className="fas fa-check mr-2"></i>
              Update Password
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Logout Dialog */}
      <Dialog open={showLogoutDialog} onOpenChange={setShowLogoutDialog}>
        <DialogContent className="bg-white border-2 border-red-200/50">
          <DialogHeader>
            <DialogTitle className="text-lg text-slate-800 flex items-center">
              <div className="p-2 backdrop-blur-md bg-red-500/40 border border-white/50 rounded-lg text-red-600 mr-3 flex items-center justify-center w-10 h-10">
                <i className="fas fa-sign-out-alt text-lg"></i>
              </div>
              Confirm Logout
            </DialogTitle>
            <DialogDescription className="text-slate-600">
              Are you sure you want to logout? You'll need to login again to access your account.
            </DialogDescription>
          </DialogHeader>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowLogoutDialog(false)}
              className="border-2 border-slate-300 text-slate-700 hover:bg-slate-100"
            >
              Cancel
            </Button>
            <Button onClick={handleLogout} className="bg-red-600 hover:bg-red-700 text-white shadow-lg font-semibold">
              <i className="fas fa-check mr-2"></i>
              Yes, Logout
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
