import { useState } from "react";
import { useSubUsers } from "@/hooks/use-subusers";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import type { SubUser, Permissions } from "@shared/schema";

const DEFAULT_PERMISSIONS: Permissions = {
  canAddVehicles: true,
  canDeleteVehicles: false,
  canUpdateVehicles: false,
  canViewVehicles: true,
  canAddCameras: false,
  canDeleteCameras: false,
  canUpdateCameras: false,
  canViewCameras: true,
  canViewDetections: true,
  canManageSubUsers: false,
  canExportData: false,
};

export interface SubUserFormData {
  username: string;
  email: string;
  fullName: string;
  mobile: string;
  password: string;
  permissions: Permissions;
}

interface SubUserManagementProps {
  parentUserId: string;
}

export default function SubUserManagement({ parentUserId }: SubUserManagementProps) {
  const { subUsers, createSubUser, updateSubUser, deleteSubUser, loading } = useSubUsers();
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  const [formData, setFormData] = useState<SubUserFormData>({
    username: "",
    email: "",
    fullName: "",
    mobile: "",
    password: "",
    permissions: DEFAULT_PERMISSIONS,
  });

  const handleCreateClick = () => {
    setEditingId(null);
    setFormData({
      username: "",
      email: "",
      fullName: "",
      mobile: "",
      password: "",
      permissions: DEFAULT_PERMISSIONS,
    });
    setIsDialogOpen(true);
  };

  const handleEditClick = (subUser: any) => {
    setEditingId(subUser.id);
    setFormData({
      username: subUser.username,
      email: subUser.email,
      fullName: subUser.fullName,
      mobile: subUser.mobile,
      password: "",
      permissions: subUser.permissions,
    });
    setIsDialogOpen(true);
  };

  const handleSubmit = async () => {
    // Validation
    if (!formData.username.trim()) {
      toast({ title: "Error", description: "Username is required", variant: "destructive" });
      return;
    }
    if (!formData.email.trim()) {
      toast({ title: "Error", description: "Email is required", variant: "destructive" });
      return;
    }
    if (!editingId && !formData.password) {
      toast({ title: "Error", description: "Password is required for new sub-users", variant: "destructive" });
      return;
    }

    try {
      if (editingId) {
        // Update
        const updateData = {
          username: formData.username,
          email: formData.email,
          fullName: formData.fullName,
          mobile: formData.mobile,
          permissions: formData.permissions,
          isActive: true,
        };
        await updateSubUser(editingId, updateData);
      } else {
        // Create
        await createSubUser({
          parentUserId,
          username: formData.username,
          email: formData.email,
          fullName: formData.fullName,
          mobile: formData.mobile,
          password: formData.password,
          permissions: formData.permissions,
          isActive: true,
        });
      }
      setIsDialogOpen(false);
    } catch (error) {
      console.error("Failed to save sub-user:", error);
    }
  };

  const handleDeleteConfirm = async (id: string) => {
    try {
      await deleteSubUser(id);
      setShowDeleteConfirm(null);
    } catch (error) {
      console.error("Failed to delete sub-user:", error);
    }
  };

  const handlePermissionChange = (key: keyof Permissions, value: boolean) => {
    setFormData((prev) => ({
      ...prev,
      permissions: {
        ...prev.permissions,
        [key]: value,
      },
    }));
  };

  const canAddMore = subUsers.length < 5;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>Sub-Users Management</CardTitle>
            <CardDescription>
              Create and manage up to 5 sub-users with customized permissions
            </CardDescription>
          </div>
          <Button
            onClick={handleCreateClick}
            disabled={!canAddMore}
            className="w-full sm:w-auto"
          >
            <i className="fas fa-plus mr-2"></i>
            Add Sub-User
          </Button>
        </CardHeader>

        <CardContent>
          {loading ? (
            <div className="text-center py-12 text-slate-400">
              <i className="fas fa-spinner fa-spin text-4xl mb-3 block"></i>
              <p className="text-sm font-medium">Loading sub-users...</p>
            </div>
          ) : subUsers.length === 0 ? (
            <div className="text-center py-8 text-slate-500 bg-slate-50 rounded-lg border border-slate-200 p-6">
              <i className="fas fa-users text-4xl mb-3 block opacity-50"></i>
              <p className="font-medium">No sub-users yet. Click "Add Sub-User" to create one.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gradient-to-r from-blue-100 to-slate-100 border-b-2 border-blue-200">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                      <i className="fas fa-user text-blue-600 mr-2"></i>Username
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                      <i className="fas fa-id-card text-blue-600 mr-2"></i>Full Name
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                      <i className="fas fa-envelope text-blue-600 mr-2"></i>Email
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                      <i className="fas fa-check-circle text-blue-600 mr-2"></i>Status
                    </th>
                    <th className="px-6 py-4 text-right text-xs font-bold text-slate-900 uppercase tracking-wider whitespace-nowrap">
                      <i className="fas fa-sliders-h text-blue-600 mr-2"></i>Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-blue-100">
                  {subUsers.map((subUser: any) => (
                    <tr key={subUser.id} className="hover:bg-blue-50 transition-colors">
                      <td className="px-6 py-4 font-medium text-slate-900">{subUser.username}</td>
                      <td className="px-6 py-4 text-slate-700">{subUser.fullName}</td>
                      <td className="px-6 py-4 text-slate-700">{subUser.email}</td>
                      <td className="px-6 py-4">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-800">
                          <i className="fas fa-check-circle mr-1"></i>Active
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right space-x-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleEditClick(subUser)}
                        >
                          <i className="fas fa-edit mr-1"></i>
                          Edit
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => setShowDeleteConfirm(subUser.id)}
                        >
                          <i className="fas fa-trash mr-1"></i>
                          Delete
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {!canAddMore && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md text-sm text-yellow-800">
              <i className="fas fa-info-circle mr-2"></i>
              You have reached the maximum of 5 sub-users per account.
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sub-User Form Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto bg-white border-slate-200">
          <DialogHeader className="border-b border-slate-200 pb-4">
            <div className="space-y-2">
              <DialogTitle className="text-2xl text-slate-800 flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <i className="fas fa-user-plus text-blue-600"></i>
                </div>
                {editingId ? "Edit Sub-User" : "Create New Sub-User"}
              </DialogTitle>
              <DialogDescription className="text-slate-600">
                {editingId
                  ? "Update the sub-user details and manage their permissions"
                  : "Create a new sub-user account with customized permissions to delegate specific tasks"}
              </DialogDescription>
            </div>
          </DialogHeader>

          <div className="space-y-6 py-4">
            {/* Basic Info Section */}
            <div>
              <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <i className="fas fa-user text-blue-600"></i>
                Basic Information
              </h3>
              <div className="grid grid-cols-2 gap-4 bg-slate-50 p-4 rounded-lg border border-slate-200">
                <div>
                  <Label htmlFor="username" className="text-slate-700 font-medium">Username <span className="text-red-500">*</span></Label>
                  <Input
                    id="username"
                    value={formData.username}
                    onChange={(e) =>
                      setFormData({ ...formData, username: e.target.value })
                    }
                    placeholder="e.g., john.doe"
                    className="mt-2 bg-white border-slate-300 text-slate-800"
                  />
                </div>
                <div>
                  <Label htmlFor="email" className="text-slate-700 font-medium">Email <span className="text-red-500">*</span></Label>
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) =>
                      setFormData({ ...formData, email: e.target.value })
                    }
                    placeholder="e.g., john@example.com"
                    className="mt-2 bg-white border-slate-300 text-slate-800"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-4">
                <div>
                  <Label htmlFor="fullName" className="text-slate-700 font-medium">Full Name</Label>
                  <Input
                    id="fullName"
                    value={formData.fullName}
                    onChange={(e) =>
                      setFormData({ ...formData, fullName: e.target.value })
                    }
                    placeholder="e.g., John Doe"
                    className="mt-2 bg-white border-slate-300 text-slate-800"
                  />
                </div>
                <div>
                  <Label htmlFor="mobile" className="text-slate-700 font-medium">Mobile Number</Label>
                  <Input
                    id="mobile"
                    value={formData.mobile}
                    onChange={(e) =>
                      setFormData({ ...formData, mobile: e.target.value })
                    }
                    placeholder="e.g., +1-234-567-8900"
                    className="mt-2 bg-white border-slate-300 text-slate-800"
                  />
                </div>
              </div>

              {!editingId && (
                <div className="mt-4">
                  <Label htmlFor="password" className="text-slate-700 font-medium">Password <span className="text-red-500">*</span></Label>
                  <Input
                    id="password"
                    type="password"
                    value={formData.password}
                    onChange={(e) =>
                      setFormData({ ...formData, password: e.target.value })
                    }
                    placeholder="Enter a secure password"
                    className="mt-2 bg-white border-slate-300 text-slate-800"
                  />
                  <p className="text-xs text-slate-500 mt-1">Minimum 6 characters recommended</p>
                </div>
              )}
            </div>

            {/* Permissions Section */}
            <div>
              <h3 className="text-lg font-semibold text-slate-800 mb-4 flex items-center gap-2">
                <i className="fas fa-lock text-blue-600"></i>
                Permissions & Access
              </h3>
              <div className="space-y-4">
                {/* Vehicle Permissions */}
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
                  <h5 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                    <i className="fas fa-car text-blue-600 w-4"></i>Vehicle Management
                  </h5>
                  <div className="space-y-3 ml-6">
                    <div className="flex items-center">
                      <Checkbox
                        id="canViewVehicles"
                        checked={formData.permissions.canViewVehicles}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canViewVehicles",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canViewVehicles" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">View Vehicles</span>
                        <span className="text-xs text-slate-500">Access vehicle list and details</span>
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Checkbox
                        id="canAddVehicles"
                        checked={formData.permissions.canAddVehicles}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canAddVehicles",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canAddVehicles" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">Add Vehicles</span>
                        <span className="text-xs text-slate-500">Create manual and ANPR entries</span>
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Checkbox
                        id="canUpdateVehicles"
                        checked={formData.permissions.canUpdateVehicles}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canUpdateVehicles",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canUpdateVehicles" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">Update Vehicles</span>
                        <span className="text-xs text-slate-500">Modify vehicle information</span>
                      </label>
                    </div>
                    <div className="flex items-center opacity-60">
                      <Checkbox
                        id="canDeleteVehicles"
                        checked={formData.permissions.canDeleteVehicles}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canDeleteVehicles",
                            checked === true
                          )
                        }
                        disabled
                        className="border-slate-300"
                      />
                      <label htmlFor="canDeleteVehicles" className="ml-3 text-sm cursor-not-allowed font-medium text-slate-600">
                        <span className="block">Delete Vehicles</span>
                        <span className="text-xs text-slate-500">Admin only</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Camera Permissions */}
                <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg border border-green-200">
                  <h5 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                    <i className="fas fa-camera text-green-600 w-4"></i>Camera Management
                  </h5>
                  <div className="space-y-3 ml-6">
                    <div className="flex items-center">
                      <Checkbox
                        id="canViewCameras"
                        checked={formData.permissions.canViewCameras}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canViewCameras",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canViewCameras" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">View Cameras</span>
                        <span className="text-xs text-slate-500">Access camera feeds and settings</span>
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Checkbox
                        id="canAddCameras"
                        checked={formData.permissions.canAddCameras}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canAddCameras",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canAddCameras" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">Create Cameras</span>
                        <span className="text-xs text-slate-500">Set up new camera connections</span>
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Checkbox
                        id="canUpdateCameras"
                        checked={formData.permissions.canUpdateCameras}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canUpdateCameras",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canUpdateCameras" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">Update Cameras</span>
                        <span className="text-xs text-slate-500">Modify camera configuration</span>
                      </label>
                    </div>
                    <div className="flex items-center opacity-60">
                      <Checkbox
                        id="canDeleteCameras"
                        checked={formData.permissions.canDeleteCameras}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canDeleteCameras",
                            checked === true
                          )
                        }
                        disabled
                        className="border-slate-300"
                      />
                      <label htmlFor="canDeleteCameras" className="ml-3 text-sm cursor-not-allowed font-medium text-slate-600">
                        <span className="block">Delete Cameras</span>
                        <span className="text-xs text-slate-500">Admin only</span>
                      </label>
                    </div>
                  </div>
                </div>

                {/* Detection & Other Permissions */}
                <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg border border-purple-200">
                  <h5 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2">
                    <i className="fas fa-tasks text-purple-600 w-4"></i>Additional Permissions
                  </h5>
                  <div className="space-y-3 ml-6">
                    <div className="flex items-center">
                      <Checkbox
                        id="canViewDetections"
                        checked={formData.permissions.canViewDetections}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canViewDetections",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canViewDetections" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">View ANPR Detections</span>
                        <span className="text-xs text-slate-500">Access detection history and data</span>
                      </label>
                    </div>
                    <div className="flex items-center">
                      <Checkbox
                        id="canExportData"
                        checked={formData.permissions.canExportData}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canExportData",
                            checked === true
                          )
                        }
                        className="border-slate-300"
                      />
                      <label htmlFor="canExportData" className="ml-3 text-sm cursor-pointer font-medium text-slate-700">
                        <span className="block">Export Data</span>
                        <span className="text-xs text-slate-500">Export detections and reports</span>
                      </label>
                    </div>
                    <div className="flex items-center opacity-60">
                      <Checkbox
                        id="canManageSubUsers"
                        checked={formData.permissions.canManageSubUsers}
                        onCheckedChange={(checked) =>
                          handlePermissionChange(
                            "canManageSubUsers",
                            checked === true
                          )
                        }
                        disabled
                        className="border-slate-300"
                      />
                      <label htmlFor="canManageSubUsers" className="ml-3 text-sm cursor-not-allowed font-medium text-slate-600">
                        <span className="block">Manage Sub-Users</span>
                        <span className="text-xs text-slate-500">Admin only</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <DialogFooter className="border-t border-slate-200 pt-4 gap-3">
            <Button 
              variant="outline" 
              onClick={() => setIsDialogOpen(false)}
              className="border-slate-300 text-slate-700 hover:bg-slate-100"
            >
              <i className="fas fa-times mr-2"></i>
              Cancel
            </Button>
            <Button 
              onClick={handleSubmit}
              className="bg-blue-600 hover:bg-blue-700 text-white"
            >
              <i className={`fas fa-${editingId ? 'save' : 'plus'} mr-2`}></i>
              {editingId ? "Update" : "Create"} Sub-User
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={!!showDeleteConfirm} onOpenChange={() => setShowDeleteConfirm(null)}>
        <DialogContent className="bg-white border-red-200">
          <DialogHeader>
            <DialogTitle className="text-red-600 flex items-center gap-2">
              <i className="fas fa-exclamation-triangle"></i>
              Delete Sub-User?
            </DialogTitle>
            <DialogDescription className="text-slate-600">
              This action cannot be undone. The sub-user account and all associated data will be permanently deleted.
            </DialogDescription>
          </DialogHeader>
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 my-4">
            <p className="text-sm text-red-700 font-medium">
              <i className="fas fa-info-circle mr-2"></i>
              Once deleted, this sub-user will lose access to all cameras, vehicles, and detection data.
            </p>
          </div>
          <DialogFooter>
            <Button 
              variant="outline" 
              onClick={() => setShowDeleteConfirm(null)}
              className="border-slate-300 text-slate-700 hover:bg-slate-100"
            >
              <i className="fas fa-times mr-2"></i>
              Cancel
            </Button>
            <Button
              onClick={() => showDeleteConfirm && handleDeleteConfirm(showDeleteConfirm)}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              <i className="fas fa-trash mr-2"></i>
              Delete Sub-User
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
