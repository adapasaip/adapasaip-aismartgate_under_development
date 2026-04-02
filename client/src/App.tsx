import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AuthProvider } from "./hooks/use-auth";
import LoginPage from "./pages/login";
import Dashboard from "./pages/dashboard";
import Vehicles from "./pages/vehicles";
import Cameras from "./pages/cameras";
import ProfilePage from "./pages/profile";
import SubUserProfilePage from "./pages/sub-user-profile";
import Sidebar from "./components/sidebar";
import { useAuth } from "./hooks/use-auth";
import { useState } from "react";

function AppRouter() {
  const { isAuthenticated, isSubUserAuthenticated, user, subUser } = useAuth();
  const [collapsed, setCollapsed] = useState(false);

  // Check if either main user or sub-user is authenticated
  const isLoggedIn = isAuthenticated || isSubUserAuthenticated;

  if (!isLoggedIn) return <LoginPage />;

  // Determine which profile component to show based on user type
  const ProfileComponent = user ? ProfilePage : SubUserProfilePage;

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50 overflow-hidden">
      {/* Sidebar */}
      <Sidebar collapsed={collapsed} setCollapsed={setCollapsed} />

      {/* Main Content */}
      <main
        className={`
          flex-1 overflow-auto transition-all duration-300 
          pt-16 lg:pt-0
          ${collapsed ? "lg:ml-20" : "lg:ml-64"}
        `}
      >
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/dashboard" component={Dashboard} />
          <Route path="/vehicles" component={Vehicles} />
          <Route path="/cameras" component={Cameras} />
          <Route path="/profile" component={ProfileComponent} />
        </Switch>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AuthProvider>
          <AppRouter />
          <Toaster />
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
