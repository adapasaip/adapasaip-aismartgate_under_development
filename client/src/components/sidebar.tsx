import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { Button } from "@/components/ui/button";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: "fas fa-gauge-high" },
  { name: "Vehicles", href: "/vehicles", icon: "fas fa-car-alt" },
  { name: "Cameras", href: "/cameras", icon: "fas fa-video" },
  { name: "Profile", href: "/profile", icon: "fas fa-user-circle" },
];

export default function Sidebar({
  collapsed,
  setCollapsed,
}: {
  collapsed: boolean;
  setCollapsed: (value: boolean) => void;
}) {
  const [location, setLocation] = useLocation();
  const { logout } = useAuth();

  const [mobileOpen, setMobileOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 1024);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleNavigate = (href: string) => {
    setLocation(href);
    if (isMobile) setMobileOpen(false); // close sidebar on mobile
  };

  return (
    <>
      {/* Sidebar - Desktop & Mobile with Premium Design */}
      <aside
        className={`
          fixed inset-y-0 left-0 text-white z-[70]
          transition-all duration-300 ease-in-out shadow-2xl
          overflow-visible
          
          bg-[#3175F1] border-r border-white/10
          
          ${collapsed ? "w-20" : "w-64"}
          ${mobileOpen ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0
          
          flex flex-col
        `}
      >
        {/* Header - Premium Design */}
        <div className="flex items-center h-20 px-4 sm:px-6 border-b border-white/15 bg-gradient-to-r from-[#3175F1] to-[#1e4ba6] shadow-lg">
          {/* Logo and Title Container */}
          <div className="flex items-center space-x-3 min-w-0 flex-1">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-white/40 to-white/20 flex items-center justify-center shadow-lg backdrop-blur-sm flex-shrink-0">
              <i className="fas fa-video text-lg text-white font-bold"></i>
            </div>
            {!collapsed && (
              <div className="flex flex-col truncate">
                <span className="text-sm sm:text-base font-bold text-white leading-tight">AI</span>
                <span className="text-xs font-semibold text-white/80 leading-none">SMARTGATE</span>
              </div>
            )}
          </div>

          {/* Desktop collapse toggle - positioned separately */}
          <button
            className="hidden lg:flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 transition-all duration-200 rounded-lg p-2 backdrop-blur-sm flex-shrink-0 ml-auto group relative"
            onClick={() => setCollapsed(!collapsed)}
          >
            <i className={`fas ${collapsed ? "fa-chevron-right" : "fa-chevron-left"} text-base transition-transform duration-300 group-hover:scale-110`}></i>
            <div className="absolute right-full mr-3 px-4 py-2.5 bg-slate-900/95 text-white text-xs rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none shadow-2xl backdrop-blur-md font-semibold z-50 border border-white/10">
              {collapsed ? "Expand" : "Collapse"}
            </div>
          </button>

          {/* Mobile close button */}
          <button
            className="lg:hidden flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 transition-all duration-200 rounded-lg p-2 backdrop-blur-sm flex-shrink-0"
            onClick={() => setMobileOpen(false)}
          >
            <i className="fas fa-xmark text-lg"></i>
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col mt-4 px-3 sm:px-4" style={{ overflowY: 'auto', overflowX: 'visible' }}>
          <div className="space-y-2">
            {navigation.map((item) => {
              const isActive =
                location === item.href ||
                (location === "/" && item.href === "/dashboard");

              return (
                <Button
                  key={item.name}
                  variant="ghost"
                  className={`
                    w-full transition-all duration-200 rounded-lg flex items-center
                    ${collapsed ? "justify-center px-2 h-12" : "justify-start px-4 h-12 space-x-3"}
                    group relative
                    ${
                      isActive
                        ? "bg-white/20 text-white shadow-lg font-semibold border border-white/20 hover:bg-white/25"
                        : "text-white/70 hover:bg-white/30 font-medium border border-transparent hover:text-slate-900 hover:border-white/15"
                    }
                  `}
                  onClick={() => handleNavigate(item.href)}
                >
                  <i className={`${item.icon} text-base flex-shrink-0 transition-transform group-hover:scale-110 duration-200`}></i>
                  {!collapsed && (
                    <span className="text-sm font-medium truncate">{item.name}</span>
                  )}
                  {collapsed && (
                    <div className="absolute left-full ml-3 px-4 py-2.5 bg-slate-900/95 text-white text-xs rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none shadow-2xl backdrop-blur-md font-semibold z-50 border border-white/10">
                      {item.name}
                    </div>
                  )}
                </Button>
              );
            })}
          </div>

          {/* Divider */}
          {!collapsed && <div className="h-px bg-white/10 my-4 mx-2"></div>}

          {/* Logout Button */}
          <div className="px-2 sm:px-3 pb-4 mt-auto">
            <Button
              variant="ghost"
              className={`
                w-full transition-all duration-200 rounded-lg flex items-center h-12
                ${collapsed ? "justify-center px-2" : "justify-start px-4 space-x-3"}
                text-white/70 hover:text-slate-900 hover:bg-red-500/70 font-medium border border-transparent hover:border-red-400/50 group
              `}
              onClick={() => {
                logout();
                setMobileOpen(false);
              }}
            >
              <i className="fas fa-sign-out-alt text-base flex-shrink-0 transition-transform group-hover:scale-110 duration-200"></i>
              {!collapsed && <span className="text-sm font-medium">Logout</span>}
              {collapsed && (
                <div className="absolute left-full ml-3 px-4 py-2.5 bg-slate-900/95 text-white text-xs rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150 pointer-events-none shadow-2xl backdrop-blur-md font-semibold z-50 border border-white/10">
                  Logout
                </div>
              )}
            </Button>
          </div>
        </nav>

        {/* Footer Info - Desktop Only with Premium Design */}
        {!collapsed && (
          <div className="border-t border-white/15 p-4 mt-auto bg-gradient-to-r from-[#3175F1]/80 to-[#1e4ba6]/80 shadow-lg">
            <div className="text-xs text-white/90 space-y-2">
              <p className="font-semibold text-white text-sm">System Status</p>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-300 animate-pulse shadow-lg"></div>
                <span className="text-white/80 text-xs">Online</span>
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* Mobile Overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[60] lg:hidden transition-all duration-200"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Mobile Top Nav Bar - Premium Design */}
      <div className="lg:hidden fixed top-0 left-0 w-full h-16 bg-[#3175F1] text-white flex items-center px-3 sm:px-4 z-[80] shadow-lg border-b border-white/10">
        {/* Hamburger */}
        <button 
          onClick={() => setMobileOpen(true)} 
          className="flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 transition-all duration-200 rounded-lg p-2 backdrop-blur-sm mr-2"
        >
          <i className="fas fa-bars text-xl"></i>
        </button>

        {/* Logo - Mobile */}
        <div className="flex items-center space-x-2 flex-1 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-white/40 to-white/20 flex items-center justify-center backdrop-blur-sm flex-shrink-0">
            <i className="fas fa-video text-sm text-white font-bold"></i>
          </div>
          <span className="font-bold text-white text-sm sm:text-base truncate">AI-SMARTGATE</span>
        </div>

        {/* Page Title */}
        <span className="text-xs sm:text-sm font-semibold text-white/80 text-right truncate ml-2">
          {location === "/dashboard" || location === "/" ? "Dashboard" : ""}
          {location === "/vehicles" ? "Vehicles" : ""}
          {location === "/cameras" ? "Cameras" : ""}
          {location === "/profile" ? "Profile" : ""}
        </span>
      </div>
    </>
  );
}
