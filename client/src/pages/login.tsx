import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [userType, setUserType] = useState<"main" | "subuser">("main");
  const [mobile, setMobile] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [, setLocation] = useLocation();
  const { toast } = useToast();

  const { login, register, subUserLogin } = useAuth();

  const handleSubUserLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      if (!username || !password) {
        throw new Error("Username and password are required");
      }

      await subUserLogin(username, password);
      setLocation("/dashboard");
    } catch (error) {
      // Error already handled by auth hook
    } finally {
      setIsLoading(false);
    }
  };

  const handleMainUserLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await login(mobile, password);
      setLocation("/dashboard");
    } catch (error) {
      // Error handled in auth hook
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isLogin) {
      setIsLoading(true);
      try {
        if (password !== confirmPassword) {
          throw new Error("Passwords do not match");
        }
        await register(fullName, mobile, password, email);
        setIsLogin(true);
        setMobile("");
        setPassword("");
        setFullName("");
        setEmail("");
        setConfirmPassword("");
      } catch (error) {
        // Error handled in auth hook
      } finally {
        setIsLoading(false);
      }
    } else if (userType === "subuser") {
      await handleSubUserLogin(e);
    } else {
      await handleMainUserLogin(e);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-blue-500 to-blue-700 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
        <div className="absolute -bottom-8 right-1/4 w-96 h-96 bg-blue-300 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-2000"></div>
        <div className="absolute top-1/2 right-0 w-96 h-96 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-4000"></div>
      </div>

      <Card className="w-full max-w-lg relative z-10 shadow-2xl border-0">
        <CardHeader className="text-center bg-gradient-to-r from-blue-50 to-slate-50 rounded-t-xl border-b-2 border-blue-200">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mb-6 mx-auto shadow-lg">
            <i className="fas fa-video text-3xl text-white"></i>
          </div>
          <CardTitle className="text-3xl font-bold text-blue-700">
            AI-SMARTGATE
          </CardTitle>
          <CardDescription className="text-sm text-slate-600 mt-2">
            ANPR Vehicle Management System
          </CardDescription>
        </CardHeader>
        
        <CardContent className="pt-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            {isLogin && (
              <div className="mb-6">
                <Label className="text-slate-700 font-semibold mb-3 block flex items-center">
                  <i className="fas fa-user-circle mr-2 text-blue-600"></i>Login As
                </Label>
                {/* Toggle Button Design */}
                <div className="flex gap-3 bg-blue-50 p-1 rounded-lg border-2 border-blue-200">
                  <button
                    type="button"
                    onClick={() => {
                      setUserType("main");
                      setMobile("");
                      setUsername("");
                      setPassword("");
                    }}
                    className={`flex-1 py-3 px-4 rounded-md font-semibold transition-all flex items-center justify-center gap-2 ${
                      userType === "main"
                        ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md"
                        : "bg-transparent text-slate-600 hover:text-blue-600"
                    }`}
                  >
                    <i className="fas fa-id-card"></i>
                    Main User
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setUserType("subuser");
                      setMobile("");
                      setUsername("");
                      setPassword("");
                    }}
                    className={`flex-1 py-3 px-4 rounded-md font-semibold transition-all flex items-center justify-center gap-2 ${
                      userType === "subuser"
                        ? "bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md"
                        : "bg-transparent text-slate-600 hover:text-blue-600"
                    }`}
                  >
                    <i className="fas fa-user-shield"></i>
                    Sub-User
                  </button>
                </div>
              </div>
            )}

            {!isLogin && (
              <>
                <div>
                  <Label htmlFor="fullName" className="text-slate-700 font-semibold flex items-center mb-2">
                    <i className="fas fa-user mr-2 text-blue-600"></i>Full Name
                  </Label>
                  <Input
                    id="fullName"
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="John Doe"
                    required
                    data-testid="input-fullname"
                    className="border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                  />
                </div>
                <div>
                  <Label htmlFor="email" className="text-slate-700 font-semibold flex items-center mb-2">
                    <i className="fas fa-envelope mr-2 text-blue-600"></i>Email (Optional)
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="your@email.com"
                    data-testid="input-email"
                    className="border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                  />
                </div>
              </>
            )}
            
            {isLogin && userType === "main" && (
              <>
                <div>
                  <Label htmlFor="mobile" className="text-slate-700 font-semibold flex items-center mb-2">
                    <i className="fas fa-phone mr-2 text-blue-600"></i>Mobile Number
                  </Label>
                  <div className="relative">
                    <i className="fas fa-phone absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400"></i>
                    <Input
                      id="mobile"
                      type="tel"
                      value={mobile}
                      onChange={(e) => setMobile(e.target.value)}
                      placeholder="1234567890"
                      className="pl-12 border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                      required
                      data-testid="input-mobile"
                    />
                  </div>
                </div>
              </>
            )}

            {isLogin && userType === "subuser" && (
              <>
                <div>
                  <Label htmlFor="username" className="text-slate-700 font-semibold flex items-center mb-2">
                    <i className="fas fa-user-tag mr-2 text-blue-600"></i>Username
                  </Label>
                  <div className="relative">
                    <i className="fas fa-user absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400"></i>
                    <Input
                      id="username"
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      placeholder="Enter username"
                      className="pl-12 border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                      required
                      data-testid="input-username"
                    />
                  </div>
                  <p className="text-xs text-slate-500 mt-2 flex items-center">
                    <i className="fas fa-info-circle mr-2"></i>Use your sub-user username (assigned by admin)
                  </p>
                </div>
              </>
            )}

            {!isLogin && (
              <div>
                <Label htmlFor="mobile" className="text-slate-700 font-semibold flex items-center mb-2">
                  <i className="fas fa-phone mr-2 text-blue-600"></i>Mobile Number
                </Label>
                <div className="relative">
                  <i className="fas fa-phone absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400"></i>
                  <Input
                    id="mobile"
                    type="tel"
                    value={mobile}
                    onChange={(e) => setMobile(e.target.value)}
                    placeholder="1234567890"
                    className="pl-12 border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                    required
                    data-testid="input-mobile-register"
                  />
                </div>
              </div>
            )}
            
            <div>
              <Label htmlFor="password" className="text-slate-700 font-semibold flex items-center mb-2">
                <i className="fas fa-lock mr-2 text-blue-600"></i>Password
              </Label>
              <div className="relative">
                <i className="fas fa-lock absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400"></i>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={isLogin ? "Enter password" : "Create a strong password"}
                  className="pl-12 border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                  required
                  data-testid="input-password"
                />
              </div>
            </div>
            
            {!isLogin && (
              <div>
                <Label htmlFor="confirmPassword" className="text-slate-700 font-semibold flex items-center mb-2">
                  <i className="fas fa-check-circle mr-2 text-blue-600"></i>Confirm Password
                </Label>
                <div className="relative">
                  <i className="fas fa-check-circle absolute left-4 top-1/2 transform -translate-y-1/2 text-blue-400"></i>
                  <Input
                    id="confirmPassword"
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="Confirm password"
                    className="pl-12 border-2 border-slate-200 hover:border-blue-400 focus:border-blue-600 focus:ring-2 focus:ring-blue-200 transition-all"
                    required
                    data-testid="input-confirm-password"
                  />
                </div>
              </div>
            )}
            
            <Button 
              type="submit" 
              className="w-full h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold text-lg rounded-lg shadow-lg hover:shadow-xl transition-all transform hover:scale-105 mt-8"
              disabled={isLoading}
              data-testid="button-submit"
            >
              {isLoading ? (
                <>
                  <i className="fas fa-spinner fa-spin mr-2"></i>
                  Loading...
                </>
              ) : (
                <>
                  {isLogin ? (
                    <>
                      <i className="fas fa-sign-in-alt mr-2"></i>Sign In
                    </>
                  ) : (
                    <>
                      <i className="fas fa-user-plus mr-2"></i>Register
                    </>
                  )}
                </>
              )}
            </Button>
          </form>
          
          <div className="mt-8 text-center border-t border-slate-200 pt-6">
            <p className="text-slate-600 mb-4">
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <Button
                variant="link"
                onClick={() => {
                  setIsLogin(!isLogin);
                  setUserType("main");
                  setMobile("");
                  setUsername("");
                  setPassword("");
                  setFullName("");
                  setEmail("");
                  setConfirmPassword("");
                }}
                className="p-0 h-auto font-bold text-blue-600 hover:text-blue-700 underline decoration-2 decoration-blue-300"
                data-testid="button-toggle-mode"
              >
                {isLogin ? "Click Register here" : "Sign In here"}
              </Button>
            </p>
          </div>
          
          {isLogin && (
            <div className="mt-6 p-5 bg-gradient-to-br from-blue-50 to-slate-50 rounded-lg border-2 border-blue-200">
              <p className="font-semibold text-slate-700 flex items-center text-sm">
                <i className="fas fa-shield-alt mr-2 text-blue-600"></i>
                Secure ANPR System
              </p>
              <p className="text-xs text-slate-600 mt-2">
                Contact your administrator for login credentials
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
