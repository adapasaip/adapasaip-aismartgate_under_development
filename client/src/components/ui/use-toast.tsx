"use client";

import * as React from "react";

// Toast type definition
type Toast = {
  id: string;
  title?: string;
  description?: string;
  action?: React.ReactNode;
};

// Function signature for toast
type ToastAction = (toast: Omit<Toast, "id">) => void;

// React context for providing toast
const ToastContext = React.createContext<{ toast: ToastAction } | undefined>(
  undefined
);

// Hook to use toast
export function useToast() {
  const context = React.useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used within a <ToastProvider />");
  }
  return context;
}

// Provider that wraps app and shows toasts
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([]);

  const toast: ToastAction = (toastData) => {
    const id = Math.random().toString(36).substring(2, 9);
    const newToast: Toast = { id, ...toastData };
    setToasts((prev) => [...prev, newToast]);

    // Auto remove toast after 3 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 3000);
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      {/* Toast container */}
      <div className="fixed bottom-4 right-4 space-y-2 z-50">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="bg-white border shadow-md rounded-lg p-3 w-72"
          >
            {t.title && (
              <div className="font-semibold text-gray-800">{t.title}</div>
            )}
            {t.description && (
              <div className="text-sm text-gray-600">{t.description}</div>
            )}
            {t.action && <div className="mt-2">{t.action}</div>}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
