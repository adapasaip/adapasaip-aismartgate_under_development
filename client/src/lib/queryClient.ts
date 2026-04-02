import { QueryClient, QueryFunction } from "@tanstack/react-query";

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  method: string,
  url: string,
  data?: unknown | undefined,
): Promise<Response> {
  // Get token from localStorage for authentication
  const token = localStorage.getItem("token");
  const isSubUser = localStorage.getItem("isSubUser") === "true";

  const headers: Record<string, string> = data ? { "Content-Type": "application/json" } : {};

  // Add Authorization header with JWT token
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Add sub-user flag header
  if (isSubUser) {
    headers["x-is-subuser"] = "true";
  }

  const res = await fetch(url, {
    method,
    headers,
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";
export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    // Get token from localStorage for authentication
    const token = localStorage.getItem("token");
    const isSubUser = localStorage.getItem("isSubUser") === "true";

    const headers: Record<string, string> = {};

    // Add Authorization header with JWT token
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    // Add sub-user flag header
    if (isSubUser) {
      headers["x-is-subuser"] = "true";
    }

    const res = await fetch(queryKey.join("/") as string, {
      credentials: "include",
      headers,
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: 0,
      gcTime: 1000 * 60 * 5, // 5 minutes garbage collection time (previously cacheTime)
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
