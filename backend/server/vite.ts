import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import type { ViteDevServer } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const log = (msg: string, src = "express") =>
  console.log(`\x1b[36m${new Date().toLocaleTimeString()}\x1b[0m \x1b[35m[${src}]\x1b[0m ${msg}`);

export async function setupVite(app: express.Application, server: any) {
  try {
    const vite = await import("vite");
    const createServer = vite.createServer || (vite as any).default?.createServer;
    
    if (!createServer) {
      console.warn("[VITE] createServer not found in vite module, using fallback");
      // Fallback: just serve static files
      return;
    }

    const viteDevServer: ViteDevServer = await createServer({
      server: { middlewareMode: true },
      appType: "spa",
    }) as ViteDevServer;

    app.use(viteDevServer.middlewares);

    app.use("*", async (req, res, next) => {
      const url = req.originalUrl;
      try {
        const clientTemplate = path.resolve(__dirname, "../../client/index.html");
        let template = fs.readFileSync(clientTemplate, "utf-8");
        template = await viteDevServer.transformIndexHtml(url, template);
        res.status(200).set({ "Content-Type": "text/html" }).end(template);
      } catch (e: any) {
        viteDevServer?.ssrFixStacktrace(e);
        next(e);
      }
    });
  } catch (error) {
    console.error("[VITE] Failed to setup Vite dev server:", error);
    // Continue without Vite HMR in development
  }
}

export function serveStatic(app: express.Application) {
  const distPath = path.resolve(__dirname, "../dist/public");
  const indexPath = path.join(distPath, "index.html");

  app.use(express.static(distPath));
  app.use("*", (req, res) => {
    res.sendFile(indexPath);
  });
}
