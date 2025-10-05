import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { createProxyMiddleware } from 'http-proxy-middleware';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const FRONT_DIR = path.join(__dirname, 'frontend');

// Serve static frontend
app.use(express.static(FRONT_DIR));

// Proxy API to FastAPI backend
// Tip: set PHENO_API_URL="http://127.0.0.1:8000/api" so /api/* → /api/*
const API_TARGET = process.env.PHENO_API_URL || 'http://127.0.0.1:8000';
const needsApiSuffix = !/\/api\/?$/.test(API_TARGET);
app.use(
  '/api',
  createProxyMiddleware({
    target: API_TARGET,
    changeOrigin: true,
    // If target doesn't end with /api, strip /api from the path
    pathRewrite: needsApiSuffix ? { '^/api': '' } : undefined,
    logLevel: 'warn',
  })
);

// SPA fallback
app.get('*', (_req, res) => res.sendFile(path.join(FRONT_DIR, 'index.html')));

const port = process.env.PORT || 5173;
app.listen(port, () => {
  console.log(`Frontend http://localhost:${port} (proxy → ${API_TARGET})`);
});