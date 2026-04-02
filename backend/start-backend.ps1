#!/usr/bin/env powershell
<#
.SYNOPSIS
    Starts the ANPR Backend System
.DESCRIPTION
    This script initializes and runs the Python-based ANPR camera system backend
.PARAMETER camera
    Camera source (default: webcam=0)
.PARAMETER host
    Host to bind to (default: 0.0.0.0)
.PARAMETER port
    Port to bind to (default: 8000)
#>

param(
    [string]$camera = "webcam=0",
    [string]$host = "0.0.0.0",
    [int]$port = 8000
)

$ErrorActionPreference = "Stop"

# Get the backend directory
$backendDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = Join-Path $backendDir "app"

Write-Host "╔════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     ANPR BACKEND SYSTEM - STARTUP SCRIPT          ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend Directory: $backendDir" -ForegroundColor Yellow
Write-Host "App Directory: $appDir" -ForegroundColor Yellow
Write-Host "Camera: $camera" -ForegroundColor Yellow
Write-Host "Host: $host" -ForegroundColor Yellow
Write-Host "Port: $port" -ForegroundColor Yellow
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "✓ Python Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.8+ or add it to PATH"
    exit 1
}

# Check if requirements are installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow

# Navigate to app directory
Set-Location $appDir
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

# Start the backend
Write-Host "Starting ANPR Backend..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

try {
    & python camera_anpr.py --cameras $camera --host $host --port $port
} catch {
    Write-Host "Error starting backend: $_" -ForegroundColor Red
    exit 1
}
