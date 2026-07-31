@echo off
cd /d "%~dp0"
title ZhiGuo AI QC System
echo ============================================
echo    ZhiGuo AI QC System - Launcher
echo ============================================
echo.
rem Stop old server listening on port 5020
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5020" ^| findstr "LISTENING"') do (
    echo [1/3] Stopping old server (PID %%a)...
    taskkill /f /pid %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
echo [2/3] Starting server (new window shows logs)...
start "ZhiGuoServer" cmd /k "python app.py"
echo [3/3] Loading models, please wait...
timeout /t 12 /nobreak >nul
echo.
echo Server ready: http://127.0.0.1:5020/
echo Phone access: http://<LAN-IP>:5020/
echo.
start http://127.0.0.1:5020/
