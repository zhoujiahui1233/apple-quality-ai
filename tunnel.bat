@echo off
cd /d "%~dp0"
title Cpolar Tunnel - ZhiGuo
set PORT=5030
if not "%1"=="" set PORT=%1
echo ============================================
echo   Cpolar Tunnel -> http://127.0.0.1:%PORT%/
echo ============================================
echo.
where cpolar >nul 2>&1
if errorlevel 1 (
  if exist "C:\Program Files\cpolar\cpolar.exe" (
    "C:\Program Files\cpolar\cpolar.exe" http %PORT%
  ) else (
    echo [ERROR] cpolar not found.
    echo Install from https://www.cpolar.com first.
    pause
  )
) else (
  cpolar http %PORT%
)
pause
