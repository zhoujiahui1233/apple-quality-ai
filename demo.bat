@echo off
cd /d "%~dp0"
title ZhiGuo Demo
echo ============================================
echo    ZhiGuo Demo - Quick Detect
echo ============================================
echo.
echo Starting demo server on port 5030...
start "ZhiGuoDemo" cmd /k "python demo_app.py"
timeout /t 2 /nobreak >nul
echo.
echo Open: http://127.0.0.1:5030/
echo (keep the ZhiGuoDemo window open)
pause
