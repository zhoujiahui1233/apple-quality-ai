@echo off
cd /d "%~dp0"
title ZhiGuo AI QC - Desktop
set SUPABASE_URL=https://uwbcveiebczmiyszsugm.supabase.co
set SUPABASE_ANON_KEY=sb_publishable_1FJ1_UA0BMc-odoPdY3syQ_5FvqiesV
set PYCMD=python
where python >nul 2>&1
if errorlevel 1 (
  if exist "E:\anaconda3\python.exe" (
    set PYCMD=E:\anaconda3\python.exe
  ) else (
    echo [ERROR] Python not found. Please install Python or Anaconda first.
    pause
    exit /b 1
  )
)
echo ============================================
echo   ZhiGuo AI QC System - Desktop Window
echo   Close this black window to stop the app.
echo ============================================
"%PYCMD%" desktop_app.py
if errorlevel 1 pause
