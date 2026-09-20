@echo off
cd /d "%~dp0"
title ZhiGuo AI 品控系统 - 启动器
if not exist logs mkdir logs
set LOGFILE=logs\startup.log
echo [%date% %time%] ===== start.bat 启动 ===== >> "%LOGFILE%"
echo ============================================
echo   智果 AI 品控系统 (ZhiGuo AI QC System)
echo ============================================
echo.
rem ===== Supabase 登录配置 =====
set SUPABASE_URL=https://uwbcveiebczmiyszsugm.supabase.co
set SUPABASE_ANON_KEY=sb_publishable_1FJ1_UA0BMc-odoPdY3syQ_5FvqiesV
rem ===== 查找 Python: python -> py -> anaconda =====
set PYCMD=
where python >nul 2>&1 && set PYCMD=python
if not defined PYCMD ( where py >nul 2>&1 && set PYCMD=py )
if not defined PYCMD ( if exist "E:\anaconda3\python.exe" set PYCMD=E:\anaconda3\python.exe )
if not defined PYCMD (
    echo [错误] 找不到 Python，请先安装 Python 或 Anaconda。
    echo [错误] 找不到 Python >> "%LOGFILE%"
    pause
    exit /b 1
)
echo 使用 Python: %PYCMD% >> "%LOGFILE%"
rem ===== 关闭占用 5020 端口的旧进程 =====
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5020" ^| findstr "LISTENING"') do (
    echo [1/4] 正在关闭旧服务 PID %%a ...
    taskkill /f /pid %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul
rem ===== 启动服务（新窗口显示日志）=====
echo [2/4] 正在启动服务（新窗口显示日志）...
start "ZhiGuoServer" cmd /k ""%PYCMD%" app.py"
rem ===== 等待端口就绪 =====
echo [3/4] 正在加载模型，请稍候（约 15-30 秒）...
set /a tries=0
:waitloop
timeout /t 2 /nobreak >nul
set /a tries+=1
netstat -ano | findstr ":5020" | findstr "LISTENING" >nul 2>&1
if not errorlevel 1 goto ready
if %tries% lss 25 goto waitloop
echo [错误] 服务未在预期时间内就绪，请查看 ZhiGuoServer 窗口中的报错信息。 >> "%LOGFILE%"
echo [错误] 服务未在预期时间内就绪，请查看 ZhiGuoServer 窗口中的报错信息。
pause
exit /b 1
:ready
echo [4/4] 服务已就绪！
echo.
echo   本机访问:  http://127.0.0.1:5020/
echo   手机访问:  http://局域网IP:5020/  （需在防火墙放行 5020 端口）
echo.
echo [%date% %time%] 服务就绪: http://127.0.0.1:5020/ >> "%LOGFILE%"
start http://127.0.0.1:5020/
echo 提示：本窗口可关闭，请保留 ZhiGuoServer 窗口以维持服务运行。
pause