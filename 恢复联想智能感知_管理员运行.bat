@echo off
echo 正在恢复联想"智能感知"服务...
sc config SmartSense start= auto >nul 2>&1
sc start SmartSense >nul 2>&1
sc config "Lenovo Smart Appearance Intelligent Sensing Aware Service" start= auto >nul 2>&1
sc start "Lenovo Smart Appearance Intelligent Sensing Aware Service" >nul 2>&1
echo 完成！已恢复。
pause
