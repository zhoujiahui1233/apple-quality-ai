@echo off
echo 正在关闭联想"智能感知"(自动锁屏/自动息屏)服务...
sc config SmartSense start= disabled >nul 2>&1
sc stop SmartSense >nul 2>&1
sc config "Lenovo Smart Appearance Intelligent Sensing Aware Service" start= disabled >nul 2>&1
sc stop "Lenovo Smart Appearance Intelligent Sensing Aware Service" >nul 2>&1
echo 完成！自动锁屏/自动息屏功能已关闭。
echo 如需恢复，请运行"恢复联想智能感知.bat"
pause
