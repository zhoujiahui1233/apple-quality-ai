@echo off
setlocal
set "CLI=%LOCALAPPDATA%\Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe"
set "ARDUINO_DIRECTORIES_DATA=E:\ArduinoData"
set "PORT=COM10"

echo ============================================
echo  ESP32 E-Nose - Compile + Upload
echo ============================================
echo Compiling...
"%CLI%" compile --fqbn esp32:esp32:esp32 --build-path E:\ArduinoBuild "E:\apple\enose\enose_sketch"
if errorlevel 1 (
  echo COMPILE FAILED
  pause
  exit /b 1
)
echo Compile OK. Uploading to %PORT% ...
"%CLI%" upload -p %PORT% --fqbn esp32:esp32:esp32 --input-dir E:\ArduinoBuild "E:\apple\enose\enose_sketch"
if errorlevel 1 (
  echo UPLOAD FAILED - check USB port and BOOT button
  pause
  exit /b 1
)
echo ============================================
echo  DONE! Firmware uploaded.
echo  (If upload stuck, hold BOOT button and retry)
echo ============================================
pause
