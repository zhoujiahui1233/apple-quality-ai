@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 正在启动 400 张种子图复核器...
echo 图片: vlm_seed_400\images
echo 草稿: vlm_seed_400\out\labels
echo 请稍候，会自动打开浏览器...
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:5070"
"E:\anaconda3\python.exe" -X utf8 review_tool.py --images "E:\apple\vlm_seed_400\images" --labels "E:\apple\vlm_seed_400\out\labels" --port 5070 --host 0.0.0.0
echo.
echo 服务已停止，按任意键关闭。
pause >nul
