@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 正在启动缺陷标注复核器...
echo 图片: defect_annotate\images
echo 草稿: auto_label_output_qwen_eval\labels
echo 会自动打开浏览器；关闭本窗口 = 停止服务
echo.
python -X utf8 review_tool.py
echo.
echo 服务已停止，按任意键关闭。
pause >nul
