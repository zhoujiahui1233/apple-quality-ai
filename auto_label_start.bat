@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ============================================================
echo  VLM 自动预标注（苹果缺陷）
echo  默认模型: 智谱 GLM-4V-Flash（免费）
echo  没配 Key 的话: 去 https://open.bigmodel.cn/ 注册并创建 Key,
echo  然后把本目录的 vlm_config.example.json 复制改名成 vlm_config.json 填进去。
echo ============================================================
echo.
if exist "vlm_config.json" (
  python auto_label_vlm.py --resume
) else (
  echo [提示] 未找到 vlm_config.json，先做 20 张试水：
  python auto_label_vlm.py --source "E:\apple\yolo_pseudo\images" --out "E:\apple\auto_label_output" --limit 20
)
echo.
echo 处理结束，按任意键关闭...
pause >nul
