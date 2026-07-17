@echo off
chcp 65001 >nul
echo 正在檢查/安裝必要套件...

pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 安裝完成！正在啟動即時影像辨識物件分類系統。
python main.py
pause