@echo off
title Cap nhat Source Code
cd /d "%~dp0"

:: Xóa bỏ file bị khóa
if exist ".git\index.lock" del ".git\index.lock"

:: Xóa chỉ mục cũ để tránh lôi kéo rác
git rm -r --cached .
:: Chỉ thêm lại các file code thiết yếu
git add app.py train.py auto.bat requirements.txt runtime.txt .gitignore

:: Commit và đẩy
git commit -m "Update code: %date% %time%"
git push origin main

echo [+] XONG! Da day code len GitHub!
pause