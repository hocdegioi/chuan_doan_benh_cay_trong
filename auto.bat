@echo off
title Cap nhat Source Code
cd /d "%~dp0"

:: Xóa lock file
if exist ".git\index.lock" del ".git\index.lock"

:: Xóa sạch cache, chỉ add file code
git rm -r --cached .
git add app.py train.py auto.bat requirements.txt runtime.txt .gitignore

:: Commit và đẩy
git commit -m "Update code: %date% %time%"
git push -u origin main
echo [+] XONG! Da day code len GitHub!
pause