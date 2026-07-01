@echo off
title Cap nhat Source Code
cd /d "%~dp0"

:: Xóa lock file nếu có
if exist ".git\index.lock" del ".git\index.lock"

:: Xóa sạch cache index (để đảm bảo Git không giữ lại các tệp cũ)
git rm -r --cached .
:: Chỉ thêm đúng các file code thiết yếu
git add app.py train.py auto.bat requirements.txt runtime.txt .gitignore

:: Commit và push
git commit -m "Update code: %date% %time%"
git push origin main

echo [+] XONG! Da day code len GitHub thanh cong!
pause