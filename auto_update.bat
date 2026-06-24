@echo off
echo --- BAT DAU DONG BO LEN GITHUB ---

:: 1. Cập nhật lịch sử mới nhất từ GitHub
git pull origin main --rebase

:: 2. Dọn dẹp lại cache để đảm bảo không dính file rác
git rm -r --cached .
git add .

:: 3. Commit và đẩy lên
git commit -m "Auto-update: Sync code"
git push origin main

echo --- DONG BO HOAN TAT! ---
pause