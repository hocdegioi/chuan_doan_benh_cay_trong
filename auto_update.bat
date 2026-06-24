@echo off
echo --- BAT DAU DONG BO APP.PY LEN GITHUB ---

:: 1. Cập nhật mới nhất từ GitHub về để tránh lỗi "rejected"
git pull origin main --rebase

:: 2. Loại bỏ tất cả file đang bị Git theo dõi khỏi cache (không xóa file thật)
:: Điều này giúp "quên" các file nặng lỡ add trước đó
git rm -r --cached .

:: 3. Thêm lại file .gitignore để đảm bảo các file nặng bị chặn vĩnh viễn
git add .gitignore

:: 4. Chỉ thêm đúng file app.py (và các file code nhẹ khác nếu bạn muốn)
git add app.py
git add train.py
git add auto_update.bat

:: 5. Commit và Đẩy lên GitHub
git commit -m "Update source code: app.py"
git push origin main

echo --- DONG BO HOAN TAT! ---
pause