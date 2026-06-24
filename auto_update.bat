@echo off
echo --- BAT DAU DONG BO "SACH" VA CUONG BUC DONG BO ---

:: 1. Cập nhật lịch sử mới nhất từ GitHub về
git pull origin main --rebase

:: 2. Xóa sạch cache để Git quên các file nặng (dataset, .tflite, .h5...)
git rm -r --cached .

:: 3. Chỉ thêm lại các file mã nguồn (TUYỆT ĐỐI KHÔNG add thư mục dataset)
git add app.py
git add labels.txt
git add train.py
git add .gitignore
git add auto_update.bat

:: 4. Commit thay đổi
git commit -m "Cleanup: Xoa bo file nang, chi giu code"

:: 5. Ép buộc đẩy lên GitHub để ghi đè lịch sử (Sửa lỗi rejected)
git push origin main --force

echo --- DONG BO HOAN TAT! ---
echo --- Kiem tra lai tren GitHub web, neu con file nang hay xoa thu cong lan cuoi. ---
pause