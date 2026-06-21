@echo off
echo --- BAT DAU QUY TRINH CAP NHAT TU DONG ---

:: 1. Chay file train và convert
echo [1/4] Dang huan luyen model...
python train.py
if %errorlevel% neq 0 (echo Loi train! & pause & exit)

echo [2/4] Dang chuyen doi sang .tflite...
python convert_to_tflite.py
if %errorlevel% neq 0 (echo Loi convert! & pause & exit)

:: 2. Sao luu file model len Google Drive (Thay duong dan cua ban vao day)
echo [3/4] Dang sao luu model len Google Drive...
copy /Y "model_cay_trong_final.tflite" "G:\My Drive\duan_lua\"

:: 3. Cap nhat "linh hon" app.py len GitHub
echo [4/4] Dang cap nhat code len GitHub...
git add app.py
git commit -m "Auto-update app.py: %date% %time%"
git pull --rebase origin master
git push origin master

echo --- HOAN TAT! MOI THU DA XONG ---
pause