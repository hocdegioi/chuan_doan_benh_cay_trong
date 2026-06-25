import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
# Cập nhật ID mới từ link Google Drive bạn cung cấp
MODEL_ID = '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3' 
LABELS_ID = '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi' 

MODEL_FILE = 'model_cay_trong_final.tflite'
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI DỮ LIỆU TỪ DRIVE ---
@st.cache_resource
def download_files_from_drive():
    """Tải mô hình và file nhãn từ Google Drive nếu chưa có trên máy."""
    try:
        # Tải mô hình
        if not os.path.exists(MODEL_FILE):
            st.info("Đang tải mô hình từ Drive, vui lòng đợi...")
            gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_FILE, quiet=False)
        
        # Tải labels
        if not os.path.exists(LABELS_FILE):
            st.info("Đang tải dữ liệu nhãn...")
            gdown.download(f'https://drive.google.com/uc?id={LABELS_ID}', LABELS_FILE, quiet=False)
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ Drive: {e}")

@st.cache_resource
def load_tflite_model():
    download_files_from_drive()
    if os.path.exists(MODEL_FILE):
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo mô hình TFLite: {e}")
    return None

@st.cache_resource
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa có dữ liệu nhãn"]

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

interpreter = load_tflite_model()
class_names = get_class_names()

if interpreter is None:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra lại quyền chia sẻ link Drive (bật chế độ 'Bất kỳ ai có liên kết đều có thể xem') hoặc kết nối mạng.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # Xử lý ảnh đầu vào
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        
        # Chạy dự đoán
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Kết quả
        idx = np.argmax(prediction)
        if idx < len(class_names):
            st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
            st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí.")
            st.info("📞 Liên hệ điều trị: 0763114770")