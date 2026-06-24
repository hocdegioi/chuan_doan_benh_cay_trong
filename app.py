import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
# Thay ID này bằng ID của file .tflite trên Drive của bạn
MODEL_ID = '17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt' 
# Thay ID này bằng ID của file labels.txt trên Drive của bạn
LABELS_ID = 'ID_FILE_LABELS_TXT_TREN_DRIVE' 

MODEL_FILE = 'model_cay_trong_final.tflite'
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI DỮ LIỆU TỪ DRIVE ---
@st.cache_resource
def download_files_from_drive():
    # Tải mô hình
    if not os.path.exists(MODEL_FILE):
        gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_FILE, quiet=False)
    # Tải labels
    if not os.path.exists(LABELS_FILE):
        gdown.download(f'https://drive.google.com/uc?id={LABELS_ID}', LABELS_FILE, quiet=False)

@st.cache_resource
def load_tflite_model():
    download_files_from_drive()
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa có dữ liệu nhãn"]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG")

interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Xử lý ảnh
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Dự đoán
    idx = np.argmax(prediction)
    if idx < len(class_names):
        st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí")
        st.info("📞 Liên hệ điều trị: 0763114770")