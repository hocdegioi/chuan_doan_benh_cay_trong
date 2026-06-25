import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
MODEL_ID = '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3' 
LABELS_ID = '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'
MODEL_FILE = 'model_light.onnx'
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI (Không dùng @st.cache_resource để tránh xung đột) ---
def download_files_from_drive():
    files = {MODEL_FILE: MODEL_ID, LABELS_FILE: LABELS_ID}
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False, fuzzy=True)

# --- KHỞI TẠO MÔ HÌNH (Lazy Import + Cache) ---
@st.cache_resource
def load_onnx_model():
    download_files_from_drive() # Gọi hàm tải thường
    import onnxruntime as ort # Lazy import để không crash lúc startup
    if os.path.exists(MODEL_FILE):
        return ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
    return None

@st.cache_data
def get_class_names():
    download_files_from_drive()
    with open(LABELS_FILE, "r", encoding='utf-8') as f:
        return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]

# --- GIAO DIỆN ---
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")
session = load_onnx_model()
class_names = get_class_names()

if session:
    uploaded_file = st.file_uploader("📂 Chọn ảnh", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img = cv2.resize(np.array(image), (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        # Dự đoán
        input_name = session.get_inputs()[0].name
        res = session.run(None, {input_name: img})[0]
        st.success(f"Kết quả: **{class_names[np.argmax(res)]}**")
        st.warning("⚠️ Cảnh báo: Điều trị ngay.")
        st.info("📞 Liên hệ: 0763114770")