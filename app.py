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

# --- HÀM TẢI DỮ LIỆU ---
def download_files_from_drive():
    files_to_download = {MODEL_FILE: MODEL_ID, LABELS_FILE: LABELS_ID}
    for filename, file_id in files_to_download.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Lỗi tải file {filename}: {e}")

# --- KHỞI TẠO MÔ HÌNH ---
@st.cache_resource
def load_onnx_model():
    download_files_from_drive()
    # LAZY IMPORT: Giải quyết lỗi ImportError/Traceback
    import onnxruntime as ort 
    
    if os.path.exists(MODEL_FILE):
        try:
            if os.path.getsize(MODEL_FILE) > 100000:
                return ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
            else:
                st.error("Mô hình bị hỏng (file quá nhỏ).")
        except Exception as e:
            st.error(f"Lỗi khởi tạo mô hình: {e}")
    return None

@st.cache_data
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa tải được nhãn"]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

session = load_onnx_model()
class_names = get_class_names()

if session is None:
    st.warning("Đang tải dữ liệu... Vui lòng đợi.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # Tiền xử lý
        img = np.array(image)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img})
            idx = np.argmax(outputs[0])
            
            if idx < len(class_names):
                st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
                st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí.")
                st.info("📞 Liên hệ điều trị: 0763114770")
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")