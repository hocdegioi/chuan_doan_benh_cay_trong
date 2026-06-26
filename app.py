import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- 1. CẤU HÌNH ---
FILE_IDS = {
    'model_light.onnx': '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3',
    'model_light.onnx.data': '1wWFts4HpzNfrpBEaLyjxVeT_JrK9NHO3',
    'labels.txt': '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'
}
MODEL_FILE = 'model_light.onnx'
LABELS_FILE = 'labels.txt'

def download_files_from_drive():
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)

@st.cache_resource
def load_onnx_model():
    download_files_from_drive()
    import onnxruntime as ort
    if os.path.exists(MODEL_FILE):
        return ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
    return None

@st.cache_data
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Lỗi tải nhãn"]

# --- 2. GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")

session = load_onnx_model()
class_names = get_class_names()

# Nút upload duy nhất hỗ trợ cả chụp ảnh (trên mobile) và chọn file
uploaded_file = st.file_uploader("Chọn ảnh từ thư viện hoặc Chụp ảnh trực tiếp", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã chọn", use_column_width=True)
    
    # --- 3. TIỀN XỬ LÝ ẢNH (PHẢI KHỚP VỚI TRAIN.PY) ---
    img = np.array(image)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    
    # Chuẩn hóa theo ImageNet (Rất quan trọng để dự đoán đúng)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # --- 4. DỰ ĐOÁN ---
    if session:
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img})
            idx = np.argmax(outputs[0])
            
            st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
            st.warning("⚠️ Cảnh báo: Điều trị sớm để giảm chi phí.") # Yêu cầu mới
            st.info("📞 Liên hệ hỗ trợ điều trị: 0763114770")
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
    else:
        st.error("Mô hình không được nạp!")