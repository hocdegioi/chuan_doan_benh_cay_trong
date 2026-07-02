import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown
import onnxruntime as ort

# --- CẤU HÌNH ---
FILE_IDS = {
    'model_light.onnx': '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3',
    'model_light.onnx.data': '1wWFts4HpzNfrpBEaLyjxVeT_JrK9NHO3',
    'labels.txt': '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'
}

def download_files():
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            gdown.download(url, filename, quiet=False)

@st.cache_resource
def load_session():
    download_files()
    return ort.InferenceSession('model_light.onnx', providers=['CPUExecutionProvider'])

@st.cache_data
def load_labels():
    download_files()
    with open('labels.txt', "r", encoding='utf-8') as f:
        return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG")

try:
    session = load_session()
    labels = load_labels()
except Exception as e:
    st.error(f"Lỗi tải file: {e}")
    st.stop()

uploaded_file = st.file_uploader("Chọn ảnh bệnh cây trồng...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # 1. Tiền xử lý chuẩn (Resize + Normalize)
    img = np.array(image)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 2. Chuyển từ (224, 224, 3) sang (3, 224, 224)
    img = img.transpose((2, 0, 1))
    
    # 3. Thêm batch dimension (1, 3, 224, 224)
    img = np.expand_dims(img, axis=0)
    
    # 4. Dự đoán (Sử dụng input name từ session)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})[0][0]
    
    # 5. Xử lý kết quả
    probs = np.exp(outputs - np.max(outputs)) / np.sum(np.exp(outputs - np.max(outputs)))
    idx = np.argmax(probs)
    
    if probs[idx] > 0.5:
        st.success(f"Kết quả dự đoán: **{labels[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị sớm để giảm chi phí.")
        st.info("📞 Liên hệ điều trị ngay: 0763114770")
    else:
        st.error("❌ Không nhận diện được bệnh (Độ tin cậy thấp).")