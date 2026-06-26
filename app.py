import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
FILE_IDS = {
    'model_light.onnx': '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3',
    'model_light.onnx.data': '1wWFts4HpzNfrpBEaLyjxVeT_JrK9NHO3',
    'labels.txt': '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'
}

def download_files():
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', filename, quiet=False)

@st.cache_resource
def load_session():
    download_files()
    import onnxruntime as ort
    return ort.InferenceSession('model_light.onnx', providers=['CPUExecutionProvider'])

@st.cache_data
def load_labels():
    download_files()
    with open('labels.txt', "r", encoding='utf-8') as f:
        return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")

session = load_session()
labels = load_labels()

uploaded_file = st.file_uploader("Chọn ảnh từ thư viện hoặc Chụp ảnh trực tiếp", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Tiền xử lý khớp với train.py
    img = np.array(image)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Dự đoán với ngưỡng 70%
    outputs = session.run(None, {session.get_inputs()[0].name: img})[0][0]
    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    idx = np.argmax(probs)
    
    if probs[idx] > 0.7:
        st.success(f"Kết quả dự đoán: **{labels[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị sớm để giảm chi phí.")
        st.info("📞 Liên hệ hỗ trợ: 0763114770")
    else:
        st.error("❌ Đối tượng không rõ ràng hoặc không phải lúa!")