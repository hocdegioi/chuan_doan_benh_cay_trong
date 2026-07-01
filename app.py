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
    if not os.path.exists('model_light.onnx'):
        raise FileNotFoundError("Không tìm thấy file mô hình!")
    return ort.InferenceSession('model_light.onnx', providers=['CPUExecutionProvider'])

@st.cache_data
def load_labels():
    with open('labels.txt', 'r', encoding='utf-8') as f:
        return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG")

try:
    session = load_session()
    labels = load_labels()
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}. Vui lòng kiểm tra lại Google Drive.")
    st.stop()

uploaded_file = st.file_uploader("Chọn ảnh từ thư viện hoặc Chụp ảnh trực tiếp", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Tiền xử lý
    img = np.array(image)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0) 

    # Dự đoán
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})[0][0]
    
    # Tính xác suất bằng numpy
    exp_out = np.exp(outputs - np.max(outputs))
    probabilities = exp_out / np.sum(exp_out)
    
    prediction = labels[np.argmax(probabilities)]
    confidence = np.max(probabilities) * 100
    
    st.success(f"Kết quả: **{prediction}**")
    st.write(f"Độ chính xác: {confidence:.2f}%")
    
    # Cảnh báo và thông tin liên hệ
    st.warning("⚠️ Điều trị sớm để giảm chi phí")
    st.error("📞 Liên hệ điều trị ngay: 0763114770")
    
    if confidence < 70:
        st.info("💡 Độ tin cậy thấp. Vui lòng thử lại với hình ảnh rõ nét hơn.")