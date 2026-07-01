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
    """Tải 3 file từ Drive về thư mục làm việc nếu chưa có"""
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            # Đã sửa: bỏ tham số fuzzy để tránh lỗi version
            gdown.download(url, filename, quiet=False)

@st.cache_resource
def load_session():
    """Load model ONNX với provider CPU"""
    download_files()
    import onnxruntime as ort
    return ort.InferenceSession('model_light.onnx', providers=['CPUExecutionProvider'])

@st.cache_data
def load_labels():
    """Load danh sách nhãn"""
    download_files()
    with open('labels.txt', "r", encoding='utf-8') as f:
        return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
st.title("🌾 CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")

# Load session và labels
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
    
    # Tiền xử lý (Giữ nguyên logic của bạn)
    img = np.array(image)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0)
    
    # --- PHẦN DỰ ĐOÁN
    
    # 1. Tự động lấy tên input từ mô hình (đã làm)
    input_name = session.get_inputs()[0].name
    
    # 2. DEBUG: In ra cấu trúc mô hình mong muốn so với cấu trúc thực tế
    input_shape = session.get_inputs()[0].shape
    if list(img.shape) != input_shape:
        st.error(f"Lỗi: Mô hình cần shape {input_shape} nhưng ảnh bạn đang gửi là {list(img.shape)}.")
        st.stop()
    
    # 3. Chạy dự đoán
    outputs = session.run(None, {input_name: img})[0][0]
    
    # --- HẾT PHẦN DỰ ĐOÁN ---
    
    # Tính xác suất (Giữ nguyên logic của bạn)
    probs = np.exp(outputs) / np.sum(np.exp(outputs))
    idx = np.argmax(probs)
    
    # Kết quả (Giữ nguyên giao diện và câu chữ của bạn)
    if probs[idx] > 0.7:
        st.success(f"Kết quả dự đoán: **{labels[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị sớm để giảm chi phí.")
        st.error("📞 Liên hệ điều trị ngay: 0763114770")
    else:
        st.error("❌ Không nhận diện được bệnh hoặc ảnh không rõ ràng!")