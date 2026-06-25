import streamlit as st
import numpy as np
import onnxruntime as ort  # Thay đổi từ torch sang onnxruntime
import cv2
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
MODEL_ID = '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3' # Hãy đảm bảo ID này trỏ đến file model_light.onnx
LABELS_ID = '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'

MODEL_FILE = 'model_light.onnx' # Cập nhật tên file
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_resource
def download_files_from_drive():
    try:
        if not os.path.exists(MODEL_FILE):
            gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_FILE, quiet=False)
        if not os.path.exists(LABELS_FILE):
            gdown.download(f'https://drive.google.com/uc?id={LABELS_ID}', LABELS_FILE, quiet=False)
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")

@st.cache_resource
def load_onnx_model():
    download_files_from_drive()
    if os.path.exists(MODEL_FILE):
        # Khởi tạo ONNX Runtime session
        return ort.InferenceSession(MODEL_FILE)
    return None

@st.cache_resource
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return []

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

session = load_onnx_model()
class_names = get_class_names()

if session is None:
    st.error("Không thể tải mô hình.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # --- XỬ LÝ ẢNH ĐẦU VÀO ---
        # Resize và chuẩn hóa giống như khi bạn train model
        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # Chuyển về (C, H, W)
        img = np.expand_dims(img, axis=0)  # Thêm batch dimension (1, 3, 224, 224)
        
        # --- CHẠY DỰ ĐOÁN ---
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        idx = np.argmax(outputs[0])
        
        # --- HIỂN THỊ KẾT QUẢ ---
        if idx < len(class_names):
            st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
            st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí.")
            st.info("📞 Liên hệ điều trị: 0763114770")