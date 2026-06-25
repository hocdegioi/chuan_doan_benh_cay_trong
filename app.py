import streamlit as st
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
# Sử dụng ID chính xác từ link bạn cung cấp
MODEL_ID = '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3' 
LABELS_ID = '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'

MODEL_FILE = 'model_light.onnx'
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI DỮ LIỆU ---
@st.cache_resource
def download_files_from_drive():
    """Tải file từ Drive và kiểm tra sự tồn tại của file."""
    files_to_download = {
        MODEL_FILE: MODEL_ID,
        LABELS_FILE: LABELS_ID
    }
    
    for filename, file_id in files_to_download.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                # Sử dụng gdown để tải
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Lỗi tải file {filename}: {e}")

@st.cache_resource
def load_onnx_model():
    download_files_from_drive()
    if os.path.exists(MODEL_FILE):
        try:
            # Kiểm tra kích thước file để đảm bảo đã tải thành công
            if os.path.getsize(MODEL_FILE) > 1000:
                return ort.InferenceSession(MODEL_FILE)
            else:
                st.error("File mô hình bị hỏng hoặc chưa tải xong.")
        except Exception as e:
            st.error(f"Lỗi khởi tạo ONNX Runtime: {e}")
    return None

@st.cache_resource
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa tải được nhãn"]

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

# Load session và labels
session = load_onnx_model()
class_names = get_class_names()

if session is None:
    st.warning("Đang tải dữ liệu từ Drive... Vui lòng đợi hoặc kiểm tra Logs nếu lỗi.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # --- TIỀN XỬ LÝ ẢNH ---
        img = np.array(image)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # (C, H, W)
        img = np.expand_dims(img, axis=0)  # (1, 3, 224, 224)
        
        # --- CHẠY DỰ ĐOÁN ---
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        idx = np.argmax(outputs[0])
        
        # --- HIỂN THỊ KẾT QUẢ ---
        if idx < len(class_names):
            st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
            st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí.")
            st.info("📞 Liên hệ điều trị: 0763114770")