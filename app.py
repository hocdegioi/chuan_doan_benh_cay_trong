import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown

# --- 1. CẤU HÌNH CÁC ID TỪ DRIVE ---
FILE_IDS = {
    'model_light.onnx': '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3',
    'model_light.onnx.data': '1wWFts4HpzNfrpBEaLyjxVeT_JrK9NHO3',
    'labels.txt': '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi'
}
MODEL_FILE = 'model_light.onnx'
LABELS_FILE = 'labels.txt'

# --- 2. HÀM TẢI DỮ LIỆU ---
def download_files_from_drive():
    """Tải các file từ Drive về cùng thư mục làm việc của App."""
    for filename, file_id in FILE_IDS.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filename, quiet=False)
            except Exception as e:
                st.error(f"Lỗi tải file {filename}: {e}")

# --- 3. HÀM LOAD MÔ HÌNH ---
@st.cache_resource
def load_onnx_model():
    """Load model ONNX, tự động nạp trọng số từ file .data đi kèm."""
    download_files_from_drive()
    import onnxruntime as ort
    
    if os.path.exists(MODEL_FILE):
        try:
            # ONNX Runtime sẽ tự động tìm model_light.onnx.data trong cùng thư mục
            return ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
        except Exception as e:
            st.error(f"Lỗi khởi tạo mô hình: {e}")
    return None

# --- 4. HÀM LẤY NHÃN ---
@st.cache_data
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Lỗi tải nhãn"]

# --- 5. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

session = load_onnx_model()
class_names = get_class_names()

if session is None:
    st.warning("Đang tải dữ liệu mô hình... Vui lòng chờ trong giây lát.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Ảnh cây trồng đã chọn", use_column_width=True)
        
        # Tiền xử lý ảnh
        img = np.array(image)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        
        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: img})
            idx = np.argmax(outputs[0])
            
            if idx < len(class_names):
                st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
                st.info("📞 Liên hệ hỗ trợ điều trị: 0763114770")
        except Exception as e:
            st.error(f"Lỗi khi thực hiện dự đoán: {e}")