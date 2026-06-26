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

# --- GIAO DIỆN ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")

session = load_onnx_model()
class_names = get_class_names()

# --- TÍCH HỢP CHỤP ẢNH VÀ TẢI ẢNH ---
tab1, tab2 = st.tabs(["📂 Tải ảnh lên", "📸 Chụp ảnh trực tiếp"])

image_to_process = None

with tab1:
    uploaded_file = st.file_uploader("Chọn ảnh từ máy", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

with tab2:
    camera_file = st.camera_input("Chụp ảnh cây trồng")
    if camera_file:
        image_to_process = Image.open(camera_file)

# --- XỬ LÝ DỰ ĐOÁN ---
if image_to_process is not None:
    st.image(image_to_process, caption="Ảnh đang kiểm tra", use_column_width=True)
    
    img = image_to_process.convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    if session:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        idx = np.argmax(outputs[0])
        
        st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị sớm để giảm chi phí.") # Dòng yêu cầu mới
        st.info("📞 Liên hệ hỗ trợ điều trị: 0763114770")
    else:
        st.error("Mô hình chưa được nạp!")