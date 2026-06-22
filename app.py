import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt'
MODEL_FILE = 'model_cay_trong_final.tflite'

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

# --- HÀM TẢI MODEL ---
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Đang tải bộ não AI về..."):
            try:
                gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
            except Exception as e:
                st.error(f"Lỗi tải model: {e}")
                return False
    return True

@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path): return []
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

@st.cache_resource
def load_tflite_model():
    if download_model():
        interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        return interpreter
    return None

# --- GIAO DIỆN CHÍNH ---
st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Nhấn để chụp hoặc chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    with st.spinner("Đang phân tích..."):
        # XỬ LÝ ẢNH
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # CHẠY AI
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        idx = np.argmax(prediction)
        
        # KẾT QUẢ
        if idx < len(class_names):
            st.success(f"Kết quả phân tích: **{class_names[idx]}**")
        else:
            st.warning("Không nhận diện được bệnh!")
            
        st.error("⚠️ Cần xử lý sớm để giảm chi phí!")
        st.markdown("---")
        st.write("📞 **Liên hệ 0763114770 để được tư vấn và điều trị ngay!**")