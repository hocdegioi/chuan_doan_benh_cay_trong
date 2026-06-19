import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH", layout="centered")

# --- HÀM HỖ TRỢ ---
@st.cache_resource
def get_class_names():
    path = "dataset"
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH")
interpreter = load_tflite_model()
class_names = get_class_names()

# Sử dụng Session State để nhớ trạng thái
if 'mode' not in st.session_state:
    st.session_state.mode = None

# Giao diện chọn
col1, col2 = st.columns(2)
if col1.button("📁 Tải ảnh"): st.session_state.mode = 'upload'
if col2.button("📸 Chụp ảnh"): st.session_state.mode = 'camera'

image = None

# Xử lý Tải ảnh
if st.session_state.mode == 'upload':
    uploaded_file = st.file_uploader("Chọn ảnh từ máy:", type=["jpg", "jpeg", "png"])
    if uploaded_file: image = Image.open(uploaded_file)

# Xử lý Chụp ảnh (Camera trực tiếp)
elif st.session_state.mode == 'camera':
    captured_image = st.camera_input("Chụp ảnh")
    if captured_image:
        image = Image.open(captured_image)

# XỬ LÝ KẾT QUẢ NGAY LẬP TỨC
if image:
    st.image(image, use_column_width=True)
    with st.spinner("Đang phân tích..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        idx = np.argmax(prediction)
        
        st.success(f"Kết quả phân tích: **{class_names[idx]}**")
        st.error("⚠️ Cần xử lý ngay để bảo vệ mùa màng!")
        st.markdown("---")
        st.write("📞 **Liên hệ 0763114770 để được tư vấn và điều trị ngay!**")