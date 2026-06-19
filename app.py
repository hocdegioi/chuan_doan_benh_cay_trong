import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

# --- HÀM HỖ TRỢ ---
@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path): return []
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
interpreter = load_tflite_model()
class_names = get_class_names()

# --- CHỈ DÙNG 1 NÚT UPLOAD DUY NHẤT ---
# Trên mobile, nút này tự hiện: Máy ảnh / Ảnh và video
uploaded_file = st.file_uploader("📂 Nhấn để chụp hoặc chọn ảnh", type=["jpg", "jpeg", "png"])

# XỬ LÝ ẢNH NGAY LẬP TỨC
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    with st.spinner("Đang phân tích..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        idx = np.argmax(prediction)
        
        # Kết quả dứt khoát
        st.success(f"Kết quả phân tích: **{class_names[idx]}**")
        st.error("⚠️ Cần xử lý sớm để giãm chi phí!")
        st.markdown("---")
        st.write("📞 **Liên hệ 0763114770 để được tư vấn và điều trị ngay!**")