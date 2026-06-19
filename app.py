import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

# --- HÀM CƠ BẢN ---
@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path): return ["Dữ liệu chưa sẵn sàng"]
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

# --- GIAO DIỆN ---
choice = st.radio("Chọn phương thức:", ["Tải ảnh có sẵn", "Chụp ảnh trực tiếp"])

image_to_process = None

if choice == "Tải ảnh có sẵn":
    uploaded_file = st.file_uploader("Chọn ảnh:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)

elif choice == "Chụp ảnh trực tiếp":
    # Nút này chỉ dùng để hiển thị nút chụp (Camera)
    if st.button("Hiện nút chụp ảnh"):
        st.session_state.open_cam = True
    
    if st.session_state.get('open_cam', False):
        captured_image = st.camera_input("Nhấn nút chụp bên dưới:")
        if captured_image:
            image_to_process = Image.open(captured_image)
            st.session_state.open_cam = False # Tắt nút chụp sau khi chụp xong

# --- XỬ LÝ AI ---
if image_to_process:
    st.image(image_to_process, use_column_width=True)
    with st.spinner("Đang phân tích..."):
        img = image_to_process.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        idx = np.argmax(prediction)
        conf = np.max(prediction) * 100
        
        if conf >= 99.0:
            st.success(f"Kết quả: **{class_names[idx]}**")
        else:
            st.warning(f"Kết quả: **{class_names[idx]}** ({conf:.2f}%). Độ tin cậy thấp!")