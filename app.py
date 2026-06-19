import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

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

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = get_class_names()

# Sử dụng tabs để người dùng tự chọn cách nhập ảnh
tab1, tab2 = st.tabs(["📸 Chụp ảnh trực tiếp", "📁 Tải ảnh có sẵn"])

image_to_process = None

with tab1:
    st.write("Nhấn nút bên dưới để mở camera:")
    captured_image = st.camera_input("Chụp ảnh")
    if captured_image is not None:
        image_to_process = Image.open(captured_image)

with tab2:
    uploaded_file = st.file_uploader("Chọn ảnh từ thiết bị:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

# Xử lý ảnh
if image_to_process is not None:
    st.image(image_to_process, caption='Ảnh đang xử lý...', use_column_width=True)
    
    with st.spinner("AI đang phân tích..."):
        img = image_to_process.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index_max = np.argmax(prediction)
        do_tin_cay = np.max(prediction) * 100
        
        # Kiểm tra ngưỡng tin cậy 99%
        if do_tin_cay >= 99.0:
            st.success(f"Kết quả dự đoán: **{class_names[index_max]}**")
            st.write(f"Độ tin cậy: {do_tin_cay:.2f}%")
        else:
            st.warning(f"Kết quả dự đoán: **{class_names[index_max]}**")
            st.info(f"Độ tin cậy chỉ đạt {do_tin_cay:.2f}%. Hãy chụp ảnh rõ nét hơn (cần >= 99%).")