import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path):
        return ["Chưa có dữ liệu"]
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = get_class_names()

# Sử dụng tabs để phân chia: 1 tab chụp ảnh, 1 tab upload file
tab1, tab2 = st.tabs(["📸 Chụp ảnh trực tiếp", "📁 Chọn ảnh có sẵn"])

image_to_process = None

with tab1:
    captured_image = st.camera_input("Chụp ảnh bệnh cây tại đây:")
    if captured_image is not None:
        image_to_process = Image.open(captured_image)

with tab2:
    uploaded_file = st.file_uploader("Hoặc chọn ảnh từ thư viện...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

# Xử lý ảnh nếu có ảnh từ một trong hai nguồn
if image_to_process is not None:
    st.image(image_to_process, caption='Ảnh đã chọn.', use_column_width=True)
    
    with st.spinner("Đang AI phân tích..."):
        img = image_to_process.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index_max = np.argmax(prediction)
        
        if index_max < len(class_names):
            ten_benh = class_names[index_max]
            do_tin_cay = np.max(prediction) * 100
            st.success(f"Kết quả dự đoán: **{ten_benh}**")
            st.write(f"Độ tin cậy: {do_tin_cay:.2f}%")
        else:
            st.error("Model không nhận diện được bệnh!")