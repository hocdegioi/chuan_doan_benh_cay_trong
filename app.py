import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

# Hàm tự động lấy danh sách bệnh (quét thư mục dataset)
@st.cache_resource
def get_class_names():
    # Lấy danh sách thư mục trong folder 'dataset'
    path = "dataset"
    if not os.path.exists(path):
        return ["Chưa có dữ liệu"]
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

# Load model TFLite
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")

# Load model và danh sách bệnh
interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = get_class_names()

# Giao diện camera cho điện thoại
captured_image = st.camera_input("Chụp ảnh bệnh cây cần kiểm tra:")

if captured_image is not None:
    # Mở ảnh
    image = Image.open(captured_image)
    st.image(image, caption='Ảnh vừa chụp.', use_column_width=True)
    
    # Xử lý phân tích ngay lập tức
    with st.spinner("Đang AI phân tích..."):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Lấy kết quả
        index_max = np.argmax(prediction)
        
        if index_max < len(class_names):
            ten_benh = class_names[index_max]
            do_tin_cay = np.max(prediction) * 100
            
            st.success(f"Kết quả dự đoán: **{ten_benh}**")
            st.write(f"Độ tin cậy của AI: {do_tin_cay:.2f}%")
        else:
            st.error("Model trả về kết quả không khớp với dữ liệu!")