import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# Cấu hình Model
MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt' # Nhớ thay ID của bạn vào đây
MODEL_FILE = 'model_cay_trong_final.tflite'

# Thiết lập giao diện với tiêu đề mới
st.set_page_config(page_title="CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_class_names():
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa có dữ liệu nhãn"]

# Hiển thị tiêu đề mới
st.title("🌾CHUẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Xử lý ảnh
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Hiển thị kết quả (đã bỏ phần else không xác định bệnh)
    idx = np.argmax(prediction)
    if idx < len(class_names):
        st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
        st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí")
        st.info("📞 Liên hệ điều trị: 0763114770")
