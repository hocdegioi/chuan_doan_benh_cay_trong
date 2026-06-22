import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt'
MODEL_FILE = 'model_cay_trong_final.tflite'

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

@st.cache_resource
def get_class_names():
    # Đọc danh sách bệnh từ file cố định được tạo bởi train.py
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return []

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    idx = np.argmax(prediction)
    if idx < len(class_names):
        st.success(f"Kết quả: **{class_names[idx]}**")
    st.write("📞 Liên hệ tư vấn: 0763114770")