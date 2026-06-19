import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH LÚA")

@st.cache_resource
def load_tflite_model():
    # Load model trực tiếp từ thư mục
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
interpreter = load_tflite_model()

uploaded_file = st.file_uploader("Chọn ảnh lá lúa...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã chọn.', use_column_width=True)
    
    if st.button("Phân tích bệnh"):
        # Xử lý ảnh
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Chạy dự đoán
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        st.success(f"Kết quả phân tích: {prediction}")