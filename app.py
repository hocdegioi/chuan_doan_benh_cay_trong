import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import os
import gdown
import numpy as np
from PIL import Image

# 1. Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH LÚA", layout="centered")

# 2. Hàm load model an toàn
@st.cache_resource
def load_model():
    model_path = 'model_cay_trong_final.h5'
    file_id = '1-dts2omYS0q9vQYkdrbJWNFAquBzrdt5'
    
    if not os.path.exists(model_path):
        st.info("Đang tải model từ Google Drive (575MB)... Vui lòng đợi trong giây lát.")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    # compile=False giúp bỏ qua lỗi xung đột phiên bản như batch_shape
    model = tf_load_model(model_path, compile=False)
    return model

# 3. Giao diện chính
st.title("🌾 AI CHẨN ĐOÁN BỆNH LÚA")
st.write("Ứng dụng sử dụng trí tuệ nhân tạo để nhận diện bệnh trên lá lúa.")

# Load model
try:
    with st.spinner("Đang khởi động AI..."):
        model = load_model()
except Exception as e:
    st.error(f"Lỗi khi nạp model: {e}")
    st.stop()

# 4. Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh lá lúa...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã chọn.', use_column_width=True)
    
    if st.button("Phân tích bệnh"):
        try:
            # Xử lý ảnh (bạn có thể thay đổi kích thước 224, 224 tùy theo model của bạn)
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Dự đoán
            prediction = model.predict(img_array)
            
            # Hiển thị kết quả (ví dụ đơn giản)
            st.success(f"Kết quả phân tích: {prediction}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {e}")