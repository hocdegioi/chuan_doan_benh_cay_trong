import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")

@st.cache_resource
def load_tflite_model():
    # Load model trực tiếp từ file .tflite đã nén
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
st.write("Hãy tải ảnh lá cây trồng cần kiểm tra lên:")

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Chọn ảnh lá cây trồng...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã chọn.', use_column_width=True)
    
    if st.button("Phân tích bệnh"):
        # Xử lý ảnh
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Chạy dự đoán
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # DANH SÁCH BỆNH (Bạn hãy thay đổi các tên này cho khớp với dữ liệu bạn đã train)
        danh_sach_benh = ['Bệnh đạo ôn', 'Bệnh đốm nâu', 'Bệnh bạc lá', 'Bệnh khô vằn', 'Bệnh vàng lá', 'Bệnh lem lép hạt', 'Cây khỏe mạnh', 'Bệnh khác']
        
        # Tìm vị trí có xác suất cao nhất
        index_max = np.argmax(prediction)
        ten_benh = danh_sach_benh[index_max]
        do_tin_cay = np.max(prediction) * 100
        
        st.success(f"Kết quả phân tích: **{ten_benh}**")
        st.info(f"Độ tin cậy của AI: {do_tin_cay:.2f}%")