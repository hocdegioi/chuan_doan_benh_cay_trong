import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")

# Hàm tự động lấy danh sách bệnh dựa trên thư mục
@st.cache_resource
def get_class_names():
    # Lấy danh sách thư mục trong folder dataset và sắp xếp theo bảng chữ cái
    folders = sorted([f for f in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", f))])
    # Tự động thay thế dấu gạch dưới bằng khoảng trắng để hiển thị đẹp hơn
    return [name.replace('_', ' ').capitalize() for name in folders]

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_cay_trong_final.tflite")
    interpreter.allocate_tensors()
    return interpreter

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
st.write("hệ thống chuẩn đoán bệnh bằng chụp ảnh .")

interpreter = load_tflite_model()
class_names = get_class_names() # Lấy danh sách bệnh tự động

uploaded_file = st.file_uploader("Chọn ảnh bệnh cây trồng...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã chọn.', use_column_width=True)
    
    if st.button("Phân tích bệnh"):
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Tìm chỉ số cao nhất
        index_max = np.argmax(prediction)
        
        # Kiểm tra xem index có nằm trong danh sách không
        if index_max < len(class_names):
            ten_benh = class_names[index_max]
            st.success(f"Kết quả phân tích: **{ten_benh}**")
        else:
            st.error("Model trả về kết quả không khớp với danh sách thư mục!")