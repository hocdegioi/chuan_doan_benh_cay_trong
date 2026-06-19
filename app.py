import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

# --- CÁC HÀM CƠ BẢN ---
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

# --- GIAO DIỆN CHỌN ---
option = st.radio("Chọn phương thức nhập ảnh:", ["Tải ảnh có sẵn", "Chụp ảnh trực tiếp"])

image_to_process = None

if option == "Tải ảnh có sẵn":
    uploaded_file = st.file_uploader("Chọn ảnh từ thiết bị:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)

elif option == "Chụp ảnh trực tiếp":
    # Dùng session_state để quản lý việc có hiển thị camera hay không
    if 'show_camera' not in st.session_state:
        st.session_state.show_camera = False
    
    if st.button("Mở Camera"):
        st.session_state.show_camera = True
        
    if st.session_state.show_camera:
        captured_image = st.camera_input("Chụp ảnh")
        if captured_image is not None:
            image_to_process = Image.open(captured_image)
            st.session_state.show_camera = False # Ẩn camera sau khi chụp xong

# --- XỬ LÝ ẢNH ---
if image_to_process is not None:
    st.image(image_to_process, caption='Ảnh đang xử lý...', use_column_width=True)
    
    with st.spinner("AI đang phân tích..."):
        img = image_to_process.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index_max = np.argmax(prediction)
        do_tin_cay = np.max(prediction) * 100
        
        if do_tin_cay >= 99.0:
            st.success(f"Kết quả dự đoán: **{class_names[index_max]}**")
        else:
            st.warning(f"Kết quả dự đoán: **{class_names[index_max]}** ({do_tin_cay:.2f}%)")
            st.info("Độ tin cậy thấp, hãy chụp lại ảnh rõ nét hơn.")