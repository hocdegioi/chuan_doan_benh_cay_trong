import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
<<<<<<< HEAD
import gdown

# --- CẤU HÌNH ---
# Lưu ý: Khi bạn có model mới, hãy cập nhật lại ID này nếu ID trên Drive thay đổi
MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt'
MODEL_FILE = 'model_cay_trong_final.tflite'

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

# --- HÀM TẢI MODEL TỪ DRIVE ---
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Đang tải bộ não AI từ Drive về..."):
            try:
                gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
            except Exception as e:
                st.error(f"Lỗi kết nối Drive: {e}")
                return False
    return True

# --- HÀM TỰ ĐỘNG QUÉT TÊN BỆNH TỪ DATASET ---
# Khi bạn thêm bệnh mới vào thư mục 'dataset', ứng dụng sẽ tự nhận diện
@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path): 
        return []
    # Lấy danh sách các thư mục con trong 'dataset'
    folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    return [name.replace('_', ' ').capitalize() for name in folders]

# --- HÀM LOAD MODEL TFLITE ---
@st.cache_resource
def load_tflite_model():
    if download_model():
        interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        return interpreter
    return None

# --- GIAO DIỆN CHÍNH ---
st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
st.write("Sử dụng AI để nhận diện bệnh hại cây trồng nhanh chóng và chính xác.")

interpreter = load_tflite_model()
class_names = get_class_names()

# Khu vực tải ảnh lên
uploaded_file = st.file_uploader("📂 Tải ảnh cây trồng cần kiểm tra tại đây...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn vừa tải lên", use_column_width=True)
    
    with st.spinner("Đang chạy AI phân tích..."):
        # Tiền xử lý ảnh (Resize về 224x224 cho đúng đầu vào mô hình)
=======

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

# --- HÀM HỖ TRỢ ---
@st.cache_resource
def get_class_names():
    path = "dataset"
    if not os.path.exists(path): return []
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

# --- CHỈ DÙNG 1 NÚT UPLOAD DUY NHẤT ---
# Trên mobile, nút này tự hiện: Máy ảnh / Ảnh và video
uploaded_file = st.file_uploader("📂 Nhấn để chụp hoặc chọn ảnh", type=["jpg", "jpeg", "png"])

# XỬ LÝ ẢNH NGAY LẬP TỨC
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    
    with st.spinner("Đang phân tích..."):
>>>>>>> de432b5cf6fe2863e7f1c6be17296fbc6b6ac986
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
<<<<<<< HEAD
        # Chạy mô hình TFLite
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Lấy kết quả có xác suất cao nhất
        idx = np.argmax(prediction)
        
        if idx < len(class_names):
            st.success(f"Kết quả chuẩn đoán: **{class_names[idx]}**")
        else:
            st.warning("Mô hình không nhận diện được bệnh này!")
            
        st.info("⚠️ Lưu ý:CẦN ĐIỀU TRỊ SỚM ĐỂ GIÃM CHI PHÍ.")
        st.markdown("---")
        st.write("📞 **GỌI NGAY ĐỂ ĐUỌC ĐIỀU TRỊ: 0763114770**")
=======
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        idx = np.argmax(prediction)
        
        # Kết quả dứt khoát
        st.success(f"Kết quả phân tích: **{class_names[idx]}**")
        st.error("⚠️ Cần xử lý sớm để giãm chi phí!")
        st.markdown("---")
        st.write("📞 **Liên hệ 0763114770 để được tư vấn và điều trị ngay!**")
>>>>>>> de432b5cf6fe2863e7f1c6be17296fbc6b6ac986
