import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# Link Google Drive chứa file .tflite (Bạn đã có link này)
MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt'
MODEL_FILE = 'model_cay_trong_final.tflite'

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")

# Hàm tải model từ Drive nếu chưa có trong thư mục của Streamlit
@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Đang tải model từ Drive..."):
            gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

# Hàm đọc nhãn từ file labels.txt do auto_update.bat đẩy lên
@st.cache_resource
def get_class_names():
    if os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding='utf-8') as f:
            # Xử lý định dạng tên bệnh
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa có dữ liệu nhãn"]

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG")
interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    # Xử lý ảnh để đưa vào model
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    # Chạy dự đoán
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Hiển thị kết quả
    idx = np.argmax(prediction)
    if idx < len(class_names):
        st.success(f"Kết quả: **{class_names[idx]}**")
    st.write("📞 Liên hệ hỗ trợ: 0763114770")import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
MODEL_URL = 'https://drive.google.com/uc?id=17t789jiASVUHvEr3PXNo8WW2ARzP3Ijt'
MODEL_FILE = 'model_cay_trong_final.tflite'

st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH", layout="centered")

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def get_class_names():
    path = "dataset"
    if os.path.exists(path):
        folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
        return [name.replace('_', ' ').capitalize() for name in folders]
    return []

st.title("🌾 AI CHẨN ĐOÁN BỆNH CÂY TRỒNG BẰNG HÌNH ẢNH")
interpreter = load_tflite_model()
class_names = get_class_names()

uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

if uploaded_file and interpreter:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    idx = np.argmax(prediction)
    if idx < len(class_names):
        st.success(f"Kết quả chuẩn đoán: **{class_names[idx]}**")
    else:
        st.warning("ĐIỀU TRỊ SỚM ĐỂ GIÃM CHI PHÍ!")
    
    st.markdown("---")
    st.write("📞 **GỌI NGAY ĐỂ ĐƯỢC ĐIỀU TRỊ: 0763114770**")
