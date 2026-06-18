import streamlit as st
import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model

# Cấu hình trang
st.set_page_config(page_title="AI CHẨN ĐOÁN BỆNH LÚA", layout="centered")

# Hàm load model với cache để tránh tải lại nhiều lần
@st.cache_resource
def load_model():
    model_path = 'model_cay_trong_final.h5'
    file_id = '1-dts2omYS0q9vQYkdrbJWNFAquBzrdt5'
    
    # Kiểm tra nếu file chưa tồn tại thì mới tải về
    if not os.path.exists(model_path):
        st.write("Đang tải model từ Google Drive (575MB). Việc này chỉ diễn ra lần đầu tiên...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    # Load model
    model = tf_load_model(model_path)
    return model

# Giao diện ứng dụng
st.title("AI CHẨN ĐOÁN BỆNH LÚA")
st.write("Ứng dụng sử dụng trí tuệ nhân tạo để nhận diện bệnh trên lá lúa.")

# Gọi hàm load model
try:
    with st.spinner("Đang khởi động AI..."):
        model = load_model()
    st.success("Đã load model thành công!")
except Exception as e:
    st.error(f"Có lỗi khi load model: {e}")
    st.stop()

# Phần xử lý ảnh người dùng upload
uploaded_file = st.file_uploader("Chọn ảnh lá lúa...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Ảnh đã chọn.', use_column_width=True)
    st.write("Đang phân tích...")
    
    # Thêm code xử lý dự đoán của bạn ở đây
    # img = preprocess_image(uploaded_file)
    # prediction = model.predict(img)
    # st.write(f"Kết quả: {prediction}")