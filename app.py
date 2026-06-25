import streamlit as st
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# --- CẤU HÌNH ---
# Giữ nguyên ID từ link Google Drive bạn cung cấp
MODEL_ID = '1j_j9tOrCb1Huw7wijks259bsj0QtF2k3' 
LABELS_ID = '1JRoe_BnwrEVbI4sOxtVYgM7nIAeGNLKi' 

# Thay đổi tên file lưu trên máy chủ cho đúng bản chất PyTorch
MODEL_FILE = 'model_cay_trong_final.pt'
LABELS_FILE = 'labels.txt'

# --- HÀM TẢI DỮ LIỆU TỪ DRIVE ---
@st.cache_resource
def download_files_from_drive():
    """Tải mô hình PyTorch và file nhãn từ Google Drive nếu chưa có trên máy."""
    try:
        # Tải mô hình
        if not os.path.exists(MODEL_FILE):
            st.info("Đang tải mô hình PyTorch từ Drive, vui lòng đợi...")
            gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_FILE, quiet=False)
        
        # Tải labels
        if not os.path.exists(LABELS_FILE):
            st.info("Đang tải dữ liệu nhãn...")
            gdown.download(f'https://drive.google.com/uc?id={LABELS_ID}', LABELS_FILE, quiet=False)
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ Drive: {e}")

@st.cache_resource
def load_pytorch_model():
    download_files_from_drive()
    if os.path.exists(MODEL_FILE):
        try:
            # Tải mô hình TorchScript, ép chạy bằng CPU trên Streamlit Cloud
            model = torch.jit.load(MODEL_FILE, map_location=torch.device('cpu'))
            model.eval()  # Chuyển sang chế độ dự đoán (Inference)
            return model
        except Exception as e:
            st.error(f"Lỗi khi khởi tạo mô hình PyTorch: {e}")
    return None

@st.cache_resource
def get_class_names():
    download_files_from_drive()
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r", encoding='utf-8') as f:
            return [line.strip().replace('_', ' ').capitalize() for line in f.readlines()]
    return ["Chưa có dữ liệu nhãn"]

# --- GIAO DIỆN STREAMLIT ---
st.set_page_config(page_title="CHẨN ĐOÁN BỆNH CÂY TRỒNG", layout="centered")
st.title("🌾 CHẨN ĐOÁN BỆNH CÂY TRỒNG")

model = load_pytorch_model()
class_names = get_class_names()

if model is None:
    st.error("Không thể tải mô hình. Vui lòng kiểm tra lại quyền chia sẻ link Drive hoặc kết nối mạng.")
else:
    uploaded_file = st.file_uploader("📂 Chọn ảnh cây trồng", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
        # --- XỬ LÝ ẢNH ĐẦU VÀO THEO CHUẨN PYTORCH ---
        # Resize về 224x224, chuyển thành Tensor và Chuẩn hóa (Normalize) theo chuẩn ImageNet
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        img_tensor = transform(image).unsqueeze(0) # Thêm batch dimension (1, 3, 224, 224)
        
        # --- CHẠY DỰ ĐOÁN ---
        with torch.no_grad():
            outputs = model(img_tensor)
            # outputs có thể là tuple hoặc tensor tùy thuộc vào cách bạn export trong train.py
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            prediction = torch.softmax(outputs, dim=1)
            idx = torch.argmax(prediction).item()
        
        # --- HIỂN THỊ KẾT QUẢ ---
        if idx < len(class_names):
            st.success(f"Kết quả dự đoán: **{class_names[idx]}**")
            st.warning("⚠️ Cảnh báo: Điều trị ngay để giảm chi phí.")
            st.info("📞 Liên hệ điều trị: 0763114770")