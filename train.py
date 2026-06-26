import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# --- CẤU HÌNH ---
DATA_DIR = "./data"  # Thư mục chứa các folder ảnh bệnh
DRIVE_DIR = r"G:\My Drive\File cây trồng"
MODEL_NAME = "model_light.onnx"
LABELS_FILE = "labels.txt"

def train_and_sync():
    print("--- Bắt đầu huấn luyện ---")
    
    # 1. Chuẩn bị dữ liệu
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Kiểm tra thư mục dữ liệu trước khi train
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại {DATA_DIR}")
        return

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    
    # Lưu nhãn (labels)
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        for class_name in dataset.classes:
            f.write(f"{class_name}\n")
    print(f"Đã lưu nhãn tại: {LABELS_FILE}")
    
    # 2. Chuẩn bị mô hình (MobileNetV2)
    # Lưu ý: 'pretrained=True' có thể gây cảnh báo, nhưng vẫn hoạt động tốt
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    model.eval()
    
    # 3. Xuất mô hình sang ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, 
        dummy_input, 
        MODEL_NAME, 
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True
    )
    print(f"Đã xuất mô hình tại: {MODEL_NAME}")
    
    # 4. Tự động đồng bộ lên Google Drive (Đã cập nhật copy file .data)
    print("--- Đang đồng bộ lên Google Drive ---")
    
    # Danh sách các file cần đồng bộ
    files_to_sync = [MODEL_NAME, LABELS_FILE]
    
    # Kiểm tra nếu có file .data phát sinh (thường là model_light.onnx.data)
    data_file = MODEL_NAME + ".data"
    if os.path.exists(data_file):
        files_to_sync.append(data_file)
        
    if os.path.exists(DRIVE_DIR):
        for file_name in files_to_sync:
            src = os.path.join(os.getcwd(), file_name)
            dst = os.path.join(DRIVE_DIR, file_name)
            try:
                shutil.copy2(src, dst)
                print(f"Đã copy {file_name} tới {DRIVE_DIR}")
            except Exception as e:
                print(f"Lỗi khi copy {file_name}: {e}")
    else:
        print(f"Lỗi: Không thấy ổ G (Google Drive) tại {DRIVE_DIR}")
        print("Vui lòng kiểm tra lại kết nối Drive của bạn.")

if __name__ == "__main__":
    train_and_sync()