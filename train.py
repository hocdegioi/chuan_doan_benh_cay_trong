import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# --- CẤU HÌNH ---
DATA_DIR = "./data"
DRIVE_DIR = r"G:\My Drive\File cây trồng"
MODEL_NAME = "model_light.onnx"
LABELS_FILE = "labels.txt"

def train_and_sync():
    print("--- Bắt đầu huấn luyện mô hình ---")
    
    # 1. Chuẩn hóa ảnh theo chuẩn ImageNet (Bắt buộc phải giống trong app.py)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Lưu nhãn (labels)
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        for class_name in dataset.classes:
            f.write(f"{class_name}\n")
    print(f"Nhãn đã lưu: {dataset.classes}")
    
    # 2. Huấn luyện thực tế
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5): # Huấn luyện 5 vòng
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(dataloader):.4f}")
    
    # 3. Xuất mô hình sang ONNX (Dùng Opset 14 để tránh lỗi)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy_input, MODEL_NAME, 
        opset_version=14, 
        input_names=['input'], output_names=['output']
    )
    print(f"Đã xuất mô hình tại: {MODEL_NAME}")
    
    # 4. Đồng bộ lên Drive
    print("--- Đang đồng bộ lên Google Drive ---")
    files_to_sync = [MODEL_NAME, MODEL_NAME + ".data", LABELS_FILE]
    if os.path.exists(DRIVE_DIR):
        for file_name in files_to_sync:
            if os.path.exists(file_name):
                shutil.copy2(file_name, os.path.join(DRIVE_DIR, file_name))
                print(f"Đã copy {file_name}")
    else:
        print("Lỗi: Không tìm thấy thư mục Google Drive.")

if __name__ == "__main__":
    train_and_sync()