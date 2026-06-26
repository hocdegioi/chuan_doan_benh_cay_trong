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
    print("--- Đang chuẩn bị dữ liệu... ---")
    
    # Chuẩn hóa chuẩn ImageNet để mô hình học chính xác
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Lưu nhãn
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        for class_name in dataset.classes:
            f.write(f"{class_name}\n")
    print(f"Nhãn đã học: {dataset.classes}")
    
    # Khởi tạo mô hình
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, len(dataset.classes))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Huấn luyện 10 vòng
    model.train()
    print("--- Đang huấn luyện mô hình (10 epochs)... ---")
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(dataloader):.4f}")
    
    # Xuất mô hình
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, MODEL_NAME, opset_version=14)
    
    # Đồng bộ lên Drive
    print("--- Đang đồng bộ lên Google Drive ---")
    for f_name in [MODEL_NAME, MODEL_NAME + ".data", LABELS_FILE]:
        if os.path.exists(f_name) and os.path.exists(DRIVE_DIR):
            shutil.copy2(f_name, os.path.join(DRIVE_DIR, f_name))
            print(f"Đã cập nhật: {f_name}")

if __name__ == "__main__":
    train_and_sync()