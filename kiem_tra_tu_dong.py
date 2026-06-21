import tensorflow as tf
import numpy as np
import cv2
import os

# 1. TỰ ĐỘNG LẤY TÊN BỆNH TỪ THƯ MỤC DATASET
dataset_path = 'dataset'
# Lấy danh sách tên thư mục, sắp xếp theo thứ tự bảng chữ cái để khớp với lúc Train
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print("AI đã học các loại bệnh sau:", class_names)

# 2. Load mô hình
model = tf.keras.models.load_model('model_benh_lua.h5')

# 3. Tự động quét thư mục ảnh cần kiểm tra
source_folder = 'anh_can_kiem_tra'

for filename in os.listdir(source_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(source_folder, filename)
        
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))
        img_array = tf.expand_dims(img_resized, 0)
        
        # Dự đoán
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Dùng danh sách class_names vừa lấy tự động
        result = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        print(f"Ảnh: {filename} -> Bệnh: {result} ({confidence:.2f}%)")
        
        cv2.putText(img, f"{result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Ket qua', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()