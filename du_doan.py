import tensorflow as tf
import numpy as np
import cv2

# 1. Load mô hình đã huấn luyện
model = tf.keras.models.load_model('model_benh_lua.h5')

# 2. Danh sách các loại bệnh (Phải đúng thứ tự như lúc bạn train)
# Bạn hãy thay thế bằng các tên thư mục bạn đã dùng
class_names = ['dao_on_co_bong', 'bo_tri'] # Sửa lại cho đúng với tên thư mục của bạn

# 3. Chọn một ảnh mới để test
img_path = 'anh_can_test.jpg' # Đặt ảnh bạn muốn kiểm tra vào cùng thư mục và đổi tên thành 'anh_can_test.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img_array = tf.expand_dims(img, 0) # Tạo batch

# 4. Dự đoán
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"Bức ảnh này khả năng cao là: {class_names[np.argmax(score)]}")
print(f"Độ tự tin: {100 * np.max(score):.2f}%")