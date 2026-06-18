import tensorflow as tf
import os

print("Đang load model...")
model = tf.keras.models.load_model('model_benh_lua.h5')

# Xóa file cũ nếu tồn tại
if os.path.exists('model_cay_trong_final.h5'):
    os.remove('model_cay_trong_final.h5')

print("Đang lưu model, vui lòng đợi...")
model.save('model_cay_trong_final.h5', include_optimizer=False)

print("Xong! Kiểm tra file trong thư mục.")