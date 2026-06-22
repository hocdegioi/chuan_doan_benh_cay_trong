import tensorflow as tf

<<<<<<< HEAD
# Đảm bảo đường dẫn đến model .h5 là đúng
model = tf.keras.models.load_model('model_benh_lua.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ĐÂY LÀ TÊN FILE MÀ HỆ THỐNG ĐANG TÌM KIẾM:
with open('model_cay_trong_final.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("Da luu file model_cay_trong_final.tflite thanh cong!")
=======
print("Đang load model... vui lòng đợi...")
# Load model h5 hiện tại
model = tf.keras.models.load_model('model_cay_trong_final.h5', compile=False)

# Chuyển đổi sang tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # Dòng này giúp giảm dung lượng
tflite_model = converter.convert()

# Lưu kết quả
with open('model_cay_trong_final.tflite', 'wb') as f:
    f.write(tflite_model)

print("Đã xong! File model_cay_trong_final.tflite đã được tạo.")
>>>>>>> de432b5cf6fe2863e7f1c6be17296fbc6b6ac986
