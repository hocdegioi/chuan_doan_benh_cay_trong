import tensorflow as tf

# Đảm bảo đường dẫn đến model .h5 là đúng
model = tf.keras.models.load_model('model_benh_lua.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ĐÂY LÀ TÊN FILE MÀ HỆ THỐNG ĐANG TÌM KIẾM:
with open('model_cay_trong_final.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("Da luu file model_cay_trong_final.tflite thanh cong!")