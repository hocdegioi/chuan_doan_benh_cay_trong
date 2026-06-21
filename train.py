import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load dữ liệu (Chỉnh sửa ở đây)
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="training", seed=123, image_size=(224, 224), batch_size=32
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset', validation_split=0.2, subset="validation", seed=123, image_size=(224, 224), batch_size=32
)

# Lấy danh sách tên các thư mục bệnh trước khi tối ưu hóa
class_names = train_ds.class_names
print("Các loại bệnh đã tìm thấy:", class_names)
num_classes = len(class_names) 

# Tối ưu hóa hiệu năng
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 2. Xây dựng Model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax') # Dùng softmax cho phân loại nhiều lớp
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 3. Huấn luyện
print("Đang bắt đầu huấn luyện...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# LƯU MÔ HÌNH SAU KHI XONG
model.save('model_benh_lua.h5')
print("Đã lưu mô hình vào file 'model_benh_lua.h5'")