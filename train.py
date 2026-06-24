import tensorflow as tf
from tensorflow.keras import layers, models
import os
import shutil

# --- CẤU HÌNH ---
DATASET_DIR = 'dataset'
MODEL_H5 = 'model_benh_lua.h5'
MODEL_TFLITE = 'model_benh_lua.tflite'
LABELS_FILE = 'labels.txt'
# Đường dẫn chính xác theo yêu cầu của bạn
DRIVE_DIR = 'G:/My Drive/File cây trồng'

def train_and_sync():
    print("--- 1. LOAD DU LIEU ---")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="training", seed=123, image_size=(224, 224), batch_size=32
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR, validation_split=0.2, subset="validation", seed=123, image_size=(224, 224), batch_size=32
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # Xuất file labels.txt
    with open(LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(class_names))
    print(f"-> Labels da cap nhat: {class_names}")

    # --- 2. HUAN LUYEN ---
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("--- 3. DANG HUAN LUYEN ---")
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # --- 4. LUU VA CHUYEN DOI ---
    model.save(MODEL_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print("-> Da tao xong file .h5 va .tflite")

    # --- 5. DONG BO LEN DRIVE ---
    if not os.path.exists DRIVE_DIR = 'G:/My Drive/File cây trồng'
        print(f"-> Loi: Khong tim thay thu muc {DRIVE_DIR}. Vui long kiem tra Google Drive.")
    else:
        files_to_sync = [MODEL_H5, MODEL_TFLITE, LABELS_FILE]
        for file_name in files_to_sync:
            if os.path.exists(file_name):
                shutil.copy2(file_name, os.path.join(DRIVE_DIR = 'G:/My Drive/File cây trồng'))
                print(f"-> Da day len Drive: {file_name}")
            else:
                print(f"-> Canh bao: Khong tim thay file {file_name}")

    print("--- DONG BO HOAN TAT ---")

if __name__ == "__main__":
    train_and_sync()