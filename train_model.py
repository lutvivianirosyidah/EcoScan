
import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Pengaturan dataset
DATASET_DIR = "dataset"
MODEL_DIR = "model"
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# Memuat gambar dan label berdasarkan nama folder
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Kategori:", class_names)

# Simpan daftar kategori
with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(class_names, f, ensure_ascii=False)

# Optimasi pemuatan data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Model dasar MobileNetV2
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Susun model klasifikasi
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Latih model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Simpan model terbaru
model.save(os.path.join(MODEL_DIR, "ecoscan_model_baru.keras"))

print("Model AI EcoScan berhasil dilatih!")
print("Kategori:", class_names)
