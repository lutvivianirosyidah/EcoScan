
import os
import json
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# PENGATURAN ECOScan - MODEL UJI 14 KATEGORI
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10
SEED = 123

# Urutan label harus konsisten dengan folder dataset
CLASS_NAMES = [
    "botol_plastik",
    "gelas_plastik",
    "kotak_makanan",
    "sachet",
    "kardus_kertas",
    "kaleng",
    "botol_kaca",
    "sampah_organik",
    "sampah_plastik",
    "puntung_rokok",
    "sampah_residu",
    "baterai_bekas",
    "sampah_elektronik",
    "aluminium",
]

MODEL_NAME = "ecoscan_model_14kelas_uji.keras"
LABEL_NAME = "class_names_14kelas_uji.json"

os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 1. PERIKSA FOLDER DATASET
# ==========================================

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {DATASET_DIR}"
    )

missing_folders = [
    name for name in CLASS_NAMES
    if not os.path.isdir(os.path.join(DATASET_DIR, name))
]

if missing_folders:
    raise FileNotFoundError(
        "Folder kategori berikut belum ditemukan:\n"
        + "\n".join(missing_folders)
    )

extra_folders = sorted(
    name for name in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, name))
    and name not in CLASS_NAMES
)

if extra_folders:
    raise ValueError(
        "Ada folder kategori yang tidak termasuk 14 kelas:\n"
        + "\n".join(extra_folders)
        + "\nPeriksa nama folder dataset terlebih dahulu."
    )

print("=" * 50)
print("ECOSCAN - TRAINING MODEL UJI 14 KATEGORI")
print("=" * 50)

# ==========================================
# 2. MUAT DATASET
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

print("\nUrutan kategori:")
for index, name in enumerate(CLASS_NAMES, start=1):
    print(f"{index:02d}. {name}")

# ==========================================
# 3. OPTIMASI DATASET
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================================
# 4. BANGUN MODEL MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Gunakan bobot dasar yang sudah dilatih sebelumnya.
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================
# 5. TRAINING
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ==========================================
# 6. EVALUASI AWAL
# ==========================================

val_loss, val_accuracy = model.evaluate(
    val_ds,
    verbose=1
)

print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

# ==========================================
# 7. SIMPAN MODEL DAN LABEL BARU
# ==========================================

model_path = os.path.join(MODEL_DIR, MODEL_NAME)
label_path = os.path.join(MODEL_DIR, LABEL_NAME)

# Nama baru agar model lama tidak tertimpa.
model.save(model_path)

with open(label_path, "w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, ensure_ascii=False, indent=4)

print("\n" + "=" * 50)
print("TRAINING MODEL UJI SELESAI")
print("=" * 50)
print(f"Model tersimpan: {model_path}")
print(f"Label tersimpan: {label_path}")
print(f"Jumlah kategori: {len(CLASS_NAMES)}")
print("Catatan: model ini masih tahap percobaan.")
