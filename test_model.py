
import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join("model", "ecoscan_model_baru.keras")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("Model baru berhasil dimuat!")
print("Kategori:", class_names)

lokasi_gambar = input(
    "\nMasukkan lokasi file gambar yang ingin dites: "
).strip().strip('"')

try:
    with Image.open(lokasi_gambar) as gambar:
        gambar = gambar.convert("RGB")
        gambar = gambar.resize((224, 224))
        gambar_array = np.array(gambar, dtype=np.float32)

    gambar_array = np.expand_dims(gambar_array, axis=0)

    prediksi = model.predict(gambar_array, verbose=0)[0]
    indeks = int(np.argmax(prediksi))

    print("\nHasil prediksi:", class_names[indeks])
    print("Confidence:", round(float(prediksi[indeks]) * 100, 2), "%")

    print("\nSemua skor kategori:")
    for nama, skor in zip(class_names, prediksi):
        print(f"{nama}: {float(skor) * 100:.2f}%")

except Exception as error:
    print("Gagal menguji gambar:", error)
