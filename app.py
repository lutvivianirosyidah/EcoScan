
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import os
import json
import uuid

app = Flask(__name__)

# =========================
# PENGATURAN APLIKASI
# =========================
UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH = os.path.join("model", "ecoscan_model_baru.keras")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


# =========================
# MEMUAT MODEL AI
# =========================
model = None
class_names = []

try:
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    print("Model AI EcoScan berhasil dimuat!")
    print("Kategori:", class_names)

except Exception as error:
    print("Model AI belum berhasil dimuat:", error)


# =========================
# DATA SARAN PENGOLAHAN
# =========================
data_saran = {
    "botol_plastik": {
        "nama": "Botol Plastik",
        "saran": [
            {
                "judul": "Pot Tanaman",
                "deskripsi": (
                    "Botol bekas dapat diubah menjadi pot tanaman. "
                    "Bersihkan botol, potong dengan hati-hati, "
                    "dan buat lubang drainase."
                ),
                "gambar": "images/pot_botol.jpg"
            },
            {
                "judul": "Kebun Vertikal",
                "deskripsi": (
                    "Manfaatkan botol bekas sebagai wadah tanaman "
                    "yang disusun pada rak atau digantung di dinding."
                ),
                "gambar": "images/kebun_vertikal.jpg"
            },
            {
                "judul": "Tempat Pensil",
                "deskripsi": (
                    "Bagian bawah botol dapat digunakan sebagai "
                    "wadah alat tulis setelah dibersihkan. "
                    "Rapikan bagian tepinya agar aman."
                ),
                "gambar": "images/tempat_pensil.jpg"
            },
            {
                "judul": "Hiasan Gantung Ubur-Ubur",
                "deskripsi": (
                    "Ubah botol plastik menjadi hiasan gantung "
                    "berbentuk ubur-ubur. Tambahkan mata dan "
                    "tentakel untuk mempercantik tampilannya."
                ),
                "gambar": "images/ubur_ubur.jpg"
            }
        ]
    },

    "gelas_plastik": {
        "nama": "Gelas Plastik",
        "saran": [
            {
                "judul": "Pot Bibit Tanaman",
                "deskripsi": (
                    "Gelas plastik yang bersih dapat digunakan "
                    "untuk menyemai bibit. Buat lubang drainase "
                    "pada bagian bawahnya."
                ),
                "gambar": "images/pot_bibit_gelas.jpg"
            },
            {
                "judul": "Wadah Alat Tulis",
                "deskripsi": (
                    "Gelas plastik dapat dijadikan wadah pensil, "
                    "pulpen, atau perlengkapan belajar kecil."
                ),
                "gambar": "images/wadah_alat_tulis_gelas.jpg"
            },
            {
                "judul": "Hiasan Gantung",
                "deskripsi": (
                    "Hias gelas plastik bekas menjadi dekorasi "
                    "gantung dengan tali dan bahan hias tambahan."
                ),
                "gambar": "images/hiasan_gantung_gelas.jpg"
            },
            {
                "judul": "Wadah Perlengkapan Kerajinan",
                "deskripsi": (
                    "Gunakan gelas yang bersih untuk menyimpan "
                    "kancing, manik-manik, atau perlengkapan "
                    "kerajinan berukuran kecil."
                ),
                "gambar": "images/wadah_kerajinan_gelas.jpg"
            }
        ]
    },

    "sachet": {
        "nama": "Kemasan Sachet",
        "saran": [
            {
                "judul": "Tas Anyaman dari Sachet",
                "deskripsi": (
                    "Bungkus sachet bekas yang sudah dibersihkan "
                    "dan dikeringkan dapat dianyam menjadi tas."
                ),
                "gambar": "images/tas_anyaman_sachet.jpg"
            },
            {
                "judul": "Dompet dari Sachet",
                "deskripsi": (
                    "Kemasan sachet dapat disusun dan dianyam "
                    "menjadi dompet kecil yang unik."
                ),
                "gambar": "images/dompet_sachet.jpg"
            },
            {
                "judul": "Bunga Hias dari Sachet",
                "deskripsi": (
                    "Potong dan bentuk kemasan sachet menjadi "
                    "kelopak bunga untuk dekorasi. Gunakan "
                    "gunting dengan hati-hati."
                ),
                "gambar": "images/bunga_sachet.jpg"
            },
            {
                "judul": "Tempat Tisu dari Sachet",
                "deskripsi": (
                    "Rangkai kemasan sachet yang bersih menjadi "
                    "lapisan dekoratif untuk kotak tempat tisu."
                ),
                "gambar": "images/tempat_tisu_sachet.jpg"
            }
        ]
    },

    "kotak_makanan": {
        "nama": "Kotak Makanan Plastik",
        "saran": [
            {
                "judul": "Organizer Penyimpanan",
                "deskripsi": (
                    "Kotak makanan bekas yang bersih dan masih "
                    "layak dapat digunakan untuk menyimpan "
                    "perlengkapan rumah tangga kecil."
                ),
                "gambar": "images/organizer_kotak.jpg"
            },
            {
                "judul": "Wadah Perlengkapan Kerajinan",
                "deskripsi": (
                    "Manfaatkan kotak bekas untuk menyimpan "
                    "alat kerajinan, benang, atau aksesori kecil."
                ),
                "gambar": "images/wadah_kerajinan_kotak.jpg"
            },
            {
                "judul": "Wadah Semai Tanaman",
                "deskripsi": (
                    "Kotak yang sesuai dapat dimanfaatkan "
                    "sebagai wadah semai tanaman. Pastikan "
                    "ada drainase yang memadai."
                ),
                "gambar": "images/wadah_semai_kotak.jpg"
            },
            {
                "judul": "Kotak Penyimpanan Serbaguna",
                "deskripsi": (
                    "Gunakan kotak yang sudah dicuci dan "
                    "dikeringkan untuk menyimpan benda nonmakanan."
                ),
                "gambar": "images/kotak_serbaguna.jpg"
            }
        ]
    }
}


# =========================
# FUNGSI VALIDASI FILE
# =========================
def file_diizinkan(nama_file):
    return (
        "." in nama_file
        and nama_file.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================
# FUNGSI PREDIKSI AI
# =========================
def prediksi_sampah(lokasi_file):
    if model is None or not class_names:
        raise RuntimeError("Model AI belum tersedia.")

    with Image.open(lokasi_file) as gambar:
        gambar = gambar.convert("RGB")
        gambar = gambar.resize((224, 224))
        gambar_array = np.array(gambar, dtype=np.float32)

    gambar_array = np.expand_dims(gambar_array, axis=0)

    hasil_prediksi = model.predict(
        gambar_array,
        verbose=0
    )[0]

    indeks = int(np.argmax(hasil_prediksi))
    kategori = class_names[indeks]
    confidence = float(hasil_prediksi[indeks]) * 100

    return kategori, round(confidence, 2)


# =========================
# HALAMAN UTAMA
# =========================
@app.route("/")
def index():
    return render_template("index.html")


# =========================
# HALAMAN KATEGORI
# =========================
@app.route("/kategori")
def kategori():
    return render_template("kategori.html")


# =========================
# HALAMAN EDUKASI
# =========================
@app.route("/edukasi")
def edukasi():
    return render_template("edukasi.html")


# =========================
# HALAMAN CARA KERJA
# =========================
@app.route("/cara-kerja")
def cara_kerja():
    return render_template("cara_kerja.html")


# =========================
# HALAMAN DETEKSI
# =========================
@app.route("/deteksi")
def deteksi():
    return render_template("deteksi.html")


# =========================
# HALAMAN SARAN PENGOLAHAN
# =========================
@app.route("/pengolahan")
def pengolahan():
    jenis = request.args.get(
        "kategori",
        "botol_plastik"
    )

    hasil = data_saran.get(jenis)

    if hasil is None:
        jenis = "botol_plastik"
        hasil = data_saran[jenis]

    return render_template(
        "pengolahan.html",
        hasil=hasil,
        jenis=jenis
    )


# =========================
# PROSES UPLOAD DAN DETEKSI
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    if "gambar" not in request.files:
        return "File gambar tidak ditemukan.", 400

    file = request.files["gambar"]

    if file.filename == "":
        return "Kamu belum memilih gambar.", 400

    if not file_diizinkan(file.filename):
        return "Format harus PNG, JPG, JPEG, atau WEBP.", 400

    if model is None or not class_names:
        return (
            "Model AI belum tersedia. Periksa folder model.",
            500
        )

    nama_asli = secure_filename(file.filename)

    if not nama_asli or "." not in nama_asli:
        return "Nama file gambar tidak valid.", 400

    ekstensi = nama_asli.rsplit(".", 1)[1].lower()
    nama_file = f"{uuid.uuid4().hex}.{ekstensi}"

    lokasi_file = os.path.join(
        app.config["UPLOAD_FOLDER"],
        nama_file
    )

    file.save(lokasi_file)

    try:
        kategori_prediksi, confidence = prediksi_sampah(
            lokasi_file
        )

    except (UnidentifiedImageError, OSError, ValueError):
        if os.path.exists(lokasi_file):
            os.remove(lokasi_file)

        return (
            "File tidak dapat dibaca sebagai gambar. "
            "Coba gunakan gambar lain.",
            400
        )

    except Exception:
        app.logger.exception("Prediksi AI gagal")

        return (
            "Terjadi kesalahan saat memproses gambar dengan AI.",
            500
        )

    return render_template(
        "hasil.html",
        nama_file=nama_file,
        nama_asli=nama_asli,
        kategori=kategori_prediksi,
        confidence=confidence
    )


# =========================
# PENANGANAN FILE TERLALU BESAR
# =========================
@app.errorhandler(413)
def file_terlalu_besar(error):
    return "Ukuran gambar terlalu besar. Maksimal 8 MB.", 413


# =========================
# JALANKAN APLIKASI
# =========================
if __name__ == "__main__":
    app.run(debug=True)
