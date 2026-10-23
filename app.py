
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(
    BASE_DIR, "model", "ecoscan_model_14kelas_uji.keras"
)
CLASS_NAMES_PATH = os.path.join(
    BASE_DIR, "model", "class_names_14kelas_uji.json"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Batas awal untuk pengujian deteksi
BATAS_CONFIDENCE = 60.0

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
                "deskripsi": "Botol bekas dapat diubah menjadi pot tanaman. Bersihkan botol, potong dengan hati-hati, dan buat lubang drainase.",
                "gambar": "images/pot_botol.jpg"
            },
            {
                "judul": "Kebun Vertikal",
                "deskripsi": "Manfaatkan botol bekas sebagai wadah tanaman yang disusun pada rak atau digantung di dinding.",
                "gambar": "images/kebun_vertikal.jpg"
            },
            {
                "judul": "Tempat Pensil",
                "deskripsi": "Bagian bawah botol dapat digunakan sebagai wadah alat tulis setelah dibersihkan. Rapikan bagian tepinya agar aman.",
                "gambar": "images/tempat_pensil.jpg"
            },
            {
                "judul": "Hiasan Gantung Ubur-Ubur",
                "deskripsi": "Ubah botol plastik menjadi hiasan gantung berbentuk ubur-ubur. Tambahkan mata dan tentakel untuk mempercantik tampilannya.",
                "gambar": "images/ubur_ubur.jpg"
            }
        ]
    },

    "gelas_plastik": {
        "nama": "Gelas Plastik",
        "saran": [
            {
                "judul": "Pot Bibit Tanaman",
                "deskripsi": "Gelas plastik yang bersih dapat digunakan untuk menyemai bibit. Buat lubang drainase pada bagian bawahnya.",
                "gambar": "images/pot_bibit_gelas.jpg"
            },
            {
                "judul": "Wadah Alat Tulis",
                "deskripsi": "Gelas plastik dapat dijadikan wadah pensil, pulpen, atau perlengkapan belajar kecil.",
                "gambar": "images/wadah_alat_tulis_gelas.jpg"
            },
            {
                "judul": "Hiasan Gantung",
                "deskripsi": "Hias gelas plastik bekas menjadi dekorasi gantung dengan tali dan bahan hias tambahan.",
                "gambar": "images/hiasan_gantung_gelas.jpg"
            },
            {
                "judul": "Wadah Perlengkapan Kerajinan",
                "deskripsi": "Gunakan gelas yang bersih untuk menyimpan kancing, manik-manik, atau perlengkapan kerajinan berukuran kecil.",
                "gambar": "images/wadah_kerajinan_gelas.jpg"
            }
        ]
    },

    "sachet": {
        "nama": "Kemasan Sachet",
        "saran": [
            {
                "judul": "Tas Anyaman dari Sachet",
                "deskripsi": "Bungkus sachet bekas yang sudah dibersihkan dan dikeringkan dapat dianyam menjadi tas.",
                "gambar": "images/tas_anyaman_sachet.jpg"
            },
            {
                "judul": "Dompet dari Sachet",
                "deskripsi": "Kemasan sachet dapat disusun dan dianyam menjadi dompet kecil yang unik.",
                "gambar": "images/dompet_sachet.jpg"
            },
            {
                "judul": "Bunga Hias dari Sachet",
                "deskripsi": "Potong dan bentuk kemasan sachet menjadi kelopak bunga untuk dekorasi. Gunakan gunting dengan hati-hati.",
                "gambar": "images/bunga_sachet.jpg"
            },
            {
                "judul": "Tempat Tisu dari Sachet",
                "deskripsi": "Rangkai kemasan sachet yang bersih menjadi lapisan dekoratif untuk kotak tempat tisu.",
                "gambar": "images/tempat_tisu_sachet.jpg"
            }
        ]
    },

    "kotak_makanan": {
        "nama": "Kotak Makanan Plastik",
        "saran": [
            {
                "judul": "Organizer Penyimpanan",
                "deskripsi": "Kotak makanan bekas yang bersih dan masih layak dapat digunakan untuk menyimpan perlengkapan rumah tangga kecil.",
                "gambar": "images/organizer_kotak.jpg"
            },
            {
                "judul": "Wadah Perlengkapan Kerajinan",
                "deskripsi": "Manfaatkan kotak bekas untuk menyimpan alat kerajinan, benang, atau aksesori kecil.",
                "gambar": "images/wadah_kerajinan_kotak.jpg"
            },
            {
                "judul": "Wadah Semai Tanaman",
                "deskripsi": "Kotak yang sesuai dapat dimanfaatkan sebagai wadah semai tanaman. Pastikan ada drainase yang memadai.",
                "gambar": "images/wadah_semai_kotak.jpg"
            },
            {
                "judul": "Kotak Penyimpanan Serbaguna",
                "deskripsi": "Gunakan kotak yang sudah dicuci dan dikeringkan untuk menyimpan benda nonmakanan.",
                "gambar": "images/kotak_serbaguna.jpg"
            }
        ]
    },

    "kardus_kertas": {
        "nama": "Kardus dan Kertas",
        "saran": [
            {
                "judul": "Organizer Meja dari Kardus",
                "deskripsi": "Kardus bersih dapat dipotong dan dirangkai menjadi organizer untuk alat tulis atau perlengkapan meja.",
                "gambar": "images/organizer_kardus.jpg"
            },
            {
                "judul": "Rak Penyimpanan Mini",
                "deskripsi": "Kardus yang cukup kokoh dapat disusun menjadi rak mini untuk benda ringan.",
                "gambar": "images/rak_kardus.jpg"
            },
            {
                "judul": "Bingkai Foto dari Kardus",
                "deskripsi": "Gunakan potongan kardus untuk membuat bingkai foto dan hias sesuai selera.",
                "gambar": "images/bingkai_foto_kardus.jpg"
            },
            {
                "judul": "Keranjang Kertas",
                "deskripsi": "Kertas bekas dapat digulung atau dianyam menjadi keranjang penyimpanan ringan.",
                "gambar": "images/keranjang_kertas.jpg"
            }
        ]
    },

    "kaleng": {
        "nama": "Kaleng",
        "saran": [
            {
                "judul": "Tempat Pensil dari Kaleng",
                "deskripsi": "Kaleng yang sudah dibersihkan dapat dijadikan tempat pensil. Pastikan seluruh bagian tepinya aman dan tidak tajam.",
                "gambar": "images/tempat_pensil_kaleng.jpg"
            },
            {
                "judul": "Pot Tanaman dari Kaleng",
                "deskripsi": "Kaleng bekas dapat digunakan sebagai pot tanaman setelah dibersihkan dan dibuat drainase dengan aman.",
                "gambar": "images/pot_kaleng.jpg"
            },
            {
                "judul": "Lentera Dekoratif",
                "deskripsi": "Kaleng dapat dijadikan lentera dekoratif oleh orang dewasa dengan alat pelindung dan tepi yang diamankan.",
                "gambar": "images/lentera_kaleng.jpg"
            },
            {
                "judul": "Tempat Sendok dan Garpu",
                "deskripsi": "Kaleng bersih dapat digunakan untuk menyimpan alat makan setelah tepinya diamankan.",
                "gambar": "images/tempat_sendok_kaleng.jpg"
            }
        ]
    },

    "botol_kaca": {
        "nama": "Botol Kaca",
        "saran": [
            {
                "judul": "Vas Bunga dari Botol Kaca",
                "deskripsi": "Botol kaca yang utuh dan bersih dapat digunakan sebagai vas bunga.",
                "gambar": "images/vas_botol_kaca.jpg"
            },
            {
                "judul": "Lampu Dekorasi Botol",
                "deskripsi": "Botol kaca dapat dijadikan dekorasi lampu menggunakan lampu LED yang sesuai dan aman.",
                "gambar": "images/lampu_botol_kaca.jpg"
            },
            {
                "judul": "Tempat Lilin dari Kaca",
                "deskripsi": "Gunakan wadah kaca yang memang tahan panas dan stabil; jangan gunakan kaca retak atau tipis.",
                "gambar": "images/tempat_lilin_kaca.jpg"
            },
            {
                "judul": "Hiasan Meja dari Botol Kaca",
                "deskripsi": "Botol kaca utuh dapat dihias menjadi dekorasi meja tanpa perlu memotong kacanya.",
                "gambar": "images/hiasan_botol_kaca.jpg"
            }
        ]
    },

    "sampah_organik": {
        "nama": "Sampah Organik",
        "saran": [
            {
                "judul": "Kompos dari Sisa Organik",
                "deskripsi": "Sisa buah, sayur, dan daun dapat diolah menjadi kompos dengan pengaturan kelembapan dan aerasi.",
                "gambar": "images/kompos_organik.jpg"
            },
            {
                "judul": "Pupuk Organik Cair",
                "deskripsi": "Sebagian sisa organik dapat diproses menjadi pupuk cair melalui metode pengolahan yang tepat.",
                "gambar": "images/pupuk_cair_organik.jpg"
            },
            {
                "judul": "Eco-Enzyme dari Sisa Buah",
                "deskripsi": "Sisa buah tertentu dapat difermentasi menjadi eco-enzyme dengan resep dan prosedur yang tepat.",
                "gambar": "images/eco_enzyme.jpg"
            },
            {
                "judul": "Mulsa dari Daun Kering",
                "deskripsi": "Daun kering dapat digunakan sebagai mulsa untuk membantu menjaga kelembapan tanah.",
                "gambar": "images/mulsa_daun_kering.jpg"
            }
        ]
    },

    "sampah_plastik": {
        "nama": "Sampah Plastik",
        "saran": [
            {
                "judul": "Mozaik dari Sampah Plastik",
                "deskripsi": "Potongan plastik bersih dapat disusun menjadi karya mozaik dekoratif.",
                "gambar": "images/mozaik_sampah_plastik.jpg"
            },
            {
                "judul": "Keranjang Anyaman Plastik Kresek",
                "deskripsi": "Kantong plastik yang bersih dan kering dapat dianyam menjadi keranjang yang dapat digunakan kembali.",
                "gambar": "images/keranjang_anyaman_kresek.jpg"
            },
            {
                "judul": "Dompet dari Plastik Kresek",
                "deskripsi": "Kantong plastik bersih dapat diolah menjadi bahan dompet dengan teknik kerajinan yang sesuai.",
                "gambar": "images/dompet_plastik_kresek.jpg"
            },
            {
                "judul": "Bunga Hias dari Sedotan",
                "deskripsi": "Sedotan bekas yang sudah dibersihkan dapat dibentuk menjadi bunga hias.",
                "gambar": "images/bunga_hias_sedotan.jpg"
            }
        ]
    },

    "puntung_rokok": {
        "nama": "Puntung Rokok",
        "saran": [
            {
                "judul": "Mozaik Puntung Rokok",
                "deskripsi": "Contoh karya edukasi yang ditangani secara khusus. Jangan membuat kerajinan rumahan dari puntung rokok karena residunya dapat berbahaya.",
                "gambar": "images/mozaik_puntung_rokok.jpg"
            },
            {
                "judul": "Bonsai Puntung Rokok",
                "deskripsi": "Jadikan gambar ini contoh edukasi pengelolaan limbah. Hindari menyentuh atau mengumpulkan puntung rokok dengan tangan telanjang.",
                "gambar": "images/bonsai_puntung_rokok.jpg"
            },
            {
                "judul": "Bunga dari Puntung Rokok",
                "deskripsi": "Puntung rokok bukan bahan kerajinan rumah yang aman. Kumpulkan secara hati-hati dan salurkan sesuai panduan pengelolaan setempat.",
                "gambar": "images/bunga_puntung_rokok.jpg"
            },
            {
                "judul": "Patung Puntung Rokok",
                "deskripsi": "Contoh karya seni khusus untuk edukasi limbah. Jangan membongkar atau menangani puntung rokok tanpa prosedur perlindungan yang tepat.",
                "gambar": "images/patung_puntung_rokok.jpg"
            }
        ]
    },

    "sampah_residu": {
        "nama": "Sampah Residu",
        "saran": [
            {
                "judul": "Buket dari Tisu",
                "deskripsi": "Tisu yang bersih dapat dibentuk menjadi buket dekoratif. Tisu yang terkontaminasi sebaiknya tidak digunakan untuk kerajinan.",
                "gambar": "images/buket_tisu.jpg"
            },
            {
                "judul": "Boneka dari Masker",
                "deskripsi": "Gunakan bahan baru atau bersih yang belum pernah dipakai untuk membuat boneka; jangan gunakan masker bekas pakai.",
                "gambar": "images/boneka_masker.jpg"
            },
            {
                "judul": "Lap Meja dari Kain Perca",
                "deskripsi": "Kain perca yang bersih dapat dijahit menjadi lap meja atau kain pembersih yang dapat digunakan kembali.",
                "gambar": "images/lap_meja_kain_perca.jpg"
            },
            {
                "judul": "Asbak dari Pecahan Keramik",
                "deskripsi": "Pecahan keramik hanya boleh digunakan setelah diolah dengan alat pelindung dan tepinya diamankan oleh orang berpengalaman.",
                "gambar": "images/asbak_pecahan_keramik.jpg"
            }
        ]
    },

    "baterai_bekas": {
        "nama": "Baterai Bekas",
        "saran": [
            {
                "judul": "Kenali Baterai Bekas",
                "deskripsi": "Pisahkan baterai bekas dari sampah rumah tangga lainnya. Jangan dibakar, dibongkar, atau direndam.",
                "gambar": "images/baterai_bekas.jpg"
            },
            {
                "judul": "Wadah Limbah Baterai",
                "deskripsi": "Simpan sementara di tempat kering dan aman, cegah terminal baterai bersentuhan, serta jauhkan dari anak-anak.",
                "gambar": "images/wadah_limbah_baterai.jpg"
            },
            {
                "judul": "Daur Ulang Baterai",
                "deskripsi": "Baterai perlu ditangani oleh fasilitas pengelolaan limbah yang sesuai, bukan dibongkar sendiri di rumah.",
                "gambar": "images/daur_ulang_baterai.jpg"
            },
            {
                "judul": "Setor Baterai ke Tempat Pengumpulan",
                "deskripsi": "Cari tempat pengumpulan yang menerima jenis baterai tersebut atau pengelola limbah elektronik yang sesuai.",
                "gambar": "images/tempat_setor_baterai.jpg"
            }
        ]
    },

    "sampah_elektronik": {
        "nama": "Sampah Elektronik",
        "saran": [
            {
                "judul": "Reuse Komponen Elektronik",
                "deskripsi": "Komponen yang masih layak dapat dimanfaatkan kembali oleh orang yang memiliki keahlian sesuai. Jangan membongkar perangkat sendiri jika tidak terlatih.",
                "gambar": "images/reuse_komponen_elektronik.jpg"
            },
            {
                "judul": "Seni Papan Sirkuit",
                "deskripsi": "Papan sirkuit dapat dijadikan karya dekoratif oleh pengolah yang memahami cara penanganan komponen elektronik bekas.",
                "gambar": "images/seni_papan_sirkuit.jpg"
            },
            {
                "judul": "Pengumpulan E-Waste",
                "deskripsi": "Kumpulkan perangkat elektronik yang tidak terpakai secara terpisah untuk disalurkan ke pengelola yang sesuai.",
                "gambar": "images/pengumpulan_ewaste.jpg"
            },
            {
                "judul": "Daur Ulang Elektronik",
                "deskripsi": "Perangkat elektronik diproses oleh fasilitas daur ulang khusus untuk memulihkan material yang masih bernilai.",
                "gambar": "images/daur_ulang_ewaste.jpg"
            }
        ]
    },

    "aluminium": {
        "nama": "Aluminium",
        "saran": [
            {
                "judul": "Tempat Pensil Aluminium",
                "deskripsi": "Kaleng aluminium yang bersih dapat dijadikan tempat pensil setelah tepinya diamankan agar tidak tajam.",
                "gambar": "images/tempat_pensil_aluminium.jpg"
            },
            {
                "judul": "Relief dari Aluminium Foil",
                "deskripsi": "Aluminium foil yang bersih dapat dibentuk menjadi pola timbul untuk karya seni dekoratif.",
                "gambar": "images/relief_aluminium_foil.jpg"
            },
            {
                "judul": "Patung Kawat Aluminium",
                "deskripsi": "Kawat aluminium dapat dibentuk menjadi patung dekoratif. Pastikan ujung kawat tidak tajam dan gunakan alat pelindung saat membentuknya.",
                "gambar": "images/patung_kawat_aluminium.jpg"
            },
            {
                "judul": "Patung Kaleng Aluminium",
                "deskripsi": "Kaleng aluminium bekas dapat dijadikan karya seni dekoratif jika diproses dengan aman dan bagian tajamnya ditutup.",
                "gambar": "images/patung_kaleng_aluminium.jpg"
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

    # Pastikan indeks sesuai dengan daftar kelas
    if indeks >= len(class_names):
        raise ValueError(
            "Jumlah keluaran model tidak sesuai dengan daftar kategori."
        )

    kategori = class_names[indeks]
    confidence = float(hasil_prediksi[indeks]) * 100
    confidence = round(confidence, 2)

    # Jika tingkat keyakinan di bawah batas,
    # tandai sebagai belum dikenali
    if confidence < BATAS_CONFIDENCE:
        return None, confidence, False

    return kategori, confidence, True



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
    jenis = request.args.get("kategori", "").strip()

    hasil = data_saran.get(jenis)

    if hasil is None:
        return (
            f"Saran pengolahan untuk kategori '{jenis}' "
            "belum tersedia. Periksa nama kategori di data_saran.",
            404
        )

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
        kategori_prediksi, confidence, dikenali = (
            prediksi_sampah(lokasi_file)
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

    # =========================
    # JIKA SAMPAH BELUM DIKENALI
    # =========================

    if not dikenali:
        return render_template(
            "hasil.html",
            nama_file=nama_file,
            nama_asli=nama_asli,
            dikenali=False,
            confidence=confidence
        )

    # =========================
    # JIKA SAMPAH DIKENALI
    # =========================

    return render_template(
        "hasil.html",
        nama_file=nama_file,
        nama_asli=nama_asli,
        kategori=kategori_prediksi,
        confidence=confidence,
        dikenali=True
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
