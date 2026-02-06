# Analisis dan Perbandingan Model

## Perbandingan Model Donut dan Florence

### Model Donut
Model Donut menghasilkan output yang lebih terstruktur dalam format yang mudah diproses (umumnya XML atau JSON). Namun, model ini masih memiliki kekurangan dalam hal akurasi pembacaan, terutama pada beberapa bagian nota yang kompleks atau kualitas gambar yang kurang baik. Struktur hasil ekstraksi cenderung konsisten, sehingga memudahkan proses parsing dan integrasi ke tahap selanjutnya.

### Model Florence
Model Florence memiliki tingkat akurasi pembacaan yang lebih baik dibandingkan Donut, terutama dalam mengenali detail pada gambar nota. Namun, hasil ekstraksi dari model ini cenderung kurang terstruktur dan membutuhkan proses parsing tambahan untuk mendapatkan data yang siap digunakan. Selain itu, model Florence relatif lebih berat dijalankan dibandingkan Donut.

### Kelemahan Keduanya
Kedua model memiliki kelemahan yang sama, yaitu performa yang menurun secara signifikan ketika melakukan inference pada gambar nota yang posisinya miring atau tidak lurus. Hal ini menyebabkan hasil ekstraksi menjadi tidak akurat atau bahkan gagal terbaca. Untuk mengatasi hal ini, diperlukan preprocessing manual pada gambar, seperti auto-crop dan deskew, yang dapat dilakukan menggunakan library OpenCV dan scikit-image.

## Alur (Flow) Aplikasi
1. **Upload Foto**: Pengguna mengunggah foto nota yang akan diproses.
2. **Isi Partisipan**: Pengguna mengisi daftar partisipan yang akan dibagi tagihannya.
3. **Pilih Model**: Pengguna memilih model AI yang akan digunakan (Donut atau Florence-2).
4. **Load Model**: Sistem memuat model yang dipilih.
5. **Inference**: Model melakukan ekstraksi data dari gambar nota.
6. **Parsing Output ke LLM**: Hasil ekstraksi diparsing dan, jika perlu, diproses lebih lanjut menggunakan LLM untuk mendapatkan struktur data yang diinginkan.
7. **Assign Menu ke Partisipan**: Pengguna meng-assign setiap menu/item ke partisipan yang sesuai.
8. **Munculkan Total**: Sistem menampilkan total tagihan per partisipan berdasarkan pembagian menu yang telah dilakukan.
# Cara Menjalankan Program dengan Docker

Berikut langkah-langkah untuk menjalankan aplikasi ini menggunakan Docker:

## 1. Persiapan
Pastikan Anda sudah menginstall:
- [Docker](https://www.docker.com/products/docker-desktop/)

## 2. Dapatkan Hugging Face Token (HF_TOKEN)
Aplikasi ini membutuhkan akses ke model dari Hugging Face. Silakan:

1. Daftar/login ke https://huggingface.co
2. Buka halaman Settings > Access Tokens
3. Buat token baru (pilih "Fine-Grained")
4. Simpan token tersebut, misal: `hf_xxxxxxxxxxxxxxxxx`

docker build -t receipt-ocr-app .
docker run -p 8501:8501 -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxx receipt-ocr-app

## 3. Build & Jalankan dengan Docker Compose
Buka terminal di folder project ini, lalu jalankan perintah berikut untuk build dan menjalankan aplikasi:

```
docker-compose up --build
```

Jika butuh environment variable (misal HF_TOKEN), bisa tambahkan pada file `.env` di root project:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

- Ganti `hf_xxxxxxxxxxxxxxxxx` dengan token Hugging Face Anda.
- Port 8501 adalah default untuk aplikasi Streamlit. Jika aplikasi Anda menggunakan port lain, sesuaikan pada docker-compose.yml.

Nama image: `streamlit-ocr-img`
Nama container: `streamlit-ocr`

## 5. Akses Aplikasi
Buka browser dan akses:

```
http://localhost:8501
```

Aplikasi akan tampil di browser Anda.

## 6. (Opsional) Mount Folder Data
Jika ingin menggunakan folder data lokal, tambahkan opsi volume:

```
docker run -p 8501:8501 -e HF_TOKEN=hf_xxxxxxxxxxxxxxxxx -v $(pwd)/data:/app/data receipt-ocr-app
```

## 7. Troubleshooting
- Pastikan file `requirements.txt` dan `Dockerfile` sudah sesuai.
- Jika ada error dependency, cek log build dan sesuaikan `requirements.txt`.
- Pastikan token Hugging Face valid dan sudah di-set di environment variable `HF_TOKEN`.

---

**Catatan:**
- Jika Anda melakukan perubahan pada kode, lakukan build ulang image Docker.
- Untuk development, Anda bisa menggunakan opsi `--rm` agar container otomatis terhapus setelah stop.

---

Selamat mencoba!
