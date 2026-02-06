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
