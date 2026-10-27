# Pj1 — Sistem Persepsi K3/APD (masker, kacamata, sarung tangan)

Team Cluster 7 · RET503 (Minggu 4–7)

Sistem ini memeriksa apakah seseorang memakai **masker (mask)**, **kacamata (glasses)**, dan **sarung tangan (gloves)**
dari kamera, dengan perangkat target Raspberry Pi 5 + webcam Logitech C270. Repositori ini memuat dataset projek,
tiga eksperimen YOLO (E1, E2, E3), hasilnya, dan analisisnya.

## Isi repositori dan kaitannya dengan tahap tugas

| Tahap | Yang dikumpulkan | Lokasi |
|---|---|---|
| 1–2 | Data projek mahasiswa | `data_projek/` (foto webcam sendiri + label YOLO + `data.yaml` + statistik), bagian "Data" di bawah |
| 3 | Hasil E1 dan E3, `results.png` kedua eksperimen | `hasil/E1_pretrained/`, `hasil/E3_tuning/` (E2 ada di `hasil/E2_scratch/`) |
| 4–6 | `hasil_praktik_Cluster 7.txt/.csv`, analisis, hasil test | `hasil_praktik_Cluster 7.txt`, `.csv`, dan README.md |
| – | Kode | `kode/` (`latih_bandingkan.py` menjalankan E1–E3) |

## 1. Lingkup persepsi dan pipeline

- **Misi:** memberi status per orang untuk tiga item APD: *sudah / belum / tidak terlihat*.
- **Kamera dan perangkat:** webcam Logitech C270 (640×480) ke Raspberry Pi 5, hanya vision.
- **Pipeline sistem (di `kode/`):** model pose (`yolo11n-pose`) mencari wajah dan tangan, lalu detektor APD dijalankan pada
  potongan wajah/tangan, dan hasilnya dihaluskan antar-frame. Eksperimen E1–E3 di bawah melatih **detektor APD** (komponen
  intinya), bukan keseluruhan pipeline. Penjelasan sistem lengkap ada di `kode/README_sistem.md`.
- **Target kinerja (dokumen desain):** [ISI: salin target mAP / FPS / latensi dari dokumen desain minggu 3]

## 2. Data (Tahap 1–2)

- **Data sendiri:** 200 foto dari webcam C270, dilabeli kotak YOLO untuk 3 kelas (`mask`, `glasses`, `gloves`) dengan
  `kode/label_tool.py`. Empat kondisi, masing-masing 50 foto: masker, kacamata, kacamata hitam + masker, dan tanpa APD (contoh negatif).
  **Foto sendiri belum memuat sarung tangan**; kelas itu dipelajari model hanya dari data publik.
  Pembagian: 140 train, 40 val, 20 test. Pembagian dibuat berurutan menurut waktu per kondisi (`make_dataset.py`), bukan acak,
  supaya foto beruntun yang hampir kembar tidak bocor dari train ke val/test.
- **Data publik (pelengkap):** dataset PPE publik berformat YOLOv8 (`PPE Detection_DATA.v3-ppe_dataset_6-class`), diunduh dari
  halaman Hugging Face `Nexus3/yolov8n-ppe-detection-6classes`. Kelasnya dipetakan ke tiga kelas kita dengan
  `kode/remap_public.py` (kelas lain dibuang). Hasilnya 3507 gambar train dan 804 gambar val. **Data publik tidak
  disertakan di repositori ini.** [ISI: sumber asli dan lisensi dataset setelah dicek di halaman unduhannya]
- **Dataset yang dipakai eksperimen** (`data_gabungan.yaml`): train = data publik + foto sendiri yang **diulang 10 kali** (agar
  tidak tenggelam di antara ribuan gambar publik); val = 40 foto sendiri + 804 publik (844 gambar); test = 20 foto sendiri.
  Sidik jari label: `c155653c2c4f` (4511 gambar, 4511 file label).
- **Ketidakseimbangan kelas:** sarung tangan hanya ada di data publik, dan **data test sendiri tidak memuat sarung tangan
  sama sekali** (lihat batasan di bagian 5).

## 3. Eksperimen (Tahap 3)

Ketiganya memakai data, epoch (30), batch (16), seed (0), dan augmentasi bawaan yang sama; hanya satu faktor yang berbeda.
GPU Tesla T4 (Google Colab), Ultralytics 8.4.189.

| Eksperimen | Model | Perubahan | imgsz | mAP50 (val) | mAP50-95 (val) | P | R | ms/gambar (T4) | Latih (menit) |
|---|---|---|---|---|---|---|---|---|---|
| E1_pretrained | `yolo26n.pt` | – (baseline) | 416 | 0.824 | 0.491 | 0.863 | 0.760 | 2.4 | 29.9 |
| E2_scratch | `yolo26n.yaml` | bobot acak | 416 | 0.627 | 0.338 | 0.746 | 0.570 | 2.0 | 29.7 |
| E3_tuning | `yolo26n.pt` | `imgsz=640` | 640 | 0.819 | **0.504** | 0.879 | 0.744 | 4.6 | 39.9 |

Grafik latihan: `hasil/E1_pretrained/results.png`, `hasil/E3_tuning/results.png` (dan `hasil/E2_scratch/results.png`).
Matriks kebingungan dan `args.yaml` ada di folder yang sama. Angka mentah per epoch ada di `results.csv`.

Per kelas pada data val:

| Kelas | E1 mAP50 | E1 mAP50-95 | E2 mAP50 | E2 mAP50-95 | E3 mAP50 | E3 mAP50-95 |
|---|---|---|---|---|---|---|
| mask | 0.823 | 0.449 | 0.610 | 0.268 | 0.821 | 0.461 |
| glasses | 0.926 | 0.502 | 0.790 | 0.409 | 0.911 | 0.506 |
| gloves | 0.723 | 0.523 | 0.480 | 0.336 | 0.724 | 0.544 |

## 4. Pemilihan model dan hasil test (Tahap 4–5)

Model dipilih dari **mAP50-95 val tertinggi**, yaitu **E3_tuning (0.504)**. Data test dipakai **sekali** pada model itu:

| Data test (20 foto sendiri) | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|
| E3_tuning | 0.995 | 0.796 | 1.000 | 1.000 |

Per kelas: mask mAP50-95 = 0.864, glasses = 0.729; sarung tangan tidak punya contoh di data test.
Data test berisi 5 foto per kondisi, termasuk 5 foto kacamata hitam + masker (kasus yang sebelumnya gagal).
Rincian lengkap: `hasil_praktik_4222411014.txt` dan `.csv`.

## 5. Analisis (Tahap 6)

**E1 vs E2 — manfaat transfer learning.** Pada epoch yang sama, E1 mencapai mAP50-95 0.491 sedangkan E2 hanya 0.338
(mAP50 0.824 vs 0.627). Kurva E2 mulai dari hampir nol (0.007 pada epoch 1, E1 sudah 0.303) dan masih naik di epoch 30,
sedangkan E1 sudah 0.303 pada epoch 1. Bobot COCO memberi pengetahuan awal tentang tepi, bentuk, dan orang
sehingga data kita yang kecil cukup untuk menyesuaikannya. Selisih terbesar ada pada kelas yang paling sulit: sarung tangan
(mAP50 0.723 vs 0.480) dan masker (0.823 vs 0.610).

**E1 vs E3 — satu langkah tuning (imgsz 416 → 640).** E3 menang tipis pada kriteria pemilihan (mAP50-95 0.504 vs 0.491,
selisih 0.013), tetapi mAP50 justru sedikit turun (0.819 vs 0.824) dan waktu inferensi naik sekitar 1,9 kali
(4.6 ms vs 2.4 ms di T4). Karena hanya satu percobaan dengan satu seed, selisih sebesar itu **belum cukup untuk disebut
peningkatan yang nyata**. Gambar yang lebih besar membantu bagian presisi (P 0.879 vs 0.863), terutama untuk masker
(P 0.938 vs 0.908), tetapi recall justru turun sedikit (0.744 vs 0.760). Untuk perangkat Raspberry Pi yang hanya CPU, kenaikan waktu hitung
sekitar dua kali lipat itu mahal; **jika latensi di Pi menjadi batas, E1 pada 416 adalah pilihan yang masuk akal meskipun
aturan pemilihan menghasilkan E3.**

**Kelas yang lemah.** Berdasarkan matriks kebingungan E3 (data val), kacamata paling mudah dikenali (≈97% terdeteksi) dan
masker paling sering terlewat (≈11% tidak terdeteksi). Alarm palsu pada latar paling banyak berupa sarung tangan
(sekitar 57% dari seluruh alarm palsu) — wajar karena sarung tangan mudah mirip tangan, lengan baju, atau benda berwarna
gelap. Pada pengamatan langsung sebelumnya (model dari data publik saja), kacamata hitam yang dipakai bersama masker sering
tidak terdeteksi; foto sendiri ditambahkan untuk memperbaikinya.

**Batasan dan hal yang tidak boleh disimpulkan berlebihan:**

1. **Angka test (mAP50 0.995) terlalu optimistis.** Hanya 20 foto (10 masker, 10 kacamata), diambil dari kamera dan sesi
   yang sama dengan foto latih, dan foto beruntun saling mirip walaupun dibagi berurutan menurut waktu. Angka val
   (0.504 mAP50-95) lebih mencerminkan kinerja umum; angka test hanya menunjukkan model bekerja pada kondisi kamera kami.
2. **Data test tidak memuat sarung tangan**, jadi kinerja sarung tangan di data test tidak dapat dilaporkan.
3. **Val didominasi data publik** (804 dari 844 gambar), sehingga angka val terutama mengukur data publik, bukan kamera C270.
4. **Satu seed, 30 epoch.** Kurva ketiga eksperimen masih naik di epoch terakhir (E3: 0.475 → 0.503 dari epoch 20 ke 30),
   jadi model belum konvergen dan ranking E1 vs E3 bisa berubah dengan pelatihan lebih lama atau seed lain.
5. Foto sendiri diulang 10 kali di train; ini menaikkan bobot data kamera kami tetapi juga berisiko menghafal foto-foto itu.
6. Ukuran data kecil: 200 foto sendiri; belum ada foto sarung tangan sendiri.

**Langkah perbaikan berikutnya:** menambah foto sarung tangan dan foto di pencahayaan/latar berbeda ke data test; melatih
lebih dari 30 epoch dengan beberapa seed; mengekspor model ke NCNN untuk menaikkan FPS di Raspberry Pi.

## 6. Uji di Raspberry Pi

[ISI setelah uji langsung: FPS, hasil kondisi masker saja / kacamata biasa / kacamata hitam + masker / tanpa APD, dan
catatan salah deteksi. Bagian ini belum dilakukan saat README ini dibuat.]

## 7. Cara menjalankan ulang

```bash
pip install ultralytics lapx
# data_gabungan.yaml: lihat kode/make_dataset.py dan kode/remap_public.py untuk membuat dataset
python kode/latih_bandingkan.py --nim 4222411014 --nama "Yasin Aprian" --data data_gabungan.yaml --out hasil_praktik --epochs 30
```

Program menulis `hasil_praktik_<NIM>.txt/.csv` dan folder `hasil_praktik/E1_pretrained|E2_scratch|E3_tuning/`.
