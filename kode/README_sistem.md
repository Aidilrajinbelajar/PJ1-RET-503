# Deteksi APD K3 (masker, kacamata, sarung tangan) di Raspberry Pi 5

## Cara kerja
1. Model YOLO mendeteksi kelas **mask**, **goggles/glasses**, dan **gloves** di tiap frame.
2. `ppe_logic.py` mengubah deteksi itu menjadi status per orang: SUDAH / BELUM / MEMERIKSA.
3. Status dihaluskan antar-frame (default: harus terdeteksi di 5 dari 8 frame terakhir) supaya tidak berkedip.
4. Hasil keseluruhan: **LENGKAP**, **APD BELUM LENGKAP**, atau **MEMERIKSA**.

Ada dua cara pakai: `detect.py` (satu model, sederhana) dan `detect_smart.py` (pose + crop, lebih akurat, lihat bagian Mode pintar). Untuk `detect.py`, dua mode otomatis sesuai model:
- **Pos pemeriksaan** (model tanpa kelas `person`): dianggap satu orang berdiri di depan kamera, cocok untuk pintu masuk atau gerbang.
- **Multi-orang** (model punya kelas `person`): APD dicocokkan ke tiap orang (masker/kacamata harus di area kepala), dengan tracking ID.

Penting: sistem hanya mengandalkan deteksi "memakai". Kelas `no_mask`, `no_glove`, dst. sengaja diabaikan karena akurasinya jauh lebih rendah. Pelanggaran = APD tidak terdeteksi cukup lama.

## Mode pintar (disarankan): pose + crop
`detect_smart.py` memakai dua model:
1. **Pose** (`yolo11n-pose`, bawaan Ultralytics) menemukan orang, wajah, dan pergelangan tangan.
2. **Model APD** dijalankan di **potongan wajah** (masker, kacamata) dan **potongan tangan** (sarung tangan), diperbesar ke `--crop-imgsz`.

Keunggulannya dibanding `detect.py`:
- Kacamata dan sarung tangan yang kecil di frame penuh jadi besar di potongan, jadi lebih mudah terdeteksi.
- Status ada empat: **SUDAH**, **BELUM** (areanya terlihat tapi APD tidak ada), **TIDAK TERLIHAT** (orang membelakangi kamera atau tangan disembunyikan, jadi tidak dituduh melanggar), dan **MEMERIKSA**.
- Sarung tangan dinilai per tangan: semua tangan yang terlihat harus bersarung tangan.
- Tidak butuh kelas `person` atau `no_*` di model APD, dan bisa multi-orang dengan tracking.

Siapkan model pose di Pi, lalu ekspor keduanya ke NCNN (ukuran ekspor harus sama dengan ukuran saat dipakai):
```bash
pip install lapx          # dibutuhkan tracker
python export_ncnn.py yolo11n-pose.pt --imgsz 416     # file akan diunduh otomatis
python export_ncnn.py best.pt --imgsz 320
python detect_smart.py --pose-model yolo11n-pose_ncnn_model --ppe-model best_ncnn_model \
    --pose-imgsz 416 --crop-imgsz 320 --source picam
```
Kalau FPS kurang, kurangi `--max-people` (default 2) atau turunkan `--pose-imgsz` ke 320.

### Mengurangi noise dan mengatur arah kamera
- **Arah gambar:** `--flip h` membalik kiri-kanan (seperti cermin), `--flip v` membalik atas-bawah, `--flip both` memutar 180 derajat. Tanpa opsi ini gambar apa adanya dari kamera.
- **Filter posisi (aktif otomatis):** kacamata hanya diterima kalau ada di garis mata, masker hanya kalau di bawah mata, sarung tangan hanya kalau di area tangan. Deteksi liar (rambut, bayangan, latar) ditolak. Matikan dengan `--no-geometry-filter`.
- **Ambang keyakinan per APD:** default mask=0.50, glasses=0.50, gloves=0.40. Kalau masih berkedip, naikkan, misalnya `--min-conf glasses=0.65 mask=0.6`.
- **Smoothing:** `--window 10 --need 7` membuat status lebih stabil tapi lebih lambat bereaksi.
- **Lihat apa yang dilihat model:** tambahkan `--debug`. Deteksi yang ditolak muncul sebagai kotak abu-abu dengan alasannya, dan terminal mencetak angka keyakinan tiap 10 frame. Dari angka itu kamu bisa menentukan ambang yang pas.

### Cara membuat vision makin pintar
Model APD paling bagus kalau dilatih dengan gambar dari kamera dan lokasi kamu sendiri. Jalankan:
```bash
python detect_smart.py ... --save-crops data_crop --save-every 15
```
Potongan wajah/tangan tersimpan di `data_crop/`. Labeli (Roboflow, CVAT, atau LabelImg) dengan kelas `mask`, `goggles`, `gloves`, gabungkan dengan dataset publik, lalu latih ulang dengan `train.py`. Targetnya sekitar 200 sampai 500 potongan dengan variasi orang, cahaya, dan jenis APD.

Catatan batas: sistem ini menilai orang yang menghadap kamera dengan wajah dan tangan terlihat. Wajah yang miring ekstrem (profil) atau tangan yang tertutup akan menjadi TIDAK TERLIHAT.

## Membuat dataset sendiri
Tidak ada jumlah minimum dari tugas, tapi makin beragam makin bagus. Sebagai patokan awal: 100-300 foto dengan variasi orang, cahaya, jarak, dan jenis APD.

1. **Ambil foto** (di Pi atau PC, webcam USB):
   ```bash
   python collect_data.py --source 0 --out dataset_raw --tag lengkap_terang
   python collect_data.py --source 0 --out dataset_raw --tag tanpa_apd_redup --auto 2
   ```
   SPASI menyimpan foto, `q` keluar, `--auto 2` menyimpan tiap 2 detik, `--flip h` kalau mau seperti cermin. Ganti `--tag` tiap kondisi (masker_saja, kacamata_saja, sarung_tangan_saja, tanpa_apd, lengkap, terang, redup).
2. **Beri label** dengan alat bawaan `python label_tool.py --dir dataset_raw` (jalan di Pi lewat VNC, tanpa akun online), atau LabelImg/CVAT, format **YOLO** (kotak). Kelas dan urutannya: `0 mask`, `1 glasses`, `2 gloves`. Label file `.txt` disimpan di folder yang sama dengan fotonya.
   - Kotak mengikuti APD saja (masker, kacamata, satu sarung tangan per kotak), bukan seluruh wajah atau tubuh.
   - Foto tanpa APD sama sekali tetap dipakai sebagai contoh negatif (tanpa label).
   - Beri label di SEMUA APD yang terlihat; APD yang lupa dilabeli membuat model belajar salah.
3. **Bagi train/val/test** dan buat `data.yaml`:
   ```bash
   python make_dataset.py --src dataset_raw --out dataset --names mask glasses gloves --empty-is-negative
   ```
   Skrip memeriksa label (kelas, koordinat), mencetak tabel jumlah foto/kotak per kelas, dan menulis `dataset/dataset_stats.txt`. Pembagian bisa diulang sama persis dengan `--seed`. Foto yang mirip (diambil beruntun) sebaiknya jangan tersebar di train dan val sekaligus; ambil dengan variasi, bukan ratusan frame yang hampir sama.
4. **Latih** (di Colab/PC GPU): `python train.py --data dataset/data.yaml --epochs 100`. Grafik `results.png` ada di `runs/detect/<nama>/`.

### Menambah dataset publik (Nexus3/yolov8n-ppe-detection-6classes)
Repo Hugging Face itu juga berisi dataset-nya: `PPE Detection_DATA.v3-ppe_dataset_6-class.yolov8.zip` (668 MB, format YOLOv8, lisensi MIT, asal data Roboflow; sebutkan sumbernya di laporan). Hanya 3 dari 6 kelasnya yang kita pakai.
```bash
mkdir public_raw && cd public_raw
unzip "../PPE Detection_DATA.v3-ppe_dataset_6-class.yolov8.zip"
cd ..
python remap_public.py --src public_raw --out public_remapped
python make_dataset.py --src dataset_raw --public public_remapped --out dataset --names mask glasses gloves --empty-is-negative
```
`remap_public.py` mengambil Gloves, goggles, mask dan membuang Vest, helmet, safety_shoe. Data publik hanya masuk ke **train**; val dan test tetap fotomu sendiri, jadi angka evaluasinya mencerminkan kamera dan lokasimu. Hanya ekstrak file gambar dan label; jangan membuka file `.pt`/`.pkl` dari sumber lain.

Izin: foto wajah orang lain hanya diambil dengan persetujuan mereka, dan jangan diunggah ke tempat publik tanpa izin.

## 1. Siapkan Raspberry Pi 5
```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv
python3 -m venv --system-site-packages ~/venv-k3
source ~/venv-k3/bin/activate
pip install ultralytics lapx
```
`--system-site-packages` wajib supaya `picamera2` dan `cv2` dari apt terbaca di venv.
Salin folder `ppe_k3` ke Pi (bisa lewat VNC atau `scp`).

## 2. Dapatkan model
**Opsi A, coba cepat tanpa training.** Ada model YOLOv8n PPE 6 kelas (gloves, vest, goggles, helmet, mask, safety_shoe) di Hugging Face: https://huggingface.co/Nexus3/yolov8n-ppe-detection-6classes. Buka tab *Files*, unduh file bobotnya (`.pt`). Nama kelasnya sudah dikenali `ppe_logic.py`. Catatan dari model card: sarung tangan paling lemah (mAP50 sekitar 0.69).

**Opsi B, latih sendiri (disarankan untuk hasil akhir).** Di PC/Colab dengan GPU:
1. Unduh dataset format YOLO yang berisi kelas masker, kacamata/goggles, dan sarung tangan (misalnya dataset PPE di Roboflow Universe atau Hugging Face yang tadi kita bahas). Idealnya tambah 200-500 foto dari kamera dan lokasi kamu sendiri.
2. Buat `data.yaml`:
   ```yaml
   path: /lokasi/dataset
   train: train/images
   val: valid/images
   names:
     0: gloves
     1: goggles
     2: mask
   ```
3. Latih: `python train.py --data data.yaml --epochs 100`
4. Ambil `runs/detect/ppe_k3/weights/best.pt`.

Foto sendiri sangat membantu karena kamera, jarak, dan pencahayaan lokasimu akan sama persis dengan saat dipakai.

## 3. Ekspor ke NCNN (di Pi)
```bash
python export_ncnn.py best.pt --imgsz 416
```
Menghasilkan folder `best_ncnn_model`. Format NCNN paling cepat untuk CPU Pi.

## 4. Jalankan (mode sederhana, satu model)
```bash
python detect.py --model best_ncnn_model --source picam --imgsz 416
```
Opsi berguna:
- `--source 0` untuk webcam USB.
- `--require mask glasses` kalau sarung tangan belum diwajibkan.
- `--gloves-count 2` kalau kedua tangan harus terlihat bersarung tangan.
- `--headless` tanpa jendela (lewat SSH).
- Tekan `q` untuk keluar.

## 5. Tuning kalau hasil kurang pas
| Masalah | Solusi |
|---|---|
| Sarung tangan sering tidak terdeteksi | Naikkan resolusi (`--imgsz 480`), dekatkan kamera, tambah foto sendiri, atau turunkan ambang `gloves` di `DEFAULT_MIN_CONF` |
| Status berkedip | Naikkan `--window` (misal 10) dan `--need` (misal 6) |
| Reaksi terlalu lambat | Turunkan `--window` dan `--need` |
| FPS rendah | Turunkan `--imgsz` (320), `--width/--height`, atau pakai model `n` |
| Salah deteksi masker di benda lain | Naikkan ambang `mask` di `DEFAULT_MIN_CONF` |

## 6. Hubungkan ke perangkat
Fungsi `on_result()` di `detect.py` dipanggil sekali setiap status berubah. Di situ kamu bisa menambahkan buzzer/LED (GPIO) atau publish MQTT ke ESP32 untuk membuka pintu.

## Uji logika (tanpa kamera/model)
```bash
python -m unittest discover -s tests -v
```
