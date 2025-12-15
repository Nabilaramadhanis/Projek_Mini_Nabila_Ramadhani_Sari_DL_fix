# 🏥 Sistem Deteksi Diabetes Berbasis Web

Aplikasi web sederhana untuk memprediksi diabetes menggunakan Machine Learning dengan metode **Ensemble Soft Voting** yang menggabungkan dua algoritma:
1. **Support Vector Machine (SVM)** - dengan kernel RBF
2. **Random Forest Classifier** - dengan 100 estimators

**Metode Ensemble:** Prediksi akhir dihitung dari rata-rata probabilitas kedua model untuk meningkatkan akurasi dan stabilitas prediksi.

## 📋 Deskripsi Proyek

Sistem ini dibuat menggunakan Python, Flask, dan Scikit-learn untuk membantu mendeteksi kemungkinan diabetes berdasarkan data medis pasien. Aplikasi ini cocok untuk:
- Pembelajaran Machine Learning
- Proyek akademik/tugas akhir
- Demo sistem prediksi kesehatan

## 🗂️ Struktur Proyek

```
diabetes_web_app/
│
├── dataset/                 # Folder dataset
│   └── diabetes.csv        # Dataset diabetes (CSV)
│
├── model/                   # Folder untuk menyimpan model
│   ├── svm_model.pkl       # Model SVM
│   ├── rf_model.pkl        # Model Random Forest
│   ├── scaler.pkl          # StandardScaler
│   └── feature_names.pkl   # Nama fitur
│
├── templates/               # Folder template HTML
│   └── index.html          # Halaman web utama
│
├── app.py                   # Aplikasi Flask (Backend)
├── train_model.py          # Script training model
├── requirements.txt        # Dependencies Python
└── README.md               # Dokumentasi (file ini)
```

## 🚀 Cara Menjalankan Aplikasi

### **Langkah 1: Persiapan Dataset**

Pastikan Anda memiliki file **`diabetes.csv`** di folder **`dataset/`**. Dataset harus memiliki:
- Beberapa kolom fitur (numerik)
- 1 kolom target dengan nilai **0** (negatif) dan **1** (positif)

Contoh format dataset:
```csv
Glucose,BloodPressure,BMI,Age,Outcome
148,72,33.6,50,1
85,66,26.6,31,0
...
```

### **Langkah 2: Install Dependencies**

Buka terminal/command prompt di folder proyek, lalu jalankan:

```bash
pip install -r requirements.txt
```

### **Langkah 3: Training Model**

Jalankan script training untuk melatih model SVM dan Random Forest:

```bash
python train_model.py
```

**Output yang diharapkan:**
- Model akan dilatih menggunakan dataset Anda
- Menampilkan accuracy dan classification report
- Menyimpan model ke folder `model/`

**Contoh output:**
```
==================================================
SISTEM DETEKSI DIABETES - TRAINING MODEL
==================================================

[1] Membaca dataset...
✓ Dataset berhasil dibaca!
  Jumlah data: 768 baris
  Jumlah kolom: 9

[2] Mendeteksi kolom target...
✓ Kolom target terdeteksi: 'Outcome'
  - Kelas 0 (Negatif): 500 data
  - Kelas 1 (Positif): 268 data

[3] Preprocessing data...
...

[4] Training model SVM...
  Accuracy: 0.7727 (77.27%)

[5] Training model Random Forest...
  Accuracy: 0.7532 (75.32%)

[6] Menyimpan model dan scaler...
✓ Semua model berhasil disimpan
```

### **Langkah 4: Jalankan Aplikasi Web**

Setelah model selesai di-training, jalankan aplikasi Flask:

```bash
python app.py
```

### **Langkah 5: Buka di Browser**

Buka browser dan akses:
```
http://127.0.0.1:5000
```

atau

```
http://localhost:5000
```

## 📊 Cara Menggunakan Aplikasi

1. **Isi Form**: Masukkan nilai untuk setiap fitur yang diminta (ada penjelasan di setiap field)
2. **Klik Prediksi**: Tekan tombol "🔍 Prediksi"
3. **Lihat Hasil**: Sistem akan menampilkan:
   - **Hasil Ensemble**: Prediksi akhir (POSITIF/NEGATIF) dengan confidence score
   - **Detail Model**: Probabilitas dari masing-masing model (SVM & Random Forest)
   - **Rumus Perhitungan**: Cara sistem menggabungkan kedua model

## 🧠 Algoritma yang Digunakan

### 1. Support Vector Machine (SVM)
- **Kernel**: RBF (Radial Basis Function)
- **Parameter**: `probability=True` untuk mendapatkan probabilitas
- **Keunggulan**: Efektif untuk dataset berdimensi tinggi

### 2. Random Forest Classifier
- **Jumlah Trees**: 100 estimators
- **Random State**: 42 (untuk reproducibility)
- **Keunggulan**: Robust terhadap overfitting, dapat menangani fitur non-linear

### 3. Ensemble Soft Voting (Metode Gabungan)
- **Cara Kerja**: Menggabungkan probabilitas dari SVM dan Random Forest
- **Rumus**: 
  ```
  Probabilitas_Akhir = (Prob_SVM + Prob_RF) / 2
  ```
- **Threshold**: Jika Probabilitas_Akhir ≥ 0.5 → POSITIF, jika < 0.5 → NEGATIF
- **Keunggulan**: 
  - Lebih stabil daripada menggunakan satu model
  - Mengurangi bias dari model individual
  - Meningkatkan generalisasi prediksi
  - Memberikan hasil yang lebih dapat diandalkan

## 🔧 Preprocessing Data

1. **Deteksi Target**: Otomatis mendeteksi kolom dengan nilai 0 dan 1
2. **Handling Missing Values**: Mengisi dengan nilai mean
3. **Standardization**: Menggunakan StandardScaler untuk normalisasi fitur
4. **Split Data**: 80% training, 20% testing (stratified)

## 📈 Evaluasi Model

Setiap model dievaluasi menggunakan:
- **Accuracy Score**: Persentase prediksi yang benar
- **Classification Report**: Precision, Recall, F1-Score untuk setiap kelas

## ⚠️ Catatan Penting

1. **Bukan untuk Diagnosis Medis**: Hasil prediksi hanya sebagai referensi. Selalu konsultasikan dengan dokter untuk diagnosis yang akurat.

2. **Dataset**: Pastikan dataset Anda berkualitas baik dan representatif.

3. **Model Performance**: Akurasi model bergantung pada kualitas dan kuantitas data training.

4. **Port Konflik**: Jika port 5000 sudah digunakan, ubah di file `app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Ganti 5001
   ```

## Troubleshooting

### Error: "File 'dataset.csv' tidak ditemukan"
- Pastikan file `diabetes.csv` ada di folder `dataset/` yang sama dengan `train_model.py`

### Error: "Model files not found"
- Jalankan `python train_model.py` terlebih dahulu sebelum menjalankan `app.py`

### Error: "ModuleNotFoundError"
- Install semua dependencies: `pip install -r requirements.txt`

### Aplikasi tidak bisa diakses
- Pastikan Flask sudah running (tidak ada error di terminal)
- Cek firewall atau antivirus yang mungkin memblokir port 5000
- Coba akses dengan IP: `http://127.0.0.1:5000`

## 📚 Dependencies

- **Flask 2.3.3**: Web framework
- **scikit-learn 1.3.0**: Machine Learning library
- **pandas 2.0.3**: Data manipulation
- **numpy 1.24.3**: Numerical computing

## 👨‍💻 Pengembangan Lebih Lanjut

Anda bisa mengembangkan proyek ini dengan:
- Menambah algoritma ML lainnya (Logistic Regression, KNN, dll)
- Membuat visualisasi data dan hasil prediksi
- Menambah fitur export hasil ke PDF
- Membuat database untuk menyimpan riwayat prediksi
- Improve UI/UX dengan framework modern (Bootstrap, Tailwind)

## 📝 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan akademik.

## 🤝 Kontributor

Dibuat dengan ❤️ menggunakan Python, Flask, dan Machine Learning

---

**Selamat mencoba! Jika ada pertanyaan atau masalah, jangan ragu untuk bertanya.**
