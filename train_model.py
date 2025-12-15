"""
SISTEM DETEKSI DIABETES - TRAINING MODEL
==========================================
Script ini digunakan untuk:
1. Membaca dataset
2. Preprocessing data
3. Melatih model SVM dan Random Forest
4. Evaluasi model
5. Menyimpan model dan scaler
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("="*50)
print("SISTEM DETEKSI DIABETES - TRAINING MODEL")
print("="*50)

# ===================================
# 1. MEMBACA DATASET
# ===================================
print("\n[1] Membaca dataset...")
try:
    df = pd.read_csv('dataset/diabetes.csv')
    print(f"✓ Dataset berhasil dibaca!")
    print(f"  Jumlah data: {len(df)} baris")
    print(f"  Jumlah kolom: {len(df.columns)}")
    print(f"\nNama kolom:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
except FileNotFoundError:
    print("✗ Error: File 'dataset/diabetes.csv' tidak ditemukan!")
    print("  Pastikan file diabetes.csv ada di folder 'dataset/'")
    exit()

# ===================================
# 2. DETEKSI KOLOM TARGET
# ===================================
print("\n[2] Mendeteksi kolom target...")
target_col = None

# Cari kolom dengan nilai unik hanya 0 dan 1
for col in df.columns:
    unique_values = df[col].dropna().unique()
    if len(unique_values) == 2 and set(unique_values) == {0, 1}:
        target_col = col
        break

if target_col is None:
    # Jika tidak ada, cek kolom yang memiliki nilai 0 dan 1 (tapi mungkin ada nilai lain)
    for col in df.columns:
        unique_values = df[col].dropna().unique()
        if 0 in unique_values and 1 in unique_values:
            target_col = col
            break

if target_col:
    print(f"✓ Kolom target terdeteksi: '{target_col}'")
    print(f"  Distribusi target:")
    print(f"  - Kelas 0 (Negatif): {len(df[df[target_col]==0])} data")
    print(f"  - Kelas 1 (Positif): {len(df[df[target_col]==1])} data")
else:
    print("✗ Error: Kolom target tidak terdeteksi!")
    print("  Pastikan ada kolom dengan nilai 0 dan 1")
    exit()

# ===================================
# 3. PREPROCESSING DATA
# ===================================
print("\n[3] Preprocessing data...")

# Pisahkan fitur dan target
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"  Jumlah fitur: {X.shape[1]}")
print(f"  Nama fitur: {list(X.columns)}")

# Cek missing value
missing_values = X.isnull().sum().sum()
print(f"\n  Missing values: {missing_values}")

if missing_values > 0:
    print("  → Mengisi missing values dengan mean...")
    X = X.fillna(X.mean())
    print("  ✓ Missing values telah ditangani")

# Cek missing value di target
if y.isnull().sum() > 0:
    print("  → Menghapus baris dengan target missing...")
    valid_indices = ~y.isnull()
    X = X[valid_indices]
    y = y[valid_indices]
    print(f"  ✓ Data valid: {len(y)} baris")

# Split data
print("\n  Split data: 80% training, 20% testing...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  ✓ Data training: {len(X_train)} baris")
print(f"  ✓ Data testing: {len(X_test)} baris")

# Standarisasi fitur
print("\n  Standarisasi fitur menggunakan StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Standarisasi selesai")

# ===================================
# 4. TRAINING MODEL SVM
# ===================================
print("\n[4] Training model SVM...")
print("  Parameter: kernel=RBF, probability=True")

svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("  ✓ Training SVM selesai")

# Prediksi dan evaluasi
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

print(f"\n  Hasil Evaluasi SVM:")
print(f"  Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, svm_pred, target_names=['Negatif', 'Positif']))

# ===================================
# 5. TRAINING MODEL RANDOM FOREST
# ===================================
print("\n[5] Training model Random Forest...")
print("  Parameter: n_estimators=100, random_state=42")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("  ✓ Training Random Forest selesai")

# Prediksi dan evaluasi
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"\n  Hasil Evaluasi Random Forest:")
print(f"  Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, rf_pred, target_names=['Negatif', 'Positif']))

# ===================================
# 6. MENYIMPAN MODEL DAN SCALER
# ===================================
print("\n[6] Menyimpan model dan scaler...")

# Buat folder model jika belum ada
os.makedirs('model', exist_ok=True)

# Simpan SVM model
with open('model/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("  ✓ SVM model disimpan: model/svm_model.pkl")

# Simpan Random Forest model
with open('model/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("  ✓ Random Forest model disimpan: model/rf_model.pkl")

# Simpan scaler
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  ✓ Scaler disimpan: model/scaler.pkl")

# Simpan nama fitur untuk keperluan web app
feature_names = list(X.columns)
with open('model/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("  ✓ Nama fitur disimpan: model/feature_names.pkl")

# ===================================
# RINGKASAN
# ===================================
print("\n" + "="*50)
print("RINGKASAN")
print("="*50)
print(f"Dataset: {len(df)} data, {len(X.columns)} fitur")
print(f"Target: {target_col}")
print(f"\nAkurasi Model:")
print(f"  • SVM (RBF): {svm_accuracy*100:.2f}%")
print(f"  • Random Forest: {rf_accuracy*100:.2f}%")
print(f"\nModel berhasil disimpan di folder 'model/'")
print("Anda sekarang bisa menjalankan aplikasi web dengan: python app.py")
print("="*50)
