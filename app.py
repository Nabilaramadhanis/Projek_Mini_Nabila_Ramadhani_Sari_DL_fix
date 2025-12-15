"""
SISTEM DETEKSI DIABETES - WEB APPLICATION
==========================================
Aplikasi web Flask untuk prediksi diabetes menggunakan:
1. Support Vector Machine (SVM)
2. Random Forest Classifier
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ===================================
# LOAD MODEL DAN SCALER
# ===================================
print("Loading models...")

try:
    # Load SVM model
    with open('model/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    print("✓ SVM model loaded")
    
    # Load Random Forest model
    with open('model/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    print("✓ Random Forest model loaded")
    
    # Load scaler
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
    
    # Load feature names
    with open('model/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    print(f"✓ Feature names loaded: {len(feature_names)} features")
    
except FileNotFoundError as e:
    print("✗ Error: Model files not found!")
    print("  Please run 'python train_model.py' first to train the models")
    exit()

# ===================================
# ROUTE: HALAMAN UTAMA
# ===================================
@app.route('/')
def home():
    """
    Halaman utama dengan form input
    """
    return render_template('index.html', features=feature_names)

# ===================================
# ROUTE: PREDIKSI
# ===================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi
    Input: Data dari form
    Output: Hasil prediksi SVM dan Random Forest
    """
    try:
        # Ambil data dari form
        input_data = []
        for feature in feature_names:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html', 
                                     features=feature_names,
                                     error=f"Error: Mohon isi semua field. Field '{feature}' kosong.")
            try:
                input_data.append(float(value))
            except ValueError:
                return render_template('index.html', 
                                     features=feature_names,
                                     error=f"Error: Field '{feature}' harus berupa angka.")
        
        # Konversi ke numpy array
        input_array = np.array(input_data).reshape(1, -1)
        
        # Standarisasi data input
        input_scaled = scaler.transform(input_array)
        
        # ===================================
        # ENSEMBLE SOFT VOTING
        # ===================================
        # Langkah 1: Dapatkan probabilitas dari SVM
        svm_probability = svm_model.predict_proba(input_scaled)[0]
        prob_svm_positive = svm_probability[1]  # Probabilitas kelas 1 (positif diabetes)
        
        # Langkah 2: Dapatkan probabilitas dari Random Forest
        rf_probability = rf_model.predict_proba(input_scaled)[0]
        prob_rf_positive = rf_probability[1]  # Probabilitas kelas 1 (positif diabetes)
        
        # Langkah 3: Hitung probabilitas akhir dengan metode Soft Voting
        # Rumus: rata-rata probabilitas kedua model
        final_probability = (prob_svm_positive + prob_rf_positive) / 2
        
        # Langkah 4: Tentukan prediksi akhir berdasarkan threshold 0.5
        if final_probability >= 0.5:
            final_prediction = "POSITIF DIABETES"
            result_color = "danger"
            result_icon = "⚠️"
        else:
            final_prediction = "NEGATIF DIABETES"
            result_color = "success"
            result_icon = "✅"
        
        # Konversi ke persentase untuk tampilan
        final_confidence = final_probability * 100
        
        # ===================================
        # DETAIL PROBABILITAS (OPSIONAL)
        # ===================================
        # Detail ini untuk transparansi, menunjukkan kontribusi masing-masing model
        svm_prob_percent = prob_svm_positive * 100
        rf_prob_percent = prob_rf_positive * 100
        
        # Kirim hasil ke template
        return render_template('index.html',
                             features=feature_names,
                             input_values=dict(zip(feature_names, input_data)),
                             # Hasil Ensemble
                             final_prediction=final_prediction,
                             final_confidence=f"{final_confidence:.2f}",
                             result_color=result_color,
                             result_icon=result_icon,
                             # Detail Model (Opsional)
                             svm_prob=f"{svm_prob_percent:.2f}",
                             rf_prob=f"{rf_prob_percent:.2f}")
    
    except Exception as e:
        return render_template('index.html',
                             features=feature_names,
                             error=f"Error: {str(e)}")

# ===================================
# JALANKAN APLIKASI
# ===================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("SISTEM DETEKSI DIABETES - WEB APP")
    print("="*50)
    print("Server berjalan di: http://127.0.0.1:5000")
    print("Tekan CTRL+C untuk menghentikan server")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
