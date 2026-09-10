"""
Pipeline Inferensi 2-Layer Machine Learning (Random Forest) + Health Assessment
==============================================================================
Menggunakan Random Forest Classifier untuk klasifikasi hierarkis:
1. Layer 1 (Random Forest): Vegetasi vs Non-Vegetasi
2. Layer 2 (Random Forest): Sawit vs Hutan / Non-Sawit
3. Layer 3 (Health Module): Multi-Spectral Vigor Score (MVS) dengan penjagaan
   boundary/mixed-pixel untuk mencegah false positive di sekitar perbatasan non-vegetasi.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load model Random Forest dan kalibrasi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_L1_PATH = os.path.join(BASE_DIR, 'models', 'model_layer1_rf_vegetasi.pkl')
MODEL_L2_PATH = os.path.join(BASE_DIR, 'models', 'model_layer2_rf_sawit.pkl')
CALIB_PATH    = os.path.join(BASE_DIR, 'models', 'health_calibration.json')

# Fallback ke XGBoost jika RF belum ada
if not os.path.exists(MODEL_L1_PATH):
    MODEL_L1_PATH = os.path.join(BASE_DIR, 'models', 'model_layer1_vegetasi.pkl')
if not os.path.exists(MODEL_L2_PATH):
    MODEL_L2_PATH = os.path.join(BASE_DIR, 'models', 'model_layer2_sawit.pkl')

model_l1 = joblib.load(MODEL_L1_PATH)
model_l2 = joblib.load(MODEL_L2_PATH)

with open(CALIB_PATH, 'r') as f:
    calib = json.load(f)

FEATURES = calib['features']

def ensure_features(df_input):
    """Memastikan seluruh 20 fitur input tersedia dan terisi dengan benar."""
    df = df_input.copy()
    
    # Hitung indeks vegetasi & air jika belum ada tapi band tersedia
    if 'MSI' not in df.columns and 'B11' in df.columns and 'B8' in df.columns:
        df['MSI'] = df['B11'] / (df['B8'] + 1e-6)
    if 'NDVI' not in df.columns and 'B8' in df.columns and 'B4' in df.columns:
        df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-6)
    if 'NDRE' not in df.columns and 'B8' in df.columns and 'B5' in df.columns:
        df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'] + 1e-6)
    if 'NDBI' not in df.columns and 'B11' in df.columns and 'B8' in df.columns:
        df['NDBI'] = (df['B11'] - df['B8']) / (df['B11'] + df['B8'] + 1e-6)
    if 'NDWI' not in df.columns and 'B3' in df.columns and 'B8' in df.columns:
        df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + 1e-6)
    if 'EVI' not in df.columns and all(b in df.columns for b in ['B8', 'B4', 'B2']):
        df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 1.0)
    if 'SAVI' not in df.columns and all(b in df.columns for b in ['B8', 'B4']):
        df['SAVI'] = 1.5 * (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)

    # Fallback tekstur GLCM jika belum terhitung
    default_textures = {
        'GLCM_Contrast': 0.84,
        'GLCM_Entropy': 2.10,
        'GLCM_Variance': 0.72
    }
    for col, default_val in default_textures.items():
        if col not in df.columns:
            df[col] = default_val

    # Pastikan seluruh kolom fitur ada (isi 0 jika ada band tersisa yang hilang)
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df

def predict_pipeline(df_input, return_timing=False):
    """
    Menerima DataFrame fitur Sentinel-2 dan memprediksi status kesehatan
    menggunakan model Random Forest 2-Layer dengan eliminasi false positive border.
    
    Returns:
        df_res: DataFrame dengan kolom pred_layer1, pred_layer2, health_status, vigor_score
        (opsional) timing: dict waktu inferensi (ms, sec, ms/sample)
    """
    t0 = time.time()
    df_prepared = ensure_features(df_input)
    df_res = df_input.copy()
    
    X = df_prepared[FEATURES]
    
    # Layer 1: Vegetatif vs Non-Vegetatif (Random Forest)
    is_veg = model_l1.predict(X)
    df_res['pred_layer1'] = np.where(is_veg == 1, 'Vegetasi', 'Non-Vegetasi')
    
    # Layer 2: Sawit vs Non-Sawit (Random Forest)
    df_res['pred_layer2'] = 'Bukan Sawit'
    veg_mask = (is_veg == 1)
    if veg_mask.sum() > 0:
        is_sawit = model_l2.predict(X[veg_mask])
        df_res.loc[veg_mask, 'pred_layer2'] = np.where(is_sawit == 1, 'Sawit', 'Hutan/Non-Sawit')
        
    # Layer 3: Health Assessment (Statistical Anomaly & Biofisika Guardrail)
    df_res['health_status'] = df_res['pred_layer2']
    df_res['vigor_score'] = np.nan
    sawit_mask = (df_res['pred_layer2'] == 'Sawit')
    
    if sawit_mask.sum() > 0:
        sub_sawit = df_prepared[sawit_mask]
        z_ndre = (sub_sawit['NDRE'] - calib['mean_ndre']) / calib['std_ndre']
        z_ndvi = (sub_sawit['NDVI'] - calib['mean_ndvi']) / calib['std_ndvi']
        z_msi  = (sub_sawit['MSI']  - calib['mean_msi'])  / calib['std_msi']
        vigor  = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
        
        health_list = []
        for idx, row in sub_sawit.iterrows():
            v = vigor.loc[idx]
            v_ndre = row['NDRE']
            v_ndvi = row['NDVI']
            v_msi = row['MSI']
            v_ndbi = row.get('NDBI', -0.3)
            v_b4 = row.get('B4', 0.03)
            v_b8 = row.get('B8', 0.3)
            nir_red = v_b8 / (v_b4 + 1e-6)
            
            # 1. Guardrail Perbatasan Non-Vegetasi (Jalan / Rumah / Tanah Terbuka / Bahu Jalan)
            # Jika NDBI tinggi atau rasio NIR/Red rendah, ini adalah piksel campuran tepi kebun!
            if (v_ndbi > -0.12) or (nir_red < 4.0) or (v_b4 > 0.065):
                health_list.append('Non-Sawit (Bahu Jalan / Pekarangan / Semak)')
            # 2. Sawit Muda / Tajuk Terbuka (kanopi belum rapat)
            elif (v_ndvi < 0.70 and v_ndre < 0.48) or (v_ndbi > -0.18):
                health_list.append('Sawit Muda / Terbuka')
            # 3. Sawit Sakit Sejati (Harus kanopi sawit dewasa rapat, tapi klorosis berat & tajuk dehidrasi)
            elif (v_ndre < 0.44 and v_ndvi < 0.72 and v_msi > 0.65) and (v < -2.0) and (v_ndbi <= -0.18):
                health_list.append('🔴 SAWIT SAKIT (Target Karantina)')
            # 4. Sawit Sehat Prima
            elif v >= -0.5 and v_ndvi >= 0.76 and v_ndre >= 0.52:
                health_list.append('🟢 Sawit Sehat Prima')
            # 5. Stres Ringan (Defisiensi Hara / Fluktuasi Musim)
            else:
                health_list.append('🟡 Sawit Stres Ringan (Defisiensi Hara)')
                
        df_res.loc[sawit_mask, 'health_status'] = health_list
        df_res.loc[sawit_mask, 'vigor_score'] = vigor.values
        
    t1 = time.time()
    elapsed_sec = t1 - t0
    elapsed_ms = elapsed_sec * 1000
    
    timing = {
        'total_samples': len(df_res),
        'elapsed_sec': round(elapsed_sec, 4),
        'elapsed_ms': round(elapsed_ms, 2),
        'ms_per_sample': round(elapsed_ms / max(1, len(df_res)), 3),
        'model_name': 'Random Forest 2-Layer'
    }
    
    if return_timing:
        return df_res, timing
    return df_res

if __name__ == '__main__':
    # Uji coba benchmark inferensi
    csv_paths = ['output/training_data_all_features.csv', '../output/training_data_all_features.csv']
    chosen_path = next((p for p in csv_paths if os.path.exists(p)), None)
    
    if chosen_path:
        print(f">> Menjalankan Benchmark Inferensi Random Forest pada {chosen_path}...")
        sample_df = pd.read_csv(chosen_path, nrows=2500)
        res, timing = predict_pipeline(sample_df, return_timing=True)
        print(f"\n=======================================================")
        print(f"  HASIL BENCHMARK KECEPATAN MODEL RANDOM FOREST")
        print(f"=======================================================")
        print(f"  Model Engine    : {timing['model_name']}")
        print(f"  Jumlah Sampel   : {timing['total_samples']:,} titik")
        print(f"  Total Waktu     : {timing['elapsed_sec']} detik ({timing['elapsed_ms']} ms)")
        print(f"  Kecepatan/Titik : {timing['ms_per_sample']} ms per pohon")
        print(f"\nDistribusi Prediksi:")
        for k, v in res['health_status'].value_counts().items():
            print(f"  {k}: {v}")
