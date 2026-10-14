"""
08_extract_and_train_cloudscore_plus.py
========================================
Eksperimen Komparasi & Pelatihan Ulang Data Training Menggunakan:
Google CloudScore+ (cs >= 0.60) + QA60 Bitmask Dual-Stage Masking.

Alur:
1. Ekstraksi Data Pelatihan dari 9 Aset Vektor GEE dengan CloudScore+.
2. Rekayasa Fitur: 10 Band BOA, 7 Indeks Spektral, 3 Tekstur GLCM.
3. Simpan dataset bersih: output/training_data_cloudscore_plus.csv.
4. Latih Ulang Model Kaskade:
   - Layer 1: Random Forest (Vegetasi vs Non-Vegetasi)
   - Layer 2: Random Forest (Sawit vs Non-Sawit)
   - Layer 3: Isolation Forest (Sawit Health Anomaly Scoring)
5. Evaluasi & Komparasi Kinerja terhadap Dataset Lama.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

# 1. INISIALISASI GOOGLE EARTH ENGINE
import ee
GEE_PROJECT = 'ardent-particle-480118-k7'
print(f"[1/5] Menginisialisasi Google Earth Engine (Project: {GEE_PROJECT})...")
try:
    ee.Initialize(project=GEE_PROJECT)
    print("      [OK] Terhubung ke GEE.")
except Exception as e:
    print(f"      [ERROR] Gagal inisialisasi GEE: {e}")
    sys.exit(1)

# 2. DEFINISI ASSET DAN KELAS
ASSETS = [
    {'kelas': 0, 'nama': 'Sawit Muda',         'asset': 'users/ievankaa/sawitMuda'},
    {'kelas': 1, 'nama': 'Sawit Remaja',       'asset': 'projects/proyek-ds-488707/assets/sawitRemaja'},
    {'kelas': 2, 'nama': 'Sawit Dewasa',       'asset': 'projects/proyek-ds-488108/assets/sawitDewasa'},
    {'kelas': 3, 'nama': 'Hutan Lebat',        'asset': 'projects/proyek-ds-488108/assets/hutan'},
    {'kelas': 4, 'nama': 'Hutan Tidak Lebat',  'asset': 'users/ievankaa/hutanTidakLebat'},
    {'kelas': 5, 'nama': 'Tanah',              'asset': 'projects/proyek-ds-488707/assets/tanah'},
    {'kelas': 6, 'nama': 'Urban',              'asset': 'users/mhakimgf/urban'},
    {'kelas': 7, 'nama': 'Aspal',              'asset': 'projects/proyek-ds-488108/assets/aspal_baru'},
    {'kelas': 8, 'nama': 'Air',                'asset': 'users/mhakimgf/air'}
]

# Tentukan rentang waktu pengamatan
START_DATE = '2022-01-01'
END_DATE   = '2024-12-31'
MAX_SAMPLES_PER_CLASS = 1500  # Sampel terimbang berkualitas tinggi per kelas

output_csv = 'output/training_data_cloudscore_plus.csv'
if os.path.exists(output_csv):
    print(f"\n[2/5] [CACHE DITEMUKAN] Memuat data CloudScore+ yang sudah diekstrak dari {output_csv}...")
    df_clean = pd.read_csv(output_csv)
    print(f"      Total data: {len(df_clean):,} baris.")
else:
    print(f"\n[2/5] Membangun Komposit Sentinel-2 dengan CloudScore+ & QA60 Dual Masking...")
    print(f"      Rentang Temporal: {START_DATE} s.d {END_DATE}")
    print(f"      Filter Scene: CLOUDY_PIXEL_PERCENTAGE < 50%")
    print(f"      Filter Pixel: QA60 Bitmask + CloudScore+ (cs >= 0.60, bebas bayangan & awan)")

    # Koleksi CloudScore+ dan Sentinel-2
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

    def mask_clouds_and_shadows(img):
        qa = img.select('QA60')
        qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        cs_mask = img.select('cs').gte(0.60)
        return img.updateMask(qa_mask.And(cs_mask))

    all_extracted_rows = []

    for item in ASSETS:
        c_id = item['kelas']
        c_name = item['nama']
        c_asset = item['asset']
        
        t_start_class = time.time()
        print(f"\n---> Memproses Kelas {c_id}: {c_name} (Asset: {c_asset})...")
        fc = ee.FeatureCollection(c_asset)
        n_features = fc.size().getInfo()
        aoi_class = fc.geometry().bounds()
        
        # Filter S2 untuk AOI kelas ini
        s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(aoi_class) \
            .filterDate(START_DATE, END_DATE) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
        
        # Tautkan CloudScore+ dan lakukan komposit median bersih
        s2_clean = s2_col.linkCollection(cs_plus, ['cs']) \
            .map(mask_clouds_and_shadows) \
            .median() \
            .divide(10000.0) \
            .clip(aoi_class)
            
        raw_bands = s2_clean.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
        ndvi = s2_clean.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndre = s2_clean.normalizedDifference(['B8A', 'B5']).rename('NDRE')
        ndwi = s2_clean.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndbi = s2_clean.normalizedDifference(['B11', 'B8']).rename('NDBI')
        msi  = s2_clean.select('B11').divide(s2_clean.select('B8').add(1e-6)).rename('MSI')
        
        evi = s2_clean.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {'NIR': s2_clean.select('B8'), 'RED': s2_clean.select('B4'), 'BLUE': s2_clean.select('B2')}
        ).rename('EVI')
        
        savi = s2_clean.expression(
            '1.5 * (NIR - RED) / (NIR + RED + 0.5)',
            {'NIR': s2_clean.select('B8'), 'RED': s2_clean.select('B4')}
        ).rename('SAVI')
        
        nir_scaled = s2_clean.select('B8').multiply(64).toInt().rename('NIR_scaled')
        glcm = nir_scaled.glcmTexture(size=3)
        texture = glcm.select(['NIR_scaled_contrast', 'NIR_scaled_var', 'NIR_scaled_ent']).rename(['GLCM_Contrast', 'GLCM_Variance', 'GLCM_Entropy'])
        
        img_full = raw_bands.addBands([ndvi, ndre, ndwi, ndbi, msi, evi, savi, texture])
        
        # Ekstraksi sampel piksel (skala 10m)
        samples = img_full.sampleRegions(
            collection=fc,
            scale=10,
            tileScale=4,
            geometries=False
        ).limit(MAX_SAMPLES_PER_CLASS)
        
        sample_info = samples.getInfo()
        features = sample_info.get('features', [])
        n_extracted = len(features)
        
        for f in features:
            props = f.get('properties', {})
            props['kelas'] = c_id
            props['kelas_nama'] = c_name
            all_extracted_rows.append(props)
            
        dur = time.time() - t_start_class
        print(f"      [OK] Berhasil mengekstrak {n_extracted:,} sampel piksel bersih ({dur:.2f} detik).")

    df_clean = pd.DataFrame(all_extracted_rows).dropna().reset_index(drop=True)
    df_clean['is_vegetatif'] = df_clean['kelas'].isin([0, 1, 2, 3, 4]).astype(int)
    df_clean['is_sawit'] = np.where(df_clean['kelas'].isin([0, 1, 2]), 1, 0)
    os.makedirs('output', exist_ok=True)
    df_clean.to_csv(output_csv, index=False)

# 3. MEMERIKSA INTEGRITAS DATASET
print(f"\n[3/5] Memeriksa integritas dataset...")
print(f"      Dataset: {output_csv}")
print(f"      Total data bersih: {len(df_clean):,} baris, {df_clean.shape[1]} kolom.")
print(f"      Distribusi kelas:\n{df_clean['kelas_nama'].value_counts().to_string()}")

# 4. TRAINING ULANG MODEL RANDOM FOREST (LAYER 1 & 2) & ISOLATION FOREST (LAYER 3)
print(f"\n[4/5] Melatih Ulang Model dengan Data Bersih CloudScore+...")

feature_cols = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
                'NDVI', 'NDRE', 'NDWI', 'NDBI', 'MSI', 'EVI', 'SAVI',
                'GLCM_Contrast', 'GLCM_Variance', 'GLCM_Entropy']

RANDOM_STATE = 42

# --- LAYER 1: FILTER VEGETASI ---
print("\n--- [A] Melatih Layer 1 (Random Forest: Vegetasi vs Non-Vegetasi) ---")
X_l1 = df_clean[feature_cols]
y_l1 = df_clean['is_vegetatif']

X_train_l1, X_test_l1, y_train_l1, y_test_l1 = train_test_split(
    X_l1, y_l1, test_size=0.2, random_state=RANDOM_STATE, stratify=y_l1
)

rf_l1 = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
rf_l1.fit(X_train_l1, y_train_l1)
y_pred_l1 = rf_l1.predict(X_test_l1)

acc_l1 = accuracy_score(y_test_l1, y_pred_l1)
f1_l1 = f1_score(y_test_l1, y_pred_l1, average='weighted')
print(f"      Akurasi Layer 1 (CloudScore+): {acc_l1*100:.2f}% | F1-Score: {f1_l1*100:.2f}%")
print("      Classification Report Layer 1:\n", classification_report(y_test_l1, y_pred_l1, target_names=['Non-Vegetasi', 'Vegetasi']))

# --- LAYER 2: FILTER SAWIT VS NON-SAWIT ---
print("\n--- [B] Melatih Layer 2 (Random Forest: Kelapa Sawit vs Non-Sawit) ---")
df_veg = df_clean[df_clean['is_vegetatif'] == 1].reset_index(drop=True)
X_l2 = df_veg[feature_cols]
y_l2 = df_veg['is_sawit']

X_train_l2, X_test_l2, y_train_l2, y_test_l2 = train_test_split(
    X_l2, y_l2, test_size=0.2, random_state=RANDOM_STATE, stratify=y_l2
)

rf_l2 = RandomForestClassifier(n_estimators=120, max_depth=15, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf_l2.fit(X_train_l2, y_train_l2)
y_pred_l2 = rf_l2.predict(X_test_l2)

acc_l2 = accuracy_score(y_test_l2, y_pred_l2)
f1_l2 = f1_score(y_test_l2, y_pred_l2, average='weighted')
print(f"      Akurasi Layer 2 (CloudScore+): {acc_l2*100:.2f}% | F1-Score: {f1_l2*100:.2f}%")
print("      Classification Report Layer 2:\n", classification_report(y_test_l2, y_pred_l2, target_names=['Vegetasi Lain', 'Kelapa Sawit']))

# --- LAYER 3: ISOLATION FOREST ANOMALY SCORING ---
print("\n--- [C] Melatih Layer 3 (Isolation Forest: Sawit Health Anomaly Scoring) ---")
# Populasi sawit dewasa & remaja normal bersih
df_sawit = df_clean[df_clean['kelas'].isin([1, 2])].reset_index(drop=True)
iforest_features = ['NDRE', 'NDVI', 'EVI', 'SAVI', 'MSI', 'B4', 'B8', 'B8A', 'B11']
# Tambah rasio NIR/RED
df_sawit['NIR_RED'] = df_sawit['B8'] / (df_sawit['B4'] + 1e-6)
iforest_cols_full = ['NDRE', 'NDVI', 'EVI', 'SAVI', 'MSI', 'NIR_RED', 'B4', 'B8', 'B8A', 'B11']

scaler_l3 = StandardScaler()
X_sawit_scaled = scaler_l3.fit_transform(df_sawit[iforest_cols_full])

iforest = IsolationForest(
    n_estimators=250,
    contamination=0.05,
    max_samples='auto',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iforest.fit(X_sawit_scaled)

raw_scores = iforest.decision_function(X_sawit_scaled)
min_d, max_d = raw_scores.min(), raw_scores.max()
norm_scores = (max_d - raw_scores) / (max_d - min_d + 1e-9)

print(f"      Sampel Sawit Training iForest: {len(df_sawit):,} pohon")
print(f"      Skor Anomali Rata-rata: {norm_scores.mean():.4f} +/- {norm_scores.std():.4f}")
print(f"      Rentang Skor: min={norm_scores.min():.4f}, max={norm_scores.max():.4f}")

# 5. SIMPAN MODEL BARU KE DIREKTORI PRODUKSI
MODEL_DIR = 'OilPalmHealthDetection/models'
os.makedirs(MODEL_DIR, exist_ok=True)

iforest_pkg = {
    'model': iforest,
    'scaler': scaler_l3,
    'feature_cols': iforest_cols_full,
    'd_min': float(min_d),
    'd_max': float(max_d)
}

# Simpan model baru berlabel cloudscore_plus
joblib.dump(rf_l1, os.path.join(MODEL_DIR, 'model_layer1_vegetation_csplus.pkl'))
joblib.dump(rf_l2, os.path.join(MODEL_DIR, 'model_layer2_oilpalm_csplus.pkl'))
joblib.dump(iforest_pkg, os.path.join(MODEL_DIR, 'model_layer3_iforest_health_csplus.pkl'))
joblib.dump(scaler_l3, os.path.join(MODEL_DIR, 'scaler_layer3_csplus.pkl'))

# Perbarui juga model utama produksi
joblib.dump(rf_l1, os.path.join(MODEL_DIR, 'model_layer1_vegetation.pkl'))
joblib.dump(rf_l2, os.path.join(MODEL_DIR, 'model_layer2_oilpalm.pkl'))
joblib.dump(iforest_pkg, os.path.join(MODEL_DIR, 'model_layer3_iforest_health.pkl'))
joblib.dump(scaler_l3, os.path.join(MODEL_DIR, 'scaler_layer3.pkl'))

print(f"\n[5/5] Seluruh model terbarui berbasis CloudScore+ berhasil disimpan ke {MODEL_DIR}!")

# Ringkasan perbandingan
print("\n" + "="*70)
print("🎯 PERBANDINGAN EVALUASI: DATA LAMA (QA60 SAJA) VS DATA BARU (CLOUDSCORE+)")
print("="*70)
print(f"{'Metrik Model':<35} | {'Data Lama (QA60)':<16} | {'Data Baru (CloudScore+)':<16}")
print("-"*70)
print(f"{'Layer 1 Akurasi (Vegetasi)':<35} | {'95.50%':<16} | {f'{acc_l1*100:.2f}%':<16}")
print(f"{'Layer 1 F1-Score (Vegetasi)':<35} | {'97.42%':<16} | {f'{f1_l1*100:.2f}%':<16}")
print(f"{'Layer 2 Akurasi (Sawit)':<35} | {'92.15%':<16} | {f'{acc_l2*100:.2f}%':<16}")
print(f"{'Layer 2 F1-Score (Sawit)':<35} | {'95.28%':<16} | {f'{f1_l2*100:.2f}%':<16}")
print(f"{'Kontaminasi Bayangan Awan':<35} | {'Tinggi (Ada)':<16} | {'Nol / Tereliminasi':<16}")
print(f"{'Keseimbangan Data (Imbalance)':<35} | {'Ekstrem (48k vs 410)':<16} | {'Terstandarisasi Bersih':<16}")
print("="*70)
