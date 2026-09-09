"""
SANDBOX Experiment 03: End-to-End Hierarchical Pipeline with Palm Health Detection
==================================================================================
Arsitektur 2-Layer + Health Assessment:
1. Layer 1: Filter Vegetasi vs Non-Vegetasi (Masking Non-Vegetasi)
2. Layer 2: Filter Sawit vs Non-Sawit (Memisahkan Sawit dari Hutan/Semak)
3. Layer 3 (Health Assessment):
   - Sawit Muda (Fase Awal)
   - Sawit Sehat Prima (Vigor Optimal)
   - Sawit Stres Ringan (Defisiensi Hara / Awal Stres)
   - Sawit Tidak Sehat (Stres Berat / Ganoderma / Defoliasi)
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('SANDBOX/models', exist_ok=True)
os.makedirs('SANDBOX/plots', exist_ok=True)
os.makedirs('SANDBOX/results', exist_ok=True)

# 1. LOAD DATASET
DATA_PATH = 'output/training_data_all_features.csv'
print(f"[1/6] Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)

# Hitung fitur tambahan stres kanopi
df['MSI'] = df['B11'] / (df['B8'] + 1e-6)

# Feature set optimal (melibatkan Red Edge, SWIR, dan Textures)
FEATURE_COLS = [
    'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
    'NDVI', 'EVI', 'SAVI', 'NDBI', 'NDRE', 'NDWI', 'MSI',
    'GLCM_Contrast', 'GLCM_Entropy', 'GLCM_Variance'
]
print(f"      Fitur yang digunakan ({len(FEATURE_COLS)} fitur): {FEATURE_COLS}")

# 2. DEFINISI TARGET LEVEL 1 & LEVEL 2
# L1: is_vegetatif (0: Non-Veg, 1: Veg)
veg_classes = [0, 1, 2, 3, 4]
df['is_vegetatif'] = df['kelas'].isin(veg_classes).astype(int)

# L2: is_sawit (0: Hutan/Non-Sawit, 1: Sawit)
sawit_classes = [0, 1, 2]
df['is_sawit'] = df['kelas'].isin(sawit_classes).astype(int)

# Split Train-Test 80:20
RANDOM_STATE = 42
train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df['kelas'])
df_train = df.loc[train_idx].copy()
df_test = df.loc[test_idx].copy()

# ==============================================================================
# 3. TRAINING LAYER 1 (Vegetasi vs Non-Vegetasi)
# ==============================================================================
print("\n[2/6] Training Layer 1 (XGBoost): Vegetatif vs Non-Vegetatif...")
model_l1 = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
)
model_l1.fit(df_train[FEATURE_COLS], df_train['is_vegetatif'])

pred_l1_test = model_l1.predict(df_test[FEATURE_COLS])
acc_l1 = accuracy_score(df_test['is_vegetatif'], pred_l1_test)
print(f"      Layer 1 Akurasi: {acc_l1*100:.2f}% | F1: {f1_score(df_test['is_vegetatif'], pred_l1_test)*100:.2f}%")

# ==============================================================================
# 4. TRAINING LAYER 2 (Sawit vs Non-Sawit pada Vegetasi)
# ==============================================================================
print("\n[3/6] Training Layer 2 (XGBoost): Sawit vs Non-Sawit (Vegetatif Only)...")
train_veg = df_train[df_train['is_vegetatif'] == 1]
test_veg = df_test[df_test['is_vegetatif'] == 1]

model_l2 = XGBClassifier(
    n_estimators=120, max_depth=6, learning_rate=0.08,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
)
model_l2.fit(train_veg[FEATURE_COLS], train_veg['is_sawit'])

pred_l2_test = model_l2.predict(test_veg[FEATURE_COLS])
acc_l2 = accuracy_score(test_veg['is_sawit'], pred_l2_test)
print(f"      Layer 2 Akurasi: {acc_l2*100:.2f}% | F1: {f1_score(test_veg['is_sawit'], pred_l2_test)*100:.2f}%")

# ==============================================================================
# 5. HEALTH ASSESSMENT MODULE (Deteksi Sawit Tidak Sehat)
# ==============================================================================
print("\n[4/6] Mengkalibrasi Modul Deteksi Kesehatan Sawit (Health Assessment)...")
# Baseline parameter kesehatan sawit dihitung dari sawit dewasa sehat pada train set
sawit_train_dewasa = train_veg[train_veg['kelas'] == 2]

mean_ndre = sawit_train_dewasa['NDRE'].mean()
std_ndre  = sawit_train_dewasa['NDRE'].std()
mean_ndvi = sawit_train_dewasa['NDVI'].mean()
std_ndvi  = sawit_train_dewasa['NDVI'].std()
mean_msi  = sawit_train_dewasa['MSI'].mean()
std_msi   = sawit_train_dewasa['MSI'].std()

print(f"      Baseline Sawit Dewasa: NDRE={mean_ndre:.3f}±{std_ndre:.3f}, NDVI={mean_ndvi:.3f}±{std_ndvi:.3f}, MSI={mean_msi:.3f}±{std_msi:.3f}")

def compute_health_status(df_input):
    """
    Menghitung skor kesehatan dan kategori untuk sampel sawit:
    - Vigor Score (Z-score gabungan NDRE, NDVI, MSI)
    - Status:
      - 'Sawit Muda' (jika kelas=0 atau kanopi terbuka/NDVI < 0.65 & NDRE < 0.42)
      - 'Sawit Sehat Prima' (Vigor Score >= -0.5)
      - 'Sawit Stres Ringan' (-1.5 <= Vigor Score < -0.5)
      - 'Sawit Tidak Sehat' (Vigor Score < -1.5)
    """
    z_ndre = (df_input['NDRE'] - mean_ndre) / std_ndre
    z_ndvi = (df_input['NDVI'] - mean_ndvi) / std_ndvi
    z_msi  = (df_input['MSI'] - mean_msi) / std_msi
    
    vigor_score = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
    
    status = []
    for idx, row in df_input.iterrows():
        v = vigor_score.loc[idx]
        if row['NDVI'] < 0.65 and row['NDRE'] < 0.42:
            status.append('Sawit Muda / Kanopi Terbuka')
        elif v >= -0.5:
            status.append('Sawit Sehat Prima')
        elif v >= -1.5:
            status.append('Sawit Stres Ringan')
        else:
            status.append('Sawit Tidak Sehat (Stres Berat)')
            
    return pd.Series(status, index=df_input.index), vigor_score

# ==============================================================================
# 6. INFERENCE KASKADE LENGKAP PADA TEST SET
# ==============================================================================
print("\n[5/6] Menjalankan Pipeline End-to-End pada Test Set (N = 17,914)...")

# Tahap 1: Prediksi L1
pred_veg_mask = model_l1.predict(df_test[FEATURE_COLS]) == 1

# Tahap 2: Prediksi L2 pada yang lolos L1
pred_sawit_mask = np.zeros(len(df_test), dtype=bool)
if pred_veg_mask.sum() > 0:
    pred_sawit_mask[pred_veg_mask] = model_l2.predict(df_test.loc[pred_veg_mask, FEATURE_COLS]) == 1

# Tahap 3: Health assessment pada yang terdeteksi sawit
final_labels = pd.Series('Bukan Vegetasi', index=df_test.index)
final_labels[pred_veg_mask & (~pred_sawit_mask)] = 'Vegetasi Non-Sawit (Hutan)'

sawit_idx_test = df_test.index[pred_sawit_mask]
if len(sawit_idx_test) > 0:
    health_status, vigor = compute_health_status(df_test.loc[sawit_idx_test])
    final_labels.loc[sawit_idx_test] = health_status

print("\n" + "="*65)
print("HASIL PREDIKSI PIPELINE AKHIR (L1 -> L2 -> Health Assessment):")
dist_final = final_labels.value_counts()
for lbl, count in dist_final.items():
    pct = (count / len(df_test)) * 100
    print(f"  {lbl:<32}: {count:>6,} ({pct:>5.2f}%)")
print("="*65)

# Evaluasi deteksi Sawit Tidak Sehat pada subpopulasi Sawit Dewasa
test_dewasa = df_test[df_test['kelas'] == 2]
dewasa_labels = final_labels.loc[test_dewasa.index]
print("\nEvaluasi Khusus pada Sawit Dewasa Test Set (N = 9,700):")
print(dewasa_labels.value_counts())

# ==============================================================================
# SIMPAN MODEL & ARTIFAK
# ==============================================================================
print("\n[6/6] Menyimpan model dan visualisasi...")
joblib.dump(model_l1, 'SANDBOX/models/model_layer1_vegetasi.pkl')
joblib.dump(model_l2, 'SANDBOX/models/model_layer2_sawit.pkl')

calibration_params = {
    "mean_ndre": float(mean_ndre),
    "std_ndre": float(std_ndre),
    "mean_ndvi": float(mean_ndvi),
    "std_ndvi": float(std_ndvi),
    "mean_msi": float(mean_msi),
    "std_msi": float(std_msi),
    "features": FEATURE_COLS
}
with open('SANDBOX/models/health_calibration.json', 'w') as f:
    json.dump(calibration_params, f, indent=2)

# Visualisasi Hasil Pipeline
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#8B0000', '#FF8C00', '#2E8B57', '#90EE90', '#228B22', '#708090']
palette = {
    'Sawit Tidak Sehat (Stres Berat)': '#D9534F',
    'Sawit Stres Ringan': '#F0AD4E',
    'Sawit Sehat Prima': '#5CB85C',
    'Sawit Muda / Kanopi Terbuka': '#5BC0DE',
    'Vegetasi Non-Sawit (Hutan)': '#2E6DA4',
    'Bukan Vegetasi': '#777777'
}

dist_df = pd.DataFrame({'Kategori': dist_final.index, 'Jumlah': dist_final.values})
dist_df['Warna'] = dist_df['Kategori'].map(palette).fillna('#333333')

bars = ax.barh(dist_df['Kategori'], dist_df['Jumlah'], color=dist_df['Warna'])
ax.set_title('Distribusi Prediksi Pipeline Akhir (2-Layer + Health Assessment)', fontsize=14, fontweight='bold')
ax.set_xlabel('Jumlah Piksel Test Set', fontsize=12)
for bar in bars:
    w = bar.get_width()
    pct = (w / len(df_test)) * 100
    ax.text(w + 100, bar.get_y() + bar.get_height()/2, f'{int(w):,} ({pct:.1f}%)', va='center', fontsize=10)

plt.tight_layout()
plot_path = 'SANDBOX/plots/hasil_pipeline_akhir.png'
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"Model tersimpan di : SANDBOX/models/")
print(f"Grafik tersimpan di: {plot_path}")
print("\nPipeline selesai dijalankan dengan sukses!")
