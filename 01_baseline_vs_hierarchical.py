"""
SANDBOX Experiment 01: Baseline Flat vs 2-Layer Hierarchical Model
==================================================================
Membandingkan model flat (9 kelas langsung) vs model hierarkis 2 layer:
- Layer 1: Vegetatif vs Non-Vegetatif
- Layer 2: Sawit vs Non-Sawit (khusus pada data vegetatif)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

# Buat direktori output plot di SANDBOX
os.makedirs('SANDBOX/plots', exist_ok=True)
os.makedirs('SANDBOX/results', exist_ok=True)

# 1. LOAD DATA
DATA_PATH = 'output/training_data_all_features.csv'
print(f"[1/5] Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
print(f"      Total data: {len(df):,} baris, {df.shape[1]} kolom.")

# 2. DEFINISI FITUR & TARGET
# Semua fitur spektral, indeks, dan tekstur (tanpa target)
feature_cols = [c for c in df.columns if c not in ['kelas', 'kelas_nama', '.geo']]
print(f"      Menggunakan {len(feature_cols)} fitur: {feature_cols}")

# Definisi Kelas:
# 0: Sawit Muda, 1: Sawit Remaja, 2: Sawit Dewasa
# 3: Hutan Lebat, 4: Hutan Tidak Lebat
# 5: Tanah, 6: Urban, 7: Aspal, 8: Air
vegetatif_classes = [0, 1, 2, 3, 4]
non_vegetatif_classes = [5, 6, 7, 8]
sawit_classes = [0, 1, 2]
non_sawit_veg_classes = [3, 4]

# Target Label Hierarkis:
# is_vegetatif: 1 jika vegetatif, 0 jika non-vegetatif
df['is_vegetatif'] = df['kelas'].isin(vegetatif_classes).astype(int)

# is_sawit (hanya valid untuk vegetatif): 1 jika sawit, 0 jika hutan/vegetasi lain
df['is_sawit'] = np.where(df['kelas'].isin(sawit_classes), 1, 0)

# Train-Test Split (80% train, 20% test) Stratified berdasarkan kelas asli
RANDOM_STATE = 42
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=df['kelas']
)

df_train = df.loc[train_idx].copy()
df_test = df.loc[test_idx].copy()

X_train_full = df_train[feature_cols]
X_test_full = df_test[feature_cols]

# ==============================================================================
# EKSPERIMEN A: BASELINE MODEL (Flat Multi-Class 9 Kelas)
# ==============================================================================
print("\n[2/5] Training Baseline Model: Flat 9-Class Classifier (XGBoost)...")
xgb_flat = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='mlogloss'
)
xgb_flat.fit(X_train_full, df_train['kelas'])
y_pred_flat = xgb_flat.predict(X_test_full)

acc_flat = accuracy_score(df_test['kelas'], y_pred_flat)
macro_f1_flat = f1_score(df_test['kelas'], y_pred_flat, average='macro')
print(f"      [Baseline Flat] Accuracy: {acc_flat:.4f} ({acc_flat*100:.2f}%)")
print(f"      [Baseline Flat] Macro F1: {macro_f1_flat:.4f} ({macro_f1_flat*100:.2f}%)")

# ==============================================================================
# EKSPERIMEN B: LAYER 1 (Vegetatif vs Non-Vegetatif)
# ==============================================================================
print("\n[3/5] Training Hierarchical Layer 1: Vegetatif vs Non-Vegetatif...")
xgb_l1 = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
)
xgb_l1.fit(X_train_full, df_train['is_vegetatif'])
y_pred_l1 = xgb_l1.predict(X_test_full)

acc_l1 = accuracy_score(df_test['is_vegetatif'], y_pred_l1)
f1_l1 = f1_score(df_test['is_vegetatif'], y_pred_l1)
print(f"      [Layer 1 - Veg vs Non] Accuracy: {acc_l1:.4f} ({acc_l1*100:.2f}%)")
print(f"      [Layer 1 - Veg vs Non] F1-Score: {f1_l1:.4f} ({f1_l1*100:.2f}%)")

# ==============================================================================
# EKSPERIMEN C: LAYER 2 (Sawit vs Non-Sawit pada Data Vegetatif)
# ==============================================================================
print("\n[4/5] Training Hierarchical Layer 2: Sawit vs Non-Sawit (Vegetatif Only)...")
# Filter train data hanya yang vegetatif
train_veg = df_train[df_train['is_vegetatif'] == 1]
test_veg = df_test[df_test['is_vegetatif'] == 1]

# Cek distribusi kelas di L2
print(f"      Data Training L2 (Vegetatif): {len(train_veg):,} sampel")
print(f"      - Sawit (Muda/Remaja/Dewasa): {(train_veg['is_sawit'] == 1).sum():,}")
print(f"      - Non-Sawit (Hutan Lebat/Tidak Lebat): {(train_veg['is_sawit'] == 0).sum():,}")

xgb_l2 = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
)
xgb_l2.fit(train_veg[feature_cols], train_veg['is_sawit'])

# Evaluasi L2 pada ground truth vegetatif
y_pred_l2_pure = xgb_l2.predict(test_veg[feature_cols])
acc_l2 = accuracy_score(test_veg['is_sawit'], y_pred_l2_pure)
f1_l2 = f1_score(test_veg['is_sawit'], y_pred_l2_pure)
print(f"      [Layer 2 - Sawit vs Hutan] Accuracy: {acc_l2:.4f} ({acc_l2*100:.2f}%)")
print(f"      [Layer 2 - Sawit vs Hutan] F1-Score: {f1_l2:.4f} ({f1_l2*100:.2f}%)")

# ==============================================================================
# EVALUASI PIPELINE END-TO-END (Cascade Layer 1 -> Layer 2)
# ==============================================================================
print("\n[5/5] Evaluasi Pipeline Cascade L1 -> L2 pada Seluruh Test Set...")
# 1. Prediksi L1 pada seluruh data test
pred_is_veg = xgb_l1.predict(X_test_full)

# 2. Inisialisasi prediksi final: default 0 (Non-Sawit)
final_pred_sawit = np.zeros(len(df_test), dtype=int)

# 3. Hanya data yang diprediksi vegetatif oleh L1 yang masuk ke L2
veg_mask = (pred_is_veg == 1)
if veg_mask.sum() > 0:
    final_pred_sawit[veg_mask] = xgb_l2.predict(X_test_full[veg_mask])

# Ground truth sawit pada seluruh test set
gt_sawit = df_test['is_sawit'].values

acc_cascade = accuracy_score(gt_sawit, final_pred_sawit)
f1_cascade = f1_score(gt_sawit, final_pred_sawit)
print("\n" + "="*65)
print("HASIL PIPELINE KASKADE HIERARKIS (SAWIT vs NON-SAWIT):")
print(f"  Akurasi Keseluruhan : {acc_cascade:.4f} ({acc_cascade*100:.2f}%)")
print(f"  F1-Score Sawit      : {f1_cascade:.4f} ({f1_cascade*100:.2f}%)")
print("="*65)

print("\nClassification Report Pipeline 2-Layer:")
print(classification_report(gt_sawit, final_pred_sawit, target_names=['Bukan Sawit', 'Sawit'], digits=4))

# Simpan Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CM Layer 1
cm_l1 = confusion_matrix(df_test['is_vegetatif'], pred_is_veg)
sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Non-Veg', 'Vegetatif'], yticklabels=['Non-Veg', 'Vegetatif'])
axes[0].set_title(f'Layer 1: Veg vs Non-Veg (Acc: {acc_l1*100:.2f}%)')
axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Aktual')

# CM Cascade L1->L2
cm_cascade = confusion_matrix(gt_sawit, final_pred_sawit)
sns.heatmap(cm_cascade, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Bukan Sawit', 'Sawit'], yticklabels=['Bukan Sawit', 'Sawit'])
axes[1].set_title(f'Pipeline 2-Layer: Sawit vs Non-Sawit (F1: {f1_cascade*100:.2f}%)')
axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Aktual')

plt.tight_layout()
plt_path = 'SANDBOX/plots/cm_2layer_pipeline.png'
plt.savefig(plt_path, dpi=150)
plt.close()
print(f"Plot disimpan di: {plt_path}")

# Simpan ringkasan hasil ke JSON
results_summary = {
    "baseline_flat": {
        "accuracy": float(acc_flat),
        "macro_f1": float(macro_f1_flat)
    },
    "layer_1_veg_nonveg": {
        "accuracy": float(acc_l1),
        "f1_score": float(f1_l1)
    },
    "layer_2_sawit_non_sawit": {
        "accuracy": float(acc_l2),
        "f1_score": float(f1_l2)
    },
    "pipeline_cascade": {
        "accuracy": float(acc_cascade),
        "f1_score_sawit": float(f1_cascade)
    }
}

with open('SANDBOX/results/exp01_hierarchical_vs_flat.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Ringkasan eksperimen 01 tersimpan di: SANDBOX/results/exp01_hierarchical_vs_flat.json")
