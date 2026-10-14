"""
OilPalmHealthDetection — Skrip 07: Deteksi Anomali Kanopi Sawit Berbasis Isolation Forest
========================================================================================
Tujuan:
1. Mengevaluasi populasi kanopi kelapa sawit menggunakan algoritma Isolation Forest.
2. Menghasilkan Continuous Anomaly Score (0.000 s.d 1.000).
3. Menerapkan Directional Gatekeeper untuk menetapkan sistem Triage Prioritas Lapangan:
   - 🔴 Tier 1: Karantina Darurat (Suspek Ganoderma / Kematian Tajuk)
   - 🟡 Tier 2: Stres Ringan (Defisiensi Hara N-P-K / Fluktuasi Air)
   - 🟢 Tier 3: Normal Sehat (Kanopi Stabil Prima)
   - 🟢 Super Sehat (Anomali Positif Kanopi Sangat Rapat)
   - 🔵 Sawit Muda (Fase Tajuk Belum Rapat)
   - ⚪ Filter Non-Sawit (Guardrail Biofisika Jalan / Tanah)
4. Menghasilkan visualisasi histogram sebaran skor anomali dan daftar target GPS ground check.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from predict_point_or_csv import predict_pipeline

print("=" * 80)
print(">> [1/4] Memuat Dataset Spektral Kelapa Sawit Sentinel-2...")
print("=" * 80)

# Cari file dataset
candidates = [
    os.path.join(BASE_DIR, '..', 'output', 'training_data_all_features.csv'),
    os.path.join(BASE_DIR, 'output', 'training_data_all_features.csv'),
    'output/training_data_all_features.csv'
]
data_path = next((p for p in candidates if os.path.exists(p)), None)

if not data_path:
    raise FileNotFoundError("Dataset output/training_data_all_features.csv tidak ditemukan.")

print(f"      Membaca data dari: {data_path}")
df_full = pd.read_csv(data_path)
print(f"      Total baris dataset: {len(df_full):,} sampel.")

# Ambil sampel representatif 5.000 titik untuk pengujian komprehensif
SAMPLE_SIZE = min(5000, len(df_full))
df_sample = df_full.sample(SAMPLE_SIZE, random_state=42).copy()

print(f"\n>> [2/4] Menjalankan Pipeline 3-Layer (Random Forest + Isolation Forest)...")
df_pred, timing = predict_pipeline(df_sample, return_timing=True)

print(f"      Selesai dalam {timing['elapsed_sec']} detik ({timing['elapsed_ms']} ms).")
print(f"      Kecepatan: {timing['ms_per_sample']} ms per pohon.")

print("\n" + "=" * 80)
print("HASIL EVALUASI SISTEM TRIAGE PRIORITAS KESEHATAN KEBUN:")
print("=" * 80)
dist = df_pred['priority_tier'].value_counts()
for tier, count in dist.items():
    pct = (count / len(df_pred)) * 100
    print(f"  • {tier:<32}: {count:>5,} titik ({pct:>5.1f}%)")

sawit_df = df_pred[df_pred['pred_layer2'] == 'Sawit'].copy()
if len(sawit_df) > 0:
    scores = sawit_df['anomaly_score']
    print(f"\nStatistik Continuous Anomaly Score pada Sawit (N={len(sawit_df):,}):")
    print(f"  - Minimum : {scores.min():.4f}")
    print(f"  - Rerata  : {scores.mean():.4f}")
    print(f"  - Median  : {scores.median():.4f}")
    print(f"  - Maksimum: {scores.max():.4f}")
    print(f"  - Std Dev : {scores.std():.4f}")

# -------------------------------------------------------------------------
# Ekspor Target Ground Check Prioritas
# -------------------------------------------------------------------------
print(f"\n>> [3/4] Mengekspor Target Ground Check Prioritas Lapangan...")
os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'plots'), exist_ok=True)

triage_mask = df_pred['priority_tier'].str.contains('Tier 1|Tier 2') | df_pred['health_status'].str.contains('SAWIT SAKIT')
target_df = df_pred[triage_mask].copy()

if len(target_df) > 0:
    target_df = target_df.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
    target_df['Ranking_Prioritas'] = target_df.index + 1
    
    # Pilih kolom yang relevan untuk tim mandor/agronomis
    cols = ['Ranking_Prioritas', 'anomaly_score', 'priority_tier', 'health_status', 'recommended_action', 'NDRE', 'NDVI', 'MSI']
    if 'lat' in target_df.columns and 'lon' in target_df.columns:
        cols = ['Ranking_Prioritas', 'lat', 'lon'] + cols[1:]
        
    out_csv = os.path.join(BASE_DIR, 'results', 'target_ground_check_prioritas.csv')
    target_df[cols].to_csv(out_csv, index=False)
    print(f"      Daftar target ground check tersimpan di: {out_csv}")
    print(f"      Total pohon target inspeksi: {len(target_df)} pohon (terurut prioritas darurat).")

# -------------------------------------------------------------------------
# Visualisasi Sebaran Skor Anomali (Distribution Plot)
# -------------------------------------------------------------------------
print(f"\n>> [4/4] Membuat Grafik Distribusi Continuous Anomaly Score...")
plt.figure(figsize=(10, 5.5))

colors = {
    '🟢 Tier 3: Normal Sehat': '#22c55e',
    '🟡 Tier 2: Stres Ringan': '#f59e0b',
    '🔴 Tier 1: Karantina Darurat': '#ef4444',
    '🟢 Super Sehat': '#10b981',
    '🔵 Sawit Muda': '#38bdf8',
    '⚪ Filter Non-Sawit': '#94a3b8'
}

clean_labels = {
    '🟢 Tier 3: Normal Sehat': '[Tier 3] Normal Sehat',
    '🟡 Tier 2: Stres Ringan': '[Tier 2] Stres Ringan',
    '🔴 Tier 1: Karantina Darurat': '[Tier 1] Karantina Darurat',
    '🟢 Super Sehat': '[Super Vigor] Kanopi Sangat Rapat',
    '🔵 Sawit Muda': '[Sawit Muda] Tajuk Terbuka',
    '⚪ Filter Non-Sawit': '[Non-Sawit] Tepi Jalan / Semak'
}

for tier in df_pred['priority_tier'].unique():
    if 'Bukan' in tier:
        continue
    sub = df_pred[df_pred['priority_tier'] == tier]
    lbl = clean_labels.get(tier, tier)
    plt.hist(
        sub['anomaly_score'],
        bins=15,
        alpha=0.65,
        label=f"{lbl} (N={len(sub)})",
        color=colors.get(tier, 'gray'),
        edgecolor='black',
        linewidth=0.5
    )

plt.axvline(0.65, color='#ef4444', linestyle='--', linewidth=1.5, label='Ambang Batas Kritis (0.65)')
plt.axvline(0.40, color='#f59e0b', linestyle=':', linewidth=1.5, label='Ambang Batas Waspada (0.40)')

plt.title('Distribusi Continuous Anomaly Score (Isolation Forest)\nSistem Triage Prioritas Kesehatan Kanopi Kelapa Sawit',
          fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Continuous Anomaly Score (0.000 = Sangat Normal, 1.000 = Ekstrem Anomali)', fontsize=11, fontweight='bold')
plt.ylabel('Frekuensi (Jumlah Piksel)', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, fontsize=9.5, loc='upper right')
plt.tight_layout()

plot_out = os.path.join(BASE_DIR, 'plots', 'anomaly_score_distribution.png')
plt.savefig(plot_out, dpi=300)
plt.close()
print(f"      Grafik distribusi tersimpan di: {plot_out}")

print("\n" + "=" * 80)
print("SKRIP 07 SELESAI DENGAN SUKSES!")
print("=" * 80)
