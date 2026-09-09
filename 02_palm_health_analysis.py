"""
SANDBOX Experiment 02: Analisis Spektral Kesehatan Sawit & Desain Palm Health Index (PHI)
========================================================================================
Menganalisis variabilitas spektral kanopi sawit (Muda, Remaja, Dewasa) dan
mengembangkan metrik kuantitatif untuk mendeteksi Sawit Tidak Sehat (Stressed/Unhealthy).
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.stdout.reconfigure(encoding='utf-8')

# 1. LOAD DATA
DATA_PATH = 'output/training_data_all_features.csv'
df = pd.read_csv(DATA_PATH)

# Filter hanya area Sawit (Kelas 0: Muda, 1: Remaja, 2: Dewasa)
palm_classes = [0, 1, 2]
df_sawit = df[df['kelas'].isin(palm_classes)].copy()

# Tambahkan fitur indeks stres kanopi:
# MSI (Moisture Stress Index) = B11 (SWIR) / B8 (NIR) -> Makin tinggi = makin kering/stres
df_sawit['MSI'] = df_sawit['B11'] / (df_sawit['B8'] + 1e-6)

# PSRI (Plant Senescence Reflectance Index) jika ada B4, B2, B6
# PSRI = (B4 - B2) / B6 -> Mengukur penuaan/kematian sel kanopi
df_sawit['PSRI'] = (df_sawit['B4'] - df_sawit['B2']) / (df_sawit['B6'] + 1e-6)

# NDRE2 = (B8A - B5) / (B8A + B5)
df_sawit['NDRE_refined'] = (df_sawit['B8A'] - df_sawit['B5']) / (df_sawit['B8A'] + df_sawit['B5'] + 1e-6)

print("=" * 65)
print(f"ANALISIS POPULASI KANOPI SAWIT (Total: {len(df_sawit):,} piksel)")
print("=" * 65)

for cls_id, cls_name in [(0, 'Sawit Muda'), (1, 'Sawit Remaja'), (2, 'Sawit Dewasa')]:
    sub = df_sawit[df_sawit['kelas'] == cls_id]
    print(f"\n--- {cls_name} (N = {len(sub):,}) ---")
    print(f"  NDVI : mean={sub['NDVI'].mean():.3f}, std={sub['NDVI'].std():.3f}, Q1={sub['NDVI'].quantile(0.25):.3f}, Median={sub['NDVI'].median():.3f}, Q3={sub['NDVI'].quantile(0.75):.3f}")
    print(f"  NDRE : mean={sub['NDRE'].mean():.3f}, std={sub['NDRE'].std():.3f}, Q1={sub['NDRE'].quantile(0.25):.3f}, Median={sub['NDRE'].median():.3f}")
    print(f"  EVI  : mean={sub['EVI'].mean():.3f}, std={sub['EVI'].std():.3f}")
    print(f"  MSI  : mean={sub['MSI'].mean():.3f}, std={sub['MSI'].std():.3f} (Moisture Stress)")

# ==============================================================================
# DESAIN PALM HEALTH INDEX (PHI)
# ==============================================================================
# Prinsip biometrik:
# - Klorofil tinggi (NDRE tinggi) -> Sehat (+)
# - Biomassa / Vigor tinggi (NDVI & EVI tinggi) -> Sehat (+)
# - Stres air tinggi (MSI tinggi) -> Sakit / Stres (-)
# - Kerusakan sel / Penuaan (PSRI tinggi) -> Sakit / Stres (-)

# Normalisasi min-max per-fase sawit untuk mendapatkan Skor Kesehatan (0 s.d 100)
# Fokus utama deteksi sawit tidak sehat adalah pada Sawit Dewasa & Remaja
# (karena Sawit Muda secara alami memang memiliki kanopi terbuka dan tanah terlihat).

sub_dewasa = df_sawit[df_sawit['kelas'] == 2].copy()

# Standarisasi Z-score untuk deteksi anomali pada Sawit Dewasa
z_ndre = (sub_dewasa['NDRE'] - sub_dewasa['NDRE'].mean()) / sub_dewasa['NDRE'].std()
z_ndvi = (sub_dewasa['NDVI'] - sub_dewasa['NDVI'].mean()) / sub_dewasa['NDVI'].std()
z_msi  = (sub_dewasa['MSI'] - sub_dewasa['MSI'].mean()) / sub_dewasa['MSI'].std()

# Multi-spectral Vigor Score (MVS)
# Nilai MVS rendah menandakan klorofil & biomassa rendah secara abnormal
sub_dewasa['vigor_score'] = (z_ndre * 0.4) + (z_ndvi * 0.4) - (z_msi * 0.2)

# Klasifikasi Tingkat Kesehatan Sawit Dewasa:
# 1. Sehat Prima (Vigor normal s.d tinggi: vigor_score >= -0.5)
# 2. Stres Ringan / Defisiensi Hara (-1.5 <= vigor_score < -0.5)
# 3. Stres Berat / Gejala Penyakit / Tidak Sehat (vigor_score < -1.5)
def categorize_health(score):
    if score >= -0.5:
        return 'Sehat Prima'
    elif score >= -1.5:
        return 'Stres Ringan'
    else:
        return 'Stres Berat / Tidak Sehat'

sub_dewasa['kategori_kesehatan'] = sub_dewasa['vigor_score'].apply(categorize_health)

print("\n" + "=" * 65)
print("DISTRIBUSI KESEHATAN SAWIT DEWASA (METODE STATISTIKAL ANOMALI):")
health_counts = sub_dewasa['kategori_kesehatan'].value_counts()
for kat, cnt in health_counts.items():
    pct = (cnt / len(sub_dewasa)) * 100
    print(f"  {kat:<28}: {cnt:>6,} ({pct:>5.2f}%)")
print("=" * 65)

# Visualisasi Distribusi Indeks Kesehatan
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram NDRE (Indikator Klorofil)
sns.histplot(data=df_sawit, x='NDRE', hue='kelas_nama', kde=True, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Distribusi NDRE (Red Edge - Klorofil Kanopi)')

# 2. Histogram MSI (Indikator Stres Air Kanopi)
sns.histplot(data=df_sawit, x='MSI', hue='kelas_nama', kde=True, ax=axes[0, 1], palette='magma')
axes[0, 1].set_title('Distribusi MSI (Moisture Stress Index = B11 / B8)')

# 3. Scatter Plot: NDVI vs NDRE dengan zona stres
sns.scatterplot(
    data=sub_dewasa.sample(min(3000, len(sub_dewasa)), random_state=42),
    x='NDVI', y='NDRE', hue='kategori_kesehatan', palette={'Sehat Prima': 'green', 'Stres Ringan': 'orange', 'Stres Berat / Tidak Sehat': 'red'},
    alpha=0.6, ax=axes[1, 0]
)
axes[1, 0].set_title('Scatter NDVI vs NDRE Sawit Dewasa (Deteksi Stres)')
axes[1, 0].axvline(0.70, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].axhline(0.35, color='gray', linestyle='--', alpha=0.5)

# 4. Boxplot Vigor Score
sns.boxplot(data=sub_dewasa, x='kategori_kesehatan', y='vigor_score', palette='RdYlGn_r', ax=axes[1, 1])
axes[1, 1].set_title('Distribusi Vigor Score per Kategori Kesehatan')

plt.tight_layout()
plot_path = 'SANDBOX/plots/analisis_kesehatan_sawit.png'
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nPlot analisis kesehatan sawit disimpan di: {plot_path}")
