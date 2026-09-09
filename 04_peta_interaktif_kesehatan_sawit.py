"""
SANDBOX Experiment 04: Pemetaan Spasial Interaktif Kesehatan Sawit
==================================================================
Memetakan hasil klasifikasi dan status kesehatan sawit ke dalam peta interaktif (HTML)
menggunakan koordinat spasial asli (.geo) dari data Sentinel-2 di Rambah Hilir, Riau.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap

sys.stdout.reconfigure(encoding='utf-8')

print("[1/5] Loading data spasial S2_training_BALANCED.csv...")
DATA_PATH = 'InspectCSV/S2_training_BALANCED.csv'
df = pd.read_csv(DATA_PATH)

# Parse koordinat dari GeoJSON string di kolom .geo
print("[2/5] Parsing koordinat GPS (Longitude, Latitude)...")
coords = [json.loads(g)['coordinates'] for g in df['.geo']]
df['lon'] = [c[0] for c in coords]
df['lat'] = [c[1] for c in coords]

# Hitung NDRE & MSI
df['NDRE'] = (df['B8A'] - df['B5']) / (df['B8A'] + df['B5'] + 1e-6)
df['MSI']  = df['B11'] / (df['B8'] + 1e-6)

# Filter hanya kelas Sawit (Kelas 0: Muda, 1: Remaja, 2: Dewasa)
palm_classes = [0, 1, 2]
df_sawit = df[df['kelas'].isin(palm_classes)].copy()
print(f"      Total data sawit dengan koordinat: {len(df_sawit):,} titik")

# [3/5] Hitung Multi-Spectral Vigor Score (MVS) & Status Kesehatan
print("[3/5] Mengkalkulasi Skor Kesehatan & Klasifikasi Stres...")
sawit_dewasa = df_sawit[df_sawit['kelas'] == 2]
mean_ndre, std_ndre = sawit_dewasa['NDRE'].mean(), sawit_dewasa['NDRE'].std()
mean_ndvi, std_ndvi = sawit_dewasa['NDVI'].mean(), sawit_dewasa['NDVI'].std()
mean_msi,  std_msi  = sawit_dewasa['MSI'].mean(),  sawit_dewasa['MSI'].std()

z_ndre = (df_sawit['NDRE'] - mean_ndre) / std_ndre
z_ndvi = (df_sawit['NDVI'] - mean_ndvi) / std_ndvi
z_msi  = (df_sawit['MSI']  - mean_msi)  / std_msi

df_sawit['vigor_score'] = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)

def assign_status(row):
    if row['kelas'] == 0:
        return 'Sawit Muda'
    v = row['vigor_score']
    if v >= -0.5:
        return 'Sawit Sehat Prima'
    elif v >= -1.5:
        return 'Sawit Stres Ringan'
    else:
        return 'Sawit Sakit / Stres Berat (Tren Menurun)'

df_sawit['status_kesehatan'] = df_sawit.apply(assign_status, axis=1)

print("\nDistribusi Status Kesehatan Spasial:")
print(df_sawit['status_kesehatan'].value_counts())

# [4/5] Membangun Peta Interaktif Folium
print("\n[4/5] Membangun Peta Interaktif Leaflet/Folium...")
center_lat = df_sawit['lat'].mean()
center_lon = df_sawit['lon'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='CartoDB positron',
    control_scale=True
)

# Tambahkan Basemap Satelit (Esri World Imagery)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri World Imagery',
    name='Citra Satelit Esri (Resolusi Tinggi)',
    overlay=False,
    control=True
).add_to(m)

# Tambahkan Layer Raster GEE: Tren Temporal Penurunan Vegetasi (2022 - 2024)
gee_tile_url = 'https://earthengine.googleapis.com/v1/projects/ardent-particle-480118-k7/maps/7db3fb367ba09196825231dc0838bdec-b9142d171cf0f2a7e3a18cc21bc1f002/tiles/{z}/{x}/{y}'
folium.TileLayer(
    tiles=gee_tile_url,
    attr='Google Earth Engine Sentinel-2 Temporal Trend',
    name='🛰️ Raster GEE: Tren Temporal Sawit Sakit (Merah=Turun Drastis)',
    overlay=True,
    control=True,
    opacity=0.75,
    show=True
).add_to(m)

# Feature Groups untuk Layer Toggle
fg_sehat  = folium.FeatureGroup(name='🟢 Sawit Sehat Prima', show=True)
fg_ringan = folium.FeatureGroup(name='🟡 Sawit Stres Ringan (Waspada)', show=True)
fg_sakit  = folium.FeatureGroup(name='🔴 Sawit Sakit / Tren Turun (KRITIS)', show=True)
fg_muda   = folium.FeatureGroup(name='🔵 Sawit Muda / Fase Awal', show=False)

# Ambil sampel representatif agar file HTML ringan dan responsif dibuka di browser
SAMPLE_SIZE = 3500
df_sample = df_sawit.sample(min(SAMPLE_SIZE, len(df_sawit)), random_state=42)

sick_points = []

for _, row in df_sample.iterrows():
    stat = row['status_kesehatan']
    lat, lon = row['lat'], row['lon']
    
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 12px; width: 220px;">
        <h4 style="margin: 0 0 5px 0; color: {'#d9534f' if 'Sakit' in stat else '#5cb85c'};">
            {stat}
        </h4>
        <hr style="margin: 3px 0 8px 0; border: 0; border-top: 1px solid #ccc;">
        <b>Koordinat:</b> {lat:.5f}, {lon:.5f}<br>
        <b>Vigor Score:</b> {row['vigor_score']:.2f}<br>
        <b>NDRE (Klorofil):</b> {row['NDRE']:.3f}<br>
        <b>NDVI (Biomassa):</b> {row['NDVI']:.3f}<br>
        <b>MSI (Stres Air):</b> {row['MSI']:.3f}<br>
        <b>Indikasi:</b> {'⚠️ Perlu Ground Check / Karantina' if 'Sakit' in stat else 'Kondisi Baik'}
    </div>
    """
    
    if stat == 'Sawit Sakit / Stres Berat (Tren Menurun)':
        sick_points.append([lat, lon, 1.0])
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='#d9534f',
            fill=True,
            fill_color='#d9534f',
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_sakit)
        
    elif stat == 'Sawit Stres Ringan':
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color='#f0ad4e',
            fill=True,
            fill_color='#f0ad4e',
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_ringan)
        
    elif stat == 'Sawit Sehat Prima':
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='#2e7d32',
            fill=True,
            fill_color='#4caf50',
            fill_opacity=0.65,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_sehat)
        
    elif stat == 'Sawit Muda':
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color='#0288d1',
            fill=True,
            fill_color='#29b6f6',
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250)
        ).add_to(fg_muda)

# Tambahkan Heatmap Khusus Area Sawit Sakit (Hotspot Penyakit)
if len(sick_points) > 0:
    fg_heatmap = folium.FeatureGroup(name='🔥 Heatmap Hotspot Sawit Sakit', show=True)
    HeatMap(
        sick_points,
        radius=18,
        blur=12,
        min_opacity=0.4,
        gradient={0.2: 'blue', 0.5: 'yellow', 0.8: 'orange', 1.0: 'red'}
    ).add_to(fg_heatmap)
    fg_heatmap.add_to(m)

# Tambahkan semua layer ke peta
fg_sehat.add_to(m)
fg_ringan.add_to(m)
fg_sakit.add_to(m)
fg_muda.add_to(m)

# Layer Control untuk menyalakan/mematikan layer di pojok kanan atas
folium.LayerControl(collapsed=False).add_to(m)

# [5/5] Simpan Peta HTML
OUTPUT_MAP = 'SANDBOX/peta_kesehatan_sawit_interaktif.html'
m.save(OUTPUT_MAP)
print(f"\n[5/5] Peta interaktif BERHASIL dibuat dan disimpan di:")
print(f"      {OUTPUT_MAP}")
print("      (Dapat dibuka langsung di browser manapun!)")
