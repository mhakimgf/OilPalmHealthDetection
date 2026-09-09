"""
SANDBOX: Modul Analisis & Pemetaan Kesehatan Sawit Berdasarkan Poligon / Area Kustom
====================================================================================
Fungsi utama:
Menerima input poligon area kebun (GeoJSON, koordinat, atau GEE Asset),
lalu otomatis:
1. Mengunduh data spektral Sentinel-2 bebas awan untuk area tersebut
2. Menjalankan filter Layer 1 (Vegetasi) & Layer 2 (Sawit)
3. Menganalisis tingkat kesehatan (Sehat, Stres Ringan, Sakit Kritis)
4. Menghasilkan peta interaktif spesifik untuk poligon tersebut yang menandai
   secara presisi koordinat pohon/blok yang sakit!
"""

import sys
import os
import json
import ee
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Inisialisasi GEE
try:
    ee.Initialize(project='ardent-particle-480118-k7')
    print("GEE terhubung dengan sukses!")
except Exception as e:
    print(f"Error inisialisasi GEE: {e}")

def inspect_polygon_health(polygon_ee, nama_blok="Blok_Kebun_Uji", year=2024, output_dir="output_polygon"):
    """
    Menganalisis dan memetakan bagian yang sakit di dalam poligon yang diberikan.
    
    Parameters:
    - polygon_ee : ee.Geometry atau ee.FeatureCollection poligon kebun
    - nama_blok  : Nama area/blok untuk penamaan file & judul peta
    - year       : Tahun citra Sentinel-2 yang dianalisis
    - output_dir : Folder penyimpanan peta hasil inspeksi
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================================")
    print(f"  MEMPROSES INSPEKSI KESEHATAN SAWIT: {nama_blok}")
    print(f"==================================================================")
    
    # 1. Bounding box & Koleksi Citra Sentinel-2
    if isinstance(polygon_ee, ee.FeatureCollection):
        aoi = polygon_ee.geometry()
    elif isinstance(polygon_ee, ee.Feature):
        aoi = polygon_ee.geometry()
    elif isinstance(polygon_ee, ee.Geometry):
        aoi = polygon_ee
    else:
        aoi = ee.Feature(polygon_ee).geometry()
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"
    
    print(f"[1/4] Mengambil citra Sentinel-2 bebas awan tahun {year}...")
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45)) \
        .median() \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']) \
        .clip(aoi)
        
    # 2. Hitung Indeks Kesehatan Kunci
    # NDVI (Biomassa), NDRE (Klorofil pelepah), MSI (Stres kelembaban)
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = s2.normalizedDifference(['B8A', 'B5']).rename('NDRE')
    ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
    msi  = s2.select('B11').divide(s2.select('B8').add(1e-6)).rename('MSI')
    nir_red = s2.select('B8').divide(s2.select('B4').add(1e-6)).rename('NIR_RED')
    b4   = s2.select('B4').divide(10000.0).rename('B4')
    
    composite = ee.Image.cat([ndvi, ndre, ndbi, msi, nir_red, b4])
    
    # 3. Sampling titik reguler di dalam poligon untuk analisis spasial
    print("[2/4] Mengekstrak grid titik piksel di dalam poligon (skala 10 meter)...")
    sampled = composite.sample(
        region=aoi,
        scale=10,
        geometries=True,
        numPixels=3000
    )
    
    features = sampled.getInfo().get('features', [])
    print(f"      Berhasil mengekstrak {len(features):,} titik observasi di dalam poligon.")
    
    if len(features) == 0:
        print("Tidak ada piksel yang berhasil diekstrak. Cek geometri poligon!")
        return None
        
    # 4. Parsing ke DataFrame
    records = []
    for f in features:
        props = f.get('properties', {})
        coords = f.get('geometry', {}).get('coordinates', [0, 0])
        records.append({
            'lon': coords[0],
            'lat': coords[1],
            'NDVI': float(props.get('NDVI', 0)),
            'NDRE': float(props.get('NDRE', 0)),
            'NDBI': float(props.get('NDBI', 0)),
            'MSI':  float(props.get('MSI', 0)),
            'NIR_RED': float(props.get('NIR_RED', 0)),
            'B4':   float(props.get('B4', 0))
        })
    df = pd.DataFrame(records)
    
    # 5. Klasifikasi Hierarkis Biofisika + Health Status
    print("[3/4] Menjalankan Pipeline Hierarkis Biofisika + Deteksi Pohon Sakit...")
    mean_ndre, std_ndre = 0.589, 0.037
    mean_ndvi, std_ndvi = 0.822, 0.031
    mean_msi,  std_msi  = 0.548, 0.052
    
    z_ndre = (df['NDRE'] - mean_ndre) / std_ndre
    z_ndvi = (df['NDVI'] - mean_ndvi) / std_ndvi
    z_msi  = (df['MSI']  - mean_msi)  / std_msi
    df['vigor_score'] = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
    
    def assign_status(row):
        v_ndvi = row['NDVI']
        v_ndre = row['NDRE']
        v_ndbi = row['NDBI']
        v_msi  = row['MSI']
        v_ratio = row['NIR_RED']
        v_b4   = row['B4']
        vigor  = row['vigor_score']

        # 1. BUKAN VEGETASI (Jalan Aspal, Atap Bangunan, Semen, Tanah Gundul)
        if (v_ndvi < 0.48) or (v_ndbi > -0.06) or (v_ratio < 2.8) or (v_b4 > 0.085):
            return 'Bukan Vegetasi (Jalan / Bangunan / Tanah)'

        # 2. BUKAN KEBUN SAWIT (Bahu Jalan Berumput, Pekarangan Rumah, Semak Liar)
        if (v_ndbi > -0.15) or (v_ratio < 4.5) or (v_ndvi < 0.65) or (v_b4 > 0.060):
            return 'Non-Sawit (Bahu Jalan / Pekarangan / Semak)'

        # 3. KEBUN SAWIT TERVERIFIKASI (Health Assessment)
        if (v_ndre < 0.46 and v_ndvi < 0.72 and v_msi > 0.65) or (vigor < -2.2):
            return '🔴 SAWIT SAKIT (Target Karantina)'
        elif v_ndvi >= 0.76 and v_ndre >= 0.52 and v_msi <= 0.65:
            return '🟢 Sawit Sehat Prima'
        else:
            return '🟡 Sawit Stres Ringan (Defisiensi Hara)'
            
    df['status'] = df.apply(assign_status, axis=1)
    
    # Ringkasan Statistik
    stats = df['status'].value_counts()
    print("\n--- RINGKASAN HASIL DIAGNOSA KESEHATAN KEBUN ---")
    total = len(df)
    for k, v in stats.items():
        pct = (v / total) * 100
        print(f"  {k:<42}: {v:>5,} titik ({pct:>5.1f}%)")
        
    sick_count = (df['status'] == '🔴 SAWIT SAKIT (Stres Berat / Ganoderma)').sum()
    sick_pct = (sick_count / total) * 100
    print(f"\n>> KESIMPULAN BLOK: {sick_pct:.1f}% area terindikasi sakit!")
    if sick_pct > 10:
        print(">> PERINGATAN: Blok ini memiliki intensitas infeksi TINGGI. Segera lakukan sanitasi/karantina!")
    elif sick_pct > 3:
        print(">> PERINGATAN: Ditemukan kluster pohon sakit. Rekomendasi: ground check target merah.")
    else:
        print(">> STATUS AMAN: Kebun mayoritas berada dalam kondisi sehat prima.")
        
    # 6. Pembuatan Peta Interaktif Spasial untuk Poligon Tersebut
    print("\n[4/4] Membangun Peta Interaktif & Menandai Titik Sakit...")
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles='CartoDB positron')
    
    # Basemap Satelit
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Citra Satelit Resolusi Tinggi',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Feature Groups
    fg_sehat  = folium.FeatureGroup(name='🟢 Sawit Sehat Prima', show=True)
    fg_ringan = folium.FeatureGroup(name='🟡 Sawit Stres Ringan', show=True)
    fg_sakit  = folium.FeatureGroup(name='🔴 SAWIT SAKIT (FOKUS INSPEKSI)', show=True)
    fg_lain   = folium.FeatureGroup(name='⚪ Non-Sawit / Jalan Kebun', show=False)
    
    sick_coords = []
    
    for _, r in df.iterrows():
        st = r['status']
        lat, lon = r['lat'], r['lon']
        
        popup_txt = f"""
        <b>Status:</b> {st}<br>
        <b>GPS:</b> {lat:.5f}, {lon:.5f}<br>
        <b>Vigor:</b> {r['vigor_score']:.2f}<br>
        <b>NDRE (Klorofil):</b> {r['NDRE']:.3f}<br>
        <b>NDVI:</b> {r['NDVI']:.3f}<br>
        <b>MSI:</b> {r['MSI']:.3f}
        """
        
        if '🔴 SAWIT SAKIT' in st:
            sick_coords.append([lat, lon, 1.0])
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color='#ff0000',
                fill=True,
                fill_color='#ff0000',
                fill_opacity=0.9,
                popup=folium.Popup(popup_txt, max_width=250)
            ).add_to(fg_sakit)
        elif '🟡 Sawit Stres' in st:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color='#f39c12',
                fill=True,
                fill_color='#f1c40f',
                fill_opacity=0.7,
                popup=folium.Popup(popup_txt, max_width=250)
            ).add_to(fg_ringan)
        elif '🟢 Sawit Sehat' in st:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='#27ae60',
                fill=True,
                fill_color='#2ecc71',
                fill_opacity=0.6,
                popup=folium.Popup(popup_txt, max_width=250)
            ).add_to(fg_sehat)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='#95a5a6',
                fill=True,
                fill_color='#bdc3c7',
                fill_opacity=0.4,
                popup=folium.Popup(popup_txt, max_width=250)
            ).add_to(fg_lain)
            
    # Tambahkan Heatmap Kluster Penyakit jika ada
    if len(sick_coords) > 0:
        fg_heat = folium.FeatureGroup(name='🔥 Hotspot Kluster Sakit', show=True)
        HeatMap(sick_coords, radius=20, blur=15, min_opacity=0.5).add_to(fg_heat)
        fg_heat.add_to(m)
        
    fg_sehat.add_to(m)
    fg_ringan.add_to(m)
    fg_sakit.add_to(m)
    fg_lain.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    
    output_html = os.path.join(output_dir, f"peta_inspeksi_{nama_blok}.html")
    m.save(output_html)
    print(f"\n>> Peta hasil inspeksi poligon tersimpan di:")
    print(f"   {output_html}")
    
    # Simpan juga koordinat titik sakit ke CSV agar bisa langsung dibawa ke GPS tim lapangan
    sick_df = df[df['status'] == '🔴 SAWIT SAKIT (Stres Berat / Ganoderma)'][['lat', 'lon', 'vigor_score', 'NDRE', 'NDVI', 'MSI']]
    output_csv = os.path.join(output_dir, f"target_ground_check_{nama_blok}.csv")
    sick_df.to_csv(output_csv, index=False)
    print(f">> Daftar koordinat GPS pohon sakit diekspor ke:")
    print(f"   {output_csv} (Siap di-load ke GPS / Google Maps tim lapangan!)")
    print("=" * 65)
    
    return output_html

if __name__ == '__main__':
    # Contoh demonstrasi menggunakan salah satu poligon kebun sawit yang ada di GEE
    print("Menguji fungsi dengan poligon kebun sawit riil di Rambah Hilir...")
    sample_poly = ee.FeatureCollection('projects/proyek-ds-488108/assets/sawitDewasa').first()
    inspect_polygon_health(sample_poly, nama_blok="Kebun_Sawit_Dewasa_Blok_A", year=2024)
