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

from predict_point_or_csv import predict_pipeline

sys.stdout.reconfigure(encoding='utf-8')

# Inisialisasi GEE
try:
    ee.Initialize(project='ardent-particle-480118-k7')
    print("GEE terhubung dengan sukses!")
except Exception as e:
    print(f"Error inisialisasi GEE: {e}")

def inspect_polygon_health(polygon_ee, nama_blok="Blok_Kebun_Uji", year=2024, quarter="Q4", output_dir="output_polygon"):
    """
    Menganalisis dan memetakan bagian yang sakit di dalam poligon yang diberikan
    berbasis periode triwulanan (per 3 bulan) dan unit spasial grid 2x2 piksel (20m x 20m).
    
    Parameters:
    - polygon_ee : ee.Geometry atau ee.FeatureCollection poligon kebun
    - nama_blok  : Nama area/blok untuk penamaan file & judul peta
    - year       : Tahun citra Sentinel-2 yang dianalisis
    - quarter    : Kuartal triwulanan ('Q1', 'Q2', 'Q3', 'Q4')
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
        
    quarter_dates = {
        'Q1': (f"{year}-01-01", f"{year}-03-31", "1 Jan - 31 Mar"),
        'Q2': (f"{year}-04-01", f"{year}-06-30", "1 Apr - 30 Jun"),
        'Q3': (f"{year}-07-01", f"{year}-09-30", "1 Jul - 30 Sep"),
        'Q4': (f"{year}-10-01", f"{year}-12-31", "1 Okt - 31 Des"),
    }
    start_date, end_date, date_label = quarter_dates.get(quarter.upper(), (f"{year}-10-01", f"{year}-12-31", "1 Okt - 31 Des"))
    
    print(f"[1/4] Mengambil citra Sentinel-2 bebas awan periode {year} {quarter} ({date_label})...")
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 75))
        
    def mask_clouds_and_shadows(img):
        qa = img.select('QA60')
        qa_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        cs_mask = img.select('cs').gte(0.55)
        return img.updateMask(qa_mask.And(cs_mask))

    s2 = s2_col.linkCollection(cs_plus, ['cs']) \
        .map(mask_clouds_and_shadows) \
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
    
    # 3. Sampling grid 2x2 piksel (20m x 20m = 400 m2) di dalam poligon
    print("[2/4] Mengekstrak grid 2x2 piksel (20m x 20m = 400 m2 / rumpun ~4-5 pokok)...")
    sampled = composite.sample(
        region=aoi,
        scale=20,
        geometries=True,
        numPixels=3500
    )
    
    features = sampled.getInfo().get('features', [])
    print(f"      Berhasil mengekstrak {len(features):,} unit grid 20m di dalam poligon.")
    
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
    
    # 5. Klasifikasi Menggunakan Pipeline Machine Learning (RF 2-Layer + iForest Anomaly Scoring)
    print("[3/4] Menjalankan Pipeline Machine Learning (RF 2-Layer + iForest Anomaly Scoring)...")
    df_pred, timing = predict_pipeline(df, return_timing=True)
    df['status'] = df_pred['health_status']
    df['priority_tier'] = df_pred['priority_tier']
    df['anomaly_score'] = df_pred['anomaly_score']
    df['recommended_action'] = df_pred['recommended_action']
    df['vigor_score'] = df_pred['vigor_score']
    
    print(f"      >> [BENCHMARK ML] Selesai memprediksi {timing['total_samples']:,} titik pohon dalam {timing['elapsed_sec']} detik ({timing['elapsed_ms']} ms)!")
    print(f"      >> [KECEPATAN ML] {timing['ms_per_sample']} ms per pohon.")
    
    # Ringkasan Statistik
    stats = df['priority_tier'].value_counts()
    print("\n--- RINGKASAN HASIL TRIAGE KESEHATAN KEBUN (iForest) ---")
    total = len(df)
    for k, v in stats.items():
        pct = (v / total) * 100
        print(f"  {k:<36}: {v:>5,} titik ({pct:>5.1f}%)")
        
    tier1_count = df['priority_tier'].str.contains('Tier 1').sum()
    tier2_count = df['priority_tier'].str.contains('Tier 2').sum()
    palm_total = max(1, total - df['status'].str.contains('Non-Sawit|Bukan').sum())
    tier1_pct = (tier1_count / palm_total) * 100
    
    print(f"\n>> KESIMPULAN BLOK: {tier1_pct:.1f}% area terindikasi Karantina Darurat (Tier 1)!")
    if tier1_pct > 10:
        print(">> PERINGATAN: Blok ini memiliki intensitas infeksi TINGGI. Segera lakukan sanitasi/karantina!")
    elif tier1_pct > 3:
        print(">> PERINGATAN: Ditemukan kluster pohon sakit. Rekomendasi: ground check target Tier 1.")
    else:
        print(">> STATUS AMAN: Kebun mayoritas berada dalam kondisi sehat prima.")
        
    # 6. Pembuatan Peta Interaktif Spasial untuk Poligon Tersebut
    print("\n[4/4] Membangun Peta Interaktif & Menandai Titik Anomali...")
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
    fg_sehat  = folium.FeatureGroup(name='🟢 Tier 3: Normal Sehat', show=True)
    fg_ringan = folium.FeatureGroup(name='🟡 Tier 2: Stres Ringan', show=True)
    fg_sakit  = folium.FeatureGroup(name='🔴 Tier 1: Karantina Darurat (URGENT)', show=True)
    fg_lain   = folium.FeatureGroup(name='⚪ Non-Sawit / Jalan Kebun', show=False)
    
    heat_coords = []
    
    for _, r in df.iterrows():
        tier = str(r['priority_tier'])
        st = str(r['status'])
        lat, lon = r['lat'], r['lon']
        score = float(r['anomaly_score'])
        act = str(r['recommended_action'])
        
        popup_txt = f"""
        <b>Prioritas:</b> {tier}<br>
        <b>Status:</b> {st}<br>
        <b>Skor Anomali:</b> <b>{score:.3f}</b><br>
        <b>GPS:</b> {lat:.5f}, {lon:.5f}<br>
        <b>NDRE (Klorofil):</b> {r['NDRE']:.3f}<br>
        <b>NDVI:</b> {r['NDVI']:.3f}<br>
        <b>MSI:</b> {r['MSI']:.3f}<br>
        <hr style="margin:4px 0;">
        <small><b>Tindakan:</b> {act}</small>
        """
        
        if 'Tier 1' in tier or 'SAWIT SAKIT' in st:
            heat_coords.append([lat, lon, score])
            folium.CircleMarker(
                location=[lat, lon],
                radius=6.5,
                color='#ff0000',
                fill=True,
                fill_color='#ff0000',
                fill_opacity=0.9,
                popup=folium.Popup(popup_txt, max_width=260)
            ).add_to(fg_sakit)
        elif 'Tier 2' in tier or 'Stres' in st:
            if score >= 0.40:
                heat_coords.append([lat, lon, score])
            folium.CircleMarker(
                location=[lat, lon],
                radius=4.5,
                color='#f39c12',
                fill=True,
                fill_color='#f1c40f',
                fill_opacity=0.75,
                popup=folium.Popup(popup_txt, max_width=260)
            ).add_to(fg_ringan)
        elif 'Tier 3' in tier or 'Sehat' in st:
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color='#27ae60',
                fill=True,
                fill_color='#2ecc71',
                fill_opacity=0.6,
                popup=folium.Popup(popup_txt, max_width=260)
            ).add_to(fg_sehat)
        else:
            folium.CircleMarker(
                location=[lat, lon],
                radius=2,
                color='#95a5a6',
                fill=True,
                fill_color='#bdc3c7',
                fill_opacity=0.35,
                popup=folium.Popup(popup_txt, max_width=260)
            ).add_to(fg_lain)
            
    # Tambahkan Weighted Anomaly Heatmap
    if len(heat_coords) > 0:
        fg_heat = folium.FeatureGroup(name='🔥 Weighted Anomaly Heatmap (iForest)', show=True)
        HeatMap(heat_coords, radius=22, blur=15, min_opacity=0.45).add_to(fg_heat)
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
    
    # Ekspor koordinat target ground check terurut berdasarkan Skor Anomali tertinggi
    triage_mask = df['priority_tier'].str.contains('Tier 1|Tier 2') | df['status'].str.contains('SAWIT SAKIT')
    target_df = df[triage_mask].copy()
    if len(target_df) > 0:
        target_df = target_df.sort_values(by='anomaly_score', ascending=False).reset_index(drop=True)
        target_df['Ranking_Prioritas'] = target_df.index + 1
        cols_export = ['Ranking_Prioritas', 'lat', 'lon', 'anomaly_score', 'priority_tier', 'status', 'recommended_action', 'NDRE', 'NDVI', 'MSI']
        target_df = target_df[cols_export]
        output_csv = os.path.join(output_dir, f"target_ground_check_{nama_blok}.csv")
        target_df.to_csv(output_csv, index=False)
        print(f">> Daftar target ground check terurut prioritas darurat diekspor ke:")
        print(f"   {output_csv} (Total: {len(target_df)} pohon prioritas)")
    print("=" * 65)
    
    return output_html

if __name__ == '__main__':
    # Contoh demonstrasi menggunakan salah satu poligon kebun sawit yang ada di GEE
    print("Menguji fungsi dengan poligon kebun sawit riil di Rambah Hilir...")
    sample_poly = ee.FeatureCollection('projects/proyek-ds-488108/assets/sawitDewasa').first()
    inspect_polygon_health(sample_poly, nama_blok="Kebun_Sawit_Dewasa_Blok_A", year=2024)
