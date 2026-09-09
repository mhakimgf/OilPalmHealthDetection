"""
Web GIS Backend: Sistem Deteksi Kesehatan Sawit Berbasis Poligon Kustom
=======================================================================
Menggunakan Flask, Google Earth Engine, dan Pipeline 2-Layer AI
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
import ee

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__, template_folder='templates', static_folder='static')

# Inisialisasi GEE
GEE_AVAILABLE = False
try:
    ee.Initialize(project='ardent-particle-480118-k7')
    GEE_AVAILABLE = True
    print(">> [GEE] Terhubung sukses dengan project ardent-particle-480118-k7!")
except Exception as e:
    print(f">> [GEE] Gagal terhubung: {e}. Menggunakan fallback simulator.")

@app.route('/')
def index():
    return render_template('index.html', gee_status=GEE_AVAILABLE)

@app.route('/api/predict_polygon', methods=['POST'])
def predict_polygon():
    """
    Menerima GeoJSON poligon dari antarmuka Web GIS,
    mengekstraksi data Sentinel-2 via GEE, dan mengembalikan hasil prediksi 2-layer.
    """
    try:
        data = request.get_json()
        if not data or 'coordinates' not in data:
            return jsonify({'success': False, 'error': 'Format koordinat poligon tidak valid.'}), 400
            
        coords = data['coordinates'] # Format: [[[lon, lat], [lon, lat], ...]]
        year = int(data.get('year', 2024))
        
        # Validasi poligon
        if len(coords[0]) < 3:
            return jsonify({'success': False, 'error': 'Poligon membutuhkan minimal 3 titik koordinat.'}), 400
            
        print(f">> [API] Menerima poligon dengan {len(coords[0])} titik. Menjalankan analisis tahun {year}...")
        
        # Ekstrak data via GEE jika online
        if GEE_AVAILABLE:
            result = process_gee_polygon(coords, year)
        else:
            result = simulate_polygon_analysis(coords)
            
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        print(f">> [API Error] {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def process_gee_polygon(coords, year):
    """Memproses poligon nyata di Google Earth Engine."""
    aoi = ee.Geometry.Polygon(coords)
    
    # Koleksi Sentinel-2
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45)) \
        .median() \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']) \
        .clip(aoi)
        
    # Band dan Indeks Spektral Sentinel-2 Komprehensif
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndre = s2.normalizedDifference(['B8A', 'B5']).rename('NDRE')
    ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
    msi  = s2.select('B11').divide(s2.select('B8').add(1e-6)).rename('MSI')
    nir_red = s2.select('B8').divide(s2.select('B4').add(1e-6)).rename('NIR_RED')
    b4   = s2.select('B4').divide(10000.0).rename('B4')
    
    composite = ee.Image.cat([ndvi, ndre, ndbi, msi, nir_red, b4])
    
    # Hitung estimasi luas area (hektar)
    area_sqm = aoi.area().getInfo()
    area_ha  = round(area_sqm / 10000.0, 2)
    
    # Sampling grid titik (skala 10m)
    scale = 10 if area_ha < 100 else 20
    sampled = composite.sample(region=aoi, scale=scale, geometries=True, numPixels=2500)
    features = sampled.getInfo().get('features', [])
    
    # Parsing ke list titik
    points = []
    sick_count = 0
    stressed_count = 0
    healthy_count = 0
    non_veg_count = 0
    non_palm_count = 0
    
    mean_ndre, std_ndre = 0.589, 0.037
    mean_ndvi, std_ndvi = 0.822, 0.031
    mean_msi,  std_msi  = 0.548, 0.052
    
    for f in features:
        props = f.get('properties', {})
        geom_coords = f.get('geometry', {}).get('coordinates', [0, 0])
        
        lon, lat = geom_coords[0], geom_coords[1]
        val_ndvi = float(props.get('NDVI', 0))
        val_ndre = float(props.get('NDRE', 0))
        val_ndbi = float(props.get('NDBI', 0))
        val_msi  = float(props.get('MSI', 0))
        val_ratio = float(props.get('NIR_RED', 0))
        val_b4   = float(props.get('B4', 0))
        
        # 1. LAYER 1: FILTER NON-VEGETASI (Jalan Aspal, Atap Seng/Rumah, Tanah Gundul, Semen, Air)
        # Karakteristik: NDBI tinggi/positif, rasio NIR/Red rendah, pantulan merah tinggi
        if (val_ndvi < 0.48) or (val_ndbi > -0.06) or (val_ratio < 2.8) or (val_b4 > 0.085):
            status = 'Bukan Vegetasi (Jalan / Bangunan / Tanah)'
            color = '#94a3b8'
            non_veg_count += 1
            vigor = -999
            
        # 2. LAYER 2: FILTER NON-SAWIT (Bahu Jalan Berumput, Pekarangan Rumah Warga, Semak Belukar)
        # Karakteristik kanopi sawit rapat: NDBI <= -0.15, NIR/Red >= 4.5, NDVI >= 0.65, B4 <= 0.060
        elif (val_ndbi > -0.15) or (val_ratio < 4.5) or (val_ndvi < 0.65) or (val_b4 > 0.060):
            status = 'Non-Sawit (Bahu Jalan / Pekarangan / Semak)'
            color = '#64748b'
            non_palm_count += 1
            vigor = -999
            
        # 3. LAYER 3: HEALTH ASSESSMENT KHUSUS KANOPÍ KELAPA SAWIT SEJATI
        else:
            z_ndre = (val_ndre - mean_ndre) / std_ndre
            z_ndvi = (val_ndvi - mean_ndvi) / std_ndvi
            z_msi  = (val_msi  - mean_msi)  / std_msi
            vigor  = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
            
            # Sawit Sakit Sejati (Ganoderma / Klorosis Berat / Crown Dieback):
            # Tajuk pohon mengalami penurunan klorofil drastis (NDRE < 0.46) DAN dehidrasi pelepah (MSI > 0.65)
            if (val_ndre < 0.46 and val_ndvi < 0.72 and val_msi > 0.65) or (vigor < -2.2):
                status = '🔴 SAWIT SAKIT (Target Karantina)'
                color = '#e74c3c'
                sick_count += 1
            elif val_ndvi >= 0.76 and val_ndre >= 0.52 and val_msi <= 0.65:
                status = '🟢 Sawit Sehat Prima'
                color = '#2ecc71'
                healthy_count += 1
            else:
                status = '🟡 Stres Ringan (Pupuk/Air)'
                color = '#f1c40f'
                stressed_count += 1
                
        points.append({
            'lat': lat,
            'lon': lon,
            'status': status,
            'color': color,
            'vigor': round(float(vigor), 2) if vigor != -999 else None,
            'ndvi': round(float(val_ndvi), 3),
            'ndre': round(float(val_ndre), 3),
            'ndbi': round(float(val_ndbi), 3),
            'msi': round(float(val_msi), 3)
        })
        
    total_pts = len(points)
    total_filtered_out = non_veg_count + non_palm_count
    palm_total = max(1, (healthy_count + stressed_count + sick_count))
    
    sick_pct = round((sick_count / palm_total) * 100, 1)
    stressed_pct = round((stressed_count / palm_total) * 100, 1)
    healthy_pct = round((healthy_count / palm_total) * 100, 1)
    
    if sick_pct > 10:
        alert = "BAHAYA: Kluster Penyakit Tinggi (>10%)"
        alert_class = "danger"
    elif sick_pct > 3:
        alert = "WASPADA: Indikasi Infeksi Terdeteksi (>3%)"
        alert_class = "warning"
    else:
        alert = "AMAN: Kanopi Sawit Mayoritas Sehat"
        alert_class = "success"
        
    return {
        'points': points,
        'summary': {
            'area_ha': area_ha,
            'total_points': total_pts,
            'palm_points': palm_total,
            'filtered_points': total_filtered_out,
            'healthy_count': healthy_count,
            'healthy_pct': healthy_pct,
            'stressed_count': stressed_count,
            'stressed_pct': stressed_pct,
            'sick_count': sick_count,
            'sick_pct': sick_pct,
            'alert': alert,
            'alert_class': alert_class
        }
    }

def simulate_polygon_analysis(coords):
    """Simulasi cerdas jika GEE offline."""
    lons = [c[0] for c in coords[0]]
    lats = [c[1] for c in coords[0]]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    points = []
    sick_count = 0
    stressed_count = 0
    healthy_count = 0
    
    for _ in range(250):
        lat = np.random.uniform(min_lat, max_lat)
        lon = np.random.uniform(min_lon, max_lon)
        rnd = np.random.rand()
        if rnd < 0.65:
            status = '🟢 Sawit Sehat Prima'
            color = '#2ecc71'
            vigor = np.random.uniform(-0.4, 1.2)
            healthy_count += 1
        elif rnd < 0.88:
            status = '🟡 Sawit Stres Ringan'
            color = '#f1c40f'
            vigor = np.random.uniform(-1.4, -0.6)
            stressed_count += 1
        else:
            status = '🔴 SAWIT SAKIT (Stres Berat / Ganoderma)'
            color = '#e74c3c'
            vigor = np.random.uniform(-3.5, -1.6)
            sick_count += 1
            
        points.append({
            'lat': lat, 'lon': lon, 'status': status, 'color': color,
            'vigor': round(float(vigor), 2),
            'ndvi': round(float(0.75 + vigor*0.05), 3),
            'ndre': round(float(0.50 + vigor*0.04), 3),
            'msi': round(float(0.60 - vigor*0.04), 3)
        })
        
    total = len(points)
    return {
        'points': points,
        'summary': {
            'area_ha': 45.2,
            'total_points': total,
            'palm_points': total,
            'healthy_count': healthy_count,
            'healthy_pct': round((healthy_count/total)*100, 1),
            'stressed_count': stressed_count,
            'stressed_pct': round((stressed_count/total)*100, 1),
            'sick_count': sick_count,
            'sick_pct': round((sick_count/total)*100, 1),
            'alert': "WASPADA: Kluster Sakit Terdeteksi",
            'alert_class': "warning"
        }
    }

if __name__ == '__main__':
    port = 5000
    print(f">> Web GIS Berjalan di: http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
