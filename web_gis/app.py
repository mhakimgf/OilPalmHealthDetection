"""
Web GIS Backend: Sistem Deteksi Kesehatan Sawit Berbasis Poligon Kustom
=======================================================================
Menggunakan Flask, Google Earth Engine, dan Pipeline Machine Learning 2-Layer (XGBoost)
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
import ee

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import module inferensi machine learning
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from predict_point_or_csv import predict_pipeline, FEATURES

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
    """Memproses poligon nyata di Google Earth Engine dan memprediksi dengan XGBoost."""
    t_start_total = time.time()
    aoi = ee.Geometry.Polygon(coords)
    
    start_date = f"{year}-01-01"
    end_date   = f"{year}-12-31"
    
    t_gee_0 = time.time()
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45)) \
        .median() \
        .divide(10000.0) \
        .clip(aoi)
        
    raw_bands = s2.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    evi = s2.expression('2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
        {'NIR': s2.select('B8'), 'RED': s2.select('B4'), 'BLUE': s2.select('B2')}).rename('EVI')
    savi = s2.expression('1.5 * (NIR - RED) / (NIR + RED + 0.5)',
        {'NIR': s2.select('B8'), 'RED': s2.select('B4')}).rename('SAVI')
    ndre = s2.normalizedDifference(['B8', 'B5']).rename('NDRE')
    ndwi = s2.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = s2.normalizedDifference(['B11', 'B8']).rename('NDBI')
    msi  = s2.select('B11').divide(s2.select('B8').add(1e-6)).rename('MSI')

    # GLCM Texture extraction
    nir_scaled = s2.select('B8').multiply(64).toInt().rename('NIR_scaled')
    glcm = nir_scaled.glcmTexture(size=3)
    texture = glcm.select(['NIR_scaled_contrast', 'NIR_scaled_var', 'NIR_scaled_ent']).rename(['GLCM_Contrast', 'GLCM_Variance', 'GLCM_Entropy'])

    composite = raw_bands.addBands([ndvi, evi, savi, ndre, ndwi, ndbi, msi, texture])
    
    # Hitung estimasi luas area (hektar)
    area_sqm = aoi.area().getInfo()
    area_ha  = round(area_sqm / 10000.0, 2)
    
    # Sampling grid titik (skala 10m)
    scale = 10 if area_ha < 100 else 20
    sampled = composite.sample(region=aoi, scale=scale, geometries=True, numPixels=2500)
    features = sampled.getInfo().get('features', [])
    t_gee_1 = time.time()
    gee_time_sec = round(t_gee_1 - t_gee_0, 2)
    
    # Parsing ke DataFrame
    records = []
    for f in features:
        props = f.get('properties', {})
        geom_coords = f.get('geometry', {}).get('coordinates', [0, 0])
        props['lon'] = geom_coords[0]
        props['lat'] = geom_coords[1]
        records.append(props)
        
    df = pd.DataFrame(records)
    if len(df) == 0:
        return {
            'points': [],
            'summary': {
                'area_ha': area_ha, 'total_points': 0, 'palm_points': 0,
                'filtered_points': 0, 'healthy_count': 0, 'healthy_pct': 0,
                'stressed_count': 0, 'stressed_pct': 0, 'sick_count': 0, 'sick_pct': 0,
                'alert': 'Tidak ada data', 'alert_class': 'warning',
                'ml_timing': {'elapsed_sec': 0, 'elapsed_ms': 0, 'ms_per_sample': 0},
                'gee_time_sec': gee_time_sec
            }
        }
        
    # Jalankan Prediksi dengan Model Machine Learning (XGBoost)
    df_pred, ml_timing = predict_pipeline(df, return_timing=True)
    
    # Pemetaan hasil prediksi ke titik visualisasi Web GIS
    points = []
    sick_count = 0
    stressed_count = 0
    healthy_count = 0
    young_count = 0
    filtered_count = 0
    
    for _, row in df_pred.iterrows():
        status = row['health_status']
        vigor = row.get('vigor_score')
        vigor_val = round(float(vigor), 2) if pd.notnull(vigor) else None
        
        if '🔴' in status or 'Sakit' in status:
            color = '#e74c3c'
            sick_count += 1
        elif '🟢' in status or 'Sehat' in status:
            color = '#2ecc71'
            healthy_count += 1
        elif '🟡' in status or 'Stres' in status:
            color = '#f1c40f'
            stressed_count += 1
        elif 'Muda' in status:
            color = '#3498db'
            young_count += 1
        else:
            color = '#94a3b8'
            filtered_count += 1
            
        points.append({
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'status': str(status),
            'color': color,
            'vigor': vigor_val,
            'ndvi': round(float(row.get('NDVI', 0)), 3),
            'ndre': round(float(row.get('NDRE', 0)), 3),
            'ndbi': round(float(row.get('NDBI', 0)), 3),
            'msi': round(float(row.get('MSI', 0)), 3)
        })
        
    total_pts = len(points)
    palm_total = max(1, (healthy_count + stressed_count + sick_count + young_count))
    
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
            'filtered_points': filtered_count,
            'healthy_count': healthy_count,
            'healthy_pct': healthy_pct,
            'stressed_count': stressed_count,
            'stressed_pct': stressed_pct,
            'sick_count': sick_count,
            'sick_pct': sick_pct,
            'alert': alert,
            'alert_class': alert_class,
            'ml_timing': ml_timing,
            'gee_time_sec': gee_time_sec,
            'model_engine': 'XGBoost 2-Layer (Hierarchical AI)'
        }
    }

def simulate_polygon_analysis(coords):
    """Simulasi cerdas menjalankan model ML pada titik sintetis jika GEE offline."""
    lons = [c[0] for c in coords[0]]
    lats = [c[1] for c in coords[0]]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    records = []
    np.random.seed(42)
    for _ in range(250):
        records.append({
            'lon': np.random.uniform(min_lon, max_lon),
            'lat': np.random.uniform(min_lat, max_lat),
            'B2': np.random.uniform(0.02, 0.05),
            'B3': np.random.uniform(0.04, 0.07),
            'B4': np.random.uniform(0.02, 0.05),
            'B5': np.random.uniform(0.08, 0.12),
            'B6': np.random.uniform(0.20, 0.30),
            'B7': np.random.uniform(0.28, 0.38),
            'B8': np.random.uniform(0.30, 0.45),
            'B8A': np.random.uniform(0.32, 0.46),
            'B11': np.random.uniform(0.15, 0.25),
            'B12': np.random.uniform(0.08, 0.15),
            'NDVI': np.random.uniform(0.70, 0.88),
            'NDRE': np.random.uniform(0.48, 0.65),
            'NDBI': np.random.uniform(-0.35, -0.18),
            'MSI': np.random.uniform(0.45, 0.65),
            'GLCM_Contrast': np.random.uniform(0.6, 1.1),
            'GLCM_Entropy': np.random.uniform(1.8, 2.3),
            'GLCM_Variance': np.random.uniform(0.5, 0.9)
        })
    df = pd.DataFrame(records)
    df_pred, ml_timing = predict_pipeline(df, return_timing=True)
    
    points = []
    sick_count = 0
    stressed_count = 0
    healthy_count = 0
    filtered_count = 0
    
    for _, row in df_pred.iterrows():
        status = row['health_status']
        vigor = row.get('vigor_score')
        vigor_val = round(float(vigor), 2) if pd.notnull(vigor) else None
        
        if '🔴' in status or 'Sakit' in status:
            color = '#e74c3c'
            sick_count += 1
        elif '🟢' in status or 'Sehat' in status:
            color = '#2ecc71'
            healthy_count += 1
        elif '🟡' in status or 'Stres' in status:
            color = '#f1c40f'
            stressed_count += 1
        else:
            color = '#94a3b8'
            filtered_count += 1
            
        points.append({
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'status': str(status),
            'color': color,
            'vigor': vigor_val,
            'ndvi': round(float(row.get('NDVI', 0)), 3),
            'ndre': round(float(row.get('NDRE', 0)), 3),
            'ndbi': round(float(row.get('NDBI', 0)), 3),
            'msi': round(float(row.get('MSI', 0)), 3)
        })
        
    total_pts = len(points)
    palm_total = max(1, (healthy_count + stressed_count + sick_count))
    sick_pct = round((sick_count / palm_total) * 100, 1)
    stressed_pct = round((stressed_count / palm_total) * 100, 1)
    healthy_pct = round((healthy_count / palm_total) * 100, 1)
    
    return {
        'points': points,
        'summary': {
            'area_ha': 4.8,
            'total_points': total_pts,
            'palm_points': palm_total,
            'filtered_points': filtered_count,
            'healthy_count': healthy_count,
            'healthy_pct': healthy_pct,
            'stressed_count': stressed_count,
            'stressed_pct': stressed_pct,
            'sick_count': sick_count,
            'sick_pct': sick_pct,
            'alert': "SIMULASI OFFLINE: Data dievaluasi via XGBoost lokal",
            'alert_class': "warning",
            'ml_timing': ml_timing,
            'gee_time_sec': 0.01,
            'model_engine': 'XGBoost 2-Layer (Simulasi)'
        }
    }

if __name__ == '__main__':
    print(">> Menjalankan Web GIS Server pada http://localhost:5000 ...")
    app.run(debug=True, port=5000)
