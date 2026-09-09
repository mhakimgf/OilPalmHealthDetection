"""
SANDBOX Experiment 05: Deteksi Tren Penurunan Temporal Sawit Sakit di Google Earth Engine
========================================================================================
Mengimplementasikan teori: Sawit yang sakit menunjukkan tren penurunan vegetasi (NDVI/NDRE)
secara konsisten dari waktu ke waktu (Negative Temporal Trend / Degradation Slope).

Metode:
1. Memuat time series citra Sentinel-2 (2022 - 2024) di ROI Rambah Hilir
2. Menghitung NDVI dan NDRE per tanggal observasi
3. Mengaplikasikan ee.Reducer.linearFit() untuk menghitung kemiringan tren (Slope d(NDVI)/dt)
4. Masking: Hanya mengevaluasi di area kebun kelapa sawit
5. Klasifikasi Tren Spasial:
   - Hijau  : Tren Stabil / Meningkat (Sawit Sehat)
   - Kuning : Penurunan Ringan / Fluktuasi Musiman
   - Merah  : Penurunan Drastis Konsisten (TERINDIKASI SAKIT / GANODERMA / MATI)
"""

import sys
import ee

sys.stdout.reconfigure(encoding='utf-8')

print("[1/5] Menginisialisasi Google Earth Engine...")
ee.Initialize(project='ardent-particle-480118-k7')
print("      Otentikasi GEE Berhasil!")

# 1. DEFINISI REGION OF INTEREST (ROI)
# Poligon sawit Rambah Hilir dari asset GEE
sawit_dewasa = ee.FeatureCollection('projects/proyek-ds-488108/assets/sawitDewasa')
sawit_remaja = ee.FeatureCollection('projects/proyek-ds-488707/assets/sawitRemaja')
kebun_sawit  = sawit_dewasa.merge(sawit_remaja)
AOI = kebun_sawit.geometry().bounds()

# 2. KOLEKSI CITRA SENTINEL-2 TIME SERIES (2022 s.d 2024)
print("[2/5] Menyiapkan Time Series Sentinel-2 (2022 - 2024)...")
START_DATE = '2022-01-01'
END_DATE   = '2024-12-31'

def prep_s2(img):
    # Hitung waktu dalam satuan tahun dari 2022
    time_years = ee.Date(img.get('system:time_start')).difference('2022-01-01', 'year');
    
    # NDVI = (B8 - B4) / (B8 + B4)
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # NDRE = (B8A - B5) / (B8A + B5)
    ndre = img.normalizedDifference(['B8A', 'B5']).rename('NDRE')
    
    # Tambahkan band waktu dan indeks
    time_band = ee.Image.constant(time_years).rename('time').toFloat()
    
    return img.addBands([time_band, ndvi, ndre]) \
              .select(['time', 'NDVI', 'NDRE']) \
              .set('system:time_start', img.get('system:time_start'))

# Load citra jernih
s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterBounds(AOI) \
    .filterDate(START_DATE, END_DATE) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
    .map(prep_s2)

count = s2_col.size().getInfo()
print(f"      Total citra Sentinel-2 dalam rentang waktu: {count} citra.")

# 3. REGRESI LINIER TEMPORAL PER PIKSEL
print("[3/5] Menghitung Linear Trend Slope per piksel (d(NDVI)/dt)...")
# ee.Reducer.linearFit() membutuhkan 2 band: [X, Y] -> [time, NDVI]
# Output: band 'scale' (slope) dan 'offset'
trend_ndvi = s2_col.select(['time', 'NDVI']).reduce(ee.Reducer.linearFit())
slope_ndvi = trend_ndvi.select('scale').rename('ndvi_slope')

trend_ndre = s2_col.select(['time', 'NDRE']).reduce(ee.Reducer.linearFit())
slope_ndre = trend_ndre.select('scale').rename('ndre_slope')

# 4. POTONG HANYA PADA KEBUN SAWIT (MASKING)
slope_sawit = slope_ndvi.clip(kebun_sawit)

# 5. KLASIFIKASI KESEHATAN BERBASIS TREN WAKTU
# Threshold slope:
# - slope >= -0.02 / tahun : Stabil / Sehat
# - -0.06 <= slope < -0.02 : Penurunan Ringan / Waspada
# - slope < -0.06 / tahun  : Penurunan Drastis (Sakit Kritis)
print("[4/5] Mengklasifikasikan Area Berdasarkan Kemiringan Tren...")

healthy_mask   = slope_sawit.gte(-0.02)
moderate_mask  = slope_sawit.lt(-0.02).And(slope_sawit.gte(-0.06))
unhealthy_mask = slope_sawit.lt(-0.06) # SAKIT / DEGRADASI TAJAM

# Buat Image 1-band klasifikasi tren:
# 1: Sehat (Stabil), 2: Waspada, 3: Sakit (Tren Turun)
classified_trend = ee.Image(0) \
    .where(healthy_mask, 1) \
    .where(moderate_mask, 2) \
    .where(unhealthy_mask, 3) \
    .clip(kebun_sawit)

# Hitung statistik persentase area sakit
print("[5/5] Menghitung estimasi proporsi sawit sakit...")
area_stats = classified_trend.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=AOI,
    scale=20,
    maxPixels=1e8
).getInfo()

print("=" * 65)
print("HASIL ANALISIS TREN TEMPORAL SAWIT (2022 - 2024):")
counts = area_stats.get('constant', {})
total_pixels = sum(counts.values()) if counts else 1
for k, v in counts.items():
    lbl = "0: Latar Belakang"
    if k == '1': lbl = "1: 🟢 Sawit Stabil / Sehat (Tren Positif / Datar)"
    elif k == '2': lbl = "2: 🟡 Sawit Penurunan Ringan (Stres Awal / Fluktuasi)"
    elif k == '3': lbl = "3: 🔴 SAWIT SAKIT (Tren Menurun Tajam d(NDVI)/dt < -0.06)"
    pct = (v / total_pixels) * 100
    print(f"  {lbl:<55}: {v:>6,} piksel ({pct:>5.1f}%)")
print("=" * 65)

# Ekspor URL Peta Tile untuk visualisasi langsung di browser
map_id = classified_trend.getMapId({
    'min': 1, 'max': 3,
    'palette': ['#2ecc71', '#f1c40f', '#e74c3c']
})
print("\nPeta raster hasil analisis tren siap divisualisasikan!")
print(f"Tile URL: {map_id['tile_fetcher'].url_format}")
