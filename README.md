# Oil Palm Health Detection (Deteksi Kesehatan Kelapa Sawit Berbasis Satelit Sentinel-2 & Web GIS)

Repositori ini berisi sistem komprehensif untuk klasifikasi tutupan lahan kelapa sawit dan deteksi tingkat kesehatan pohon sawit (Sehat Prima, Stres Ringan, dan Sawit Sakit/Kritis) menggunakan citra satelit **Sentinel-2**, **Google Earth Engine (GEE)**, dan pemodelan **Hierarchical Machine Learning (XGBoost)**.

---

## 🌟 Fitur Utama

1. **Model Hierarkis 2-Layer + Health Assessment**:
   - **Layer 1 (XGBoost)**: Memisahkan Vegetasi vs Non-Vegetasi (Akurasi 95.50%, F1 97.42%).
   - **Layer 2 (XGBoost)**: Memisahkan Kelapa Sawit vs Hutan / Non-Sawit (Akurasi 92.15%, F1 95.28%).
   - **Layer 3 (Modul Kesehatan)**: Deteksi tingkat kesehatan kanopi sawit menggunakan *Multi-Spectral Vigor Score (MVS)* berbasis Z-score deviasi statistik dari parameter baseline sawit dewasa sehat.
2. **Eliminasi False Positive Non-Sawit & Jalanan**:
   - Filter biofisika multi-indeks (`NDBI`, `NIR/Red`, `B4`, `NDVI`, `NDRE`, `MSI`) untuk mencegah bahu jalan, aspal, tanah terbuka, dan atap perumahan warga terdeteksi sebagai "sawit sakit".
3. **Web GIS & Visualisasi Interaktif**:
   - Dashboard Web GIS (Flask + Leaflet) interaktif untuk menggambar poligon kebun sawit, inspeksi kesehatan kanopi per pohon (grid 10 meter), visualisasi *disease heatmap*, dan ekspor koordinat target *ground check*.
4. **Validasi Tren Temporal (Google Earth Engine)**:
   - Analisis kemiringan tren temporal (*time series slope*) multi-tahun untuk mengonfirmasi infeksi kronis (misal *Ganoderma boninense*).

---

## 📁 Struktur Direktori

- **`web_gis/`**: Aplikasi Web GIS interaktif (Backend Flask, Leaflet map, drawing tools, live GEE analysis).
- **`models/`**: Model machine learning terlatih (`model_layer1_vegetasi.pkl`, `model_layer2_sawit.pkl`) dan file parameter kalibrasi (`health_calibration.json`).
- **`plots/`**: Visualisasi evaluasi model (Confusion Matrix kaskade, grafik biometrik kesehatan).
- **`output_polygon/`**: Contoh output inspeksi poligon kebun (peta HTML mandiri dan target ground check CSV).
- **`results/`**: Log evaluasi metrik kinerja model (JSON).

### Daftar Skrip Penting

1. **`01_baseline_vs_hierarchical.py`**: Perbandingan langsung model baseline flat 9-class vs model hierarkis 2-layer.
2. **`02_palm_health_analysis.py`**: Analisis statistik biometrik spektral kanopi sawit (fase muda, remaja, dewasa).
3. **`03_hierarchical_palm_health_pipeline.py`**: Pipeline training end-to-end model hierarkis dan kalibrasi modul kesehatan.
4. **`04_peta_interaktif_kesehatan_sawit.py`**: Pembuatan peta interaktif Folium berbasis data spasial tabular.
5. **`05_gee_temporal_trend_detector.py`**: Deteksi tren temporal kemiringan vegetasi (*slope analysis*) via Google Earth Engine.
6. **`06_predict_custom_polygon.py`**: CLI untuk inspeksi poligon kustom kebun dengan ekstraksi Sentinel-2 GEE dan pelaporan target sakit.
7. **`predict_point_or_csv.py`**: Skrip inferensi siap pakai untuk memprediksi file CSV baru menggunakan model terlatih.
8. **`METODOLOGI_KOMPREHENSIF_SAWIT.ipynb`**: Dokumentasi saintifik lengkap, teori agronomi remote sensing, dan kode eksperimen interaktif.

---

## 🚀 Cara Menjalankan Web GIS

1. Pastikan dependensi terpasang:
   ```bash
   pip install flask earthengine-api pandas numpy scikit-learn xgboost joblib
   ```
2. Autentikasi Google Earth Engine (jika belum):
   ```bash
   earthengine authenticate
   ```
3. Jalankan server Flask:
   ```bash
   cd web_gis
   python app.py
   ```
4. Buka browser pada alamat `http://localhost:5000`.

---

## 📄 Lisensi & Kontribusi
Dikembangkan untuk riset dan monitoring kesehatan perkebunan kelapa sawit presisi tinggi berbasis penginderaan jauh (*precision agriculture*).
