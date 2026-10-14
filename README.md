# Oil Palm Health Detection (Deteksi Kesehatan Kelapa Sawit Berbasis Satelit Sentinel-2 & Web GIS)

Sistem cerdas komprehensif untuk klasifikasi tutupan lahan kelapa sawit dan diagnosa tingkat kesehatan pohon sawit menggunakan citra satelit **Sentinel-2**, **Google Earth Engine (GEE)**, dan pemodelan **Hybrid Machine Learning (Random Forest 2-Layer + Isolation Forest Anomaly Scoring)**.

---

## 🌟 Fitur Utama

1. **Pipeline 3-Layer (Hybrid Supervised + Unsupervised AI)**:
   - **Layer 1 (Random Forest)**: Memisahkan Vegetasi vs Non-Vegetasi (Akurasi 95.50%, F1 97.42%).
   - **Layer 2 (Random Forest)**: Memisahkan Kelapa Sawit vs Hutan / Non-Sawit (Akurasi 92.15%, F1 95.28%).
   - **Layer 3 (Isolation Forest Anomaly Scoring & Directional Gatekeeper)**:
     - Menghasilkan **Continuous Anomaly Score ($0.000$ s.d $1.000$)** untuk setiap pohon sawit.
     - Mengatasi masalah *class imbalance* ekstrem di mana pohon sakit adalah kasus langka ($<5\%$).
     - Dilengkapi **Directional Gatekeeper** untuk membedakan secara biologis antara penyakit kronis (*Ganoderma*), stres pupuk/air, replanting, dan kanopi super subur.

2. **Sistem Triage Prioritas Lapangan**:
   | Tingkat Prioritas | Skor Anomali | Indikator Spektral | Tindakan Rekomendasi Lapangan |
   | :--- | :---: | :--- | :--- |
   | **🔴 Tier 1: Karantina Darurat** | $\ge 0.65$ | Klorosis parah (NDRE anjlok), tajuk kering (MSI melonjak) | **URGENT**: Cek jamur Ganoderma di pangkal batang; buat parit isolasi (*trenching*). |
   | **🟡 Tier 2: Stres Ringan** | $0.40 - 0.65$ | Penurunan biomassa ringan, fluktuasi hara | **Inspeksi Agronomi**: Cek pelepah kuning/kering, dosis pupuk N-P-K, atau drainase. |
   | **🟢 Tier 3: Normal Sehat** | $< 0.40$ | Kanopi rimbun, klorofil tinggi stabil | **Perawatan Rutin**: Pemupukan berkala dan panen standar. |
   | **🟢 Super Sehat** | $\ge 0.60$ (Arah Positif) | NDRE $\ge 0.62$, NDVI $\ge 0.84$ | **Super Vigor**: Pohon sangat produktif; pertahankan manajemen kebun. |

3. **Eliminasi False Positive Non-Sawit & Jalanan (Guardrail Biofisika)**:
   - Filter biofisika multi-indeks (`NDBI`, `NIR/Red`, `B4`, `NDVI`, `NDRE`, `MSI`) untuk mencegah bahu jalan, aspal, tanah terbuka, dan atap perumahan warga terdeteksi sebagai "sawit sakit".

4. **Kecepatan Inferensi Ultra Cepat (<0.06 ms per pohon)**:
   - Pipeline Machine Learning yang telah dioptimasi mampu memproses hingga **5.000 pohon dalam ~0.25 detik** pada CPU biasa secara real-time.

5. **Web GIS Interaktif & Weighted Heatmap**:
   - Dashboard Web GIS (Flask + Leaflet) interaktif untuk menggambar poligon bebas di peta satelit.
   - **Weighted Heatmap Anomali** kontinu berbobot skor Isolation Forest.
   - Ekspor target GPS ground check terurut berdasarkan ranking skor anomali tertinggi.

---

## 📁 Struktur Direktori

```
OilPalmHealthDetection/
├── models/
│   ├── model_layer1_rf_vegetasi.pkl       # Layer 1: Random Forest Vegetasi
│   ├── model_layer2_rf_sawit.pkl          # Layer 2: Random Forest Sawit
│   ├── model_layer3_iforest_health.pkl    # Layer 3: Isolation Forest Multi-Spektral
│   ├── model_layer3_iforest_temporal.pkl  # Layer 3: Isolation Forest Dinamika Temporal
│   └── health_calibration.json            # Parameter baseline kanopi sawit sehat
├── plots/
│   ├── anomaly_score_distribution.png     # Grafik sebaran Continuous Anomaly Score
│   └── hasil_pipeline_akhir.png           # Evaluasi perbandingan model
├── results/
│   └── target_ground_check_prioritas.csv  # Daftar koordinat GPS urut prioritas darurat
├── web_gis/
│   ├── static/                            # CSS modern dark-mode, Leaflet app.js
│   ├── templates/                         # Antarmuka Web GIS (index.html)
│   └── app.py                             # Server Flask backend Web GIS
├── 01_baseline_vs_hierarchical.py         # Skrip komparasi flat vs hierarkis
├── 02_palm_health_analysis.py             # Analisis biometrik spektral kanopi
├── 03_hierarchical_palm_health_pipeline.py# Pipeline training awal model hierarkis
├── 04_peta_interaktif_kesehatan_sawit.py  # Visualisasi peta interaktif Folium
├── 05_gee_temporal_trend_detector.py      # Analisis kemiringan tren temporal Sentinel-2 GEE
├── 06_predict_custom_polygon.py           # CLI inspeksi poligon kustom kebun
├── 07_anomaly_scoring_detector.py         # Evaluasi mandiri Isolation Forest Anomaly Scoring
├── predict_point_or_csv.py                # Core inference engine 3-layer
└── README.md                              # Dokumentasi teknis sistem
```

---

## 🚀 Panduan Eksekusi

### 1. Menjalankan Evaluasi Anomaly Scoring
```bash
python 07_anomaly_scoring_detector.py
```

### 2. Menjalankan Web GIS Interaktif
1. Pastikan dependensi terpasang:
   ```bash
   pip install flask earthengine-api pandas numpy scikit-learn joblib folium
   ```
2. Jalankan server Flask:
   ```bash
   cd web_gis
   python app.py
   ```
3. Buka browser pada alamat: **`http://localhost:5000`**

### 3. Menginspeksi Poligon Kustom via CLI
```bash
python 06_predict_custom_polygon.py
```

---

## 📄 Lisensi & Kontribusi
Dikembangkan untuk riset dan monitoring presisi kesehatan perkebunan kelapa sawit (*precision agriculture*) berbasis satelit Sentinel-2.
