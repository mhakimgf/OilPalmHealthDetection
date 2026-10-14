"""
Pipeline Inferensi 3-Layer: Random Forest 2-Layer + Isolation Forest Anomaly Scoring
=====================================================================================
Arsitektur Sistem:
1. Layer 1 (Random Forest): Filter Vegetasi vs Non-Vegetasi
2. Layer 2 (Random Forest): Filter Kelapa Sawit vs Hutan / Non-Sawit
3. Layer 3 (Isolation Forest Anomaly Scoring & Directional Gatekeeper):
   - Continuous Anomaly Score (0.000 s.d 1.000)
   - Directional Gatekeeper: Membedakan penyakit sejati vs stres hara vs kanopi super lebat
   - Sistem Triage Prioritas Lapangan:
     * 🔴 Tier 1: Karantina Darurat (Suspek Ganoderma / Sakit Kronis)
     * 🟡 Tier 2: Stres Ringan (Defisiensi Hara N-P-K / Drainase)
     * 🟢 Tier 3: Normal Sehat (Kanopi Prima)
     * 🟢 Super Sehat (Anomali Positif Kanopi Super Rapat)
     * ⚪ Filter Tepi Jalan / Non-Sawit (Guardrail Biofisika)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Path Model Layer 1 & Layer 2 (Random Forest dengan Fallback XGBoost)
MODEL_L1_PATH = os.path.join(BASE_DIR, 'models', 'model_layer1_rf_vegetasi.pkl')
MODEL_L2_PATH = os.path.join(BASE_DIR, 'models', 'model_layer2_rf_sawit.pkl')
CALIB_PATH    = os.path.join(BASE_DIR, 'models', 'health_calibration.json')

if not os.path.exists(MODEL_L1_PATH):
    MODEL_L1_PATH = os.path.join(BASE_DIR, 'models', 'model_layer1_vegetasi.pkl')
if not os.path.exists(MODEL_L2_PATH):
    MODEL_L2_PATH = os.path.join(BASE_DIR, 'models', 'model_layer2_sawit.pkl')

model_l1 = joblib.load(MODEL_L1_PATH)
model_l2 = joblib.load(MODEL_L2_PATH)

with open(CALIB_PATH, 'r') as f:
    calib = json.load(f)

FEATURES = calib['features']

# 2. Path Model Layer 3 (Isolation Forest Anomaly Scoring)
IFOREST_HEALTH_PATH = os.path.join(BASE_DIR, 'models', 'model_layer3_iforest_health.pkl')
IFOREST_TEMPORAL_PATH = os.path.join(BASE_DIR, 'models', 'model_layer3_iforest_temporal.pkl')

iforest_health_pkg = None
if os.path.exists(IFOREST_HEALTH_PATH):
    iforest_health_pkg = joblib.load(IFOREST_HEALTH_PATH)

iforest_temporal_pkg = None
if os.path.exists(IFOREST_TEMPORAL_PATH):
    iforest_temporal_pkg = joblib.load(IFOREST_TEMPORAL_PATH)


def ensure_features(df_input):
    """Memastikan seluruh fitur biofisika dan spektral Sentinel-2 tersedia."""
    df = df_input.copy()
    
    # Hitung indeks vegetasi & air jika belum ada tapi band tersedia
    if 'MSI' not in df.columns and 'B11' in df.columns and 'B8' in df.columns:
        df['MSI'] = df['B11'] / (df['B8'] + 1e-6)
    if 'NIR_RED' not in df.columns and 'B8' in df.columns and 'B4' in df.columns:
        df['NIR_RED'] = df['B8'] / (df['B4'] + 1e-6)
    if 'NDVI' not in df.columns and 'B8' in df.columns and 'B4' in df.columns:
        df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 1e-6)
    if 'NDRE' not in df.columns and 'B8' in df.columns and 'B5' in df.columns:
        df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'] + 1e-6)
    if 'NDBI' not in df.columns and 'B11' in df.columns and 'B8' in df.columns:
        df['NDBI'] = (df['B11'] - df['B8']) / (df['B11'] + df['B8'] + 1e-6)
    if 'NDWI' not in df.columns and 'B3' in df.columns and 'B8' in df.columns:
        df['NDWI'] = (df['B3'] - df['B8']) / (df['B3'] + df['B8'] + 1e-6)
    if 'EVI' not in df.columns and all(b in df.columns for b in ['B8', 'B4', 'B2']):
        df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B2'] + 1.0)
    if 'SAVI' not in df.columns and all(b in df.columns for b in ['B8', 'B4']):
        df['SAVI'] = 1.5 * (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)

    # Fallback tekstur GLCM jika belum terhitung
    default_textures = {
        'GLCM_Contrast': 0.84,
        'GLCM_Entropy': 2.10,
        'GLCM_Variance': 0.72
    }
    for col, default_val in default_textures.items():
        if col not in df.columns:
            df[col] = default_val

    # Pastikan seluruh 20 kolom fitur L1/L2 ada
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def predict_pipeline(df_input, return_timing=False):
    """
    Pipeline 3-Layer Machine Learning:
    - Layer 1: Random Forest (Vegetasi vs Non-Vegetasi)
    - Layer 2: Random Forest (Sawit vs Non-Sawit)
    - Layer 3: Isolation Forest Anomaly Scoring + Directional Gatekeeper
    
    Returns:
        df_res: DataFrame dengan kolom:
                - pred_layer1, pred_layer2
                - anomaly_score (0.000 s.d 1.000)
                - priority_tier (Tier 1 Karantina, Tier 2 Stres, Tier 3 Normal, dll.)
                - health_status
                - recommended_action
                - vigor_score (MVS Z-score)
        (opsional) timing: dict waktu inferensi
    """
    t0 = time.time()
    df_prepared = ensure_features(df_input)
    df_res = df_prepared.copy()
    
    X = df_prepared[FEATURES]
    
    # -------------------------------------------------------------
    # Layer 1: Filter Vegetatif vs Non-Vegetatif (Random Forest)
    # -------------------------------------------------------------
    is_veg = model_l1.predict(X)
    df_res['pred_layer1'] = np.where(is_veg == 1, 'Vegetasi', 'Non-Vegetasi')
    
    # -------------------------------------------------------------
    # Layer 2: Filter Sawit vs Non-Sawit (Random Forest)
    # -------------------------------------------------------------
    df_res['pred_layer2'] = 'Bukan Sawit'
    veg_mask = (is_veg == 1)
    if veg_mask.sum() > 0:
        is_sawit = model_l2.predict(X[veg_mask])
        df_res.loc[veg_mask, 'pred_layer2'] = np.where(is_sawit == 1, 'Sawit', 'Hutan/Non-Sawit')
        
    # -------------------------------------------------------------
    # Layer 3: Isolation Forest Anomaly Scoring & Directional Gatekeeper
    # -------------------------------------------------------------
    df_res['health_status'] = df_res['pred_layer2']
    df_res['anomaly_score'] = 0.0
    df_res['priority_tier'] = np.where(df_res['pred_layer2'] == 'Bukan Sawit', '⚪ Bukan Sawit', '🟢 Tier 3: Normal Sehat')
    df_res['recommended_action'] = '-'
    df_res['vigor_score'] = np.nan
    
    sawit_mask = (df_res['pred_layer2'] == 'Sawit')
    
    # Pastikan memuat paket model iForest terkini secara aman
    curr_iforest_pkg = iforest_health_pkg
    if os.path.exists(IFOREST_HEALTH_PATH):
        try:
            loaded_pkg = joblib.load(IFOREST_HEALTH_PATH)
            if isinstance(loaded_pkg, dict):
                curr_iforest_pkg = loaded_pkg
            else:
                scaler_path = os.path.join(BASE_DIR, 'models', 'scaler_layer3.pkl')
                scaler_obj = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
                curr_iforest_pkg = {
                    'model': loaded_pkg,
                    'scaler': scaler_obj,
                    'feature_cols': ['NDRE', 'NDVI', 'EVI', 'SAVI', 'MSI', 'NIR_RED', 'B4', 'B8', 'B8A', 'B11'],
                    'd_min': -0.35,
                    'd_max': 0.15
                }
        except Exception:
            pass

    if sawit_mask.sum() > 0 and curr_iforest_pkg is not None:
        sub_sawit = df_prepared[sawit_mask].copy()
        
        # Ekstraksi fitur untuk Isolation Forest
        if isinstance(curr_iforest_pkg, dict):
            iforest_model = curr_iforest_pkg.get('model', curr_iforest_pkg)
            scaler = curr_iforest_pkg.get('scaler', None)
            feat_cols = curr_iforest_pkg.get('feature_cols', ['NDRE', 'NDVI', 'EVI', 'SAVI', 'MSI', 'NIR_RED', 'B4', 'B8', 'B8A', 'B11'])
            d_min = curr_iforest_pkg.get('d_min', -0.35)
            d_max = curr_iforest_pkg.get('d_max', 0.15)
        else:
            iforest_model = curr_iforest_pkg
            scaler_path = os.path.join(BASE_DIR, 'models', 'scaler_layer3.pkl')
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            feat_cols = ['NDRE', 'NDVI', 'EVI', 'SAVI', 'MSI', 'NIR_RED', 'B4', 'B8', 'B8A', 'B11']
            d_min = -0.35
            d_max = 0.15
        
        X_health = sub_sawit[feat_cols].values
        if scaler is not None:
            X_scaled = scaler.transform(X_health)
        else:
            X_scaled = X_health
        
        # Hitung raw decision function & normalisasi ke Continuous Anomaly Score (0.0 s.d 1.0)
        raw_decision = iforest_model.decision_function(X_scaled)
        anomaly_scores = np.clip((d_max - raw_decision) / (d_max - d_min + 1e-6), 0.0, 1.0)
        
        # Hitung juga baseline Z-score Vigor (MVS) untuk metrik komparasi
        z_ndre = (sub_sawit['NDRE'] - calib['mean_ndre']) / calib['std_ndre']
        z_ndvi = (sub_sawit['NDVI'] - calib['mean_ndvi']) / calib['std_ndvi']
        z_msi  = (sub_sawit['MSI']  - calib['mean_msi'])  / calib['std_msi']
        vigor  = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
        
        status_list = []
        tier_list = []
        action_list = []
        
        # DIRECTIONAL GATEKEEPER:
        # Menentukan arah anomali (degradasi penyakit vs pemupukan vs kanopi super lebat)
        for i, (idx, row) in enumerate(sub_sawit.iterrows()):
            score = float(anomaly_scores[i])
            v_ndre = float(row['NDRE'])
            v_ndvi = float(row['NDVI'])
            v_msi  = float(row['MSI'])
            v_ndbi = float(row.get('NDBI', -0.3))
            v_b4   = float(row.get('B4', 0.03))
            v_b8   = float(row.get('B8', 0.3))
            nir_red = v_b8 / (v_b4 + 1e-6)
            
            # 1. Guardrail Perbatasan Non-Vegetasi (Jalan / Tanah Terbuka / Bahu Jalan)
            if (v_ndbi > -0.12) or (nir_red < 4.0) or (v_b4 > 0.065):
                status_list.append('Non-Sawit (Bahu Jalan / Pekarangan / Semak)')
                tier_list.append('⚪ Filter Non-Sawit')
                action_list.append('Piksel campuran tepi kebun atau bahu jalan; dikecualikan dari monitoring pohon.')
                
            # 2. Sawit Muda / Tajuk Terbuka (Fase kanopi belum menutup sempurna)
            elif (v_ndvi < 0.70 and v_ndre < 0.48) or (v_ndbi > -0.18):
                status_list.append('Sawit Muda / Tajuk Terbuka')
                tier_list.append('🔵 Sawit Muda')
                action_list.append('Monitoring fase vegetatif muda; pertumbuhan tajuk belum rapat sempurna.')
                
            # 3. Super Vigor (Anomali Positif: Kanopi Sangat Rapat & Produktif)
            elif score >= 0.60 and v_ndre >= 0.62 and v_ndvi >= 0.84 and v_msi < 0.52:
                status_list.append('🟢 Sawit Super Vigor (Anomali Positif)')
                tier_list.append('🟢 Super Sehat')
                action_list.append('Pohon sangat produktif & prima; pertahankan dosis pemupukan saat ini.')
                
            # 4. Tier 1: Karantina Darurat (Suspek Sakit Kronis / Ganoderma / Kematian Tajuk)
            # Indikator: Anomali tinggi + Klorosis parah (NDRE anjlok) + Tajuk kering (MSI melonjak)
            elif (score >= 0.65 and (v_ndre < 0.48 or v_ndvi < 0.74 or v_msi > 0.62)) or \
                 (score >= 0.55 and v_ndre < 0.42 and v_msi > 0.65):
                status_list.append('🔴 SAWIT SAKIT (Target Karantina / Ganoderma)')
                tier_list.append('🔴 Tier 1: Karantina Darurat')
                action_list.append('URGENT: Cek pangkal batang untuk jamur Ganoderma; buat parit isolasi (trenching) segera.')
                
            # 5. Tier 2: Stres Ringan (Defisiensi Hara N-P-K / Fluktuasi Air / Stres Ringan)
            elif (0.40 <= score < 0.65 and (v_ndre < 0.54 or v_msi > 0.58)) or \
                 (v_ndvi < 0.78 and score >= 0.35):
                status_list.append('🟡 Sawit Stres Ringan (Defisiensi Hara)')
                tier_list.append('🟡 Tier 2: Stres Ringan')
                action_list.append('Inspeksi lapangan: Cek pelepah menguning/kering, defisiensi hara N-P-K atau drainase air.')
                
            # 6. Tier 3: Sawit Normal Sehat (Kondisi Standar Stabil Prima)
            else:
                status_list.append('🟢 Sawit Sehat Prima')
                tier_list.append('🟢 Tier 3: Normal Sehat')
                action_list.append('Kondisi kanopi stabil prima; lanjutkan pemupukan dan panen rutin.')

        df_res.loc[sawit_mask, 'health_status'] = status_list
        df_res.loc[sawit_mask, 'priority_tier'] = tier_list
        df_res.loc[sawit_mask, 'recommended_action'] = action_list
        df_res.loc[sawit_mask, 'anomaly_score'] = np.round(anomaly_scores, 4)
        df_res.loc[sawit_mask, 'vigor_score'] = np.round(vigor.values, 3)

    t1 = time.time()
    elapsed_sec = t1 - t0
    elapsed_ms = elapsed_sec * 1000
    
    timing = {
        'total_samples': len(df_res),
        'elapsed_sec': round(elapsed_sec, 4),
        'elapsed_ms': round(elapsed_ms, 2),
        'ms_per_sample': round(elapsed_ms / max(1, len(df_res)), 3),
        'model_name': 'Random Forest 2-Layer + Isolation Forest Anomaly Scoring'
    }
    
    if return_timing:
        return df_res, timing
    return df_res


if __name__ == '__main__':
    # Uji coba inferensi dengan dataset sampel
    csv_paths = ['output/training_data_all_features.csv', '../output/training_data_all_features.csv']
    chosen_path = next((p for p in csv_paths if os.path.exists(p)), None)
    
    if chosen_path:
        print(f">> Menjalankan Benchmark Pipeline (Random Forest + Isolation Forest) pada {chosen_path}...")
        sample_df = pd.read_csv(chosen_path).sample(2500, random_state=42)
        res, timing = predict_pipeline(sample_df, return_timing=True)
        print(f"\n==================================================================")
        print(f"  BENCHMARK PIPELINE KESEHATAN KELAPA SAWIT (3-LAYER)")
        print(f"==================================================================")
        print(f"  Model Engine    : {timing['model_name']}")
        print(f"  Total Sampel    : {timing['total_samples']:,} titik")
        print(f"  Waktu Eksekusi  : {timing['elapsed_sec']} detik ({timing['elapsed_ms']} ms)")
        print(f"  Kecepatan/Pohon : {timing['ms_per_sample']} ms per pohon")
        print(f"\nDistribusi Priority Tier:")
        for k, v in res['priority_tier'].value_counts().items():
            pct = (v / len(res)) * 100
            print(f"  • {k:<30}: {v:>5,} ({pct:>5.1f}%)")
        print(f"\nRata-rata Skor Anomali pada Sawit: {res[res['pred_layer2'] == 'Sawit']['anomaly_score'].mean():.3f}")
        print("==================================================================")
