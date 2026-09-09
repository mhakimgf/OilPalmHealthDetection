"""
SANDBOX: Script Inferensi Pipeline 2-Layer + Health Assessment
==============================================================
Contoh penggunaan untuk memprediksi data tabular baru (CSV atau dictionary nilai fitur).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Load model dan kalibrasi (fleksibel baik di root repo maupun folder lokal)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_L1_PATH = os.path.join(BASE_DIR, 'models', 'model_layer1_vegetasi.pkl')
MODEL_L2_PATH = os.path.join(BASE_DIR, 'models', 'model_layer2_sawit.pkl')
CALIB_PATH    = os.path.join(BASE_DIR, 'models', 'health_calibration.json')

model_l1 = joblib.load(MODEL_L1_PATH)
model_l2 = joblib.load(MODEL_L2_PATH)

with open(CALIB_PATH, 'r') as f:
    calib = json.load(f)

FEATURES = calib['features']

def predict_pipeline(df_input):
    """
    Menerima DataFrame yang memiliki kolom fitur-fitur Sentinel-2 dan
    mengembalikan DataFrame dengan label hierarkis dan status kesehatan.
    """
    df_res = df_input.copy()
    
    # Hitung MSI jika belum ada
    if 'MSI' not in df_res.columns:
        df_res['MSI'] = df_res['B11'] / (df_res['B8'] + 1e-6)
        
    # Pastikan semua fitur ada
    X = df_res[FEATURES]
    
    # Layer 1: Vegetatif vs Non-Vegetatif
    is_veg = model_l1.predict(X)
    df_res['pred_layer1'] = np.where(is_veg == 1, 'Vegetasi', 'Non-Vegetasi')
    
    # Layer 2: Sawit vs Non-Sawit
    df_res['pred_layer2'] = 'Bukan Sawit'
    veg_mask = (is_veg == 1)
    if veg_mask.sum() > 0:
        is_sawit = model_l2.predict(X[veg_mask])
        df_res.loc[veg_mask, 'pred_layer2'] = np.where(is_sawit == 1, 'Sawit', 'Hutan/Non-Sawit')
        
    # Layer 3: Health Assessment
    df_res['health_status'] = df_res['pred_layer2']
    sawit_mask = (df_res['pred_layer2'] == 'Sawit')
    
    if sawit_mask.sum() > 0:
        sub_sawit = df_res[sawit_mask]
        z_ndre = (sub_sawit['NDRE'] - calib['mean_ndre']) / calib['std_ndre']
        z_ndvi = (sub_sawit['NDVI'] - calib['mean_ndvi']) / calib['std_ndvi']
        z_msi  = (sub_sawit['MSI'] - calib['mean_msi']) / calib['std_msi']
        vigor  = (0.45 * z_ndre) + (0.35 * z_ndvi) - (0.20 * z_msi)
        
        health_list = []
        for idx, row in sub_sawit.iterrows():
            v = vigor.loc[idx]
            if row['NDVI'] < 0.65 and row['NDRE'] < 0.42:
                health_list.append('Sawit Muda / Terbuka')
            elif v >= -0.5:
                health_list.append('Sawit Sehat Prima')
            elif v >= -1.5:
                health_list.append('Sawit Stres Ringan')
            else:
                health_list.append('Sawit Tidak Sehat (Stres Berat)')
                
        df_res.loc[sawit_mask, 'health_status'] = health_list
        df_res.loc[sawit_mask, 'vigor_score'] = vigor
        
    return df_res

if __name__ == '__main__':
    # Test prediksi dengan sample data
    sample_df = pd.read_csv('output/training_data_all_features.csv', nrows=10)
    result = predict_pipeline(sample_df)
    print("Contoh Hasil Prediksi 10 Baris Pertama:")
    print(result[['kelas_nama', 'pred_layer1', 'pred_layer2', 'health_status']])
