import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("Memulai proses pelatihan model XGBoost secara lokal...")

# 1. Load Data
data_path = r'c:\Users\LENOVO\Documents\skripsi\Data_Skripsi_Halodoc_Alodokter'
df_alo = pd.read_csv(os.path.join(data_path, 'DATASET_MASTER_ALODOKTER_LABELED.csv'))
df_halo = pd.read_csv(os.path.join(data_path, 'DATASET_MASTER_HALODOC_LABELED.csv'))

# Gabungkan dataset
df = pd.concat([df_alo, df_halo], ignore_index=True)

# 2. Preprocessing Data (Buang baris kosong)
df = df.dropna(subset=['teks_bersih', 'label_sentimen'])
df['teks_bersih'] = df['teks_bersih'].astype(str)

X = df['teks_bersih']
y = df['label_sentimen']

print(f"Total data siap latih: {len(df)} baris")

# 3. Encoding Label (Positif, Netral, Negatif -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. TF-IDF Vectorization
print("Melakukan ekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 5. Train XGBoost Model
print("Melatih model XGBoost... (Ini mungkin memakan waktu beberapa detik)")
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)
xgb_model.fit(X_tfidf, y_encoded)

# 6. Save Models
os.makedirs('models', exist_ok=True)

print("Menyimpan model ke folder models/ ...")
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("PELATIHAN SELESAI! Model berhasil diekspor.")
print("File tersimpan: tfidf.pkl, label_encoder.pkl, xgboost_model.pkl")
