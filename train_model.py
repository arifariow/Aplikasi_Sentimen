import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print("Memulai proses pelatihan ulang model XGBoost (Versi 2.0)...")

# --- NLTK & Sastrawi Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Stopword removal
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    text = ' '.join(tokens)
    
    # Stemming
    text = stemmer.stem(text)
    return text.strip()

# 1. Load Data
data_path = r'c:\Users\LENOVO\Documents\skripsi\Data_Skripsi_Halodoc_Alodokter'
try:
    df_alo = pd.read_csv(os.path.join(data_path, 'DATASET_MASTER_ALODOKTER_LABELED.csv'))
    df_halo = pd.read_csv(os.path.join(data_path, 'DATASET_MASTER_HALODOC_LABELED.csv'))
    df = pd.concat([df_alo, df_halo], ignore_index=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Preprocessing Data (Buang baris kosong)
df = df.dropna(subset=['teks', 'label_sentimen'])

# Terapkan fungsi preprocessing baru (Stopword + Sastrawi)
print("Melakukan preprocessing (Stopword + Stemming)... Ini memakan waktu beberapa menit.")
# Jika di dataset sudah ada teks_bersih, kita timpa dengan preprocessing terbaru untuk memastikan stopword hilang
df['teks_bersih_v2'] = df['teks'].apply(preprocess_text)
df = df[df['teks_bersih_v2'] != '']

X = df['teks_bersih_v2']
y = df['label_sentimen']

print(f"Total data siap latih: {len(df)} baris")

# 3. Encoding Label (Positif, Netral, Negatif -> 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. TF-IDF Vectorization dengan N-Gram (1,2)
print("Melakukan ekstraksi fitur TF-IDF (N-Gram 1,2)...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X)

# 5. Train XGBoost Model
print("Melatih model XGBoost... ")
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
