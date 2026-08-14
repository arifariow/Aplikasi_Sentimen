import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import io
import altair as alt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- CONFIG ---
st.set_page_config(page_title="Prediksi Sentimen Telemedicine", layout="wide")

# CSS dan SVG Icons Custom
st.markdown("""
    <style>
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #1f77b4;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .result-box-positif {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #28a745;
        margin-bottom: 10px;
    }
    .result-box-negatif {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #dc3545;
        margin-bottom: 10px;
    }
    .result-box-netral {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #6c757d;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# SVG Icons
ICON_POS = '<svg style="width:24px;height:24px;fill:currentColor;vertical-align:middle" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>'
ICON_NEG = '<svg style="width:24px;height:24px;fill:currentColor;vertical-align:middle" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>'
ICON_NET = '<svg style="width:24px;height:24px;fill:currentColor;vertical-align:middle" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm4 11H8v-2h8v2z"/></svg>'

# --- INITIALIZE STEMMER ---
@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = get_stemmer()

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    model_dir = "models"
    try:
        with open(os.path.join(model_dir, 'tfidf.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            le = pickle.load(f)
        with open(os.path.join(model_dir, 'xgboost_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        return vectorizer, le, model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None, None

vectorizer, le, model = load_models()

# --- PREPROCESSING & ABSA FUNCTIONS ---
def clean_text_with_steps(text):
    steps = {}
    steps['1. Teks Asli'] = text
    text = str(text).lower()
    steps['2. Case Folding (Huruf Kecil)'] = text
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    steps['3. Hapus URL & Mention'] = text
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    steps['4. Hapus Tanda Baca & Angka'] = text
    text = stemmer.stem(text)
    steps['5. Hasil Akhir (Stemming)'] = text
    return text.strip(), steps

def clean_text_batch(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return stemmer.stem(text).strip()

def detect_aspect(text):
    text_lower = str(text).lower()
    if any(word in text_lower for word in ['lemot', 'error', 'bug', 'crash', 'lag', 'loading', 'sistem', 'aplikasi', 'berat', 'buka']):
        return "Sistem & Performa"
    elif any(word in text_lower for word in ['dokter', 'pelayanan', 'ramah', 'balas', 'konsultasi', 'cs', 'admin', 'bantu', 'tanya']):
        return "Pelayanan Medis & CS"
    elif any(word in text_lower for word in ['mahal', 'murah', 'harga', 'biaya', 'bayar', 'promo', 'transaksi', 'saldo', 'potong']):
        return "Harga & Transaksi"
    else:
        return "Umum (Tidak Spesifik)"

# --- UI LAYOUT ---
st.title("Aplikasi Prediksi Sentimen & Analisis Aspek (ABSA)")
st.markdown("Sistem *Enterprise-grade* ini menggunakan algoritma **XGBoost** untuk mengklasifikasikan sentimen dan mendeteksi aspek keluhan/pujian pengguna secara otomatis.")
st.markdown("---")

# Membuat Tabs
tab1, tab2, tab3 = st.tabs(["💬 Analisis Teks Tunggal", "📁 Analisis Massal (Batch Upload)", "📊 Dashboard Dataset"])

# TAB 1: SINGLE TEXT
with tab1:
    col1A, col1B = st.columns([2, 1])
    with col1A:
        st.markdown("**Form Input Ulasan Pengguna:**")
        user_input = st.text_area("", height=120, placeholder="Contoh: Aplikasi ini sangat lambat saat memuat resep obat namun dokternya ramah.", label_visibility="collapsed")
        
        if st.button("Jalankan Analisis Sentimen", type="primary", use_container_width=True, icon=":material/analytics:"):
            if not user_input.strip():
                st.warning("Peringatan: Silakan masukkan teks ulasan terlebih dahulu!", icon=":material/warning:")
            elif model is None:
                st.error("Error: Model klasifikasi tidak tersedia.", icon=":material/error:")
            else:
                with st.spinner("Sistem sedang memproses teks..."):
                    cleaned_text, steps_dict = clean_text_with_steps(user_input)
                    X_input = vectorizer.transform([cleaned_text])
                    
                    pred = model.predict(X_input)[0]
                    pred_proba = model.predict_proba(X_input)[0]
                    sentiment = le.inverse_transform([pred])[0]
                    confidence = np.max(pred_proba) * 100
                    
                    # Deteksi Aspek
                    aspek = detect_aspect(user_input)
                    
                    if sentiment.upper() == 'POSITIF':
                        st.success(f"**KATEGORI SENTIMEN:** {sentiment.upper()}", icon=":material/check_circle:")
                    elif sentiment.upper() == 'NEGATIF':
                        st.error(f"**KATEGORI SENTIMEN:** {sentiment.upper()}", icon=":material/cancel:")
                    else:
                        st.info(f"**KATEGORI SENTIMEN:** {sentiment.upper()}", icon=":material/remove_circle:")
                        
                    st.info(f"**DETEKSI ASPEK (ABSA):** {aspek}", icon=":material/category:")
                    
                    st.write("**Probabilitas Keakuratan (Confidence Score):**")
                    st.progress(float(np.max(pred_proba)))
                    st.caption(f"Tingkat Keyakinan Model XGBoost: {confidence:.2f}%")
                    
                    st.markdown("### Detail Preprocessing (Penelusuran White Box)")
                    for step_name, step_result in steps_dict.items():
                        st.markdown(f"**{step_name}:**")
                        st.code(step_result)
    with col1B:
         st.info("Fitur *Single Text* ini digunakan untuk menguji fungsionalitas model *XGBoost* secara spesifik pada satu buah ulasan beserta detail *White Box* preprocessing-nya.", icon=":material/info:")

# TAB 2: BATCH PROCESSING
with tab2:
    st.markdown("### Analisis Sentimen Skala Besar (Batch Processing)")
    st.markdown("Fitur ini dirancang untuk memproses ribuan ulasan sekaligus, cocok digunakan oleh divisi *Customer Service* atau *Product Manager* untuk mengetahui sentimen massal pengguna.")
    
    uploaded_file = st.file_uploader("Upload dataset ulasan (Format CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview Data Original:")
            st.dataframe(df_batch.head(3))
            
            # Cari kolom teks
            text_col = None
            for col in df_batch.columns:
                if col.lower() in ['teks', 'ulasan', 'review', 'content']:
                    text_col = col
                    break
                    
            if not text_col:
                st.error("Error: Tidak dapat menemukan kolom bernama 'teks' atau 'ulasan' dalam file CSV.")
            else:
                if st.button("Mulai Proses Batch", type="primary", icon=":material/play_arrow:"):
                    with st.spinner(f"Memproses {len(df_batch)} baris data... (Ini mungkin memakan waktu beberapa saat)"):
                        # Preprocess massal
                        df_batch['teks_bersih'] = df_batch[text_col].apply(lambda x: clean_text_batch(x))
                        
                        # Extract Feature
                        X_batch = vectorizer.transform(df_batch['teks_bersih'])
                        
                        # Prediksi Sentimen
                        preds = model.predict(X_batch)
                        df_batch['Prediksi_Sentimen'] = le.inverse_transform(preds)
                        
                        # Prediksi Aspek
                        df_batch['Deteksi_Aspek'] = df_batch[text_col].apply(lambda x: detect_aspect(x))
                        
                        st.success(f"✅ Berhasil memproses {len(df_batch)} baris data!")
                        
                        # Tampilkan Grafik (Pie Chart Altair)
                        sentiment_counts = df_batch['Prediksi_Sentimen'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentimen', 'Jumlah']
                        
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            st.markdown("#### Distribusi Sentimen")
                            pie_chart = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                                theta=alt.Theta(field="Jumlah", type="quantitative"),
                                color=alt.Color(field="Sentimen", type="nominal", 
                                              scale=alt.Scale(domain=['Positif', 'Netral', 'Negatif'], 
                                                            range=['#28a745', '#6c757d', '#dc3545'])),
                                tooltip=['Sentimen', 'Jumlah']
                            ).properties(height=300)
                            st.altair_chart(pie_chart, use_container_width=True)
                            
                        with col_chart2:
                            st.markdown("#### Preview Hasil Analisis")
                            st.dataframe(df_batch[[text_col, 'Prediksi_Sentimen', 'Deteksi_Aspek']].head(10))
                        
                        # Tombol Download
                        csv = df_batch.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Laporan Hasil Analisis (CSV)",
                            data=csv,
                            file_name='hasil_analisis_batch.csv',
                            mime='text/csv',
                            type="primary"
                        )
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

# TAB 3: DASHBOARD
with tab3:
    st.markdown("### Karakteristik Data Latih Skripsi")
    st.info("Dashboard ini menampilkan visualisasi statis dari dataset pelatihan (*Training Data*) berjumlah 3.352 ulasan dari Halodoc dan Alodokter.", icon=":material/bar_chart:")
    
    colA, colB, colC = st.columns(3)
    img_dir = "assets"
    
    with colA:
        wc_pos = os.path.join(img_dir, "WordCloud_Positif_Halodoc.png")
        if os.path.exists(wc_pos):
            st.image(wc_pos, caption="Kata Paling Sering Muncul (Sentimen Positif)", use_container_width=True)
    with colB:
        wc_neg = os.path.join(img_dir, "WordCloud_Negatif_Alodokter.png")
        if os.path.exists(wc_neg):
            st.image(wc_neg, caption="Kata Paling Sering Muncul (Sentimen Negatif)", use_container_width=True)
    with colC:
        cm_xgb = os.path.join(img_dir, "CM_XGBoost_Halodoc.png")
        if os.path.exists(cm_xgb):
            st.image(cm_xgb, caption="Confusion Matrix Model XGBoost", use_container_width=True)
