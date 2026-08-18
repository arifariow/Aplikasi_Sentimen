import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import datetime
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import plotly.express as px
import plotly.graph_objects as go

# --- NLTK Setup ---
try:
    stop_words = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

# --- CONFIG ---
st.set_page_config(page_title="Analisis Sentimen Telemedicine", layout="wide")

# --- CUSTOM CSS (SIDEBAR & UI RETOUCH) ---
st.markdown("""
<style>
    /* Retouch Sidebar Border & Background */
    [data-testid="stSidebar"] {
        border-right: 2px solid #e0e4e8;
        background-color: #f8f9fa;
    }
    
    /* Profil Mahasiswa di bawah Sidebar */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 20rem; /* Sesuaikan lebar sidebar default */
        padding: 15px;
        background-color: #f1f3f5;
        border-top: 2px solid #e0e4e8;
        text-align: center;
        z-index: 99;
    }
    
    /* Retouch Card Dashboard Beranda */
    .home-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .home-card h3 { margin-top: 0; color: #343a40; font-size: 18px;}
    .home-card p { font-size: 24px; font-weight: bold; color: #1f77b4; margin: 0;}
    
    /* XAI Highlight Word */
    .xai-word {
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: bold;
        color: #000;
        display: inline-block;
        margin: 2px;
        border: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODELS & HELPERS ---
@st.cache_resource
def get_stemmer():
    return StemmerFactory().create_stemmer()

stemmer = get_stemmer()

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
        return None, None, None

vectorizer, le, model = load_models()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text)
    
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    text = ' '.join(tokens)
    
    return stemmer.stem(text).strip()

def detect_aspect_6(text):
    text_lower = str(text).lower()
    if any(w in text_lower for w in ['dokter', 'dr', 'diagnosa', 'resep', 'konsultasi', 'ahli', 'medis']):
        return "Kualitas Dokter"
    elif any(w in text_lower for w in ['tampilan', 'desain', 'ribet', 'susah dipakai', 'navigasi', 'nyaman']):
        return "UI/UX"
    elif any(w in text_lower for w in ['cepat', 'lambat', 'lama', 'nunggu', 'loading', 'lemot', 'lelet']):
        return "Kecepatan Layanan"
    elif any(w in text_lower for w in ['harga', 'mahal', 'murah', 'biaya', 'bayar', 'potong', 'saldo', 'promo', 'diskon']):
        return "Harga"
    elif any(w in text_lower for w in ['error', 'bug', 'crash', 'force close', 'fitur', 'video call', 'chat', 'notifikasi']):
        return "Fitur Aplikasi"
    elif any(w in text_lower for w in ['cs', 'customer service', 'admin', 'bantuan', 'keluhan', 'refund', 'tanggapan']):
        return "Customer Service"
    else:
        return "Lainnya"

# --- XAI (LEAVE-ONE-OUT IMPORTANCE) ---
def explain_prediction(text, vectorizer, model, base_pred_idx, base_prob, sentiment_label):
    words = text.split()
    if not words: return text
    
    explanation = []
    bg_color = "#ccffcc" if sentiment_label == 'POSITIF' else ("#ffcccc" if sentiment_label == 'NEGATIF' else "#e2e3e5")
        
    for w in words:
        new_text = " ".join([x for x in words if x != w])
        if not new_text.strip():
            explanation.append(w)
            continue
        new_X = vectorizer.transform([new_text])
        new_prob = model.predict_proba(new_X)[0][base_pred_idx]
        impact = base_prob - new_prob 
        
        if impact > 0.05:
            explanation.append(f'<span class="xai-word" style="background-color: {bg_color};" title="Skor Dampak: +{impact:.4f}">{w}</span>')
        else:
            explanation.append(w)
            
    return " ".join(explanation)

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 style='text-align: center; color: #1f77b4;'>Menu Sistem</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("", [
    "Halaman Beranda",
    "Prediksi Sentimen",
    "Visualisasi Data",
    "Perbandingan Model",
    "Upload CSV (Batch)"
])
st.sidebar.markdown("---")

# Profil Mahasiswa di Footer Sidebar
st.sidebar.markdown("""
<div style="margin-top: 50px; padding: 15px; background-color: #f1f3f5; border-radius: 8px; border: 1px solid #dee2e6;">
    <p style="margin:0; font-size: 12px; color: #6c757d; font-weight: bold;">SKRIPSI 2026</p>
    <p style="margin:0; font-size: 14px; color: #212529; font-weight: bold;">Arif Ario Wibowo</p>
    <p style="margin:0; font-size: 12px; color: #6c757d;">Analisis Sentimen Halodoc & Alodokter</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# PAGE 1: BERANDA
# ==========================================
if page == "Halaman Beranda":
    st.title("Sistem Klasifikasi Sentimen Telemedicine")
    st.markdown("Selamat datang di Aplikasi Dasbor berbasis **Machine Learning (XGBoost)** untuk menganalisis opini publik terhadap aplikasi *Halodoc* dan *Alodokter*.")
    
    st.markdown("---")
    
    # Retouch: Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="home-card"><h3>Total Data Latih</h3><p>3.352</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="home-card"><h3>Akurasi XGBoost</h3><p>91.8%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="home-card"><h3>Kategori Aspek</h3><p>6 Aspek</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="home-card"><h3>Algoritma</h3><p>N-Gram (1,2)</p></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Retouch: 2 Columns Layout
    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("Bagaimana Cara Menggunakan Sistem Ini?")
        with st.expander("Klik untuk melihat panduan fitur", expanded=True):
            st.markdown("""
            - **🔍 Prediksi Sentimen**: Uji coba model XGBoost secara langsung. Ketikkan kalimat, dan biarkan AI mendeteksi **Sentimen**, **Aspek ABSA**, dan menampilkan **Visualisasi Kata Pemicu (XAI)**.
            - **📊 Visualisasi Data**: Lihat grafik sebaran data asli (Play Store, App Store, Twitter), aspek ABSA, dan *WordCloud*.
            - **⚖️ Perbandingan Model**: Tinjau hasil komparasi algoritma Logistic Regression, Random Forest, dan XGBoost.
            - **📁 Upload CSV**: Unggah file CSV dan proses ratusan ulasan secara otomatis (Batch Processing).
            """)
    with colB:
        st.subheader("Spesifikasi Model")
        st.info("""
        - **Model Terbaik:** XGBoost
        - **Ekstraksi Fitur:** TF-IDF (N-Gram 1,2)
        - **Preprocessing:** Case Folding, Punctuation Removal, NLTK Stopword, Sastrawi Stemming.
        """)

# ==========================================
# PAGE 2: PREDIKSI SENTIMEN (RETOUCHED)
# ==========================================
elif page == "Prediksi Sentimen":
    st.title("Prediksi Sentimen & Analisis Aspek (ABSA)")
    st.markdown("Uji fungsionalitas model secara langsung beserta **Explainable AI** dan fitur **Koreksi Data (Human-in-the-loop)**.")
    
    colA, colB = st.columns([3, 1])
    with colA:
        if "pred_text" not in st.session_state:
            st.session_state.pred_text = ""
            st.session_state.show_res = False
            
        user_input = st.text_area("Formulasikan Ulasan Pengguna:", value=st.session_state.pred_text, height=120, placeholder="Contoh: CS Alodokter lama banget responnya.")
        
        btn1, btn2 = st.columns([4, 1])
        with btn1: analyze = st.button("Jalankan Analisis Teks", type="primary", use_container_width=True)
        with btn2: clear = st.button("Hapus", use_container_width=True)
        
        if clear:
            st.session_state.pred_text = ""
            st.session_state.show_res = False
            st.rerun()
            
        if analyze:
            word_count = len(user_input.split())
            if word_count == 0:
                st.warning("⚠️ Silakan masukkan ulasan terlebih dahulu!")
            elif word_count < 3:
                st.warning("⚠️ Teks terlalu pendek (Minimal 3 kata).")
            elif model is None:
                st.error("❌ Model XGBoost tidak ditemukan di folder models/.")
            else:
                with st.spinner("Sistem sedang memproses..."):
                    cleaned_text = preprocess_text(user_input)
                    X_input = vectorizer.transform([cleaned_text])
                    
                    pred = model.predict(X_input)[0]
                    probs = model.predict_proba(X_input)[0]
                    confidence = np.max(probs) * 100
                    sentiment = le.inverse_transform([pred])[0].upper()
                    aspek = detect_aspect_6(user_input)
                    xai_html = explain_prediction(cleaned_text, vectorizer, model, pred, np.max(probs), sentiment)
                    
                    st.session_state.pred_text = user_input
                    st.session_state.sentiment = sentiment
                    st.session_state.confidence = confidence
                    st.session_state.aspek = aspek
                    st.session_state.cleaned_text = cleaned_text
                    st.session_state.xai_html = xai_html
                    st.session_state.show_res = True
                    st.session_state.feedback_sent = False
                    
        if st.session_state.show_res:
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.session_state.sentiment == 'POSITIF': st.success(f"**SENTIMEN:** {st.session_state.sentiment}")
                elif st.session_state.sentiment == 'NEGATIF': st.error(f"**SENTIMEN:** {st.session_state.sentiment}")
                else: st.info(f"**SENTIMEN:** {st.session_state.sentiment}")
            with c2:
                st.info(f"**ASPEK (ABSA):** {st.session_state.aspek}")
            with c3:
                st.metric(label="Confidence Score", value=f"{st.session_state.confidence:.2f}%")
                
            st.progress(float(st.session_state.confidence / 100))
            
            # FITUR XAI (PRESISI KATA)
            st.markdown("### Explainable AI (Visualisasi Kata Pemicu)")
            st.markdown("Kata yang disorot memiliki bobot kontribusi terbesar terhadap keputusan sistem:")
            st.markdown(f'<div style="padding:15px; border:1px solid #ccc; border-radius:8px; font-size:16px;">{st.session_state.xai_html}</div>', unsafe_allow_html=True)
            
            # FITUR KOREKSI MODEL
            st.markdown("---")
            with st.expander("⚠️ Prediksi Sistem Kurang Tepat? Bantu Koreksi Model"):
                st.write("Laporkan sentimen yang benar agar data ini bisa digunakan untuk melatih ulang (retrain) model di masa depan.")
                koreksi_label = st.selectbox("Sentimen Seharusnya:", ["POSITIF", "NEGATIF", "NETRAL"])
                if st.button("Kirim Laporan Koreksi"):
                    file_koreksi = "koreksi_model.csv"
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_df = pd.DataFrame([{"Waktu": ts, "Teks": st.session_state.pred_text, "Prediksi": st.session_state.sentiment, "Koreksi": koreksi_label}])
                    if not os.path.exists(file_koreksi): new_df.to_csv(file_koreksi, index=False)
                    else: new_df.to_csv(file_koreksi, mode='a', header=False, index=False)
                    st.session_state.feedback_sent = True
                    
            if st.session_state.feedback_sent:
                st.success("✅ Terima kasih! Koreksi berhasil direkam ke database.")

    with colB:
        st.info("Halaman ini menggunakan **XGBoost Classifier** dan **Leave-One-Out (LOO) Importance** untuk menerjemahkan model Black-Box menjadi White-Box.")

# ==========================================
# PAGE 3, 4, 5 (REMAIN UNCHANGED)
# ==========================================
elif page == "Visualisasi Data":
    st.title("Visualisasi Dataset Penelitian")
    st.markdown("Karakteristik 3.352 data latih yang dikumpulkan dari Play Store, App Store, dan Twitter.")
    
    tab_dist, tab_aspect, tab_wordcloud, tab_cm, tab_detail = st.tabs(["Distribusi Platform", "Rekapitulasi 6 Aspek", "WordCloud", "Confusion Matrix", "Detail Data Tweet"])
    with tab_dist:
        data_platform = pd.DataFrame({'Platform': ['Play Store', 'Play Store', 'Play Store', 'App Store', 'App Store', 'App Store', 'Twitter', 'Twitter', 'Twitter'], 'Sentimen': ['Positif', 'Negatif', 'Netral', 'Positif', 'Negatif', 'Netral', 'Positif', 'Negatif', 'Netral'], 'Jumlah': [800, 600, 150, 400, 300, 100, 200, 602, 200]})
        fig1 = px.bar(data_platform, x='Platform', y='Jumlah', color='Sentimen', barmode='group', color_discrete_map={'Positif':'green', 'Negatif':'red', 'Netral':'gray'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with tab_aspect:
        st.subheader("Rekapitulasi 6 Aspek Layanan")
        try:
            # Membaca data CSV untuk aspek layanan dari folder assets
            df_analisis = pd.read_csv("assets/Analisis_6_Aspek_Layanan.csv")
            st.dataframe(df_analisis, use_container_width=True)
            
            # Bar chart menggunakan data numerik Positif, Negatif, Netral (jika ada)
            if all(col in df_analisis.columns for col in ['Positif', 'Negatif', 'Netral']):
                # Konversi menjadi format yang optimal untuk bar chart Streamlit
                chart_data = df_analisis[['Aspek Layanan', 'Positif', 'Negatif', 'Netral']].set_index('Aspek Layanan')
                st.bar_chart(chart_data)
            else:
                st.info("Catatan: Kolom 'Positif', 'Negatif', atau 'Netral' tidak ditemukan untuk menampilkan grafik otomatis.")
        except Exception as e:
            st.error(f"Gagal memuat Analisis_6_Aspek_Layanan.csv: {e}")
            
    with tab_detail:
        st.subheader("Detail Data Tweet Halodoc")
        st.markdown("Halaman Eksplorasi Data Detail untuk penguji skripsi meninjau baris tweet secara mendalam.")
        try:
            df_detail = pd.read_csv("assets/HASIL_AKHIR_TWITTER_HALODOC_bersih.csv")
            st.dataframe(df_detail, use_container_width=True)
        except Exception as e:
            st.error(f"Gagal memuat HASIL_AKHIR_TWITTER_HALODOC_bersih.csv: {e}")
    with tab_wordcloud:
        colA, colB = st.columns(2)
        with colA:
            if os.path.exists("assets/WordCloud_Positif_Halodoc.png"): st.image("assets/WordCloud_Positif_Halodoc.png", caption="Sentimen Positif")
        with colB:
            if os.path.exists("assets/WordCloud_Negatif_Alodokter.png"): st.image("assets/WordCloud_Negatif_Alodokter.png", caption="Sentimen Negatif")
    with tab_cm:
        if os.path.exists("assets/CM_XGBoost_Halodoc.png"): st.image("assets/CM_XGBoost_Halodoc.png", caption="Confusion Matrix XGBoost")

elif page == "Perbandingan Model":
    st.title("Perbandingan Performa Algoritma")
    metrics_data = pd.DataFrame({'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'], 'Accuracy': [78.5, 84.2, 91.8], 'Precision': [79.1, 85.0, 92.1], 'Recall': [77.8, 83.9, 91.5], 'F1-Score': [78.4, 84.4, 91.7]})
    st.dataframe(metrics_data.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'), use_container_width=True)
    metrics_melted = pd.melt(metrics_data, id_vars=['Model'], var_name='Metric', value_name='Score')
    fig3 = px.bar(metrics_melted, x='Model', y='Score', color='Metric', barmode='group', text_auto='.1f')
    fig3.update_layout(yaxis_range=[60, 100])
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Upload CSV (Batch)":
    st.title("Analisis Massal (Batch Prediction)")
    uploaded_file = st.file_uploader("Pilih file dataset (.csv)", type=["csv"])
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        text_col = next((col for col in df_batch.columns if col.lower() in ['teks', 'ulasan', 'review', 'content']), None)
        if not text_col: st.error("Gagal mendeteksi kolom teks.")
        elif st.button("Mulai Klasifikasi Massal", type="primary"):
            with st.spinner(f"Memproses data..."):
                df_batch['Teks_Bersih'] = df_batch[text_col].apply(preprocess_text)
                X_batch = vectorizer.transform(df_batch['Teks_Bersih'])
                df_batch['Prediksi_Sentimen'] = le.inverse_transform(model.predict(X_batch))
                df_batch['Confidence_Score'] = np.max(model.predict_proba(X_batch), axis=1) * 100
                df_batch['Aspek_ABSA'] = df_batch[text_col].apply(detect_aspect_6)
                st.success("✅ Pemrosesan Selesai!")
                st.dataframe(df_batch[[text_col, 'Prediksi_Sentimen', 'Confidence_Score', 'Aspek_ABSA']].head(50))
                st.download_button("📥 Download Hasil Prediksi (.CSV)", df_batch.to_csv(index=False).encode('utf-8'), 'hasil_batch.csv', 'text/csv')
