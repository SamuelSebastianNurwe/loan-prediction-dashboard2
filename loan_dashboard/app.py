"""
app.py — Home page of Loan Prediction Intelligence Dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="Loan Prediction Intelligence Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar global (deskripsi sesuai halaman aktif)
from utils.sidebar import render_sidebar

# ─── Global Dark Theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root */
:root {
    --bg-primary   : #0E1117;
    --bg-card      : #1A1F2E;
    --bg-card2     : #1E2435;
    --accent-blue  : #1e3a6b;
    --accent-light : #4E9AF1;
    --accent-purple: #7C5CBF;
    --accent-orange: #F97316;
    --text-primary : #E2E8F0;
    --text-muted   : #94A3B8;
    --border       : #2D3748;
    --success      : #22C55E;
    --danger       : #EF4444;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #1A1F2E 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* Main background */
.main { background-color: var(--bg-primary) !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-light)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover { 
    transform: translateY(-2px) !important; 
    box-shadow: 0 6px 20px rgba(78,154,241,0.4) !important;
}

/* Metric */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
}
[data-testid="stMetric"] label { color: var(--text-muted) !important; font-size: 0.82rem !important; }
[data-testid="stMetricValue"] { color: var(--accent-light) !important; font-weight: 700 !important; }

/* Tabs */
.stTabs [role="tablist"] { border-bottom: 2px solid var(--border) !important; }
.stTabs [role="tab"] { color: var(--text-muted) !important; padding: 0.5rem 1.2rem !important; font-weight: 500 !important; }
.stTabs [aria-selected="true"] { 
    color: var(--accent-light) !important; 
    border-bottom: 2px solid var(--accent-light) !important;
    font-weight: 600 !important;
}

/* Info / Warning boxes */
.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }

/* KPI Card custom */
.kpi-card {
    background: var(--bg-card);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(78,154,241,0.15);
}
.kpi-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.kpi-label { color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
.kpi-value { font-size: 1.8rem; font-weight: 700; }
.kpi-delta { font-size: 0.85rem; margin-top: 0.2rem; }

/* Section header */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--accent-blue);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0D1526 0%, #1e3a6b 50%, #0f2952 100%);
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(78,154,241,0.25);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(78,154,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #fff 30%, var(--accent-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* Feature card grid */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    height: 100%;
    transition: all 0.25s ease;
}
.feature-card:hover {
    border-color: var(--accent-light);
    box-shadow: 0 6px 20px rgba(78,154,241,0.12);
    transform: translateY(-2px);
}
.feature-card h4 { color: var(--accent-light); font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem; }
.feature-card p  { color: var(--text-muted); font-size: 0.875rem; line-height: 1.55; }

/* Badge */
.badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
}
.badge-blue   { background: rgba(78,154,241,0.15); color: #4E9AF1; border: 1px solid rgba(78,154,241,0.3); }
.badge-purple { background: rgba(124,92,191,0.15); color: #7C5CBF; border: 1px solid rgba(124,92,191,0.3); }
.badge-orange { background: rgba(249,115,22,0.15); color: #F97316; border: 1px solid rgba(249,115,22,0.3); }
.badge-green  { background: rgba(34,197,94,0.15);  color: #22C55E; border: 1px solid rgba(34,197,94,0.3); }

/* Tables */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
thead th { background-color: var(--accent-blue) !important; color: white !important; }

/* Selectbox / Input */
.stSelectbox > div, .stNumberInput > div, .stSlider { color: var(--text-primary) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-light); }
</style>
""", unsafe_allow_html=True)

# ─── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🏦 Loan Prediction Intelligence Dashboard</div>
    <div class="hero-subtitle">Loan Eligibility Classification &amp; Loan Amount Regression Analytics</div>
    <div>
        <span class="badge badge-blue">NN / MLP</span>
        <span class="badge badge-purple">SVM</span>
        <span class="badge badge-orange">Decision Tree</span>
        <span class="badge badge-green">HPO Optimized</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Purpose Cards ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Dashboard Ini Digunakan Untuk</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Evaluasi Klasifikasi Loan Eligibility</h4>
        <p>Dashboard ini menyediakan platform komprehensif untuk <strong>menganalisis dan membandingkan performa algoritma klasifikasi</strong> (NN/MLP, SVM, Decision Tree) dalam memprediksi status kelayakan pinjaman secara akurat dan efisien.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📈 Estimasi Jumlah Pinjaman</h4>
        <p>Gunakan dashboard untuk <strong>membandingkan dan mengoptimalkan model regresi</strong> yang memprediksi besaran pinjaman yang tepat berdasarkan profil finansial pemohon dengan metrik evaluasi yang komprehensif.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🚀 Simulasi & Prediksi Real-time</h4>
        <p>Manfaatkan dashboard untuk <strong>simulasi interaktif dan prediksi real-time</strong> terhadap aplikasi pinjaman baru dengan visualisasi mendalam tentang faktor-faktor yang mempengaruhi keputusan.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("""
    <div class="feature-card">
        <h4>⚙️ Evaluasi HPO</h4>
        <p>Perbandingan menyeluruh dampak <strong>Hyperparameter Optimization (HPO)</strong> 
        terhadap akurasi, F1-Score, MAE, dan RMSE masing-masing model pada dataset pelatihan.</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <h4>🔬 Prediksi & Simulator</h4>
        <p>Simulasi interaktif: masukkan profil pemohon dan dapatkan prediksi kelayakan <em>Loan_Status</em> pada halaman <b>Classification Simulator</b>, serta estimasi <em>LoanAmount</em> pada halaman <b>Regression Simulator</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Dataset Overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📁 Tentang Dataset</div>', unsafe_allow_html=True)

info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.metric("Total Fitur Input", "11", help="Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area")
with info_col2:
    st.metric("Target Klasifikasi", "Loan_Status", help="Binary: Approved (1) / Rejected (0)")
with info_col3:
    st.metric("Target Regresi", "LoanAmount", help="Nilai numerik kontinyu dalam ribuan USD")
with info_col4:
    st.metric("Algoritma Diuji", "3 × 2 = 6", help="NN/MLP, SVM, Decision Tree — masing-masing versi HPO & Non-HPO")

# ─── Navigation Guide ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Panduan Navigasi</div>', unsafe_allow_html=True)

nav_data = {
    "Halaman": [
        "1. Executive Overview",
        "2. Classification Lab",
        "3. Regression Lab",
        "4. Feature Analytics",
        "5. Explainability",
        "6. Classification Simulator",
        "7. Regression Simulator",
    ],
    "Deskripsi": [
        "Ringkasan KPI, distribusi data, dan visualisasi profil dataset keseluruhan",
        "Perbandingan model klasifikasi (NN/MLP, SVM/DT) dengan confusion matrix & analisis HPO",
        "Perbandingan model regresi (NN/MLP, SVM/DT) dengan scatter plot & analisis HPO",
        "Heatmap korelasi, boxplot distribusi LoanAmount, & analisis Credit_History",
        "Deskripsi fitur, feature importance Decision Tree, & panduan interpretasi model",
        "Input profil pemohon, prediksi kelayakan Loan_Status (Approved/Rejected), & rekomendasi",
        "Input profil pemohon, estimasi LoanAmount (ribuan USD), radar chart & rekomendasi",
    ],
}

import pandas as pd
st.dataframe(
    pd.DataFrame(nav_data),
    use_container_width=True,
    hide_index=True,
)

# Sidebar global untuk semua halaman
render_sidebar("Executive Overview")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2024 Loan Prediction Intelligence Dashboard — Tugas Besar Kelompok 8")
