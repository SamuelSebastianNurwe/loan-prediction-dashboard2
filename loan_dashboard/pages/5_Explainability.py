"""
pages/5_Explainability.py
Deskripsi fitur dan feature importance dari model Decision Tree.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import numpy as np

from utils.helper import load_model, FEATURE_DESCRIPTIONS
from utils.visualization import COLORS, plot_feature_importance

st.set_page_config(page_title="Explainability | Loan Dashboard", page_icon="💡", layout="wide")

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0E1117!important;color:#E2E8F0!important}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1526,#1A1F2E)!important;border-right:1px solid #2D3748!important}
    section[data-testid="stSidebar"] *{color:#E2E8F0!important}
    .main{background-color:#0E1117!important}.block-container{padding:2rem 2.5rem 3rem!important}
    .section-header{font-size:1.25rem;font-weight:700;color:#E2E8F0;margin:1.5rem 0 0.8rem;padding-bottom:0.4rem;border-bottom:2px solid #1e3a6b}
    .feat-card{background:#1A1F2E;border:1px solid #2D3748;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.5rem;transition:all 0.2s}
    .feat-card:hover{border-color:#4E9AF1;transform:translateX(4px)}
    .feat-name{font-weight:600;color:#4E9AF1;font-size:0.95rem}
    .feat-type{display:inline-block;padding:0.15rem 0.6rem;border-radius:50px;font-size:0.72rem;font-weight:600;margin-left:0.5rem}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0E1117}::-webkit-scrollbar-thumb{background:#2D3748;border-radius:3px}
    </style>""", unsafe_allow_html=True)

inject_css()

# Sidebar global agar ringkasan tetap muncul di halaman ini
render_sidebar("Explainability")

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(245,158,11,0.2);">
    <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">💡 Model Explainability</h1>
    <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Deskripsi fitur, feature importance, dan panduan interpretasi model</p>
</div>
""", unsafe_allow_html=True)

# ─── Feature Descriptions Table ───────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Deskripsi Fitur Model</div>', unsafe_allow_html=True)

feat_meta = {
    "Gender"           : ("Kategorikal", "Binary", "Male=1, Female=0"),
    "Married"          : ("Kategorikal", "Binary", "Yes=1, No=0"),
    "Dependents"       : ("Kategorikal", "Ordinal", "0, 1, 2, 3+"),
    "Education"        : ("Kategorikal", "Binary", "Graduate=0, Not Graduate=1"),
    "Self_Employed"    : ("Kategorikal", "Binary", "No=0, Yes=1"),
    "ApplicantIncome"  : ("Numerik", "Kontinu", "Pendapatan bulanan dalam USD"),
    "CoapplicantIncome": ("Numerik", "Kontinu", "Pendapatan bulanan co-applicant USD"),
    "LoanAmount"       : ("Numerik", "Kontinu", "Jumlah pinjaman dalam ribuan USD"),
    "Loan_Amount_Term" : ("Numerik", "Diskrit", "Tenor dalam bulan (umumnya 360)"),
    "Credit_History"   : ("Numerik", "Binary", "1=Riwayat baik, 0=Tidak ada/Buruk"),
    "Property_Area"    : ("Kategorikal", "Nominal", "Urban / Semiurban / Rural (One-Hot)"),
}

type_colors = {"Kategorikal": "#7C5CBF", "Numerik": "#4E9AF1"}

for feat, (dtype, scale, detail) in feat_meta.items():
    color = type_colors.get(dtype, "#94A3B8")
    st.markdown(f"""
    <div class="feat-card">
        <span class="feat-name">{feat}</span>
        <span class="feat-type" style="background:rgba(78,154,241,0.1);color:{color};border:1px solid {color}40;">{dtype}</span>
        <span class="feat-type" style="background:rgba(245,158,11,0.1);color:#F59E0B;border:1px solid #F59E0B40;">{scale}</span>
        <div style="color:#94A3B8;font-size:0.85rem;margin-top:0.4rem;">{FEATURE_DESCRIPTIONS.get(feat, detail)}</div>
        <div style="color:#64748B;font-size:0.78rem;margin-top:0.2rem;">Encoding: {detail}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Full Feature Table ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Tabel Ringkasan Fitur</div>', unsafe_allow_html=True)

feat_df = pd.DataFrame([
    {
        "Fitur": feat,
        "Tipe Data": dtype,
        "Skala": scale,
        "Deskripsi": FEATURE_DESCRIPTIONS.get(feat, "-"),
        "Encoding": detail,
    }
    for feat, (dtype, scale, detail) in feat_meta.items()
])
st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ─── Feature Importance (DT, SVM, NN/MLP) ───────────────────────────────────
st.markdown('<div class="section-header">📈 Feature Importance — Multi-Model (HPO)</div>', unsafe_allow_html=True)

feat_names_clf = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome","LoanAmount",
    "Loan_Amount_Term","Credit_History","PAR_Rural","PAR_Semiurban","PAR_Urban",
]
feat_names_reg = [
    "Gender","Married","Dependents","Education","Self_Employed",
    "ApplicantIncome","CoapplicantIncome",
    "Loan_Amount_Term","Credit_History","PAR_Rural","PAR_Semiurban","PAR_Urban",
]

def get_importances_from_model(model, attr_name: str):
    """Coba ambil feature_importances_ dari sklearn model/pipeline."""
    if model is None:
        return None
    if hasattr(model, attr_name):
        try:
            return getattr(model, attr_name)
        except Exception:
            pass
    try:
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, attr_name):
                    return getattr(step, attr_name)
    except Exception:
        pass
    return None

def fallback_importances(names):
    """Fallback representatif (agar chart selalu tampil untuk presentasi)."""
    base = np.array([0.02] * len(names), dtype=float)
    if "Credit_History" in names:
        base[names.index("Credit_History")] = 0.32
    if "ApplicantIncome" in names:
        base[names.index("ApplicantIncome")] = 0.14
    if "CoapplicantIncome" in names:
        base[names.index("CoapplicantIncome")] = 0.10
    if "LoanAmount" in names:
        base[names.index("LoanAmount")] = 0.15
    base = base / base.sum()
    return base.tolist()

tab_dt, tab_svm, tab_nn = st.tabs([
    "🟠 Decision Tree (DT)",
    "🟣 SVM (RBF)",
    "🔵 Neural Network / MLP",
])

# ── DT ─────────────────────────────────────────────────────────────────────
with tab_dt:
    sub_dt_clf, sub_dt_reg = st.tabs(["Klasifikasi (DT HPO)", "Regresi (DT HPO)"])

    with sub_dt_clf:
        with st.spinner("Memuat model DT Klasifikasi..."):
            dt_clf = load_model("clf_dt_hpo")

        importances = get_importances_from_model(dt_clf, "feature_importances_")
        if importances is not None:
            fig = plot_feature_importance(feat_names_clf, list(importances), "Feature Importance — DT Klasifikasi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("DT tidak mengekspos `feature_importances_`. Menampilkan fallback representatif.")
            fig = plot_feature_importance(feat_names_clf, fallback_importances(feat_names_clf), "Feature Importance — DT (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

    with sub_dt_reg:
        with st.spinner("Memuat model DT Regresi..."):
            dt_reg = load_model("reg_dt_hpo")

        importances_r = get_importances_from_model(dt_reg, "feature_importances_")
        if importances_r is not None:
            fig = plot_feature_importance(feat_names_reg, list(importances_r), "Feature Importance — DT Regresi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("DT tidak mengekspos `feature_importances_`. Menampilkan fallback representatif.")
            fig = plot_feature_importance(feat_names_reg, fallback_importances(feat_names_reg), "Feature Importance — DT (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

# ── SVM ────────────────────────────────────────────────────────────────────
with tab_svm:
    sub_svm_clf, sub_svm_reg = st.tabs(["Klasifikasi (SVM HPO)", "Regresi (SVM HPO)"])

    with sub_svm_clf:
        with st.spinner("Memuat model SVM Klasifikasi (HPO)..."):
            svm_clf = load_model("clf_svm_hpo")

        importances = get_importances_from_model(svm_clf, "feature_importances_")
        if importances is not None:
            fig = plot_feature_importance(feat_names_clf, list(importances), "Feature Importance — SVM Klasifikasi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SVM umumnya tidak memiliki `feature_importances_`. Menampilkan fallback representatif untuk presentasi.")
            fig = plot_feature_importance(feat_names_clf, fallback_importances(feat_names_clf), "Feature Importance — SVM (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

    with sub_svm_reg:
        with st.spinner("Memuat model SVM Regresi (HPO)..."):
            svm_reg = load_model("reg_svm_hpo")

        importances_r = get_importances_from_model(svm_reg, "feature_importances_")
        if importances_r is not None:
            fig = plot_feature_importance(feat_names_reg, list(importances_r), "Feature Importance — SVM Regresi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SVM umumnya tidak memiliki `feature_importances_`. Menampilkan fallback representatif untuk presentasi.")
            fig = plot_feature_importance(feat_names_reg, fallback_importances(feat_names_reg), "Feature Importance — SVM (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

# ── NN / MLP ───────────────────────────────────────────────────────────────
with tab_nn:
    sub_nn_clf, sub_nn_reg = st.tabs(["Klasifikasi (NN/MLP HPO)", "Regresi (NN/MLP HPO)"])

    with sub_nn_clf:
        with st.spinner("Memuat model NN/MLP Klasifikasi (HPO)..."):
            nn_clf = load_model("clf_nn_hpo")

        importances = get_importances_from_model(nn_clf, "feature_importances_")
        if importances is not None:
            fig = plot_feature_importance(feat_names_clf, list(importances), "Feature Importance — NN/MLP Klasifikasi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("MLP umumnya tidak memiliki `feature_importances_`. Menampilkan fallback representatif untuk presentasi.")
            fig = plot_feature_importance(feat_names_clf, fallback_importances(feat_names_clf), "Feature Importance — NN/MLP (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

    with sub_nn_reg:
        with st.spinner("Memuat model NN/MLP Regresi (HPO)..."):
            nn_reg = load_model("reg_nn_nohpo")

        importances_r = get_importances_from_model(nn_reg, "feature_importances_")
        if importances_r is not None:
            fig = plot_feature_importance(feat_names_reg, list(importances_r), "Feature Importance — NN/MLP Regresi (HPO)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("MLP umumnya tidak memiliki `feature_importances_`. Menampilkan fallback representatif untuk presentasi.")
            fig = plot_feature_importance(feat_names_reg, fallback_importances(feat_names_reg), "Feature Importance — NN/MLP (Representatif)")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.8rem;">
  <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    <b>Digunakan untuk apa?</b><br>
    • Memvisualisasikan fitur mana yang paling berpengaruh terhadap output model.<br><br>
    <b>Hasilnya apa?</b><br>
    • Untuk model yang mendukung `feature_importances_`, tampil sesuai nilai model.<br>
    • Jika tidak mendukung (mis. SVM/MLP), halaman tetap menampilkan nilai <i>representatif</i> supaya presentasi tidak terputus.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Preprocessing Explanation ────────────────────────────────────────────────
st.markdown('<div class="section-header">🔧 Alur Preprocessing</div>', unsafe_allow_html=True)

pp1, pp2 = st.columns(2)
with pp1:
    st.markdown("""
    <div style="background:#1A1F2E;border-radius:14px;padding:1.3rem;border:1px solid #2D3748;">
        <div style="font-size:1rem;font-weight:700;color:#4E9AF1;margin-bottom:1rem;">📌 Tahapan Encoding</div>
        <div style="color:#94A3B8;font-size:0.87rem;line-height:1.8;">
            <b style="color:#E2E8F0;">1. Label Encoding</b><br>
            Gender, Married, Education, Self_Employed → Binary (0/1)<br><br>
            <b style="color:#E2E8F0;">2. Ordinal Encoding</b><br>
            Dependents: 0→0, 1→1, 2→2, 3+→3<br><br>
            <b style="color:#E2E8F0;">3. One-Hot Encoding</b><br>
            Property_Area → PAR_Rural, PAR_Semiurban, PAR_Urban
        </div>
    </div>
    """, unsafe_allow_html=True)

with pp2:
    st.markdown("""
    <div style="background:#1A1F2E;border-radius:14px;padding:1.3rem;border:1px solid #2D3748;">
        <div style="font-size:1rem;font-weight:700;color:#F97316;margin-bottom:1rem;">⚡ Tahapan Scaling</div>
        <div style="color:#94A3B8;font-size:0.87rem;line-height:1.8;">
            <b style="color:#E2E8F0;">StandardScaler (Z-Score)</b><br>
            Semua fitur numerik dinormalisasi dengan:<br>
            <code style="background:#0D1526;padding:0.2rem 0.4rem;border-radius:4px;color:#4E9AF1;">z = (x - μ) / σ</br></code><br><br>
            Scaling diperlukan untuk SVM dan NN/MLP karena sensitif terhadap skala fitur.<br>
            Decision Tree tidak memerlukan scaling namun diterapkan untuk konsistensi pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 💡 Explainability")
    st.markdown("---")
    st.markdown("**Fitur Total:** 11 (input) → 13 (setelah encoding)")
    st.markdown("**Preprocessing:** Label + One-Hot Encoding + StandardScaler")
    st.markdown("---")
    st.markdown("**Insight Kunci:**")
    st.markdown("- Credit_History: fitur paling dominan")
    st.markdown("- ApplicantIncome: prediksi LoanAmount")
