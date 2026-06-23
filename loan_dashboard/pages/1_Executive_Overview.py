"""
pages/1_Executive_Overview.py
Ringkasan KPI, distribusi data, dan visualisasi profil dataset keseluruhan.
Metrik klasifikasi & regresi dihitung langsung dari CSV aktual.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import numpy as np

from utils.helper import load_raw_data, get_clf_metrics, get_reg_metrics, BASE_DIR
from utils.visualization import (
    COLORS, plot_histogram, plot_pie, plot_stacked_bar, plot_correlation_heatmap
)

st.set_page_config(page_title="Executive Overview | Loan Dashboard", page_icon="📊", layout="wide")

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #0E1117 !important; color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1526 0%, #1A1F2E 100%) !important; border-right: 1px solid #2D3748 !important; }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    .main { background-color: #0E1117 !important; }
    .block-container { padding: 2rem 2.5rem 3rem !important; }
    [data-testid="stMetric"] { background: #1A1F2E !important; border: 1px solid #2D3748 !important; border-radius: 12px !important; padding: 1.1rem 1.3rem !important; }
    [data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.82rem !important; }
    [data-testid="stMetricValue"] { color: #4E9AF1 !important; font-weight: 700 !important; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #E2E8F0; margin: 1.5rem 0 0.8rem; padding-bottom: 0.4rem; border-bottom: 2px solid #1e3a6b; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

inject_css()
render_sidebar("Executive Overview")

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(78,154,241,0.2);">
    <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">📊 Executive Overview</h1>
    <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Ringkasan metrik utama & distribusi dataset pinjaman</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Memuat data..."):
    df_raw = load_raw_data()

if df_raw.empty:
    st.error("❌ Dataset tidak ditemukan. Pastikan file CSV tersedia di folder `data/`.")
    st.stop()

df = df_raw.copy()
loan_col   = "LoanAmount"
status_col = "label_asli"
income_col = "ApplicantIncome"
co_income  = "CoapplicantIncome"

# ─── KPI Section ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

total_records = len(df)
approved      = int(df[status_col].sum()) if status_col in df.columns else 0
rejected      = total_records - approved
pct_approved  = approved / total_records * 100 if total_records > 0 else 0
avg_loan      = df[loan_col].mean() if loan_col in df.columns else 0
avg_income    = df[income_col].mean() if income_col in df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📁 Total Pemohon", f"{total_records:,}")
k2.metric("✅ Approved", f"{approved:,}", delta=f"{pct_approved:.1f}%")
k3.metric("❌ Rejected", f"{rejected:,}", delta=f"-{100-pct_approved:.1f}%")
k4.metric("💰 Avg LoanAmount", f"${avg_loan:,.1f}K" if avg_loan > 0 else "N/A")
k5.metric("💵 Avg ApplicantIncome", f"${avg_income:,.0f}" if avg_income > 0 else "N/A")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📉 Distribusi LoanAmount & Proporsi Loan_Status</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if loan_col in df.columns:
        fig = plot_histogram(df[loan_col], title="Distribusi LoanAmount (ribuan USD)", xlab="LoanAmount (K)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
            <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
            📌 <b>Penjelasan:</b> Histogram distribusi frekuensi LoanAmount dari data training aktual (ribuan USD).
            Mayoritas pemohon mengajukan pinjaman dalam rentang tertentu. Data ini berasal langsung dari
            <code>train_u6lujuX_CVtuZ9i.csv</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

with c2:
    label_col = status_col if status_col in df.columns else "prediksi"
    if label_col in df.columns:
        counts  = df[label_col].value_counts().dropna()
        labels_ = ["Approved" if l == 1 else "Rejected" for l in counts.index]
        fig = plot_pie(labels_, counts.values, title="Proporsi Loan_Status")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
            <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
            📌 <b>Penjelasan:</b> Proporsi Loan_Status dari dataset training nyata. Imbalans kelas ini
            penting diperhatikan dalam pelatihan model klasifikasi.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Charts Row 2 ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏘️ Distribusi Property Area & Credit History</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    if all(c in df.columns for c in ["PAR_Rural", "PAR_Semiurban", "PAR_Urban"]):
        area_counts = {
            "Rural"    : int(df["PAR_Rural"].sum()),
            "Semiurban": int(df["PAR_Semiurban"].sum()),
            "Urban"    : int(df["PAR_Urban"].sum()),
        }
        area_series = pd.Series(area_counts)
        area_series = area_series[area_series > 0]
        if not area_series.empty:
            fig = plot_pie(area_series.index.tolist(), area_series.values.tolist(), title="Distribusi Property Area")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
                <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
                📌 <b>Penjelasan:</b> Distribusi lokasi properti pemohon dari dataset training aktual.
                </p>
            </div>
            """, unsafe_allow_html=True)

with c4:
    if "Credit_History" in df.columns and label_col in df.columns:
        fig = plot_stacked_bar(df, x_col="Credit_History", hue_col=label_col,
                               title="Credit_History vs Loan_Status")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
            <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
            📌 <b>Penjelasan:</b> Hubungan antara riwayat kredit dan status pinjaman dari data nyata.
            Pemohon dengan Credit_History=1 memiliki tingkat persetujuan jauh lebih tinggi.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ─── Income Distribution ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">💵 Distribusi Pendapatan Pemohon</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    if income_col in df.columns:
        fig = plot_histogram(df[income_col], title="Distribusi ApplicantIncome", xlab="Income (USD)", color=COLORS["accent"])
        st.plotly_chart(fig, use_container_width=True)

with c6:
    if co_income in df.columns:
        fig = plot_histogram(df[co_income], title="Distribusi CoapplicantIncome", xlab="Income (USD)", color=COLORS["secondary"])
        st.plotly_chart(fig, use_container_width=True)

# ─── Correlation Heatmap ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Korelasi Fitur Numerik</div>', unsafe_allow_html=True)

num_cols = [c for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
            if c in df.columns]
if len(num_cols) > 1:
    fig = plot_correlation_heatmap(df, num_cols, "Heatmap Korelasi Fitur Numerik")
    st.plotly_chart(fig, use_container_width=True)

# ─── Evaluation Metrics Comparison Table — dihitung dari CSV aktual ────────────
st.markdown('<div class="section-header">📊 Perbandingan Metrik Evaluasi Model (dari CSV Aktual)</div>', unsafe_allow_html=True)

CLF_METRICS = get_clf_metrics()
REG_METRICS = get_reg_metrics()

clf_rows = []
for algo in ["SVM", "DT", "NN"]:
    for variant in ["hpo", "nohpo"]:
        m = CLF_METRICS[algo][variant]
        if not m:
            continue
        clf_rows.append({
            "Algoritma": "Neural Network / MLP" if algo == "NN" else algo,
            "Versi": "HPO" if variant == "hpo" else "Non-HPO",
            "Accuracy" : f"{m.get('accuracy', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall"   : f"{m.get('recall', 0):.4f}",
            "F1-Score" : f"{m.get('f1', 0):.4f}",
        })

clf_tab, reg_tab = st.tabs(["Classification Metrics", "Regression Metrics"])

with clf_tab:
    if clf_rows:
        st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Data klasifikasi tidak tersedia.")

with reg_tab:
    reg_rows = []
    for algo in ["SVM", "DT", "NN"]:
        for variant in ["hpo", "nohpo"]:
            m = REG_METRICS[algo][variant]
            if not m:
                continue
            reg_rows.append({
                "Algoritma": "Neural Network / MLP" if algo == "NN" else algo,
                "Versi"    : "HPO" if variant == "hpo" else "Non-HPO",
                "R² Score" : f"{m.get('r2', 0):.4f}",
                "MAE (K)"  : f"{m.get('mae', 0):.2f}",
                "RMSE (K)" : f"{m.get('rmse', 0):.2f}",
            })
    if reg_rows:
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Data regresi tidak tersedia.")

# ─── Data Preview ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Preview Dataset Training (10 Baris Pertama)</div>', unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True, hide_index=True)
st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
    <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    📌 Data ditampilkan langsung dari <code>train_u6lujuX_CVtuZ9i.csv</code> setelah preprocessing (encoding & one-hot).
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Executive Overview")
    st.markdown("---")
    st.metric("Total Records", f"{total_records:,}")
    st.metric("Approval Rate", f"{pct_approved:.1f}%")
    st.markdown("---")
    st.caption("Data dari training set model klasifikasi.")
