"""
pages/4_Feature_Analytics.py
Analisis hubungan antar fitur: korelasi, distribusi, dan boxplot.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import numpy as np

from utils.helper import load_raw_data
from utils.visualization import (
    COLORS, plot_correlation_heatmap, plot_boxplot,
    plot_stacked_bar, plot_histogram, plot_pie
)

st.set_page_config(page_title="Feature Analytics | Loan Dashboard", page_icon="🔍", layout="wide")

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0E1117!important;color:#E2E8F0!important}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1526,#1A1F2E)!important;border-right:1px solid #2D3748!important}
    section[data-testid="stSidebar"] *{color:#E2E8F0!important}
    .main{background-color:#0E1117!important}.block-container{padding:2rem 2.5rem 3rem!important}
    .section-header{font-size:1.25rem;font-weight:700;color:#E2E8F0;margin:1.5rem 0 0.8rem;padding-bottom:0.4rem;border-bottom:2px solid #1e3a6b}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0E1117}::-webkit-scrollbar-thumb{background:#2D3748;border-radius:3px}
    </style>""", unsafe_allow_html=True)

inject_css()

# Sidebar global agar ringkasan tetap muncul di halaman ini
render_sidebar("Feature Analytics")

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(34,197,94,0.2);">
    <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">🔍 Feature Analytics</h1>
    <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Analisis korelasi, distribusi, dan hubungan antar fitur dataset pinjaman</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("Memuat data..."):
    df = load_raw_data()

if df.empty:
    st.error("❌ Dataset tidak ditemukan.")
    st.stop()

# ─── Correlation Heatmap ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Heatmap Korelasi Fitur Numerik</div>', unsafe_allow_html=True)

corr_cols = [c for c in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
             if c in df.columns]

if len(corr_cols) >= 2:
    fig = plot_correlation_heatmap(df, corr_cols, "Heatmap Korelasi Fitur Numerik")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
        <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
        📌 <b>Interpretasi Heatmap Korelasi:</b><br>
        • <b>ApplicantIncome ↔ LoanAmount (positif):</b> Penghasilan lebih tinggi → pinjaman yang diminta lebih besar<br>
        • <b>CoapplicantIncome ↔ LoanAmount:</b> Kontribusi co-applicant mempengaruhi kemampuan pinjaman<br>
        • <b>Credit_History:</b> Mempengaruhi keputusan approval, namun tidak selalu besaran pinjaman<br>
        • <b>Loan_Amount_Term:</b> Relatif independen, dapat fleksibel diatur berdasarkan kesepakatan
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Kolom numerik tidak cukup untuk heatmap korelasi.")

# ─── Boxplot LoanAmount ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📦 Distribusi LoanAmount berdasarkan Kategori</div>', unsafe_allow_html=True)

if "LoanAmount" in df.columns:
    b1, b2, b3 = st.columns(3)

    with b1:
        if "Education" in df.columns:
            # Decode Education: 0=Graduate, 1=Not Graduate
            df_plot = df.copy()
            df_plot["Education_Label"] = df_plot["Education"].map({0:"Graduate", 1:"Not Graduate"})
            fig = plot_boxplot(df_plot, "Education_Label", "LoanAmount", "LoanAmount vs Education")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="background:#1A1F2E;border-radius:12px;padding:0.8rem;border:1px solid #2D3748;font-size:0.85rem;color:#94A3B8;">
            📌 <b>Insight:</b> Graduate cenderung mengajukan pinjaman lebih tinggi karena penghasilan rata-rata lebih besar dan kepercayaan bank lebih tinggi.
            </div>
            """, unsafe_allow_html=True)

    with b2:
        if all(c in df.columns for c in ["PAR_Rural", "PAR_Semiurban", "PAR_Urban"]):
            df_plot2 = df.copy()
            def get_area(row):
                if row["PAR_Rural"] == 1: return "Rural"
                if row["PAR_Semiurban"] == 1: return "Semiurban"
                return "Urban"
            df_plot2["Property_Area"] = df_plot2.apply(get_area, axis=1)
            fig = plot_boxplot(df_plot2, "Property_Area", "LoanAmount", "LoanAmount vs Property Area")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="background:#1A1F2E;border-radius:12px;padding:0.8rem;border:1px solid #2D3748;font-size:0.85rem;color:#94A3B8;">
            📌 <b>Insight:</b> Area Urban menunjukkan median LoanAmount tertinggi. Ini mencerminkan nilai properti lebih tinggi dan daya beli lebih kuat di area perkotaan.
            </div>
            """, unsafe_allow_html=True)

    with b3:
        if "Self_Employed" in df.columns:
            df_plot3 = df.copy()
            df_plot3["Self_Employed_Label"] = df_plot3["Self_Employed"].map({0:"No", 1:"Yes"})
            fig = plot_boxplot(df_plot3, "Self_Employed_Label", "LoanAmount", "LoanAmount vs Self Employed")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <div style="background:#1A1F2E;border-radius:12px;padding:0.8rem;border:1px solid #2D3748;font-size:0.85rem;color:#94A3B8;">
            📌 <b>Insight:</b> Self-employed tidak menunjukkan perbedaan signifikan dalam besaran pinjaman, namun memiliki variabilitas lebih tinggi karena penghasilan lebih fluktuatif.
            </div>
            """, unsafe_allow_html=True)

# ─── Credit History vs Loan Status ────────────────────────────────────────────
st.markdown('<div class="section-header">💳 Distribusi Credit_History vs Loan_Status</div>', unsafe_allow_html=True)

label_col = "label_asli" if "label_asli" in df.columns else "prediksi"
if "Credit_History" in df.columns and label_col in df.columns:
    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        fig = plot_stacked_bar(
            df, x_col="Credit_History", hue_col=label_col,
            title="Credit_History vs Loan_Status (Stacked Bar)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch_col2:
        # Compute rates
        credit_stats = df.groupby("Credit_History")[label_col].agg(["sum","count"])
        credit_stats["approval_rate"] = credit_stats["sum"] / credit_stats["count"] * 100
        credit_stats.index = ["No Credit (0)" if i == 0 else "Good Credit (1)" for i in credit_stats.index]

        import plotly.graph_objects as go
        fig2 = go.Figure(go.Bar(
            x=credit_stats.index,
            y=credit_stats["approval_rate"],
            marker_color=[COLORS["danger"], COLORS["success"]],
            text=[f"{v:.1f}%" for v in credit_stats["approval_rate"]],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
        fig2.update_layout(
            title=dict(text="Tingkat Approval per Credit History", font=dict(color=COLORS["text"], size=15)),
            xaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
            yaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"], range=[0, 110]),
            paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
            font=dict(color=COLORS["text"]), height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
        <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
        📌 <b>Penjelasan Chart:</b> Stacked bar chart menunjukkan komposisi approval/rejection per Credit History status. 
        Bar chart kedua menampilkan persentase approval rate — pemohon dengan riwayat kredit baik memiliki tingkat persetujuan 
        jauh lebih tinggi. <b>Credit_History adalah fitur paling diskriminatif dalam klasifikasi Loan_Status.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Income Distribution ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">💰 Analisis Distribusi Pendapatan</div>', unsafe_allow_html=True)

inc_col1, inc_col2 = st.columns(2)
with inc_col1:
    if "ApplicantIncome" in df.columns:
        fig = plot_histogram(df["ApplicantIncome"], "Distribusi ApplicantIncome", "Income (USD)", COLORS["primary"])
        st.plotly_chart(fig, use_container_width=True)
with inc_col2:
    if "CoapplicantIncome" in df.columns:
        # Only show non-zero
        co_inc_nonzero = df["CoapplicantIncome"][df["CoapplicantIncome"] > 0]
        fig = plot_histogram(co_inc_nonzero, "Distribusi CoapplicantIncome (Non-Zero)", "Income (USD)", COLORS["accent"])
        st.plotly_chart(fig, use_container_width=True)

# ─── Gender & Marital Analysis ────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Profil Demografis Pemohon</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3)
with d1:
    if "Gender" in df.columns:
        g_counts = df["Gender"].value_counts()
        labels_ = ["Male" if i == 1 else "Female" for i in g_counts.index]
        fig = plot_pie(labels_, g_counts.values, "Distribusi Gender")
        st.plotly_chart(fig, use_container_width=True)

with d2:
    if "Married" in df.columns:
        m_counts = df["Married"].value_counts()
        labels_ = ["Married" if i == 1 else "Single" for i in m_counts.index]
        fig = plot_pie(labels_, m_counts.values, "Status Pernikahan")
        st.plotly_chart(fig, use_container_width=True)

with d3:
    if "Education" in df.columns:
        e_counts = df["Education"].value_counts()
        labels_ = ["Graduate" if i == 0 else "Not Graduate" for i in e_counts.index]
        fig = plot_pie(labels_, e_counts.values, "Tingkat Pendidikan")
        st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.markdown("### 🔍 Feature Analytics")
    st.markdown("---")
    st.markdown("**Analisis tersedia:**")
    st.markdown("- Korelasi fitur numerik")
    st.markdown("- Distribusi LoanAmount")
    st.markdown("- Credit History vs Status")
    st.markdown("- Distribusi pendapatan")
    st.markdown("- Profil demografis")
