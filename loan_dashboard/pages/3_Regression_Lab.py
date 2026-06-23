"""
pages/3_Regression_Lab.py
Komparasi model regresi NN/MLP, SVM, Decision Tree untuk prediksi LoanAmount.
Metrik dihitung langsung dari CSV aktual.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.helper import load_reg_data, get_reg_metrics, BEST_PARAMS
from utils.visualization import (
    COLORS, plot_actual_vs_predicted, plot_model_comparison_bar
)

st.set_page_config(page_title="Regression Lab | Loan Dashboard", page_icon="📈", layout="wide")

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0E1117!important;color:#E2E8F0!important}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1526,#1A1F2E)!important;border-right:1px solid #2D3748!important}
    section[data-testid="stSidebar"] *{color:#E2E8F0!important}
    .main{background-color:#0E1117!important}.block-container{padding:2rem 2.5rem 3rem!important}
    [data-testid="stMetric"]{background:#1A1F2E!important;border:1px solid #2D3748!important;border-radius:12px!important;padding:1.1rem 1.3rem!important}
    [data-testid="stMetric"] label{color:#94A3B8!important;font-size:0.82rem!important}
    [data-testid="stMetricValue"]{color:#4E9AF1!important;font-weight:700!important}
    .section-header{font-size:1.25rem;font-weight:700;color:#E2E8F0;margin:1.5rem 0 0.8rem;padding-bottom:0.4rem;border-bottom:2px solid #1e3a6b}
    .algo-card{background:#1A1F2E;border:1px solid #2D3748;border-radius:14px;padding:1.3rem 1.5rem;transition:all 0.2s}
    .param-box{background:#0D1526;border-left:3px solid #F97316;border-radius:0 8px 8px 0;padding:0.8rem 1rem;font-size:0.85rem;color:#94A3B8;margin-top:0.8rem}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0E1117}::-webkit-scrollbar-thumb{background:#2D3748;border-radius:3px}
    </style>""", unsafe_allow_html=True)

inject_css()
render_sidebar("Regression Lab")

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(249,115,22,0.2);">
    <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">📈 Regression Lab</h1>
    <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Komparasi Algoritma Regresi LoanAmount — NN/MLP, SVM, Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ─── Load metrics dari CSV aktual ─────────────────────────────────────────────
REG_METRICS = get_reg_metrics()

algo_keys  = ["SVM", "DT", "NN"]
algo_names = {"SVM": "Support Vector Machine", "DT": "Decision Tree", "NN": "Neural Network / MLP"}
algo_icons = {"SVM": "🟣", "DT": "🟠", "NN": "🟢"}
algo_colors_map = {"SVM": "#7C5CBF", "DT": "#F97316", "NN": "#22C55E"}

best_algo = max(algo_keys, key=lambda a: REG_METRICS[a]["hpo"].get("r2", 0))

st.info("""
**💡 Mengapa LoanAmount Dipilih Sebagai Label Regresi?**

- **Bersifat numerik kontinu** — cocok untuk task regresi
- **Prediksi besaran pinjaman** berdasarkan profil finansial pemohon
- **Berkaitan erat** dengan fitur: `ApplicantIncome`, `CoapplicantIncome`, `Credit_History`, `Loan_Amount_Term`, `Property_Area`
""")

# ─── Algorithm Performance Cards ──────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Performa Algoritma Regresi (dari CSV Aktual)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
for col, algo in zip([col1, col2, col3], algo_keys):
    m = REG_METRICS[algo]["hpo"]
    color = algo_colors_map[algo]
    is_best = (algo == best_algo)
    border_style = f"border-top:4px solid {color}; border:2px solid {color};" if is_best else f"border-top:4px solid {color};"
    with col:
        st.markdown(f"""
        <div class="algo-card" style="{border_style}">
            <div style="font-size:1.1rem;font-weight:700;color:{color};">{algo_icons[algo]} {algo_names[algo]}</div>
            <div style="color:#94A3B8;font-size:0.8rem;margin-bottom:0.8rem;">{'⭐ Best Model' if is_best else 'NoHPO'}</div>
        </div>
        """, unsafe_allow_html=True)
        st.metric("🎯 R² Score", f"{m.get('r2', 0):.4f}")
        st.metric("📊 MAE", f"{m.get('mae', 0):.2f} K")
        st.metric("📈 RMSE", f"{m.get('rmse', 0):.2f} K")

# ─── HPO Parameter Summary ────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Parameter Terbaik Hasil HPO (Regresi)</div>', unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns(3)
param_keys = ["NN Regresi", "SVM Regresi", "DT Regresi"]
for pcol, pkey in zip([pc1, pc2, pc3], param_keys):
    with pcol:
        params = BEST_PARAMS[pkey]
        param_str = " ".join([f"<div>• <b>{k}</b>: {v}</div>" for k, v in params.items()])
        st.markdown(f"""
        <div class="param-box">
            <div style="color:#F97316;font-weight:600;margin-bottom:0.4rem;">{pkey}</div>
            {param_str}
        </div>
        """, unsafe_allow_html=True)

# ─── Metrics Comparison Table ─────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Tabel Perbandingan Metrik Evaluasi Regresi (dari CSV Aktual)</div>', unsafe_allow_html=True)

rows = []
for algo in algo_keys:
    m = REG_METRICS[algo]["hpo"]
    if not m:
        continue
    rows.append({
        "Algoritma": algo_names[algo],
        "R² Score" : m.get("r2", 0),
        "MAE (K)"  : m.get("mae", 0),
        "RMSE (K)" : m.get("rmse", 0),
    })

df_cmp = pd.DataFrame(rows)
if not df_cmp.empty:
    st.dataframe(
        df_cmp.style
            .format({"R² Score":"{:.4f}", "MAE (K)":"{:.2f}", "RMSE (K)":"{:.2f}"})
            .highlight_max(subset=["R² Score"], color="#1e3a6b")
            .highlight_min(subset=["MAE (K)", "RMSE (K)"], color="#0f2952"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
    <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    📌 Semua nilai metrik dihitung langsung dari kolom <code>prediksi</code> vs <code>label_asli</code>
    pada file CSV di folder <code>data/</code>. Highlight biru = R² tertinggi, gelap = MAE/RMSE terendah.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Metric comparison bar charts ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visualisasi Perbandingan Metrik Regresi</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["R² Score", "MAE", "RMSE"])

with tab1:
    fig = go.Figure()
    for var, color_ in [("hpo", COLORS["primary"]), ("nohpo", COLORS["muted"])]:
        fig.add_trace(go.Bar(
            name="HPO" if var == "hpo" else "Non-HPO",
            x=algo_keys,
            y=[REG_METRICS[a][var].get("r2", 0) for a in algo_keys],
            marker_color=color_,
            text=[f"{REG_METRICS[a][var].get('r2', 0):.4f}" for a in algo_keys],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
    fig.update_layout(
        barmode="group", title=dict(text="Perbandingan R² Score (dari CSV Aktual)", font=dict(color=COLORS["text"], size=15)),
        xaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]), legend=dict(font=dict(color=COLORS["text"])), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = go.Figure()
    for var, color_ in [("hpo", COLORS["accent"]), ("nohpo", COLORS["muted"])]:
        fig2.add_trace(go.Bar(
            name="HPO" if var == "hpo" else "Non-HPO",
            x=algo_keys,
            y=[REG_METRICS[a][var].get("mae", 0) for a in algo_keys],
            marker_color=color_,
            text=[f"{REG_METRICS[a][var].get('mae', 0):.2f}" for a in algo_keys],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
    fig2.update_layout(
        barmode="group", title=dict(text="Perbandingan MAE (K) — dari CSV Aktual", font=dict(color=COLORS["text"], size=15)),
        xaxis=dict(color=COLORS["text"]), yaxis=dict(color=COLORS["text"]),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]), legend=dict(font=dict(color=COLORS["text"])), height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = go.Figure()
    for var, color_ in [("hpo", COLORS["secondary"]), ("nohpo", COLORS["muted"])]:
        fig3.add_trace(go.Bar(
            name="HPO" if var == "hpo" else "Non-HPO",
            x=algo_keys,
            y=[REG_METRICS[a][var].get("rmse", 0) for a in algo_keys],
            marker_color=color_,
            text=[f"{REG_METRICS[a][var].get('rmse', 0):.2f}" for a in algo_keys],
            textposition="outside",
            textfont=dict(color=COLORS["text"]),
        ))
    fig3.update_layout(
        barmode="group", title=dict(text="Perbandingan RMSE (K) — dari CSV Aktual", font=dict(color=COLORS["text"], size=15)),
        xaxis=dict(color=COLORS["text"]), yaxis=dict(color=COLORS["text"]),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]), legend=dict(font=dict(color=COLORS["text"])), height=380,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ─── Scatter Plots: Actual vs Predicted — semua model ─────────────────────────
st.markdown('<div class="section-header">📉 Scatter Plot: Actual vs Predicted LoanAmount — Semua Model</div>', unsafe_allow_html=True)

reg_csv_map = {
    "SVM": "reg_svm_nohpo",
    "DT" : "reg_dt_nohpo",
    "NN" : "reg_nn_nohpo",
}

sp_cols = st.columns(3)
for col, algo in zip(sp_cols, algo_keys):
    with col:
        df_reg = load_reg_data(reg_csv_map[algo])
        st.markdown(f"**{algo_icons[algo]} {algo_names[algo]}**")
        if not df_reg.empty and "prediksi" in df_reg.columns and "label_asli" in df_reg.columns:
            y_true = df_reg["label_asli"].values
            y_pred = df_reg["prediksi"].values
            fig = plot_actual_vs_predicted(y_true, y_pred, f"Actual vs Predicted — {algo}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data tidak tersedia.")

st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
    <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    📌 Scatter plot menampilkan data nyata dari kolom <code>label_asli</code> (aktual) vs <code>prediksi</code>
    pada CSV regresi. Garis diagonal = prediksi sempurna. Semakin dekat titik ke garis, semakin akurat model.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Kesimpulan (Berdasarkan Data CSV Aktual)</div>', unsafe_allow_html=True)

best_m = REG_METRICS[best_algo]["hpo"]
st.success(f"""
**🏆 Model Terbaik: {algo_names[best_algo]}**

Berdasarkan evaluasi metrik dari CSV hasil prediksi aktual di folder `data/`:

- **R² Score:** {best_m.get('r2', 0):.4f}
- **MAE:** {best_m.get('mae', 0):.2f} K
- **RMSE:** {best_m.get('rmse', 0):.2f} K

Model ini memiliki R² Score tertinggi dari ketiga algoritma yang dievaluasi.
""")
