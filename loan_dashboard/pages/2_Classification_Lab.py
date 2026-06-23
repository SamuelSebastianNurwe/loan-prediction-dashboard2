"""
pages/2_Classification_Lab.py
Perbandingan model klasifikasi NN/MLP, SVM, Decision Tree.
Metrik dihitung langsung dari CSV aktual.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils.sidebar import render_sidebar
import pandas as pd
import numpy as np

from utils.helper import (
    load_clf_data, get_clf_metrics, BEST_PARAMS,
)
from utils.visualization import (
    COLORS, plot_confusion_matrix, plot_model_comparison_bar, ALGO_COLORS
)

st.set_page_config(page_title="Classification Lab | Loan Dashboard", page_icon="🔬", layout="wide")

def inject_css():
    st.markdown("""
    <style>
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
    .algo-card:hover{border-color:#4E9AF1;box-shadow:0 6px 20px rgba(78,154,241,0.12)}
    .param-box{background:#0D1526;border-left:3px solid #4E9AF1;border-radius:0 8px 8px 0;padding:0.8rem 1rem;font-size:0.85rem;color:#94A3B8;margin-top:0.8rem}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#0E1117}::-webkit-scrollbar-thumb{background:#2D3748;border-radius:3px}
    </style>""", unsafe_allow_html=True)

inject_css()
render_sidebar("Classification Lab")

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(78,154,241,0.2);">
    <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">🔬 Classification Lab</h1>
    <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Komparasi Algoritma Klasifikasi Loan_Status — NN/MLP, SVM, Decision Tree</p>
</div>
""", unsafe_allow_html=True)

# ─── Load metrics dari CSV aktual ─────────────────────────────────────────────
CLF_METRICS = get_clf_metrics()

algo_keys  = ["SVM", "DT", "NN"]
algo_names = {"SVM": "Support Vector Machine", "DT": "Decision Tree", "NN": "Neural Network / MLP"}
algo_icons = {"SVM": "🟣", "DT": "🟠", "NN": "🟢"}
algo_colors_map = {"SVM": "#7C5CBF", "DT": "#F97316", "NN": "#22C55E"}

# ─── Determine best model berdasarkan F1 ──────────────────────────────────────
best_algo = max(algo_keys, key=lambda a: CLF_METRICS[a]["hpo"].get("f1", 0))

st.markdown('<div class="section-header">🏆 Performa Algoritma (dari CSV Aktual)</div>', unsafe_allow_html=True)

cols = st.columns(len(algo_keys))
for col, algo in zip(cols, algo_keys):
    m = CLF_METRICS[algo]["hpo"]
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
        st.metric("Accuracy",  f"{m.get('accuracy', 0):.4f}")
        st.metric("Precision", f"{m.get('precision', 0):.4f}")
        st.metric("Recall",    f"{m.get('recall', 0):.4f}")
        st.metric("F1-Score",  f"{m.get('f1', 0):.4f}")

# ─── HPO Parameter Summary ────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚙️ Parameter Terbaik Hasil HPO</div>', unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns(3)
param_keys = ["NN Klasifikasi", "SVM Klasifikasi", "DT Klasifikasi"]
for pcol, pkey in zip([pc1, pc2, pc3], param_keys):
    with pcol:
        params = BEST_PARAMS[pkey]
        param_str = "\n".join([f"• <b>{k}</b>: {v}" for k, v in params.items()])
        st.markdown(f"""
        <div class="param-box">
            <div style="color:#4E9AF1;font-weight:600;margin-bottom:0.4rem;">{pkey}</div>
            {param_str}
        </div>
        """, unsafe_allow_html=True)

# ─── Metrics Comparison Table ─────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Tabel Perbandingan Metrik Evaluasi (dari CSV Aktual)</div>', unsafe_allow_html=True)

rows = []
for algo in algo_keys:
    m = CLF_METRICS[algo]["hpo"]
    if not m:
        continue
    rows.append({
        "Algoritma" : algo_names[algo],
        "Accuracy"  : m.get("accuracy", 0),
        "Precision" : m.get("precision", 0),
        "Recall"    : m.get("recall", 0),
        "F1-Score"  : m.get("f1", 0),
    })

df_cmp = pd.DataFrame(rows)
if not df_cmp.empty:
    st.dataframe(
        df_cmp.style
            .format({"Accuracy":"{:.4f}", "Precision":"{:.4f}", "Recall":"{:.4f}", "F1-Score":"{:.4f}"})
            .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"], color="#1e3a6b"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
    <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    📌 <b>Catatan:</b> Semua nilai metrik dihitung langsung dari kolom <code>prediksi</code> dan <code>label_asli</code>
    pada masing-masing file CSV di folder <code>data/</code>.
    Highlight biru menunjukkan nilai terbaik untuk setiap metrik.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Metric comparison bar charts ─────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visualisasi Perbandingan Metrik</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "Precision", "Recall", "F1-Score"])
for tab, metric in zip([tab1, tab2, tab3, tab4], ["Accuracy", "Precision", "Recall", "F1-Score"]):
    with tab:
        fig = plot_model_comparison_bar(CLF_METRICS, metric=metric.lower(), title=f"Perbandingan {metric} — Klasifikasi")
        st.plotly_chart(fig, use_container_width=True)

# ─── Confusion Matrices dari CSV aktual ───────────────────────────────────────
st.markdown('<div class="section-header">🔢 Confusion Matrix — Semua Model (dari CSV Aktual)</div>', unsafe_allow_html=True)

cm_csv_map = {
    "SVM": "clf_svm_hpo",
    "DT" : "clf_dt_hpo",
    "NN" : "clf_nn_hpo",
}

cm_cols = st.columns(3)
for col, algo in zip(cm_cols, algo_keys):
    with col:
        df_cm = load_clf_data(cm_csv_map[algo])
        st.markdown(f"**{algo_icons[algo]} {algo_names[algo]}**")
        if not df_cm.empty and "prediksi" in df_cm.columns and "label_asli" in df_cm.columns:
            y_pred = df_cm["prediksi"].astype(int)
            y_true = df_cm["label_asli"].astype(int)
            fig = plot_confusion_matrix(y_true, y_pred, f"CM — {algo}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data tidak tersedia.")

st.markdown("""
<div style="background:#1A1F2E;border-radius:12px;padding:1rem;border:1px solid #2D3748;margin-top:0.5rem;">
    <p style="color:#94A3B8;font-size:0.9rem;margin:0;line-height:1.6;">
    📌 Confusion matrix dihitung dari kolom <code>prediksi</code> vs <code>label_asli</code> pada CSV aktual masing-masing model.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Conclusion ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Kesimpulan (Berdasarkan Data CSV Aktual)</div>', unsafe_allow_html=True)

best_m = CLF_METRICS[best_algo]["hpo"]
st.success(f"""
**🏆 Model Terbaik: {algo_names[best_algo]}**

Berdasarkan evaluasi metrik dari CSV hasil prediksi aktual:

- **Accuracy:** {best_m.get('accuracy', 0):.4f} ({best_m.get('accuracy', 0)*100:.2f}%)
- **Precision:** {best_m.get('precision', 0):.4f}
- **Recall:** {best_m.get('recall', 0):.4f}
- **F1-Score:** {best_m.get('f1', 0):.4f}

Model ini dipilih berdasarkan F1-Score tertinggi dari hasil prediksi nyata pada file CSV di folder `data/`.
""")
