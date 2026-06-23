"""
pages/6_Classification_Simulator.py
Simulasi prediksi kelayakan pinjaman (Loan_Status).
Input numerik langsung (number_input + radio) — tidak ada slider/selectbox kategori.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from utils.sidebar import render_sidebar
from utils import helper as _helper
from utils.helper import (
    load_model,
    preprocess_for_model,
    prepare_batch_features,
    CLF_FEATURE_COLS,
    predict_loan_status,
    build_applicant_narrative,
    build_recommendations,
    FEATURE_DESCRIPTIONS,
    CLF_METRICS,
)
from utils.visualization import COLORS, plot_radar_chart, plot_pie, plot_stacked_bar

st.set_page_config(
    page_title="Classification Simulator | Loan Dashboard",
    page_icon="🔬",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:#0E1117!important;color:#E2E8F0!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1526,#1A1F2E)!important;border-right:1px solid #2D3748!important}
section[data-testid="stSidebar"] *{color:#E2E8F0!important}
.main{background-color:#0E1117!important}
.block-container{padding:2rem 2.5rem 3rem!important}
[data-testid="stMetric"]{background:#1A1F2E!important;border:1px solid #2D3748!important;border-radius:12px!important;padding:1.1rem 1.3rem!important}
[data-testid="stMetric"] label{color:#94A3B8!important;font-size:0.82rem!important}
[data-testid="stMetricValue"]{color:#4E9AF1!important;font-weight:700!important}
.section-header{font-size:1.25rem;font-weight:700;color:#E2E8F0;margin:1.5rem 0 0.8rem;padding-bottom:0.4rem;border-bottom:2px solid #1e3a6b}
.input-section{background:#1A1F2E;border:1px solid #2D3748;border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.2rem}
.input-group-title{font-size:1rem;font-weight:700;color:#4E9AF1;margin-bottom:1.1rem;letter-spacing:0.02em}
.stButton>button{background:linear-gradient(135deg,#1e3a6b,#4E9AF1)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:700!important;padding:0.7rem 2rem!important;font-size:1rem!important;transition:all 0.25s ease!important;width:100%!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(78,154,241,0.4)!important}
.result-approved{background:linear-gradient(135deg,#064e3b,#065f46);border:2px solid #22C55E;border-radius:16px;padding:1.5rem 2rem;text-align:center;margin:1rem 0}
.result-rejected{background:linear-gradient(135deg,#450a0a,#7f1d1d);border:2px solid #EF4444;border-radius:16px;padding:1.5rem 2rem;text-align:center;margin:1rem 0}
.rec-item{background:#1A1F2E;border:1px solid #2D3748;border-radius:10px;padding:0.8rem 1.1rem;margin-bottom:0.5rem;font-size:0.9rem;line-height:1.5}
.explain-card{background:#1A1F2E;border-radius:14px;padding:1.3rem;border:1px solid #2D3748;margin-top:0.5rem}
.explain-title{font-size:1rem;font-weight:700;color:#4E9AF1;margin-bottom:0.7rem}
.explain-text{color:#94A3B8;font-size:0.87rem;line-height:1.7}
div[data-testid="stNumberInput"] input{background:#0D1526!important;border:1px solid #2D3748!important;border-radius:8px!important;color:#E2E8F0!important;font-size:0.95rem!important}
div[data-testid="stNumberInput"] button{background:#1e3a6b!important;border:1px solid #2D3748!important;color:#E2E8F0!important}
div[data-testid="stRadio"] label{color:#E2E8F0!important;font-size:0.9rem!important}
</style>
""", unsafe_allow_html=True)

render_sidebar("Classification Simulator")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(78,154,241,0.25);">
  <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">🔬 Classification Simulator</h1>
  <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Simulasi prediksi kelayakan kredit (Loan_Status) — masukkan nilai numerik secara langsung</p>
</div>
""", unsafe_allow_html=True)

def explain_card(title, text):
    st.markdown(f"""
<div class="explain-card">
  <div class="explain-title">{title}</div>
  <div class="explain-text">{text}</div>
</div>""", unsafe_allow_html=True)

# ─── Sidebar: mode & model ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧰 Mode Prediksi")
    pred_mode = st.radio("Pilih mode", ["Prediksi Manual", "Prediksi CSV/Excel"], index=0)
    st.markdown("---")
    st.markdown("## 🧠 Pilih Model")
    clf_choice = st.selectbox("📊 Model Klasifikasi", ["SVM", "NN (MLP)", "DT"], key="sim_clf_model")
    st.markdown("---")
    st.markdown("### 🏆 Performa Model")
    algo_key = "SVM" if clf_choice == "SVM" else ("NN" if clf_choice == "NN (MLP)" else "DT")
    m = CLF_METRICS[algo_key]["hpo"]
    st.metric("Accuracy",  f"{m.get('accuracy',0):.4f}")
    st.metric("F1-Score",  f"{m.get('f1',0):.4f}")
    st.markdown("---")

# ─── Helper ───────────────────────────────────────────────────────────────────
def get_clf_model(choice):
    return load_model({"SVM":"clf_svm_hpo","NN (MLP)":"clf_nn_hpo","DT":"clf_dt_hpo"}[choice])

# ══════════════════════════════════════════════════════════════════════════════
# MODE: PREDIKSI MANUAL
# ══════════════════════════════════════════════════════════════════════════════
if pred_mode == "Prediksi Manual":

    st.markdown('<div class="section-header">📝 Input Profil Pemohon</div>', unsafe_allow_html=True)
    st.caption("Masukkan nilai numerik langsung untuk setiap fitur pemohon. Klik ＋/－ untuk mengubah nilai.")

    # ── Blok 1: Fitur Numerik ─────────────────────────────────────────────────
    st.markdown('<div class="input-group-title">💵 Fitur Numerik</div>', unsafe_allow_html=True)

    n1, n2, n3 = st.columns(3)
    with n1:
        applicant_income = st.number_input(
            "ApplicantIncome (USD/bulan)",
            min_value=0, max_value=100000, value=5000, step=100,
            help="Pendapatan bulanan pemohon utama dalam USD"
        )
    with n2:
        coapplicant_income = st.number_input(
            "CoapplicantIncome (USD/bulan)",
            min_value=0, max_value=50000, value=0, step=100,
            help="Pendapatan bulanan co-applicant (0 jika tidak ada)"
        )
    with n3:
        loan_amount = st.number_input(
            "LoanAmount (ribuan USD)",
            min_value=1, max_value=700, value=150, step=1,
            help="Jumlah pinjaman yang diajukan dalam ribuan USD"
        )

    n4, n5 = st.columns(2)
    with n4:
        loan_term = st.number_input(
            "Loan_Amount_Term (bulan)",
            min_value=12, max_value=480, value=360, step=12,
            help="Tenor/durasi pinjaman dalam bulan (misal: 360 = 30 tahun)"
        )
    with n5:
        dependents = st.number_input(
            "Dependents (jumlah tanggungan)",
            min_value=0, max_value=10, value=0, step=1,
            help="Jumlah anggota keluarga yang ditanggung pemohon"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Blok 2: Fitur Kategorikal (radio) ─────────────────────────────────────
    st.markdown('<div class="input-group-title">🗂️ Fitur Kategorikal</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**👤 Gender**")
        gender = st.radio("Gender", ["Male", "Female"], index=0, key="clf_gender", label_visibility="collapsed")

        st.markdown("**💍 Married**")
        married = st.radio("Married", ["Yes", "No"], index=0, key="clf_married", label_visibility="collapsed")

    with r2:
        st.markdown("**🎓 Education**")
        education = st.radio("Education", ["Graduate", "Not Graduate"], index=0, key="clf_edu", label_visibility="collapsed")

        st.markdown("**💼 Self Employed**")
        self_emp = st.radio("Self Employed", ["No", "Yes"], index=0, key="clf_selfemp", label_visibility="collapsed")

    with r3:
        st.markdown("**💳 Credit History**")
        credit_opt = st.radio("Credit History", ["1 — Good Credit", "0 — No/Bad Credit"], index=0, key="clf_credit", label_visibility="collapsed")

        st.markdown("**🏘️ Property Area**")
        prop_area = st.radio("Property Area", ["Urban", "Semiurban", "Rural"], index=0, key="clf_area", label_visibility="collapsed")

    credit_val = 1.0 if credit_opt.startswith("1") else 0.0

    # ── Tombol prediksi ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Prediksi Kelayakan Kredit", use_container_width=True, key="clf_predict_btn")

    # ── Tabel preview input ───────────────────────────────────────────────────
    with st.expander("📋 Lihat Data Input", expanded=False):
        dep_str = f"{dependents}" if dependents < 3 else "3+"
        preview = {
            "Fitur": ["Gender","Married","Dependents","Education","Self_Employed",
                      "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"],
            "Nilai": [gender, married, dep_str, education, self_emp,
                      f"{applicant_income:,}", f"{coapplicant_income:,}",
                      f"{loan_amount:,}K", f"{loan_term} bln",
                      "1 (Baik)" if credit_val == 1.0 else "0 (Buruk)", prop_area],
            "Keterangan": [
                "Male=1 / Female=0","Yes=1 / No=0","0–3+","Graduate=0 / Not Graduate=1",
                "No=0 / Yes=1","USD per bulan","USD per bulan","Ribuan USD","Bulan",
                "Riwayat kredit bersih=1","Lokasi properti"
            ],
        }
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ── Hasil prediksi ────────────────────────────────────────────────────────
    if predict_btn:
        dep_str = f"{dependents}" if dependents < 3 else "3+"
        input_dict = {
            "Gender": gender, "Married": married, "Dependents": dep_str,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": credit_val, "Property_Area": prop_area,
        }

        clf_model = get_clf_model(clf_choice)
        with st.spinner("⚙️ Memproses prediksi..."):
            clf_feats = preprocess_for_model(input_dict, clf_model, CLF_FEATURE_COLS)
            pred_int, clf_label = predict_loan_status(clf_feats, clf_model)

        st.markdown('<div class="section-header">📊 Hasil Prediksi Kelayakan Kredit</div>', unsafe_allow_html=True)

        res_col, metric_col = st.columns(2)
        with res_col:
            approved = (clf_label == "Disetujui")
            if approved:
                st.markdown(f"""
<div class="result-approved">
  <div style="color:#22C55E;font-size:2.2rem;font-weight:800;">✅ DISETUJUI</div>
  <div style="font-size:0.95rem;margin-top:0.4rem;">Pemohon dinyatakan <b>LAYAK</b> mendapatkan pinjaman</div>
  <div style="margin-top:0.6rem;font-size:0.82rem;color:#86efac;">Model: {clf_choice}</div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-rejected">
  <div style="color:#EF4444;font-size:2.2rem;font-weight:800;">❌ DITOLAK</div>
  <div style="font-size:0.95rem;margin-top:0.4rem;">Pemohon dinyatakan <b>TIDAK LAYAK</b> saat ini</div>
  <div style="margin-top:0.6rem;font-size:0.82rem;color:#fca5a5;">Model: {clf_choice}</div>
</div>""", unsafe_allow_html=True)

        with metric_col:
            st.metric("Keputusan", clf_label)
            st.metric("Pengajuan Pinjaman", f"${loan_amount:,}K")
            st.metric("Pendapatan Total", f"${applicant_income + coapplicant_income:,}/bln")
            dti = (loan_amount * 1000 / loan_term) / max(applicant_income + coapplicant_income, 1) * 100
            st.metric("Debt-to-Income Est.", f"{dti:.1f}%")

        st.info(f"""
**📌 Interpretasi Hasil:**
- Model **{clf_choice}** memprediksi pemohon **{"LAYAK (Disetujui)" if approved else "TIDAK LAYAK (Ditolak)"}**.
- Hasil ini bersifat prediktif berbasis data historis dan tidak menggantikan keputusan kredit formal perbankan.
""")

        # Insight & Rekomendasi
        st.markdown('<div class="section-header">📊 Insight Profil Pemohon</div>', unsafe_allow_html=True)
        st.markdown(build_applicant_narrative(input_dict, clf_label, float(loan_amount)))

        st.markdown('<div class="section-header">🎯 Rekomendasi</div>', unsafe_allow_html=True)
        for rec in build_recommendations(input_dict, clf_label):
            st.markdown(f'<div class="rec-item">{rec}</div>', unsafe_allow_html=True)

        # Radar chart
        st.markdown('<div class="section-header">🕸️ Profil Finansial Pemohon (Radar Chart)</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            fig_radar = plot_radar_chart(
                applicant_income=float(applicant_income),
                coapplicant_income=float(coapplicant_income),
                loan_amount=float(loan_amount),
                loan_term=float(loan_term),
                credit_history=float(credit_val),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        with rc2:
            explain_card("🕸️ Radar Chart",
                "Merangkum 5 dimensi profil finansial pemohon (dinormalisasi 0–100). "
                "Semakin luas polygon, semakin kuat profil keuangan. "
                "<b>Credit_History</b> yang tinggi umumnya menjadi penentu utama persetujuan.")

# ══════════════════════════════════════════════════════════════════════════════
# MODE: PREDIKSI CSV/EXCEL
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">📤 Prediksi Batch: CSV / Excel</div>', unsafe_allow_html=True)
    st.info("""
**Format kolom yang dibutuhkan:**
`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area`

Nilai numerik sudah dalam bentuk aslinya (belum di-scale). Pipeline akan melakukan encoding & scaling otomatis.
""")

    uploaded = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}"); st.stop()

        if df_in.empty:
            st.warning("File kosong."); st.stop()

        clf_model = get_clf_model(clf_choice)
        if clf_model is None:
            st.error("Model tidak ditemukan di folder model."); st.stop()

        with st.spinner("Memprediksi..."):
            clf_feats = prepare_batch_features(df_in, clf_model, CLF_FEATURE_COLS)
            preds = clf_model.predict(clf_feats)
            df_out = df_in.copy()
            df_out["Predicted_Loan_Status"] = ["Disetujui" if int(p) == 0 else "Ditolak" for p in preds]

        st.dataframe(df_out.head(10), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Hasil (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            "prediksi_klasifikasi.csv", "text/csv")

        st.markdown('<div class="section-header">📊 Approval Rate</div>', unsafe_allow_html=True)
        vc = df_out["Predicted_Loan_Status"].value_counts()
        pc1, pc2 = st.columns([2, 1])
        with pc1:
            st.plotly_chart(plot_pie(
                labels=vc.index.tolist(), values=vc.values.tolist(),
                title="Approval vs Rejection Rate"
            ), use_container_width=True)
        with pc2:
            explain_card("📊 Pie Chart — Approval Rate",
                f"Proporsi hasil prediksi batch.<br>"
                f"<b>Disetujui:</b> {vc.get('Disetujui',0)} pemohon<br>"
                f"<b>Ditolak:</b> {vc.get('Ditolak',0)} pemohon")

        if "Credit_History" in df_out.columns:
            st.markdown('<div class="section-header">🧩 Predicted Status vs Credit History</div>', unsafe_allow_html=True)
            df_tmp = df_out.copy()
            df_tmp["Credit_History_Bucket"] = df_tmp["Credit_History"].apply(
                lambda x: "Good (1)" if float(x) == 1.0 else "Bad (0)")
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                st.plotly_chart(plot_stacked_bar(
                    df_tmp, x_col="Credit_History_Bucket",
                    hue_col="Predicted_Loan_Status",
                    title="Predicted Status vs Credit History"
                ), use_container_width=True)
            with sc2:
                explain_card("🧩 Stacked Bar",
                    "Hubungan antara riwayat kredit (Good/Bad) dengan hasil prediksi. "
                    "Pemohon dengan Credit_History=1 umumnya memiliki rasio persetujuan jauh lebih tinggi.")
