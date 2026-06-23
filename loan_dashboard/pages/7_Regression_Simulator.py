"""
pages/7_Regression_Simulator.py
Simulasi prediksi estimasi besaran pinjaman (LoanAmount).
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
    REG_FEATURE_COLS,
    predict_loan_amount,
    build_applicant_narrative,
    FEATURE_DESCRIPTIONS,
    REG_METRICS,
    SCALE_STATS,
)
from utils.visualization import COLORS, plot_radar_chart, plot_histogram

st.set_page_config(
    page_title="Regression Simulator | Loan Dashboard",
    page_icon="📈",
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
.input-group-title{font-size:1rem;font-weight:700;color:#F97316;margin-bottom:1.1rem;letter-spacing:0.02em}
.stButton>button{background:linear-gradient(135deg,#1e3a6b,#4E9AF1)!important;color:white!important;border:none!important;border-radius:10px!important;font-weight:700!important;padding:0.7rem 2rem!important;font-size:1rem!important;transition:all 0.25s ease!important;width:100%!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(78,154,241,0.4)!important}
.result-box{background:linear-gradient(135deg,#1e3a6b,#0f2952);border:2px solid #4E9AF1;border-radius:16px;padding:1.8rem 2rem;text-align:center;margin:1rem 0}
.explain-card{background:#1A1F2E;border-radius:14px;padding:1.3rem;border:1px solid #2D3748;margin-top:0.5rem}
.explain-title{font-size:1rem;font-weight:700;color:#F97316;margin-bottom:0.7rem}
.explain-text{color:#94A3B8;font-size:0.87rem;line-height:1.7}
div[data-testid="stNumberInput"] input{background:#0D1526!important;border:1px solid #2D3748!important;border-radius:8px!important;color:#E2E8F0!important;font-size:0.95rem!important}
div[data-testid="stNumberInput"] button{background:#1e3a6b!important;border:1px solid #2D3748!important;color:#E2E8F0!important}
div[data-testid="stRadio"] label{color:#E2E8F0!important;font-size:0.9rem!important}
</style>
""", unsafe_allow_html=True)

render_sidebar("Regression Simulator")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0D1526,#1e3a6b,#0f2952);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.8rem;border:1px solid rgba(249,115,22,0.25);">
  <h1 style="font-size:1.9rem;font-weight:800;color:#fff;margin:0 0 0.3rem;">📈 Regression Simulator</h1>
  <p style="color:#94A3B8;margin:0;font-size:0.95rem;">Simulasi prediksi estimasi plafon pinjaman (LoanAmount) — masukkan nilai numerik secara langsung</p>
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
    reg_choice = st.selectbox("📈 Model Regresi", ["SVM", "NN (MLP)", "DT"], key="sim_reg_model")
    st.markdown("---")
    st.markdown("### 🏆 Performa Model")
    algo_key = "SVM" if reg_choice == "SVM" else ("NN" if reg_choice == "NN (MLP)" else "DT")
    m = REG_METRICS[algo_key]["hpo"]
    st.metric("R² Score", f"{m.get('r2',0):.4f}")
    st.metric("MAE",      f"{m.get('mae',0):.2f} K")
    st.metric("RMSE",     f"{m.get('rmse',0):.2f} K")
    st.markdown("---")

def get_reg_model(choice):
    return load_model({"SVM":"reg_svm_nohpo","NN (MLP)":"reg_nn_nohpo","DT":"reg_dt_nohpo"}[choice])

# ══════════════════════════════════════════════════════════════════════════════
# MODE: PREDIKSI MANUAL
# ══════════════════════════════════════════════════════════════════════════════
if pred_mode == "Prediksi Manual":

    st.markdown('<div class="section-header">📝 Input Profil Pemohon</div>', unsafe_allow_html=True)
    st.caption("Masukkan nilai numerik untuk setiap fitur. Model regresi akan memprediksi besaran LoanAmount yang sesuai.")

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
        loan_term = st.number_input(
            "Loan_Amount_Term (bulan)",
            min_value=12, max_value=480, value=360, step=12,
            help="Tenor/durasi pinjaman dalam bulan"
        )

    n4, _ = st.columns([1, 2])
    with n4:
        dependents = st.number_input(
            "Dependents (jumlah tanggungan)",
            min_value=0, max_value=10, value=0, step=1,
            help="Jumlah anggota keluarga yang ditanggung"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Blok 2: Status Pinjaman (digunakan sebagai fitur oleh model regresi) ──
    st.markdown('<div class="input-group-title">📋 Status Kelayakan Kredit (Hasil Klasifikasi)</div>', unsafe_allow_html=True)
    st.caption("Model regresi menggunakan Loan_Status sebagai salah satu fitur prediktor.")

    loan_status_opt = st.radio(
        "Loan_Status (status kelayakan kredit pemohon)",
        ["1 — Approved (Disetujui)", "0 — Rejected (Ditolak)"],
        index=0, key="reg_loan_status", horizontal=True
    )
    loan_status_val = 1.0 if loan_status_opt.startswith("1") else 0.0

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Blok 3: Fitur Kategorikal ─────────────────────────────────────────────
    st.markdown('<div class="input-group-title">🗂️ Fitur Kategorikal</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**👤 Gender**")
        gender = st.radio("Gender", ["Male", "Female"], index=0, key="reg_gender", label_visibility="collapsed")

        st.markdown("**💍 Married**")
        married = st.radio("Married", ["Yes", "No"], index=0, key="reg_married", label_visibility="collapsed")

    with r2:
        st.markdown("**🎓 Education**")
        education = st.radio("Education", ["Graduate", "Not Graduate"], index=0, key="reg_edu", label_visibility="collapsed")

        st.markdown("**💼 Self Employed**")
        self_emp = st.radio("Self Employed", ["No", "Yes"], index=0, key="reg_selfemp", label_visibility="collapsed")

    with r3:
        st.markdown("**💳 Credit History**")
        credit_opt = st.radio("Credit History", ["1 — Good Credit", "0 — No/Bad Credit"], index=0, key="reg_credit", label_visibility="collapsed")

        st.markdown("**🏘️ Property Area**")
        prop_area = st.radio("Property Area", ["Urban", "Semiurban", "Rural"], index=0, key="reg_area", label_visibility="collapsed")

    credit_val = 1.0 if credit_opt.startswith("1") else 0.0

    # ── Tombol prediksi ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Estimasi Plafon Pinjaman", use_container_width=True, key="reg_predict_btn")

    # ── Preview input ─────────────────────────────────────────────────────────
    with st.expander("📋 Lihat Data Input", expanded=False):
        dep_str = f"{dependents}" if dependents < 3 else "3+"
        preview = {
            "Fitur": ["Gender","Married","Dependents","Education","Self_Employed",
                      "ApplicantIncome","CoapplicantIncome","Loan_Amount_Term",
                      "Credit_History","Loan_Status","Property_Area"],
            "Nilai": [gender, married, dep_str, education, self_emp,
                      f"{applicant_income:,}", f"{coapplicant_income:,}",
                      f"{loan_term} bln",
                      "1 (Baik)" if credit_val==1.0 else "0 (Buruk)",
                      "1 (Approved)" if loan_status_val==1.0 else "0 (Rejected)",
                      prop_area],
        }
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

    # ── Hasil prediksi ────────────────────────────────────────────────────────
    if predict_btn:
        dep_str = f"{dependents}" if dependents < 3 else "3+"
        input_dict = {
            "Gender": gender, "Married": married, "Dependents": dep_str,
            "Education": education, "Self_Employed": self_emp,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "Loan_Amount_Term": loan_term, "Credit_History": credit_val,
            "Loan_Status": loan_status_val, "Property_Area": prop_area,
            "LoanAmount": 150,  # placeholder — bukan dipakai di reg pipeline
        }

        reg_model = get_reg_model(reg_choice)
        with st.spinner("⚙️ Memproses prediksi..."):
            reg_feats = preprocess_for_model(input_dict, reg_model, REG_FEATURE_COLS)
            loan_pred = predict_loan_amount(reg_feats, reg_model)

        st.markdown('<div class="section-header">📊 Hasil Estimasi Plafon Pinjaman</div>', unsafe_allow_html=True)

        res_col, metric_col = st.columns(2)
        with res_col:
            st.markdown(f"""
<div class="result-box">
  <div style="color:#94A3B8;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;">💰 Estimasi Plafon Maksimal</div>
  <div style="color:#4E9AF1;font-size:2.8rem;font-weight:800;">${loan_pred:,.1f}K</div>
  <div style="color:#E2E8F0;font-size:0.9rem;margin-top:0.4rem;">≈ <b>${loan_pred*1000:,.0f}</b> USD</div>
  <div style="margin-top:0.8rem;font-size:0.82rem;color:#94A3B8;">Model: <b>{reg_choice}</b></div>
</div>""", unsafe_allow_html=True)

        with metric_col:
            st.metric("💰 Estimasi LoanAmount", f"${loan_pred:,.1f}K")
            st.metric("📅 Tenor", f"{loan_term} bulan")
            monthly_cicilan = (loan_pred * 1000) / max(loan_term, 1)
            st.metric("📆 Est. Cicilan/Bulan", f"${monthly_cicilan:,.0f}")
            income_total = applicant_income + coapplicant_income
            ratio = (monthly_cicilan / max(income_total, 1)) * 100
            st.metric("📊 Rasio Cicilan/Income", f"{ratio:.1f}%")

        st.info(f"""
**📌 Interpretasi Hasil:**
- Model regresi **{reg_choice}** mengestimasikan plafon pinjaman sebesar **${loan_pred:,.1f}K** (ribuan USD).
- Estimasi cicilan per bulan ≈ **${monthly_cicilan:,.0f}** dengan tenor **{loan_term} bulan**.
- Rasio cicilan terhadap pendapatan ≈ **{ratio:.1f}%** {'✅ (< 30%, sehat)' if ratio < 30 else '⚠️ (> 30%, perlu evaluasi)'}.
- Nilai ini adalah estimasi model dan tidak menggantikan perhitungan kredit resmi bank.
""")

        # Insight
        st.markdown('<div class="section-header">📊 Insight Profil Pemohon</div>', unsafe_allow_html=True)
        st.markdown(build_applicant_narrative(input_dict, "Disetujui", loan_pred))

        # Radar chart
        st.markdown('<div class="section-header">🕸️ Profil Finansial Pemohon (Radar Chart)</div>', unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1])
        with rc1:
            fig_radar = plot_radar_chart(
                applicant_income=float(applicant_income),
                coapplicant_income=float(coapplicant_income),
                loan_amount=float(loan_pred),
                loan_term=float(loan_term),
                credit_history=float(credit_val),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        with rc2:
            explain_card("🕸️ Radar Chart",
                "Merangkum profil finansial dalam 5 dimensi (dinormalisasi 0–100). "
                "Nilai LoanAmount yang ditampilkan adalah <b>hasil prediksi model</b>, bukan input. "
                "Semakin luas area polygon, semakin besar kapasitas pinjaman yang disarankan.")

# ══════════════════════════════════════════════════════════════════════════════
# MODE: PREDIKSI CSV/EXCEL
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-header">📤 Prediksi Batch: CSV / Excel</div>', unsafe_allow_html=True)
    st.info("""
**Format kolom yang dibutuhkan:**
`Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term, Credit_History, Loan_Status, Property_Area`

Nilai numerik dalam bentuk aslinya (belum di-scale). `Loan_Status` = 1 (Approved) / 0 (Rejected).
""")

    uploaded = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}"); st.stop()

        if df_in.empty:
            st.warning("File kosong."); st.stop()

        reg_model = get_reg_model(reg_choice)
        if reg_model is None:
            st.error("Model tidak ditemukan di folder model."); st.stop()

        with st.spinner("Memprediksi..."):
            reg_feats = prepare_batch_features(df_in, reg_model, REG_FEATURE_COLS)
            raw_preds = reg_model.predict(reg_feats)
            pred_list = []
            for raw in raw_preds:
                raw = float(raw)
                m_s, s_s = SCALE_STATS["LoanAmount"]
                pred_list.append(float(raw * s_s + m_s) if abs(raw) < 20 else raw)
            df_out = df_in.copy()
            df_out["Predicted_LoanAmount_K"] = pred_list

        st.dataframe(df_out.head(10), use_container_width=True, hide_index=True)

        col_sum1, col_sum2, col_sum3 = st.columns(3)
        col_sum1.metric("Jumlah Baris", f"{len(df_out):,}")
        col_sum2.metric("Rata-rata Prediksi", f"${pd.Series(pred_list).mean():,.1f}K")
        col_sum3.metric("Maks. Prediksi", f"${pd.Series(pred_list).max():,.1f}K")

        st.download_button("⬇️ Download Hasil (CSV)",
            df_out.to_csv(index=False).encode("utf-8"),
            "prediksi_regresi.csv", "text/csv")

        st.markdown('<div class="section-header">📈 Distribusi Predicted LoanAmount</div>', unsafe_allow_html=True)
        hc1, hc2 = st.columns([2, 1])
        with hc1:
            st.plotly_chart(plot_histogram(
                df_out["Predicted_LoanAmount_K"],
                title="Distribusi Predicted LoanAmount (K)",
                xlab="LoanAmount prediksi (ribuan USD)",
                color=COLORS["primary"]
            ), use_container_width=True)
        with hc2:
            explain_card("📈 Histogram Distribusi",
                "Menampilkan sebaran nilai prediksi LoanAmount dari seluruh baris file. "
                "Konsentrasi di suatu rentang menunjukkan plafon yang paling sering "
                "diestimasikan model untuk profil pemohon dalam dataset ini.")
