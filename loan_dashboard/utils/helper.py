"""
utils/helper.py — Loan Prediction Intelligence Dashboard
Semua metrik dihitung langsung dari CSV hasil prediksi aktual.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

# ─── Path constants ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR  = os.path.join(BASE_DIR, "data")


def _read_csv_flexible(filename: str) -> pd.DataFrame:
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(BASE_DIR, "Data", filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    return pd.DataFrame()


# ─── Model filenames ────────────────────────────────────────────────────────────
MODEL_FILES = {
    "clf_dt_hpo"    : "clf_dt_hpo.joblib",
    "clf_nn_hpo"    : "clf_nn_nohpo.joblib",
    "clf_svm_hpo"   : "clf_svm_nohpo.joblib",
    "reg_dt_nohpo"  : "reg_dt_nohpo.joblib",
    "reg_svm_nohpo" : "reg_svm_nohpo.joblib",
    "reg_nn_nohpo"  : "reg_nn_nohpo.joblib",
}

# ─── CSV result filenames ───────────────────────────────────────────────────────
CSV_FILES = {
    "clf_dt_hpo"    : "Klasifikasi_DT-HPO.csv",
    "clf_nn_hpo"    : "Klasifikasi_NN_NoHPO.csv",
    "clf_svm_hpo"   : "Klasifikasi_SVM-NoHPO.csv",
    "reg_dt_nohpo"  : "hasilRegresi_DT_NoHPO.csv",
    "reg_svm_nohpo" : "hasilRegresi_SVM_NoHpo.csv",
    "reg_nn_nohpo"  : "hasilRegresi_NN_NoHPO.csv",
}

# ─── Feature columns ────────────────────────────────────────────────────────────
CLF_FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
    "PAR_Rural", "PAR_Semiurban", "PAR_Urban",
]

REG_FEATURE_COLS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term",
    "Credit_History", "Loan_Status", "PAR_Rural", "PAR_Semiurban",
]

# ─── Scaling statistics ─────────────────────────────────────────────────────────
SCALE_STATS = {
    "Unnamed: 0"        : (300.0,   180.0),
    "Gender"            : (0.8054,  0.3963),
    "Married"           : (0.6702,  0.4708),
    "Dependents"        : (0.8453,  1.0027),
    "Education"         : (0.7838,  0.4119),
    "Self_Employed"     : (0.1351,  0.3424),
    "ApplicantIncome"   : (5403.46, 6109.04),
    "CoapplicantIncome" : (1621.25, 2926.25),
    "LoanAmount"        : (146.41,  85.59),
    "Loan_Amount_Term"  : (342.0,   65.12),
    "Credit_History"    : (0.8423,  0.3647),
    "PAR_Rural"         : (0.3189,  0.4665),
    "PAR_Semiurban"     : (0.3811,  0.4860),
    "PAR_Urban"         : (0.2973,  0.4574),
    # Loan_Status dipakai sebagai fitur pada model regresi. Di dataset training:
    # Y/Approved = 0 dan N/Rejected = 1, lalu distandardisasi.
    "Loan_Status"       : (0.3127,  0.4639),
}


def explain_chart(title: str, text_md: str):
    """Tampilkan kartu penjelasan singkat untuk diagram/chart di setiap page."""
    st.info(f"**{title}**\n\n{text_md}")


def get_model_feature_columns(model, fallback_cols: list) -> list:
    """Ambil nama fitur langsung dari model jika tersedia, agar DT/NN/SVM tetap sesuai."""
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        return [str(c) for c in list(cols)]
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        return list(fallback_cols)[: int(n_features)]
    return list(fallback_cols)


# ─── Compute metrics from CSV (cached) ─────────────────────────────────────────
def _compute_clf_metrics_from_csv(csv_key: str) -> dict:
    """Hitung metrik klasifikasi dari CSV aktual."""
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix
        )
        df = _read_csv_flexible(CSV_FILES[csv_key])
        if df.empty or "prediksi" not in df.columns or "label_asli" not in df.columns:
            return {}
        y_true = df["label_asli"].astype(int)
        y_pred = df["prediksi"].astype(int)
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy" : float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall"   : float(recall_score(y_true, y_pred, zero_division=0)),
            "f1"       : float(f1_score(y_true, y_pred, zero_division=0)),
            "cm"       : cm,
        }
    except Exception:
        return {}


def _compute_reg_metrics_from_csv(csv_key: str) -> dict:
    """Hitung metrik regresi dari CSV aktual."""
    try:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        df = _read_csv_flexible(CSV_FILES[csv_key])
        if df.empty or "prediksi" not in df.columns or "label_asli" not in df.columns:
            return {}
        y_true = df["label_asli"].values.astype(float)
        y_pred = df["prediksi"].values.astype(float)
        r2   = float(r2_score(y_true, y_pred))
        mae  = float(mean_absolute_error(y_true, y_pred))
        mse  = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _build_clf_metrics() -> dict:
    """Build CLF_METRICS dict dari CSV aktual."""
    return {
        "DT": {
            "hpo"   : _compute_clf_metrics_from_csv("clf_dt_hpo"),
            "nohpo" : _compute_clf_metrics_from_csv("clf_dt_hpo"),   # DT hanya ada HPO
        },
        "NN": {
            "hpo"   : _compute_clf_metrics_from_csv("clf_nn_hpo"),
            "nohpo" : _compute_clf_metrics_from_csv("clf_nn_hpo"),
        },
        "SVM": {
            "hpo"   : _compute_clf_metrics_from_csv("clf_svm_hpo"),
            "nohpo" : _compute_clf_metrics_from_csv("clf_svm_hpo"),
        },
    }


@st.cache_data(show_spinner=False)
def _build_reg_metrics() -> dict:
    """Build REG_METRICS dict dari CSV aktual."""
    return {
        "DT": {
            "hpo"   : _compute_reg_metrics_from_csv("reg_dt_nohpo"),
            "nohpo" : _compute_reg_metrics_from_csv("reg_dt_nohpo"),
        },
        "SVM": {
            "hpo"   : _compute_reg_metrics_from_csv("reg_svm_nohpo"),
            "nohpo" : _compute_reg_metrics_from_csv("reg_svm_nohpo"),
        },
        "NN": {
            "hpo"   : _compute_reg_metrics_from_csv("reg_nn_nohpo"),
            "nohpo" : _compute_reg_metrics_from_csv("reg_nn_nohpo"),
        },
    }


def get_clf_metrics() -> dict:
    """Return CLF_METRICS dihitung dari CSV (cached)."""
    m = _build_clf_metrics()
    # Alias non_hpo → nohpo
    for algo in m:
        m[algo]["non_hpo"] = m[algo]["nohpo"]
    return m


def get_reg_metrics() -> dict:
    """Return REG_METRICS dihitung dari CSV (cached)."""
    m = _build_reg_metrics()
    for algo in m:
        m[algo]["non_hpo"] = m[algo]["nohpo"]
    return m


# Lazy-loaded globals — diisi saat pertama kali diakses di dalam Streamlit context
# Halaman yang masih import CLF_METRICS / REG_METRICS langsung tetap bekerja
# karena kita override di sini dengan property-like approach.
# Namun karena Python module-level, kita gunakan fungsi getter saja.
# Untuk backward-compat dengan halaman lama yg masih "from utils.helper import CLF_METRICS":
# kita definisikan sebagai fungsi yang akan di-monkey-patch saat pertama kali diimport dalam streamlit.

# ─── Lazy fallback: jika diimport di luar Streamlit context (misal saat testing) ───
try:
    CLF_METRICS = _build_clf_metrics()
    for _algo in CLF_METRICS:
        CLF_METRICS[_algo]["non_hpo"] = CLF_METRICS[_algo]["nohpo"]
except Exception:
    CLF_METRICS = {
        "DT":  {"hpo": {}, "nohpo": {}, "non_hpo": {}},
        "NN":  {"hpo": {}, "nohpo": {}, "non_hpo": {}},
        "SVM": {"hpo": {}, "nohpo": {}, "non_hpo": {}},
    }

try:
    REG_METRICS = _build_reg_metrics()
    for _algo in REG_METRICS:
        REG_METRICS[_algo]["non_hpo"] = REG_METRICS[_algo]["nohpo"]
except Exception:
    REG_METRICS = {
        "DT":  {"hpo": {}, "nohpo": {}, "non_hpo": {}},
        "SVM": {"hpo": {}, "nohpo": {}, "non_hpo": {}},
        "NN":  {"hpo": {}, "nohpo": {}, "non_hpo": {}},
    }


# ─── Best HPO parameters ───────────────────────────────────────────────────────
BEST_PARAMS = {
    "DT Klasifikasi" : {"max_depth": 5, "min_samples_split": 3, "criterion": "gini"},
    "NN Klasifikasi" : {"hidden_layer_sizes": "(100,)", "activation": "relu", "solver": "adam"},
    "SVM Klasifikasi": {"C": 1.5, "kernel": "linear", "gamma": "scale"},
    "DT Regresi"     : {"max_depth": 5, "min_samples_split": 3, "criterion": "squared_error"},
    "SVM Regresi"    : {"C": 1.5, "kernel": "linear", "epsilon": 0.1},
    "NN Regresi"     : {"hidden_layer_sizes": "(100,)", "activation": "relu", "solver": "adam"},
}

# ─── Feature descriptions ──────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "Gender"           : "Jenis kelamin pemohon (Male=1 / Female=0)",
    "Married"          : "Status pernikahan (Married=1 / Single=0)",
    "Dependents"       : "Jumlah tanggungan (0, 1, 2, 3+)",
    "Education"        : "Tingkat pendidikan (Graduate=0 / Not Graduate=1)",
    "Self_Employed"    : "Status wirausaha (Yes=1 / No=0)",
    "ApplicantIncome"  : "Pendapatan bulanan pemohon utama (USD)",
    "CoapplicantIncome": "Pendapatan bulanan co-applicant (USD)",
    "LoanAmount"       : "Jumlah pinjaman yang diajukan (ribuan USD) — Target Regresi",
    "Loan_Amount_Term" : "Tenor pinjaman dalam bulan",
    "Credit_History"   : "Riwayat kredit (Baik=1 / Buruk atau Tidak Ada=0)",
    "Property_Area"    : "Lokasi properti (Urban / Semiurban / Rural)",
}

# ─── Loaders ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_key: str):
    """Load model joblib. Return None jika tidak ditemukan (TANPA st.error)."""
    filename = MODEL_FILES.get(model_key)
    if filename is None:
        return None
    candidates = [
        os.path.join(MODEL_DIR, filename),
        os.path.join(BASE_DIR, "Model", filename),
        os.path.join(BASE_DIR, filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return joblib.load(path)
            except Exception:
                pass
    return None  # Silent — halaman menangani sendiri jika None


@st.cache_data(show_spinner=False)
def load_clf_data(model_key: str) -> pd.DataFrame:
    fname = CSV_FILES.get(model_key, "Klasifikasi_DT-HPO.csv")
    return _read_csv_flexible(fname)


@st.cache_data(show_spinner=False)
def load_reg_data(model_key: str) -> pd.DataFrame:
    fname = CSV_FILES.get(model_key, "hasilRegresi_SVM_NoHpo.csv")
    return _read_csv_flexible(fname)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = _read_csv_flexible("train_u6lujuX_CVtuZ9i.csv")
    if df is not None and not df.empty:
        df_processed = df.copy()
        if "Loan_Status" in df_processed.columns:
            df_processed["label_asli"] = df_processed["Loan_Status"].map({"Y": 1, "N": 0})
        if "Gender" in df_processed.columns:
            df_processed["Gender"] = df_processed["Gender"].map({"Male": 1, "Female": 0})
        if "Married" in df_processed.columns:
            df_processed["Married"] = df_processed["Married"].map({"Yes": 1, "No": 0})
        if "Education" in df_processed.columns:
            df_processed["Education"] = df_processed["Education"].map({"Graduate": 0, "Not Graduate": 1})
        if "Self_Employed" in df_processed.columns:
            df_processed["Self_Employed"] = df_processed["Self_Employed"].map({"Yes": 1, "No": 0})
        if "Property_Area" in df_processed.columns:
            df_processed["PAR_Rural"]     = (df_processed["Property_Area"] == "Rural").astype(float)
            df_processed["PAR_Semiurban"] = (df_processed["Property_Area"] == "Semiurban").astype(float)
            df_processed["PAR_Urban"]     = (df_processed["Property_Area"] == "Urban").astype(float)
        return df_processed
    return pd.DataFrame()


# ─── Preprocessing ─────────────────────────────────────────────────────────────
def encode_input(input_dict: dict) -> dict:
    encoded = {}
    encoded["Unnamed: 0"]        = float(input_dict.get("Unnamed: 0", 300.0))
    encoded["Gender"]            = 1 if input_dict.get("Gender") == "Male" else 0
    encoded["Married"]           = 1 if input_dict.get("Married") == "Yes" else 0
    dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    encoded["Dependents"]        = dep_map.get(str(input_dict.get("Dependents", "0")), 0)
    encoded["Education"]         = 0 if input_dict.get("Education") == "Graduate" else 1
    encoded["Self_Employed"]     = 1 if input_dict.get("Self_Employed") == "Yes" else 0
    encoded["ApplicantIncome"]   = float(input_dict.get("ApplicantIncome", 0))
    encoded["CoapplicantIncome"] = float(input_dict.get("CoapplicantIncome", 0))
    encoded["LoanAmount"]        = float(input_dict.get("LoanAmount", 0))
    encoded["Loan_Amount_Term"]  = float(input_dict.get("Loan_Amount_Term", 360))
    encoded["Credit_History"]    = float(input_dict.get("Credit_History", 1))
    status = input_dict.get("Loan_Status", input_dict.get("Predicted_Loan_Status", "Y"))
    if isinstance(status, str):
        status_clean = status.strip().lower()
        encoded["Loan_Status"] = 1 if status_clean in ["n", "no", "rejected", "ditolak", "1"] else 0
    else:
        encoded["Loan_Status"] = int(float(status))

    area = input_dict.get("Property_Area", None)
    if area is not None:
        area = str(area).strip().lower()
        encoded["PAR_Rural"] = 1 if area == "rural" else 0
        encoded["PAR_Semiurban"] = 1 if area == "semiurban" else 0
        encoded["PAR_Urban"] = 1 if area == "urban" else 0
    else:
        encoded["PAR_Rural"] = float(input_dict.get("PAR_Rural", 0))
        encoded["PAR_Semiurban"] = float(input_dict.get("PAR_Semiurban", 0))
        encoded["PAR_Urban"] = float(input_dict.get("PAR_Urban", 0))
    return encoded


def scale_features(encoded: dict, cols: list) -> np.ndarray:
    row = []
    for col in cols:
        val  = encoded.get(col, 0)
        mean, std = SCALE_STATS.get(col, (0.0, 1.0))
        row.append((val - mean) / std if std != 0 else 0.0)
    return np.array(row).reshape(1, -1)


def preprocess_for_clf(input_dict: dict) -> pd.DataFrame:
    encoded = encode_input(input_dict)
    scaled  = scale_features(encoded, CLF_FEATURE_COLS)
    return pd.DataFrame(scaled, columns=CLF_FEATURE_COLS)


def preprocess_for_reg(input_dict: dict) -> pd.DataFrame:
    encoded = encode_input(input_dict)
    scaled = scale_features(encoded, REG_FEATURE_COLS)
    return pd.DataFrame(scaled, columns=REG_FEATURE_COLS)


def preprocess_for_model(input_dict: dict, model, fallback_cols: list) -> pd.DataFrame:
    """Preprocessing manual yang mengikuti fitur asli model (.joblib)."""
    cols = get_model_feature_columns(model, fallback_cols)
    encoded = encode_input(input_dict)
    scaled = scale_features(encoded, cols)
    return pd.DataFrame(scaled, columns=cols)


def prepare_batch_features(df_in: pd.DataFrame, model, fallback_cols: list) -> pd.DataFrame:
    """
    Siapkan fitur batch.
    - Jika file upload sudah berisi kolom numerik/scaled sesuai model, pakai langsung tanpa encode ulang.
    - Jika file upload masih mentah seperti data asli, encode + scaling sesuai statistik training.
    """
    cols = get_model_feature_columns(model, fallback_cols)
    df = df_in.copy()

    if all(c in df.columns for c in cols):
        numeric = df[cols].apply(pd.to_numeric, errors="coerce")
        if not numeric.isna().all(axis=None):
            return numeric.fillna(0.0)

    rows = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        encoded = encode_input(row_dict)
        scaled = scale_features(encoded, cols).reshape(-1)
        rows.append(scaled)
    return pd.DataFrame(rows, columns=cols)


# ─── Prediction helpers ────────────────────────────────────────────────────────
def normalize_prediction_status(pred: int) -> str:
    return "Disetujui" if int(pred) == 0 else "Ditolak"


def predict_loan_status(features, model) -> tuple:
    if model is None:
        return 0, "Model tidak tersedia"
    try:
        pred = int(model.predict(features)[0])
        label = normalize_prediction_status(pred)
        return pred, label
    except Exception as e:
        return 0, f"Error: {e}"


def predict_loan_status_svm_threshold(features, svm_model, svm_threshold: float = 0.0) -> tuple:
    if svm_model is None:
        return 0, "Model tidak tersedia"
    try:
        score = float(svm_model.decision_function(features)[0])
        pred  = 1 if score >= svm_threshold else 0
        label = normalize_prediction_status(pred)
        return pred, label
    except Exception:
        return predict_loan_status(features, svm_model)


def predict_loan_amount(features, model) -> float:
    if model is None:
        return 0.0
    try:
        raw = model.predict(features)[0]
        mean, std = SCALE_STATS["LoanAmount"]
        if abs(raw) < 20:
            return float(raw * std + mean)
        return float(raw)
    except Exception:
        return 0.0


# ─── Narrative builder ─────────────────────────────────────────────────────────
def build_applicant_narrative(input_dict: dict, clf_result: str, loan_amount: float) -> str:
    income   = float(input_dict.get("ApplicantIncome", 0))
    co_inc   = float(input_dict.get("CoapplicantIncome", 0))
    credit   = float(input_dict.get("Credit_History", 1))
    area     = input_dict.get("Property_Area", "Urban")
    edu      = input_dict.get("Education", "Graduate")
    employed = input_dict.get("Self_Employed", "No")

    parts = []
    if income > 8000:
        parts.append("Pemohon memiliki **pendapatan tinggi** yang mendukung kemampuan cicilan.")
    elif income > 4000:
        parts.append("Pendapatan pemohon **cukup memadai** untuk cicilan pinjaman standar.")
    else:
        parts.append("Pendapatan pemohon **tergolong rendah**, perlu evaluasi rasio cicilan lebih ketat.")

    if co_inc > 2000:
        parts.append(f"Co-applicant berkontribusi **${co_inc:,.0f}/bulan**, memperkuat profil kredit.")
    elif co_inc > 0:
        parts.append("Co-applicant memberikan kontribusi penghasilan tambahan.")

    if credit == 1.0:
        parts.append("**Riwayat kredit bersih** — faktor positif utama dalam keputusan persetujuan.")
    else:
        parts.append("⚠️ **Tidak ada riwayat kredit** yang terdeteksi — risiko kredit meningkat.")

    area_desc = {
        "Urban"    : "Properti di area **Urban** memiliki nilai jaminan yang tinggi.",
        "Semiurban": "Properti di area **Semiurban** relatif stabil secara nilai.",
        "Rural"    : "Properti di area **Rural** — perlu penilaian agunan lebih menyeluruh.",
    }
    parts.append(area_desc.get(area, ""))

    if edu == "Graduate":
        parts.append("Status **Sarjana** meningkatkan profil profesional pemohon.")
    if employed == "Yes":
        parts.append("Sebagai **wirausaha**, pendapatan bersifat variabel; verifikasi laporan keuangan disarankan.")

    return "\n\n".join(p for p in parts if p)


def build_recommendations(input_dict: dict, clf_result: str) -> list:
    recs     = []
    credit   = float(input_dict.get("Credit_History", 1))
    income   = float(input_dict.get("ApplicantIncome", 0))
    co_inc   = float(input_dict.get("CoapplicantIncome", 0))
    term     = float(input_dict.get("Loan_Amount_Term", 360))
    employed = input_dict.get("Self_Employed", "No")

    if clf_result == "Disetujui":
        recs.append("✅ **Persiapkan Dokumen Kelengkapan**: KTP, slip gaji 3 bulan terakhir, NPWP, & rekening koran.")
        if credit == 1.0:
            recs.append("✅ **Pertahankan Riwayat Kredit Baik**: Bayar tagihan tepat waktu untuk menjaga skor kredit.")
        recs.append("✅ **Asuransi Kredit**: Pertimbangkan perlindungan jiwa & PHK agar cicilan tetap aman.")
        if term > 300:
            recs.append("💡 **Penyesuaian Tenor**: Pertimbangkan tenor lebih pendek untuk mengurangi total bunga.")
        if co_inc == 0:
            recs.append("💡 **Tambahkan Co-Applicant**: Co-applicant berpenghasilan akan meningkatkan plafon pinjaman.")
    else:
        recs.append("⚠️ **Perbaiki Riwayat Kredit**: Lunasi tunggakan dan bangun catatan kredit positif minimal 12 bulan.")
        if income < 4000:
            recs.append("⚠️ **Tingkatkan Pendapatan / Tambah Co-Applicant**: Pendapatan gabungan memperkuat kelayakan.")
        recs.append("📋 **Ajukan Kembali dalam 6-12 Bulan**: Setelah memperbaiki profil keuangan, ajukan ulang permohonan.")
        recs.append("💼 **Konsultasi dengan Petugas Kredit**: Dapatkan saran spesifik untuk situasi finansial Anda.")
        if employed == "Yes":
            recs.append("📊 **Siapkan Laporan Keuangan Usaha**: Bukti omzet & laba rugi minimal 2 tahun terakhir.")

    return recs
