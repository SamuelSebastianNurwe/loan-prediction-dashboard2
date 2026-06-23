"""
utils/sidebar.py — Sidebar info per halaman.
Metrik dibaca langsung dari CSV aktual via helper.
"""

import streamlit as st


def render_sidebar(active_page: str):
    """Tampilkan ringkasan halaman aktif di sidebar."""
    with st.sidebar:
        st.markdown("---")

        if active_page == "Executive Overview":
            st.markdown("### 📊 Executive Overview")
            st.markdown("Ringkasan metrik utama, distribusi dataset, dan perbandingan model.")
            try:
                from utils.helper import get_clf_metrics, get_reg_metrics
                clf = get_clf_metrics()
                reg = get_reg_metrics()
                # Cari best classifier (F1 tertinggi)
                best_clf = max(clf, key=lambda a: clf[a]["hpo"].get("f1", 0))
                best_reg = max(reg, key=lambda a: reg[a]["hpo"].get("r2", 0))
                st.markdown(f"**Best Classifier:** {best_clf}")
                st.markdown(f"**Best Regressor:** {best_reg}")
            except Exception:
                st.markdown("**Best Classifier:** SVM")
                st.markdown("**Best Regressor:** DT")

        elif active_page == "Classification Lab":
            st.markdown("### 🔬 Classification Lab")
            try:
                from utils.helper import get_clf_metrics
                clf = get_clf_metrics()
                best = max(clf, key=lambda a: clf[a]["hpo"].get("f1", 0))
                m = clf[best]["hpo"]
                st.markdown(f"**Best Model:** {best}")
                st.markdown(f"**Accuracy:** {m.get('accuracy', 0):.4f}")
                st.markdown(f"**Precision:** {m.get('precision', 0):.4f}")
                st.markdown(f"**Recall:** {m.get('recall', 0):.4f}")
                st.markdown(f"**F1-Score:** {m.get('f1', 0):.4f}")
            except Exception:
                st.markdown("Memuat metrik...")

        elif active_page == "Regression Lab":
            st.markdown("### 📈 Regression Lab")
            try:
                from utils.helper import get_reg_metrics
                reg = get_reg_metrics()
                best = max(reg, key=lambda a: reg[a]["hpo"].get("r2", 0))
                m = reg[best]["hpo"]
                st.markdown(f"**Best Model:** {best}")
                st.markdown(f"**R² Score:** {m.get('r2', 0):.4f}")
                st.markdown(f"**MAE:** {m.get('mae', 0):.2f} K")
                st.markdown(f"**RMSE:** {m.get('rmse', 0):.2f} K")
            except Exception:
                st.markdown("Memuat metrik...")

        elif active_page == "Feature Analytics":
            st.markdown("### 🔍 Feature Analytics")
            st.markdown("Analisis distribusi, korelasi, dan hubungan antar fitur dataset pinjaman.")
            st.markdown("Fokus: ApplicantIncome, CoapplicantIncome, LoanAmount, Credit_History.")

        elif active_page == "Explainability":
            st.markdown("### 💡 Explainability")
            st.markdown("Feature importance DT, analisis SVM, dan panduan interpretasi model.")

        elif active_page == "Classification Simulator":
            st.markdown("### 🔬 Classification Simulator")
            st.markdown("Simulasi prediksi kelayakan pinjaman (Approved/Rejected) secara real-time.")
            try:
                from utils.helper import get_clf_metrics
                clf = get_clf_metrics()
                best = max(clf, key=lambda a: clf[a]["hpo"].get("accuracy", 0))
                m = clf[best]["hpo"]
                st.markdown(f"**Best Classifier:** {best}")
                st.markdown(f"**Accuracy:** {m.get('accuracy', 0):.4f}")
            except Exception:
                pass

        elif active_page == "Regression Simulator":
            st.markdown("### 📈 Regression Simulator")
            st.markdown("Simulasi estimasi besaran pinjaman (LoanAmount) secara real-time.")
            try:
                from utils.helper import get_reg_metrics
                reg = get_reg_metrics()
                best = max(reg, key=lambda a: reg[a]["hpo"].get("r2", 0))
                m = reg[best]["hpo"]
                st.markdown(f"**Best Regressor:** {best}")
                st.markdown(f"**R² Score:** {m.get('r2', 0):.4f}")
            except Exception:
                pass
