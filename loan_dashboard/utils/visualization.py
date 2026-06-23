"""
utils/visualization.py
Fungsi visualisasi reusable untuk seluruh halaman dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix


# ─── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary"   : "#4E9AF1",
    "secondary" : "#7C5CBF",
    "accent"    : "#F97316",
    "success"   : "#22C55E",
    "danger"    : "#EF4444",
    "warning"   : "#F59E0B",
    "bg"        : "#0E1117",
    "card"      : "#1A1F2E",
    "border"    : "#2D3748",
    "text"      : "#E2E8F0",
    "muted"     : "#94A3B8",
}

ALGO_COLORS = {
    "SVM": "#7C5CBF",
    "DT" : "#F97316",
    "NN" : "#22C55E",
}


# ─── KPI Card ─────────────────────────────────────────────────────────────────
def render_kpi_card(label: str, value: str, delta: str = "", icon: str = "📊", color: str = "#4E9AF1"):
    """Render a styled KPI metric card using st.markdown."""
    delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 4px solid {color};">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ─── Correlation Heatmap ───────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, cols: list, title: str = "Korelasi Fitur") -> go.Figure:
    corr = df[cols].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
        colorbar=dict(title="r", tickfont=dict(color=COLORS["text"])),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(tickfont=dict(color=COLORS["text"])),
        yaxis=dict(tickfont=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ─── Confusion Matrix Heatmap ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title: str = "Confusion Matrix") -> go.Figure:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: Rejected", "Predicted: Approved"],
        y=["Actual: Rejected", "Actual: Approved"],
        colorscale=[[0, "#1A1F2E"], [0.5, "#4E9AF1"], [1, "#22C55E"]],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        xaxis=dict(side="bottom", tickfont=dict(color=COLORS["text"])),
        yaxis=dict(autorange="reversed", tickfont=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


# ─── Scatter: Actual vs Predicted ─────────────────────────────────────────────
def plot_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted LoanAmount") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color=COLORS["primary"], size=7, opacity=0.7, line=dict(color="white", width=0.5)),
        name="Prediksi",
    ))
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color=COLORS["danger"], dash="dash", width=2),
        name="Ideal (y=x)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Actual LoanAmount", gridcolor=COLORS["border"], color=COLORS["text"]),
        yaxis=dict(title="Predicted LoanAmount", gridcolor=COLORS["border"], color=COLORS["text"]),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
    )
    return fig


# ─── Radar Chart ──────────────────────────────────────────────────────────────
def plot_radar_chart(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float,
    loan_term: float,
    credit_history: float,
) -> go.Figure:
    """Normalized radar chart of applicant financial profile."""
    # Normalization ranges (domain knowledge)
    norms = {
        "ApplicantIncome"   : (0, 20000),
        "CoapplicantIncome" : (0, 10000),
        "LoanAmount"        : (0, 500),
        "Loan_Amount_Term"  : (0, 480),
        "Credit_History"    : (0, 1),
    }
    values_raw = {
        "ApplicantIncome"   : applicant_income,
        "CoapplicantIncome" : coapplicant_income,
        "LoanAmount"        : loan_amount,
        "Loan_Amount_Term"  : loan_term,
        "Credit_History"    : credit_history,
    }
    normalized = []
    for key, (lo, hi) in norms.items():
        val = values_raw[key]
        normalized.append(round((val - lo) / (hi - lo) * 100, 1) if hi != lo else 0)
    normalized.append(normalized[0])  # close the polygon

    categories = list(norms.keys()) + [list(norms.keys())[0]]

    fig = go.Figure(go.Scatterpolar(
        r=normalized,
        theta=categories,
        fill="toself",
        fillcolor="rgba(78, 154, 241, 0.2)",
        line=dict(color=COLORS["primary"], width=2),
        marker=dict(color=COLORS["primary"], size=8),
        name="Profil Pemohon",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["card"],
            angularaxis=dict(
                tickcolor=COLORS["text"],
                linecolor=COLORS["border"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["text"], size=11),
            ),
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickcolor=COLORS["text"],
                gridcolor=COLORS["border"],
                tickfont=dict(color=COLORS["muted"], size=9),
                linecolor=COLORS["border"],
            ),
        ),
        paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        margin=dict(l=30, r=30, t=30, b=30),
        height=400,
    )
    return fig


# ─── Bar chart: Model Comparison ──────────────────────────────────────────────
def _get_metric_value(model_metrics: dict, variant: str, metric: str) -> float:
    """Ambil nilai metrik dengan aman untuk key huruf besar/kecil dan alias non_hpo/nohpo."""
    if variant == "non_hpo":
        variant_dict = model_metrics.get("non_hpo") or model_metrics.get("nohpo") or {}
    else:
        variant_dict = model_metrics.get(variant, {})
    aliases = [metric, metric.lower(), metric.upper(), metric.capitalize()]
    if metric.lower() in ["f1-score", "f1"]:
        aliases += ["f1", "F1", "F1-Score"]
    if metric.lower() in ["r2", "r² score", "r2 score"]:
        aliases += ["r2", "R2", "R² Score"]
    for key in aliases:
        if key in variant_dict:
            return float(variant_dict[key])
    return 0.0


def plot_model_comparison_bar(metrics_dict: dict, metric: str = "Accuracy", title: str = "") -> go.Figure:
    """
    metrics_dict: { "SVM": {"hpo": {...}, "non_hpo": {...}}, ... }
    Aman dari KeyError 'nohpo' dan key metrik huruf besar/kecil.
    """
    algos = list(metrics_dict.keys())
    hpo_vals = [_get_metric_value(metrics_dict[a], "hpo", metric) for a in algos]
    nohpo_vals = [_get_metric_value(metrics_dict[a], "non_hpo", metric) for a in algos]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="HPO",
        x=algos,
        y=hpo_vals,
        marker_color=COLORS["primary"],
        text=[f"{v:.4f}" for v in hpo_vals],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.add_trace(go.Bar(
        name="Non-HPO",
        x=algos,
        y=nohpo_vals,
        marker_color=COLORS["muted"],
        text=[f"{v:.4f}" for v in nohpo_vals],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(
        title=dict(text=title or f"Perbandingan {metric}", font=dict(color=COLORS["text"], size=15)),
        barmode="group",
        xaxis=dict(title="Algoritma", color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(
            title=metric,
            color=COLORS["text"],
            gridcolor=COLORS["border"],
            range=[max(0, min(hpo_vals + nohpo_vals) - 0.05), max(hpo_vals + nohpo_vals + [1.0]) * 1.15],
        ),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


# ─── Pie chart ────────────────────────────────────────────────────────────────
def plot_pie(labels, values, title: str = "") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=[COLORS["success"], COLORS["danger"], COLORS["warning"], COLORS["secondary"]]),
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=15)),
        paper_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


# ─── Histogram ────────────────────────────────────────────────────────────────
def plot_histogram(series: pd.Series, title: str = "", xlab: str = "", color: str = None) -> go.Figure:
    color = color or COLORS["primary"]
    fig = go.Figure(go.Histogram(
        x=series,
        nbinsx=30,
        marker_color=color,
        opacity=0.85,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=15)),
        xaxis=dict(title=xlab, color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(title="Frekuensi", color=COLORS["text"], gridcolor=COLORS["border"]),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        bargap=0.05,
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
    )
    return fig


# ─── Box plot ─────────────────────────────────────────────────────────────────
def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str = "") -> go.Figure:
    fig = px.box(
        df, x=x_col, y=y_col,
        color=x_col,
        color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["success"]],
    )
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=15)),
        xaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


# ─── Stacked bar ──────────────────────────────────────────────────────────────
def plot_stacked_bar(df: pd.DataFrame, x_col: str, hue_col: str, title: str = "") -> go.Figure:
    groups = df[x_col].unique()
    hue_vals = df[hue_col].unique()
    color_seq = [COLORS["success"], COLORS["danger"]]
    fig = go.Figure()
    for i, hv in enumerate(sorted(hue_vals)):
        counts = [len(df[(df[x_col] == g) & (df[hue_col] == hv)]) for g in groups]
        fig.add_trace(go.Bar(
            name=str(hv),
            x=groups,
            y=counts,
            marker_color=color_seq[i % len(color_seq)],
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=15)),
        barmode="stack",
        xaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"], gridcolor=COLORS["border"]),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        legend=dict(font=dict(color=COLORS["text"])),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


# ─── Feature Importance Bar ────────────────────────────────────────────────────
def plot_feature_importance(feature_names: list, importances: list, title: str = "Feature Importance") -> go.Figure:
    pairs = sorted(zip(importances, feature_names), reverse=True)
    imps, names = zip(*pairs)
    fig = go.Figure(go.Bar(
        x=list(imps),
        y=list(names),
        orientation="h",
        marker=dict(
            color=list(imps),
            colorscale=[[0, COLORS["muted"]], [1, COLORS["primary"]]],
            showscale=False,
        ),
        text=[f"{v:.4f}" for v in imps],
        textposition="outside",
        textfont=dict(color=COLORS["text"]),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        xaxis=dict(title="Importance", color=COLORS["text"], gridcolor=COLORS["border"]),
        yaxis=dict(color=COLORS["text"], autorange="reversed"),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=10, r=100, t=50, b=10),
        height=400,
    )
    return fig
