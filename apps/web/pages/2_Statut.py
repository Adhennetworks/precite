"""Statut du dataset Precite."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import get_status_report

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False



st.set_page_config(page_title="Statut — Precite", page_icon="📊", layout="wide")
st.title("📊 Statut du dataset")
st.caption("Vue d'ensemble de la couverture et de la qualité des données")


with st.spinner("Chargement des statistiques..."):
    try:
        report = get_status_report()
    except Exception as exc:
        st.error(f"Erreur : {exc}")
        st.stop()


# === Metriques globales ===
c1, c2, c3 = st.columns(3)
c1.metric("Articles", f"{report['total_articles']:,}")
c2.metric("Prédictions", f"{report['total_predictions']:,}")
c3.metric("Modèles", f"{report['total_models']:,}")


# === Breakdown global par statut MEDLINE ===
st.subheader("Couverture MEDLINE (tous articles)")

df = pd.DataFrame(report["medline_breakdown"])
if len(df):
    fig = px.pie(
        df,
        values="n",
        names="status",
        hole=0.4,
        color="status",
        color_discrete_map={
            "MEDLINE": "#2e7d32",
            "OldMEDLINE": "#1b5e20",
            "In-Process": "#f9a825",
            "In-Data-Review": "#ef6c00",
            "Publisher": "#e65100",
            "PubMed-not-MEDLINE": "#616161",
            "NULL": "#9e9e9e",
        },
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


# === Breakdown recent (2020+) ===
st.subheader("Couverture MEDLINE (articles 2020+)")

df_recent = pd.DataFrame(report["medline_breakdown_recent"])
col1, col2 = st.columns([1, 1])

with col1:
    if len(df_recent):
        fig2 = px.pie(
            df_recent,
            values="n",
            names="status",
            hole=0.4,
            color="status",
            color_discrete_map={
                "MEDLINE": "#2e7d32",
                "OldMEDLINE": "#1b5e20",
                "In-Process": "#f9a825",
                "In-Data-Review": "#ef6c00",
                "Publisher": "#e65100",
                "PubMed-not-MEDLINE": "#616161",
                "NULL": "#9e9e9e",
            },
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    if len(df_recent):
        st.dataframe(
            df_recent.rename(columns={
                "status": "Statut",
                "n": "Articles",
                "pct": "%",
            }),
            use_container_width=True,
            hide_index=True,
        )


# === Evolution par annee ===
st.subheader("Évolution annuelle (2020+)")

df_year = pd.DataFrame(report["by_year_medline"])
if len(df_year):
    fig3 = px.bar(
        df_year,
        x="year",
        y="n",
        color="status",
        color_discrete_map={
            "MEDLINE": "#2e7d32",
            "OldMEDLINE": "#1b5e20",
            "In-Process": "#f9a825",
            "In-Data-Review": "#ef6c00",
            "Publisher": "#e65100",
            "PubMed-not-MEDLINE": "#616161",
            "NULL": "#9e9e9e",
        },
        labels={"year": "Année", "n": "Articles", "status": "Statut"},
    )
    fig3.update_layout(barmode="stack")
    st.plotly_chart(fig3, use_container_width=True)


# === Note methodologique ===
with st.expander("i À propos de ces statistiques"):
    st.markdown("""
    **Statut MEDLINE** : PubMed indexe chaque article avec un statut reflétant son niveau d'indexation :

    - 🟢 **MEDLINE** : article complètement indexé (features fiables)
    - 🟢 **OldMEDLINE** : article legacy indexé (avant 2000)
    - 🟡 **In-Process** : en cours d'indexation
    - 🟠 **In-Data-Review** : en révision NLM
    - 🟠 **Publisher** : publié mais pas encore indexé (features partielles)
    - ⚪ **PubMed-not-MEDLINE** : hors périmètre NLM (revues non-indexées)

    Les articles **non-MEDLINE** peuvent avoir des features incomplètes
    (0 références, 0 MeSH). Les prédictions sur ces articles sont **indicatives**.

    **Mise à jour** : chaque dimanche, le pipeline ré-extrait les statuts
    depuis les fichiers XML PubMed et met à jour la base.
    """)