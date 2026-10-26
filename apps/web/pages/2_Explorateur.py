"""Exploration par filtres."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import search_papers

@st.cache_data(ttl=300)
def cached_search(year_min, year_max, limit):
    return search_papers(year_min, year_max, limit)

st.set_page_config(page_title="Explorateur — Precite", page_icon="🗂️", layout="wide")
st.title("🗂️ Explorateur")
st.caption("Parcourir les articles et leurs prédictions")

col1, col2, col3 = st.columns(3)
with col1:
    year_min = st.number_input("Année min", min_value=1970, max_value=2030, value=2015)
with col2:
    year_max = st.number_input("Année max", min_value=1970, max_value=2030, value=2023)
with col3:
    limit = st.number_input("Limite", min_value=10, max_value=1000, value=100)

try:
    data = cached_search(year_min=int(year_min), year_max=int(year_max), limit=int(limit))
except Exception as exc:
    st.error(f"Erreur : {exc}")
    st.stop()

papers = data.get("papers", [])
st.write(f"**{len(papers)}** articles affichés (limite {limit})")

if not papers:
    st.info("Aucun article pour ces critères.")
    st.stop()

df = pd.DataFrame(papers)
st.dataframe(df, use_container_width=True)

if "citations_3y" in df.columns:
    st.subheader("Distribution des citations observées")
    counts = df["citations_3y"].value_counts().sort_index()
    st.bar_chart(counts.head(50))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Exporter en CSV",
    data=csv,
    file_name=f"precite_{year_min}_{year_max}.csv",
    mime="text/csv",
)