"""Page d'accueil Precite."""

from __future__ import annotations

import streamlit as st
from api_client import health, stats

st.set_page_config(page_title="Precite", page_icon="📚", layout="wide")

st.title("📚 Precite")
st.caption("Prédiction des citations à partir des caractéristiques observables à la soumission")

st.markdown(
    """
    **Precite** prédit les citations d'articles PubMed en utilisant
    uniquement des variables **accessibles à la soumission** (avant peer review).
    Aucun auteur, journal ou prestige institutionnel n'est utilisé,
    ce qui **neutralise l'effet Matthieu**.

    - **Recherche** : consulter les prédictions pour un PMID
    - **Explorateur** : filtrer par année, batch
    - **Modèles** : liste des modèles (LM, NB, production, DLE)
    - **Monitoring** : suivi de la performance estimée par année
    """
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("État du service")
    try:
        h = health()
        if h.get("status") == "ok":
            st.success("API en ligne")
        else:
            st.warning(f"État : {h.get('status')}")
    except Exception as exc:
        st.error(f"API inaccessible : {exc}")

with col2:
    st.subheader("Base")
    try:
        s = stats()
        st.metric("Articles", f"{s['papers']:,}")
        st.metric("Prédictions", f"{s['predictions']:,}")
        st.metric("Modèles", s["models"])
    except Exception as exc:
        st.warning(f"Stats indisponibles : {exc}")