"""Recherche par PMID."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import get_paper


# Mapping status MEDLINE -> (icone, libelle, couleur)
MEDLINE_BADGES = {
    "MEDLINE": ("🟢", "Article indexé MEDLINE — prédiction fiable", "green"),
    "OldMEDLINE": ("🟢", "Article indexé MEDLINE (archive)", "green"),
    "Publisher": ("🟠", "Article publié, en attente d'indexation NLM — features partielles", "orange"),
    "In-Process": ("🟠", "Article en cours d'indexation — features partielles", "orange"),
    "In-Data-Review": ("🟠", "Article en révision NLM — features partielles", "orange"),
    "PubMed-not-MEDLINE": ("⚪", "Hors périmètre MEDLINE", "gray"),
}


def render_medline_badge(status: str | None) -> None:
    if status is None:
        st.warning("⚪ Statut MEDLINE inconnu")
        return
    icon, label, _ = MEDLINE_BADGES.get(status, ("❓", f"Statut : {status}", "gray"))
    if status in ("MEDLINE", "OldMEDLINE"):
        st.success(f"{icon} {label}")
    elif status in ("Publisher", "In-Process", "In-Data-Review"):
        st.warning(f"{icon} {label}")
    else:
        st.info(f"{icon} {label}")


st.set_page_config(page_title="Recherche — Precite", page_icon="🔎", layout="wide")
st.title("🔎 Recherche par PMID")
st.caption("Consulter les prédictions pré-calculées pour un article")

pmid = st.text_input("PMID", placeholder="ex: 21229385")

if pmid:
    with st.spinner("Recherche..."):
        try:
            paper = get_paper(pmid)
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            st.stop()

    if paper is None:
        st.warning(f"Aucun article avec le PMID {pmid}")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"PMID {paper['pmid']}")
        st.write(f"**Année** : {paper.get('year', '—')}")

        # Badge MEDLINE
        render_medline_badge(paper.get("medline_status"))

        preds = paper.get("predictions", [])
        if preds:
            df = pd.DataFrame(preds)
            df = df.rename(columns={
                "model_name": "Modèle",
                "model_version": "Version",
                "predicted_citations": "Citations prédites",
            })
            st.subheader("Prédictions par modèle")
            st.dataframe(
                df[["Modèle", "Version", "Citations prédites"]],
                use_container_width=True,
            )

            lm = next((p for p in preds if "precite-lm" in p["model_name"]), None)
            nb = next((p for p in preds if "precite-nb" in p["model_name"]), None)
            if lm and nb:
                c1, c2 = st.columns(2)
                c1.metric("Prédiction LM", f"{lm['predicted_citations']:.2f}")
                c2.metric("Prédiction NB", f"{nb['predicted_citations']:.2f}")
        else:
            st.info("Pas encore de prédiction pour cet article.")

        c3, c4 = st.columns(2)
        c3.metric(
            "Citations réelles (3y)",
            f"{paper.get('citations_3y', 0):.0f}" if paper.get("citations_3y") is not None else "—",
        )
        c4.metric(
            "SJR pondéré (3y)",
            f"{paper.get('weighted_citations_3y', 0):.1f}" if paper.get("weighted_citations_3y") is not None else "—",
        )

    with col2:
        st.subheader("Features")
        st.json(paper.get("features", {}))