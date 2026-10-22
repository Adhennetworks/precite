"""Liste des modèles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import list_models

st.set_page_config(page_title="Modèles — Precite", page_icon="🧠", layout="wide")
st.title("🧠 Modèles")
st.caption("Modèles entraînés pour la prédiction de citations")

model_type = st.selectbox(
    "Filtrer par type",
    [
        "Tous",
        "production-lm", "production-nb",
        "lm", "nb",
        "dle-lm", "dle-nb",
    ],
)

try:
    mt = None if model_type == "Tous" else model_type
    data = list_models(model_type=mt)
except Exception as exc:
    st.error(f"Erreur : {exc}")
    st.stop()

models = data.get("models", [])
st.write(f"**{len(models)}** modèles")

if not models:
    st.info("Aucun modèle pour ce filtre.")
    st.stop()

# Tableau récap
rows = []
for m in models:
    met = m.get("metrics", {})
    rows.append({
        "Nom": m["model_name"],
        "Type": m["model_type"],
        "v": m["model_version"],
        "Valide": m["valid_year"],
        "Train": m.get("n_train"),
        "Spearman": met.get("spearman"),
        "MAE": met.get("mae"),
        "R²": met.get("r2"),
        "Nagelkerke": met.get("pseudo_r2_nagelkerke"),
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)

# Détail d'un modèle
st.subheader("Détail")
selected = st.selectbox("Choisir un modèle", [m["model_name"] for m in models])
for m in models:
    if m["model_name"] == selected:
        st.json(m)
        break