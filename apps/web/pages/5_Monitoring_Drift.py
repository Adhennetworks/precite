"""Monitoring du drift des données."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import streamlit as st

HAS_EVIDENTLY = importlib.util.find_spec("evidently") is not None

st.set_page_config(page_title="Monitoring — Precite", page_icon="📈", layout="wide")
st.title("📈 Monitoring du drift")
st.caption("Surveillance de la stabilité des features dans le temps")

# Chargement du rapport de drift (généré par le workflow GitHub Actions)
drift_report_path = Path("data/drift_report.json")

if not drift_report_path.exists():
    st.warning(
        "⚠️ Aucun rapport de drift disponible. "
        "Lance le workflow `Drift Monitoring` sur GitHub Actions pour en générer un."
    )
    st.stop()

with open(drift_report_path) as f:
    drift_data = json.load(f)

# Résumé
st.subheader("Résumé")
if drift_data.get("metrics"):
    # Chercher le résultat du preset DataDrift
    for metric in drift_data["metrics"]:
        if metric.get("metric") == "DataDriftPreset":
            result = metric.get("result", {})
            dataset_drift = result.get("dataset_drift", False)
            if dataset_drift:
                st.error("⚠️ Drift détecté sur les données récentes.")
            else:
                st.success("✅ Aucun drift significatif détecté.")
            break
else:
    st.info("Rapport vide ou format inattendu.")

st.divider()

# Affichage détaillé (si Evidently est disponible)
if HAS_EVIDENTLY:
    st.subheader("Rapport détaillé Evidently")
    # On peut ré-afficher le rapport HTML si on l'a sauvegardé
    html_path = Path("data/drift_report.html")
    if html_path.exists():
        with open(html_path) as f:
            st.components.v1.html(f.read(), height=800, scrolling=True)
    else:
        st.info(
            "Le rapport HTML n'est pas disponible. "
            "Ajoute `result.save_html('data/drift_report.html')` dans `drift_monitor.py`."
        )
else:
    st.warning(
        "Evidently n'est pas installé. "
        "Installe-le avec `pip install evidently` pour afficher le rapport détaillé."
    )

# Historique des rapports (si on les stocke)
st.subheader("Historique des rapports")
st.info(
    "Pour un historique, stocke chaque rapport dans "
    "`data/drift_reports/drift_YYYYMMDD.json` et affiche-les avec un sélecteur de date."
)