"""Monitoring du drift des données."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Monitoring Drift — Precite", page_icon="📈", layout="wide")
st.title("📈 Monitoring du drift")
st.caption("Surveillance de la stabilité des features dans le temps")


@st.cache_data(ttl=3600)
def load_drift_report() -> dict | None:
    """Charge le rapport de drift depuis HF ou local."""
    # Essai 1 : local (si lancé en dev)
    local_path = Path("data/drift_report.json")
    if local_path.exists():
        with open(local_path) as f:
            return json.load(f)

    # Essai 2 : depuis HF
    try:
        from huggingface_hub import hf_hub_download

        token = os.environ.get("HF_TOKEN")
        path = hf_hub_download(
            repo_id="preciteproject/precite-data",
            filename="drift_report.json",
            repo_type="dataset",
            token=token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Impossible de charger le rapport : {exc}")
        return None


report = load_drift_report()

if report is None:
    st.warning(
        "⚠️ Aucun rapport de drift disponible. "
        "Lance le workflow `Drift Monitoring` sur GitHub Actions pour en générer un."
    )
    st.stop()

# === Résumé ===
st.subheader("Résumé")

metrics = report.get("metrics", [])
drift_summary = None
for m in metrics:
    name = m.get("metric_name", "")
    if "DriftedColumnsCount" in name:
        drift_summary = m.get("value", {})
        break

if drift_summary:
    count = drift_summary.get("count", 0)
    share = drift_summary.get("share", 0)

    c1, c2 = st.columns(2)
    c1.metric("Features driftées", f"{int(count)}")
    c2.metric("Part des features", f"{share*100:.0f}%")

    if share > 0.5:
        st.warning(
            f"⚠️ {share*100:.0f}% des features ont un drift significatif. "
            "Cela peut indiquer un changement dans la distribution des données PubMed."
        )
    else:
        st.success("✅ Drift modéré ou absent.")

# === Détail par feature ===
st.divider()
st.subheader("Détail par feature")

feature_rows = []
for m in metrics:
    name = m.get("metric_name", "")
    if "ValueDrift" in name or "ColumnDrift" in name:
        config = m.get("config", {})
        column = config.get("column", "?")
        value = m.get("value", {})

        # value peut etre un dict (format standard) ou un float (format simplifie)
        if isinstance(value, dict):
            drift_score = value.get("drift_score", value.get("score", None))
            drift_detected = value.get("drift_detected", None)
        elif isinstance(value, (int, float)):
            drift_score = float(value)
            drift_detected = drift_score > 0.1  # seuil Evidently par defaut
        else:
            drift_score = None
            drift_detected = None
            
        feature_rows.append({
            "Feature": column,
            "Score de drift": round(drift_score, 4) if drift_score is not None else "—",
            "Drift détecté": "⚠️" if drift_detected else "✅",
        })

if feature_rows:
    import pandas as pd
    df = pd.DataFrame(feature_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Aucun détail par feature disponible dans ce rapport.")

# === Rapport brut ===
with st.expander("📄 Rapport JSON brut"):
    st.json(report)