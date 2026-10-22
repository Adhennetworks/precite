"""Monitoring : distributions, volumes, DLE."""

from __future__ import annotations

import duckdb
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Monitoring — Precite", page_icon="📊", layout="wide")
st.title("📊 Monitoring")
st.caption("Distributions observées et performances estimées")


@st.cache_resource
def get_con():
    return duckdb.connect()


con = get_con()

# ─── Volume par annee ──────────────────────────────────────────────────────

st.subheader("Volume de prédictions par année (2000+)")

try:
    df = con.execute("""
        SELECT year, COUNT(*) as n
        FROM 'data/predictions/family=lm/*.parquet'
        WHERE year >= 2000
        GROUP BY year
        ORDER BY year
    """).fetchdf()
    st.bar_chart(df.set_index("year"))
except Exception as exc:
    st.error(f"Erreur : {exc}")
    st.stop()

# ─── Statistiques completes ────────────────────────────────────────────────

st.subheader("Statistiques des prédictions LM (2020-2022)")

try:
    stats = con.execute("""
        SELECT
            COUNT(*)                                   as n,
            MIN(predicted_citations)                   as min,
            quantile_cont(predicted_citations, 0.01)   as p01,
            quantile_cont(predicted_citations, 0.25)   as p25,
            quantile_cont(predicted_citations, 0.50)   as median,
            quantile_cont(predicted_citations, 0.75)   as p75,
            quantile_cont(predicted_citations, 0.95)   as p95,
            quantile_cont(predicted_citations, 0.99)   as p99,
            quantile_cont(predicted_citations, 0.999)  as p999,
            quantile_cont(predicted_citations, 0.9999) as p9999,
            MAX(predicted_citations)                   as max,
            AVG(predicted_citations)                   as mean
        FROM 'data/predictions/family=lm/*.parquet'
        WHERE year BETWEEN 2020 AND 2022
    """).fetchdf().iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles", f"{int(stats['n']):,}")
    c2.metric("Médiane", f"{stats['median']:.2f}")
    c3.metric("Moyenne", f"{stats['mean']:,.0f}")
    c4.metric("Maximum", f"{stats['max']:.2e}")

    c5, c6, c7, c8, c9, c10 = st.columns(6)
    c5.metric("p01", f"{stats['p01']:.2f}")
    c6.metric("p25", f"{stats['p25']:.2f}")
    c7.metric("p75", f"{stats['p75']:.2f}")
    c8.metric("p95", f"{stats['p95']:.2f}")
    c9.metric("p99", f"{stats['p99']:.2f}")
    c10.metric("p999", f"{stats['p999']:,.0f}")
except Exception as exc:
    st.error(f"Erreur stats : {exc}")
    st.stop()

# ─── Distribution sur echelle arsinh ───────────────────────────────────────

st.subheader("Distribution des prédictions (échelle arsinh)")

try:
    # Bin en arsinh, comme la cible d'entrainement
    df_hist = con.execute("""
        SELECT
            ROUND(ASINH(predicted_citations) * 4) / 4 as arsinh_bin,
            COUNT(*) as n
        FROM 'data/predictions/family=lm/*.parquet'
        WHERE year BETWEEN 2020 AND 2022
        GROUP BY arsinh_bin
        ORDER BY arsinh_bin
    """).fetchdf()

    df_hist = df_hist.set_index("arsinh_bin")
    st.bar_chart(df_hist["n"])

    st.caption(
        "Échelle arsinh (identique à la cible d'entraînement). "
        "Repères : 0 = 0 citation, 3 ≈ 10, 5 ≈ 74, 7 ≈ 550, 9 ≈ 4 050, 11 ≈ 30 000. "
        "Aucune troncature : toutes les prédictions sont incluses, y compris les extrêmes."
    )
except Exception as exc:
    st.error(f"Erreur histogramme : {exc}")

# ─── Distribution sur echelle arsinh (LM vs NB) ────────────────────────────

st.subheader("Comparaison LM vs NB (échelle arsinh)")

try:
    df_cmp = con.execute("""
        SELECT
            'LM' as family,
            ROUND(ASINH(predicted_citations) * 4) / 4 as arsinh_bin,
            COUNT(*) as n
        FROM 'data/predictions/family=lm/*.parquet'
        WHERE year BETWEEN 2020 AND 2022
        GROUP BY arsinh_bin
        UNION ALL
        SELECT
            'NB' as family,
            ROUND(ASINH(predicted_citations) * 4) / 4 as arsinh_bin,
            COUNT(*) as n
        FROM 'data/predictions/family=nb/*.parquet'
        WHERE year BETWEEN 2020 AND 2022
        GROUP BY arsinh_bin
    """).fetchdf()

    # Pivot pour bar_chart
    pivot = df_cmp.pivot(index="arsinh_bin", columns="family", values="n").fillna(0)
    st.bar_chart(pivot)
    st.caption(
        "Les deux modèles ont des distributions comparables en arsinh. "
        "Le NB produit légèrement plus de valeurs proches de 0 (articles non cités)."
    )
except Exception as exc:
    st.error(f"Erreur comparaison : {exc}")

# ─── Repartition par modele ────────────────────────────────────────────────

st.subheader("Statistiques par modèle (LM, 2020-2022)")

try:
    df_models = con.execute("""
        SELECT
            model_name,
            COUNT(*) as n,
            AVG(predicted_citations) as mean,
            quantile_cont(predicted_citations, 0.5) as median,
            quantile_cont(predicted_citations, 0.99) as p99,
            MAX(predicted_citations) as max
        FROM 'data/predictions/family=lm/*.parquet'
        WHERE year BETWEEN 2020 AND 2022
        GROUP BY model_name
        ORDER BY model_name
    """).fetchdf()

    df_models = df_models.rename(columns={
        "model_name": "Modèle",
        "n": "N",
        "mean": "Moyenne",
        "median": "Médiane",
        "p99": "p99",
        "max": "Max",
    })
    st.dataframe(df_models, use_container_width=True)
except Exception as exc:
    st.error(f"Erreur : {exc}")

# ─── Info DLE ──────────────────────────────────────────────────────────────

st.divider()
st.subheader("DLE multi-horizon")

st.markdown(
    """
    Les modèles **DLE** (Direct Loss Estimation) estiment la performance du
    modèle de production **sans cible connue**, en utilisant les citations
    partielles observées :

    - **DLE-1y** : utilise les citations à 1 an
    - **DLE-2y** : utilise les citations à 2 ans
    - **DLE-3y** : utilise les citations à 3 ans

    Pour les articles récents (2024+) dont la cible 3 ans n'est pas encore
    disponible, le DLE-1y permet d'estimer la MAE finale.
    """
)