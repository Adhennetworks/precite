import nannyml as nml
import pandas as pd
import duckdb
from pathlib import Path

# 1. Charger les données de référence (période d'entraînement où on a les cibles)
# Ex: données 2018-2022 avec les vraies citations
con = duckdb.connect()
reference = con.execute("""
    SELECT pmid, year, n_pages, n_references, abstract_len, 
           predicted_citations, weighted_citations_3y as target
    FROM read_parquet('data/features/chunk-*.parquet')
    WHERE year BETWEEN 2018 AND 2022
    LIMIT 50000
""").fetchdf()

# 2. Charger les données de production (nouvelles prédictions SANS cibles)
analysis = con.execute("""
    SELECT pmid, year, n_pages, n_references, abstract_len, predicted_citations
    FROM read_parquet('data/features/chunk-*.parquet')
    WHERE year = 2026
    LIMIT 50000
""").fetchdf()

# 3. Initialiser le CBPE pour la régression
# Note: NannyML supporte la régression via 'regression' problem_type
estimator = nml.CBPE(
    y_pred='predicted_citations',
    y_true='target',
    metrics=['r2', 'mae'],  # ou 'mse', 'rmse'
    problem_type='regression',
    chunk_size=5000,  # taille des chunks pour l'estimation
)

# 4. Fit sur la référence
estimator.fit(reference)

# 5. Estimer la performance sur la production
results = estimator.estimate(analysis)

# 6. Sauvegarder le rapport
results.to_df().to_parquet('data/performance_estimate.parquet')
print(results.to_df().tail())