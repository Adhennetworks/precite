#!/bin/bash
set -e

# 1. Telecharger les donnees depuis HF si pas presentes
if [ ! -d "data/final" ] || [ -z "$(ls -A data/final 2>/dev/null)" ]; then
    echo "Telechargement des features depuis HF..."
    python scripts/download_from_hf.py \
        --source ${HF_DATASET_FEATURES:-precite/features} \
        --target data/final
fi

# 2. Lancer l'API
echo "Demarrage de l'API..."
uvicorn precite.api.main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 2