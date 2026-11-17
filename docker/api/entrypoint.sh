#!/bin/bash
set -e

# Port : SnapDeploy injecte PORT, sinon 8000 par défaut
PORT="${PORT:-8000}"

echo "Demarrage de l'API Precite sur le port ${PORT}"
echo "PRECITE_REMOTE=${PRECITE_REMOTE:-false}"

exec uvicorn precite.api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info