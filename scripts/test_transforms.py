"""Test unitaire : verifie que les transformations arsinh sont appliquees."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from precite.datasets.loader import _apply_transforms


def main() -> int:
    # Test direct sur _apply_transforms (sans build_design_matrix)
    df = pd.DataFrame({
        "pmid": ["a", "b"],
        "n_references": [4000, 30],
        "mean_ref_age": [15.0, 5.0],
        "abstract_len": [250, 180],
        "title_len": [12, 14],
        "n_mesh": [8, 10],
    })

    print("=== Avant transform ===")
    print(f"  n_references = {df['n_references'].tolist()}")
    print(f"  mean_ref_age = {df['mean_ref_age'].tolist()}")
    print(f"  abstract_len = {df['abstract_len'].tolist()}")
    print(f"  title_len    = {df['title_len'].tolist()}")

    X = df[["n_references", "mean_ref_age", "abstract_len", "title_len", "n_mesh"]].copy()
    X = _apply_transforms(X)

    print()
    print("=== Apres _apply_transforms ===")
    all_ok = True

    for col in ["n_references", "mean_ref_age", "abstract_len"]:
        for i, raw in enumerate(df[col]):
            val = X[col].iloc[i]
            expected = np.arcsinh(raw)
            status = "OK" if abs(val - expected) < 1e-6 else "ECHEC"
            if status == "ECHEC":
                all_ok = False
            print(f"  {col:<15} brut={raw:<8} -> {val:.4f} (attendu {expected:.4f}) [{status}]")

    print()
    print("=== Features non transformees ===")
    for col in ["title_len", "n_mesh"]:
        for i, raw in enumerate(df[col]):
            val = X[col].iloc[i]
            status = "OK" if val == raw else "ECHEC (transform inattendue)"
            if status != "OK":
                all_ok = False
            print(f"  {col:<15} brut={raw:<8} -> {val:.4f} [{status}]")

    print()
    if all_ok:
        print("[OK] Transform arsinh correctement applique sur les 3 features")
        return 0
    else:
        print("[ECHEC]")
        return 1


if __name__ == "__main__":
    sys.exit(main())