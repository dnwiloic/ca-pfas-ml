"""Assemble le dataset final en joignant les features dérivées validées.

Prend CA-PFAS-ASGWS.parquet (46 338 lignes × 201 cols, niveau prélèvement) et
attache les features dérivées au niveau puits (many-to-one sur gm_well_id) :

    well_features_extra.parquet   → 12 features transférables (land use, topo,
                                     profondeur nappe, profondeur puits)
    well_hydraulic_gradient.csv   → 6 features mécanistes (gradient hydraulique
                                     DWR, direction d'écoulement)

Sortie : CA-PFAS-ASGWS_v2.parquet dans data/processed/ ET dans pfas-gnn/data/.
Le suffixe _v2 préserve l'original intact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---- chemins
ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
GNN_DATA = ROOT.parent / "pfas-gnn" / "data"

BASE = PROC / "CA-PFAS-ASGWS.parquet"
EXTRA = PROC / "well_features_extra.parquet"
GRAD = PROC / "well_hydraulic_gradient.csv"
OUT_NAME = "CA-PFAS-ASGWS_v2.parquet"

# Colonnes du gradient à retenir (hydr_head_m et hydr_grad_mag sont redondants
# avec hydr_grad_mag_permil — on ne les garde pas pour éviter la multicolinéarité)
GRAD_COLS = ["gm_well_id", "hydr_grad_mag_permil", "flow_dir_sin", "flow_dir_cos",
             "dist_nearest_gwl_km"]


def main() -> None:
    df = pd.read_parquet(BASE)
    print(f"Base                  : {df.shape}")

    extra = pd.read_parquet(EXTRA).drop(columns=["latitude", "longitude"], errors="ignore")
    grad = pd.read_csv(GRAD)[GRAD_COLS]

    n0 = len(df)
    df = df.merge(extra, on="gm_well_id", how="left")
    df = df.merge(grad,  on="gm_well_id", how="left")
    assert len(df) == n0, "merge a changé le nombre de lignes"

    new_cols = [c for c in df.columns if c not in pd.read_parquet(BASE).columns]
    print(f"Nouvelles colonnes    : {len(new_cols)}  {new_cols}")

    # couverture des nouvelles features
    for c in new_cols:
        fill = df[c].notna().mean()
        print(f"  {c:<30} fill={fill:.0%}")

    for dest in [PROC, GNN_DATA]:
        out = dest / OUT_NAME
        df.to_parquet(out, index=False)
        print(f"\nÉcrit → {out}")
        print(f"  Shape finale : {df.shape}")


if __name__ == "__main__":
    main()
