"""Consolide les features dérivées VALIDÉES en un seul fichier joignable.

Rassemble les features qui ont passé le protocole d'audit (AUC intra-bloc CV
spatiale k=8 ≈ AUC globale, après croisement avec le confondant `gm_well_category`)
en un `well_features_extra.parquet` prêt à brancher dans le pipeline pfas-gnn
(jointure sur `gm_well_id`, nœud `well`).

N'inclut PAS les features rejetées (pop_density, pH/redox, slope, type d'aquifère,
wetlands) ni lat/lon comme features (C-LOC.1). Ces fichiers restent sur disque
pour un éventuel re-audit contre la cible EPA-2024 (seuil 4 ng/L).

Sortie : data/processed/well_features_extra.parquet (+ _dict.csv)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "well_features_extra.parquet"

# fichier source → colonnes validées à conserver
SOURCES = {
    "well_depth_to_water.csv": ["depth_to_water_m", "depth_to_water_log1p", "dtw_far"],
    "well_construction.csv": ["depth_eff_ft", "depth_eff_log1p", "screen_mid_ft",
                              "screen_length_ft", "depth_missing"],
    "well_landuse.csv": ["dev_intensity", "lc_developed"],
    "well_topography.csv": ["elevation_m", "topo_missing"],
}

# dictionnaire : colonne → (mécanisme, AUC intra-bloc « propre », note)
DICT = {
    "depth_to_water_m": ("épaisseur zone non-saturée", 0.59, "transférable"),
    "depth_to_water_log1p": ("idem, log1p", 0.59, "variante"),
    "dtw_far": ("fiabilité : >20 km d'une station DWR", None, "flag manquant"),
    "depth_eff_ft": ("profondeur d'extraction (crépine sinon total)", 0.58,
                     "fort mais ~0,2 d'AUC brute = confondant puits-surveillance ; valeur propre 0,58"),
    "depth_eff_log1p": ("idem, log1p", 0.58, "variante"),
    "screen_mid_ft": ("milieu de crépine", 0.58, "≈ depth_eff, couverture 45%"),
    "screen_length_ft": ("longueur de crépine", None, "long→propre (inversé)"),
    "depth_missing": ("profondeur indisponible", None, "flag manquant (biais −6pt, modéré)"),
    "dev_intensity": ("intensité de développement urbain NLCD (0-4)", 0.61,
                      "meilleure feature transférable"),
    "lc_developed": ("occupation développée (binaire)", 0.56, "transférable, mineure"),
    "elevation_m": ("altitude du terrain (DEM 3DEP)", 0.57,
                    "transférable propre ; redondant +0,40 avec depth_to_water"),
    "topo_missing": ("élévation indisponible", None, "flag manquant (côtier/bordure)"),
}


def main() -> None:
    base = pd.read_csv(PROC / "well_hydraulic_gradient.csv")[["gm_well_id", "latitude", "longitude"]]
    out = base.copy()
    for fname, cols in SOURCES.items():
        df = pd.read_csv(PROC / fname)
        keep = ["gm_well_id"] + [c for c in cols if c in df.columns]
        out = out.merge(df[keep], on="gm_well_id", how="left")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"⚠️ {fname}: colonnes absentes {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    out.to_csv(OUT.with_suffix(".csv"), index=False)

    # dictionnaire de données
    feat_cols = [c for c in out.columns if c not in ("gm_well_id", "latitude", "longitude")]
    dd = pd.DataFrame([
        {"column": c, "mecanisme": DICT.get(c, ("", None, ""))[0],
         "auc_intra_bloc": DICT.get(c, ("", None, ""))[1],
         "note": DICT.get(c, ("", None, ""))[2],
         "pct_non_na": round(100 * out[c].notna().mean(), 1)}
        for c in feat_cols
    ])
    dd.to_csv(PROC / "well_features_extra_dict.csv", index=False)

    print(f"écrit {OUT} ({out.shape[0]} puits × {len(feat_cols)} features)")
    print("\nfeatures + couverture :")
    print(dd[["column", "auc_intra_bloc", "pct_non_na", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
