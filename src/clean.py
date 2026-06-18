"""
Phase 2 — Nettoyage et prétraitement (pandas).

Règles (article + SI) :
  - Supprimer les colonnes avec >90 % de NA
  - Filtrer les 2 % outliers sur les concentrations PFAS
  - Retenir un échantillon par (well_id, collection_date)
  - Imputer les non-détectés à RL/2
  - Encoder les variables catégorielles

Sorties :
  - data/processed/wells_pfas_long.parquet   — long format (mesures)
  - data/processed/wells_pfas_clean.parquet  — wide format prêt pour merge
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NA_COLUMN_THRESHOLD = 0.90
PFAS_OUTLIER_QUANTILE = 0.98

# VVL code → nom canonique
PFAS_CANONICAL: dict[str, str] = {
    # PFCA (acides carboxyliques)
    "PFBTA": "PFBA",
    "PFPA": "PFPeA",
    "PFHA": "PFHxA",
    "PFHPA": "PFHpA",
    "PFOA": "PFOA",
    "PFNA": "PFNA",
    "PFNDCA": "PFDA",
    "PFUNDCA": "PFUnDA",
    "PFDOA": "PFDoDA",
    "PFTRIDA": "PFTrDA",
    "PFTEDA": "PFTeDA",
    # PFSA (acides sulfoniques)
    "PFBSA": "PFBS",
    "PFPES": "PFPeS",
    "PFHXSA": "PFHxS",
    "PFHPSA": "PFHpS",
    "PFOS": "PFOS",
    "PFNS": "PFNS",
    "PFDSA": "PFDS",
    # FTS (fluorotelomers)
    "4:2FTS": "FTS_4_2",
    "6:2FTS": "FTS_6_2",
    "8:2FTS": "FTS_8_2",
    "10:2FTS": "FTS_10_2",
    # FTCA
    "3:3FTCA": "FTCA_3_3",
    "5:3FTCA": "FTCA_5_3",
    "7:3FTCA": "FTCA_7_3",
    # FOSA / sulfonamides
    "MEFOSA": "MeFOSA",
    "ETFOSA": "EtFOSA",
    "PFOSA": "PFOSAm",
    "NETFOSAA": "NEtFOSAA",
    "NMEFOSAA": "NMeFOSAA",
    "MEFOSE": "MeFOSE",
    "ETFOSE": "EtFOSE",
    # Ether PFAS
    "HFPA-DA": "HFPO_DA",
    "ADONA": "ADONA",
    "9ClPF3ONS": "F53B_major",
    "11ClPF3OUDS": "F53B_minor",
    "PFEESA": "PFEESA",
    "PFMBA": "PFMBA",
    "PFMPA": "PFMPA",
    "NFDHA": "NFDHA",
    # PFCA longs
    "PFODA": "PFOdDA",
    "PFHXDA": "PFHxDA",
}

# Modifieurs → détecté / non-détecté
DETECTED_MODIFIERS = frozenset({"=", "J", ">"})
NON_DETECT_MODIFIERS = frozenset({"<", "ND", "DN"})


def _load_pfas_long(pfas_path: Path, allwells_path: Path) -> pd.DataFrame:
    """Charge pfas.csv + métadonnées puits ; retourne le format long nettoyé."""
    logger.info("Lecture %s…", pfas_path.name)
    df = pd.read_csv(
        pfas_path,
        encoding="latin-1",
        low_memory=False,
        usecols=[
            "gm_dataset_name", "gm_well_id", "gm_chemical_vvl",
            "gm_result_modifier", "gm_result", "gm_reporting_limit",
            "gm_samp_collection_date", "gm_latitude", "gm_longitude",
        ],
        dtype={"gm_result": str, "gm_reporting_limit": str},
    )

    # Garder uniquement les PFAS connus
    df = df[df["gm_chemical_vvl"].isin(PFAS_CANONICAL)].copy()
    df["pfas"] = df["gm_chemical_vvl"].map(PFAS_CANONICAL)

    # Nettoyage numérique
    df["gm_result"] = pd.to_numeric(df["gm_result"], errors="coerce")
    df["gm_reporting_limit"] = pd.to_numeric(df["gm_reporting_limit"], errors="coerce")

    # Flags détection
    df["detected"] = df["gm_result_modifier"].isin(DETECTED_MODIFIERS)

    # Concentration : valeur mesurée si détecté, sinon RL/2
    df["concentration_ngL"] = np.where(
        df["detected"],
        df["gm_result"],
        df["gm_reporting_limit"] / 2.0,
    )
    # Valeur brute (NaN si non-détecté)
    df["concentration_raw_ngL"] = np.where(df["detected"], df["gm_result"], np.nan)

    df["collection_date"] = pd.to_datetime(df["gm_samp_collection_date"], errors="coerce")
    df = df.dropna(subset=["gm_well_id", "collection_date"])

    # Métadonnées puits
    logger.info("Lecture %s…", allwells_path.name)
    wells = pd.read_csv(
        allwells_path,
        encoding="latin-1",
        low_memory=False,
        usecols=[
            "gm_well_id", "gm_latitude", "gm_longitude",
            "gm_well_depth_ft", "gm_gis_county", "gm_gis_dwr_basin",
            "gm_gis_regional_board", "gm_gis_dwr_region",
        ],
    ).drop_duplicates("gm_well_id")

    # Préférer lat/lon de allwells si disponible, sinon pfas.csv
    df = df.drop(columns=["gm_latitude", "gm_longitude"], errors="ignore")
    df = df.merge(
        wells.rename(columns={
            "gm_latitude": "latitude",
            "gm_longitude": "longitude",
            "gm_well_depth_ft": "well_depth_ft",
            "gm_gis_county": "county",
            "gm_gis_dwr_basin": "dwr_basin",
            "gm_gis_regional_board": "regional_board",
            "gm_gis_dwr_region": "dwr_region",
        }),
        on="gm_well_id",
        how="left",
    )

    return df


def _pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long → wide : une ligne par (well_id, collection_date)."""
    logger.info("Pivot wide…")

    key_cols = ["gm_well_id", "collection_date", "gm_dataset_name",
                "latitude", "longitude", "well_depth_ft",
                "county", "dwr_basin", "regional_board", "dwr_region"]

    # Conserver une seule mesure par (well, date, pfas) — prendre la max si dupliqué
    agg = (
        long_df
        .groupby(["gm_well_id", "collection_date", "pfas"], as_index=False)
        .agg(
            concentration_ngL=("concentration_ngL", "max"),
            concentration_raw_ngL=("concentration_raw_ngL", "max"),
            detected=("detected", "any"),
        )
    )

    # Pivot concentration imputée (ngL)
    conc_wide = agg.pivot_table(
        index=["gm_well_id", "collection_date"],
        columns="pfas",
        values="concentration_ngL",
        aggfunc="max",
    )
    conc_wide.columns = [f"{c}_ngL" for c in conc_wide.columns]

    # Pivot flag détection
    det_wide = agg.pivot_table(
        index=["gm_well_id", "collection_date"],
        columns="pfas",
        values="detected",
        aggfunc="any",
    ).fillna(False)
    det_wide.columns = [f"{c}_detected" for c in det_wide.columns]

    wide = pd.concat([conc_wide, det_wide], axis=1).reset_index()

    # Ré-associer les métadonnées puits (dédupliquées par well+date)
    meta = (
        long_df[key_cols]
        .drop_duplicates(["gm_well_id", "collection_date"])
    )
    wide = wide.merge(meta, on=["gm_well_id", "collection_date"], how="left")

    return wide


def drop_sparse_columns(df: pd.DataFrame, threshold: float = NA_COLUMN_THRESHOLD) -> pd.DataFrame:
    """Supprime les colonnes dont la proportion de NA dépasse le seuil.

    Travaille sur _ngL comme référence (NA = PFAS jamais mesuré pour ce
    puits/échantillon). Droppe la paire _ngL + _detected correspondante.
    """
    ngL_cols = [c for c in df.columns if c.endswith("_ngL")]
    na_frac = df[ngL_cols].isna().mean()
    sparse_pfas = [c.replace("_ngL", "") for c in na_frac[na_frac > threshold].index]
    drop = []
    for pfas in sparse_pfas:
        for suffix in ("_ngL", "_detected"):
            col = f"{pfas}{suffix}"
            if col in df.columns:
                drop.append(col)
    if drop:
        logger.info(
            "Drop %d colonnes sparse (%d PFAS, NA > %.0f%%): %s",
            len(drop), len(sparse_pfas), threshold * 100, sparse_pfas,
        )
    return df.drop(columns=drop)


def clip_pfas_outliers(df: pd.DataFrame, q: float = PFAS_OUTLIER_QUANTILE) -> pd.DataFrame:
    """Plafonne les concentrations PFAS au quantile q (top 2 % par défaut)."""
    ngL_cols = [c for c in df.columns if c.endswith("_ngL")]
    for col in ngL_cols:
        cap = df[col].quantile(q)
        df[col] = df[col].clip(upper=cap)
    return df


def main() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pfas_path = RAW_DIR / "gama" / "pfas.csv"
    allwells_path = RAW_DIR / "gama" / "gama_allwells.csv"

    long_df = _load_pfas_long(pfas_path, allwells_path)
    logger.info("Long format : %d lignes, %d puits uniques", len(long_df), long_df["gm_well_id"].nunique())

    # Sauvegarde long format
    long_out = PROCESSED_DIR / "wells_pfas_long.parquet"
    long_df.to_parquet(long_out, index=False)
    logger.info("Sauvegardé : %s", long_out)

    # Pivot wide
    wide = _pivot_wide(long_df)
    logger.info("Wide format avant nettoyage : %s", wide.shape)

    # Drop colonnes trop sparse
    wide = drop_sparse_columns(wide)

    # Clip outliers PFAS
    wide = clip_pfas_outliers(wide)

    # Trier par well + date
    wide = wide.sort_values(["gm_well_id", "collection_date"]).reset_index(drop=True)

    out = PROCESSED_DIR / "wells_pfas_clean.parquet"
    wide.to_parquet(out, index=False)
    logger.info("Sauvegardé : %s  (%d lignes × %d colonnes)", out, *wide.shape)
    return wide


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
