"""
Extraction des variables GLDAS NOAH025 Monthly v2.1 pour la Californie.

Pour chaque fichier .nc4 (un par mois), extrait la grille CA (32-42°N, 124-114°W)
et sauvegarde un parquet agrégé : (year, month, lat, lon) × variables hydrologiques.

Variables extraites :
  Rainf_f_tavg          → rainfall_mm_month      (kg m-2 s-1 → mm/mois)
  Evap_tavg             → et_mm_month            (évapotranspiration totale)
  Qs_acc                → runoff_mm              (surface runoff, kg m-2 ≈ mm)
  SoilMoi0_10cm_inst    → soil_moi_0_10_kg_m2
  SoilMoi10_40cm_inst   → soil_moi_10_40_kg_m2
  SoilMoi40_100cm_inst  → soil_moi_40_100_kg_m2  (NEW)
  SoilMoi100_200cm_inst → soil_moi_100_200_kg_m2 (NEW)
  RootMoist_inst        → root_zone_moist_kg_m2
  Tair_f_inst           → temp_c                 (K → °C, NEW)
  SWE_inst              → snowpack_mm            (kg m-2 ≈ mm, NEW)

Usage :
    python -m src.process_gldas
"""

from __future__ import annotations

import logging
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLDAS_DIR = PROJECT_ROOT / "data" / "raw" / "environment" / "nasa_gldas"
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "environment" / "gldas_ca_monthly.parquet"

# Bounding box Californie avec marge
LAT_MIN, LAT_MAX = 32.0, 42.5
LON_MIN, LON_MAX = -124.5, -113.5

# Secondes dans un mois moyen (30.44 j × 86400 s)
SECONDS_PER_MONTH = 30.44 * 86400

# Variables à extraire → (nom_sortie, facteur_multiplication, offset_additif)
# offset = -273.15 pour convertir K → °C, 0 sinon
VARIABLES: dict[str, tuple[str, float, float]] = {
    "Rainf_f_tavg":           ("rainfall_mm_month",       SECONDS_PER_MONTH, 0.0),
    "Evap_tavg":              ("et_mm_month",              SECONDS_PER_MONTH, 0.0),
    "Qs_acc":                 ("runoff_mm",                1.0,               0.0),
    "SoilMoi0_10cm_inst":     ("soil_moi_0_10_kg_m2",     1.0,               0.0),
    "SoilMoi10_40cm_inst":    ("soil_moi_10_40_kg_m2",    1.0,               0.0),
    "SoilMoi40_100cm_inst":   ("soil_moi_40_100_kg_m2",   1.0,               0.0),
    "SoilMoi100_200cm_inst":  ("soil_moi_100_200_kg_m2",  1.0,               0.0),
    "RootMoist_inst":         ("root_zone_moist_kg_m2",   1.0,               0.0),
    "Tair_f_inst":            ("temp_c",                   1.0,            -273.15),  # K → °C
    "SWE_inst":               ("snowpack_mm",              1.0,               0.0),
}


def _parse_ym(fname: str) -> tuple[int, int]:
    """GLDAS_NOAH025_M.AYYYYMM.021.nc4 → (year, month)"""
    part = fname.split(".A")[1][:6]
    return int(part[:4]), int(part[4:6])


def process_all() -> pd.DataFrame:
    nc4_files = sorted(GLDAS_DIR.glob("GLDAS_NOAH025_M.A*.nc4"))
    if not nc4_files:
        raise FileNotFoundError(f"Aucun fichier .nc4 dans {GLDAS_DIR}")
    logger.info("%d fichiers GLDAS à traiter", len(nc4_files))

    records: list[dict] = []

    for path in nc4_files:
        year, month = _parse_ym(path.name)
        try:
            ds = nc.Dataset(path)

            # Masque géographique (fait une seule fois sur le premier fichier)
            lats = ds.variables["lat"][:]
            lons = ds.variables["lon"][:]
            lat_mask = (lats >= LAT_MIN) & (lats <= LAT_MAX)
            lon_mask = (lons >= LON_MIN) & (lons <= LON_MAX)

            grid_lats = lats[lat_mask]
            grid_lons = lons[lon_mask]
            lat_idx = np.where(lat_mask)[0]
            lon_idx = np.where(lon_mask)[0]

            # Extraire chaque variable dans la bounding box CA
            slices: dict[str, np.ndarray] = {}
            for nc_var, (out_name, factor, offset) in VARIABLES.items():
                if nc_var not in ds.variables:
                    continue
                arr = ds.variables[nc_var][0, lat_idx[0]:lat_idx[-1]+1,
                                                 lon_idx[0]:lon_idx[-1]+1]
                arr = np.ma.filled(arr.astype(float), np.nan)
                slices[out_name] = arr * factor + offset

            ds.close()

            # Construire les lignes (lat, lon) × variables
            ny, nx = len(grid_lats), len(grid_lons)
            for i in range(ny):
                for j in range(nx):
                    row: dict = {"year": year, "month": month,
                                 "lat": float(grid_lats[i]),
                                 "lon": float(grid_lons[j])}
                    for out_name, arr in slices.items():
                        val = arr[i, j]
                        row[out_name] = float(val) if not np.isnan(val) else np.nan
                    records.append(row)

        except Exception as e:
            logger.warning("  erreur %s : %s", path.name, e)
            continue

        logger.info("  %d-%02d traité", year, month)

    if not records:
        logger.error("Aucun enregistrement extrait")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Arrondir lat/lon à la précision GLDAS (0.25°)
    df["lat"] = df["lat"].round(3)
    df["lon"] = df["lon"].round(3)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Sauvegardé : %s (%d lignes, %d mois × %d cellules CA)",
                OUT_PATH.name, len(df), df[["year", "month"]].drop_duplicates().__len__(),
                df[["lat", "lon"]].drop_duplicates().__len__())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    process_all()
