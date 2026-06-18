"""
Phase 2 (suite) + Phase 3 — Fusion géospatiale et construction du dataset CA-PFAS-ASGWS.

Étapes :
  1. Cibles ML : sum_pfas > 70 ng/L (binaire), chaque PFAS > 2 ng/L (multilabel),
                 classes 0–3 selon le volume d'observations par analyte.
  2. Bassins SGMA (DWR) — point-in-polygon (geopandas).
  2b. Catégorie du puits (gm_well_category) depuis allwells.
  3. Sources de contamination (GeoTracker PFAS) — KD-Tree à 1/3/10/50 km + type.
  3b. Co-contaminants GAMA (VOCs, nitrate, solvants) — merge_asof ±365 j.
  4. Météo NOAA GHCND — agrégats 30/90/365 j avant la date de collecte.
  5. Qualité de l'air EPA AQS — PM2.5, PM10, NO2, SO2, vent, humidité (annuel).
  6. Hydrologie NASA GLDAS — précip., ET, ruissellement, humidité du sol (mensuel).
  7. Export final : data/processed/CA-PFAS-ASGWS.{parquet,csv}
                    data/processed/data_dict.csv

Usage :
    python -m src.merge
    python -m src.merge --no-weather   # skip NOAA (résultat partiel mais rapide)
    python -m src.merge --no-aqs       # skip AQS
    python -m src.merge --no-gldas     # skip GLDAS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from shapely.geometry import Point

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# ──────────────────────────────────────────────────────────
# Seuils article Dong et al. 2024
# ──────────────────────────────────────────────────────────
TOTAL_PFAS_THRESHOLD_NG_L = 70.0
INDIVIDUAL_PFAS_THRESHOLD_NG_L = 2.0

# Classes multilabel (nombre de détections par analyte dans le dataset)
# 0 : > 25 000 obs  1 : > 5 000  2 : > 1 000  3 : > 100
CLASS_THRESHOLDS = [(25_000, 0), (5_000, 1), (1_000, 2), (100, 3)]

# Rayons KD-Tree (km)
CONTAM_RADII_KM = (1, 3, 10, 50)

# Co-contaminants GAMA (Dong et al. 2024 Table S4 + S9)
CO_CONTAMINANTS = [
    # Originaux (15)
    "NO3N",    # Nitrate, rang 13
    "BTBZS",   # sec-Butylbenzène, rang 83
    "PBZN",    # n-Propylbenzène, rang 79
    "EDB",     # 1,2-Dibromoéthane, rang 34
    "DBCP",    # 1,2-Dibromo-3-chloropropane, rang 48
    "TCPR123", # 1,2,3-Trichloropropane, rang 44
    "TCE",     # Trichloroéthylène, rang 52
    "PCE",     # Tétrachloroéthylène, rang 43
    "MTBE",    # Méthyl tert-butyl éther, rang 54
    "TCA111",  # 1,1,1-Trichloroéthane, rang 97
    "TCA112",  # 1,1,2-Trichloroéthane, rang 91
    "DCE12T",  # trans-1,2-Dichloroéthylène, rang 67
    "VC",      # Chlorure de vinyle, rang 74
    "STY",     # Styrène, rang 98
    "EBZ",     # Éthylbenzène, rang 59
    # Chimie de l'eau (rangs 28, 30, 33, 35, 45)
    "TDS",     # Solides dissous totaux
    "MN",      # Manganèse
    "AS",      # Arsenic
    "SO4",     # Sulfate
    "FE",      # Fer
    # VOCs aromatiques (rangs 53, 58, 60, 61, 68)
    "NAPH",    # Naphtalène, rang 53
    "BZ",      # Benzène, rang 58
    "XYLENES", # Xylènes, rang 60
    "BZME",    # Toluène, rang 61
    "TMB124",  # 1,2,4-Triméthylbenzène, rang 68
    # Chlorinés aliphatiques (rangs 56, 70, 75, 86)
    "DCE12C",  # cis-1,2-Dichloroéthylène, rang 56
    "DCE11",   # 1,1-Dichloroéthylène, rang 70
    "DCA12",   # 1,2-Dichloroéthane, rang 75
    "DCA11",   # 1,1-Dichloroéthane, rang 86
    # Halogénés divers (rangs 71, 72, 81, 84, 85, 87, 88, 89, 93, 94, 95, 99, 100, 101, 104)
    "BTBZN",   # n-Butylbenzène, rang 71
    "FC113",   # Fréon 113, rang 72
    "BDCME",   # Bromodichlorométhane, rang 81
    "TCB124",  # 1,2,4-Trichlorobenzène, rang 84
    "DBCME",   # Dibromochlorométhane, rang 85
    "FC12",    # Fréon 12, rang 87
    "TBME",    # tert-Butyl méthyl éther, rang 88
    "CTCL",    # Tétrachlorure de carbone, rang 89
    "CLBZ",    # Chlorobenzène, rang 93
    "BTBZT",   # tert-Butylbenzène, rang 94
    "FC11",    # Fréon 11, rang 95
    "PCA",     # rang 99
    "DCBZ12",  # 1,2-Dichlorobenzène, rang 100
    "DCBZ13",  # 1,3-Dichlorobenzène, rang 101
    "DCPA12",  # 1,2-Dichloropropane, rang 104
]

# NOAA
API_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
DATASET_GHCND = "GHCND"
DATATYPES = ("PRCP", "TMAX", "TMIN")
MIN_COVERAGE = 0.7
N_WORKERS = 4
REQUEST_TIMEOUT = 60
EARTH_R_KM = 6371.0088


# ══════════════════════════════════════════════════════════
# Utilitaires spatiaux
# ══════════════════════════════════════════════════════════

def latlon_to_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = EARTH_R_KM * np.cos(lat_r) * np.cos(lon_r)
    y = EARTH_R_KM * np.cos(lat_r) * np.sin(lon_r)
    z = EARTH_R_KM * np.sin(lat_r)
    return np.column_stack([x, y, z])


def build_kdtree(lat: np.ndarray, lon: np.ndarray) -> cKDTree:
    return cKDTree(latlon_to_xyz(lat, lon))


def nearest_neighbor(lat: np.ndarray, lon: np.ndarray,
                     tree: cKDTree) -> tuple[np.ndarray, np.ndarray]:
    xyz = latlon_to_xyz(lat, lon)
    dist_xyz, idx = tree.query(xyz)
    dist_km = 2 * EARTH_R_KM * np.arcsin(np.clip(dist_xyz / (2 * EARTH_R_KM), 0, 1))
    return dist_km, idx


def neighbors_within(lat: float, lon: float, tree: cKDTree,
                     radius_km: float) -> np.ndarray:
    xyz = latlon_to_xyz(np.array([lat]), np.array([lon]))
    chord = 2 * EARTH_R_KM * np.sin(radius_km / (2 * EARTH_R_KM))
    return tree.query_ball_point(xyz[0], chord)


# ══════════════════════════════════════════════════════════
# Chargement .env
# ══════════════════════════════════════════════════════════

def _load_env_var(name: str) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return None


# ══════════════════════════════════════════════════════════
# 1. Cibles ML
# ══════════════════════════════════════════════════════════

def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes cibles et l'attribution de classe par analyte."""
    ngL_cols = [c for c in df.columns if c.endswith("_ngL")]
    pfas_names = [c.replace("_ngL", "") for c in ngL_cols]

    # Somme des concentrations (imputées)
    df["sum_pfas_ngL"] = df[ngL_cols].fillna(0).sum(axis=1)

    # Cible binaire article : sum > 70 ng/L
    df["target_sum_gt70"] = (df["sum_pfas_ngL"] > TOTAL_PFAS_THRESHOLD_NG_L).astype(int)

    # Cibles multilabel : chaque PFAS > 2 ng/L
    for p in pfas_names:
        raw_col = f"{p}_ngL"
        df[f"label_{p}"] = (df[raw_col].fillna(0) > INDIVIDUAL_PFAS_THRESHOLD_NG_L).astype(int)

    # Classes multilabel (0–3) selon le volume de détections
    det_cols = [f"{p}_detected" for p in pfas_names if f"{p}_detected" in df.columns]
    n_det = {col.replace("_detected", ""): df[col].sum() for col in det_cols}
    classes: dict[str, int] = {}
    for pfas, count in n_det.items():
        assigned = -1
        for threshold, cls in CLASS_THRESHOLDS:
            if count > threshold:
                assigned = cls
                break
        classes[pfas] = assigned  # -1 = exclu (< 100 détections)

    df["pfas_class_assignment"] = df.apply(
        lambda _: json.dumps({k: v for k, v in classes.items() if v >= 0}),
        axis=1,
    )

    logger.info(
        "Cibles : sum>70=%d (%.1f%%)  PFAS classes: %s",
        df["target_sum_gt70"].sum(),
        100 * df["target_sum_gt70"].mean(),
        {c: v for c, v in classes.items() if v >= 0},
    )
    return df


# ══════════════════════════════════════════════════════════
# 1b. Catégorie du puits
# ══════════════════════════════════════════════════════════

def add_well_category(df: pd.DataFrame, allwells_path: Path) -> pd.DataFrame:
    """Ajoute gm_well_category depuis allwells.csv (Municipal/Domestic/Monitoring…)."""
    logger.info("Well category — lecture allwells…")
    wells = pd.read_csv(
        allwells_path, encoding="latin-1", low_memory=False,
        usecols=["gm_well_id", "gm_well_category"],
    ).drop_duplicates("gm_well_id")
    df = df.merge(wells, on="gm_well_id", how="left")
    n_ok = df["gm_well_category"].notna().sum()
    logger.info("  %d/%d puits avec catégorie", n_ok, len(df))
    return df


# ══════════════════════════════════════════════════════════
# 2. Bassins SGMA
# ══════════════════════════════════════════════════════════

def add_sgma_basins(df: pd.DataFrame, basins_path: Path) -> pd.DataFrame:
    logger.info("SGMA basins — point-in-polygon…")
    basins = gpd.read_file(basins_path)[
        ["Basin_Name", "Basin_Subbasin_Name", "Region_Office", "geometry"]
    ].to_crs("EPSG:4326")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(gdf, basins, how="left", predicate="within")

    df["sgma_basin_name"] = joined["Basin_Name"].values
    df["sgma_subbasin_name"] = joined["Basin_Subbasin_Name"].values
    df["sgma_region_office"] = joined["Region_Office"].values
    in_basin = df["sgma_basin_name"].notna().sum()
    logger.info("  %d/%d points dans un bassin SGMA", in_basin, len(df))
    return df.drop(columns=["geometry"], errors="ignore")


# ══════════════════════════════════════════════════════════
# 3. Sources de contamination (GeoTracker PFAS)
# ══════════════════════════════════════════════════════════

def add_contamination(df: pd.DataFrame, gt_path: Path) -> pd.DataFrame:
    logger.info("Contamination GeoTracker — KD-Tree %s km…", CONTAM_RADII_KM)
    gt = pd.read_csv(gt_path).dropna(subset=["latitude", "longitude"])
    tree = build_kdtree(gt["latitude"].values, gt["longitude"].values)

    dist_km, nearest_idx = nearest_neighbor(
        df["latitude"].values, df["longitude"].values, tree
    )
    df["dist_geotracker_km"] = dist_km.round(3)

    # Type du site PFAS le plus proche (Fac_Conf_type du papier)
    if "site_type" in gt.columns:
        df["nearest_geotracker_type"] = gt["site_type"].values[nearest_idx]

    for r in CONTAM_RADII_KM:
        counts = []
        for lat, lon in zip(df["latitude"].values, df["longitude"].values):
            idx = neighbors_within(lat, lon, tree, r)
            counts.append(len(idx))
        df[f"n_geotracker_within_{r}km"] = counts

    logger.info(
        "  dist médiane nearest site : %.1f km", float(np.median(dist_km))
    )
    return df


# ══════════════════════════════════════════════════════════
# 3b. Co-contaminants GAMA
# ══════════════════════════════════════════════════════════

def add_co_contaminants(df: pd.DataFrame, gama_path: Path) -> pd.DataFrame:
    """
    Fusionne les co-contaminants GAMA (VOCs, nitrate, solvants) avec les puits PFAS.

    Stratégie (article §2.1.2) :
      1. Match exact sur (gm_well_id, date).
      2. Pour les non-matchés : merge_asof ±365 j (mesure la plus proche par puits).
    """
    logger.info("Co-contaminants GAMA — lecture %s…", gama_path.name)
    raw = pd.read_csv(
        gama_path, encoding="latin-1", low_memory=False,
        usecols=[
            "gm_well_id", "gm_chemical_vvl", "gm_result_modifier",
            "gm_result", "gm_reporting_limit", "gm_samp_collection_date",
        ],
        dtype={"gm_result": str, "gm_reporting_limit": str},
    )
    raw = raw[raw["gm_chemical_vvl"].isin(CO_CONTAMINANTS)].copy()
    if raw.empty:
        logger.warning("  aucun co-contaminant trouvé dans %s", gama_path.name)
        return df

    raw["gm_result"] = pd.to_numeric(raw["gm_result"], errors="coerce")
    raw["gm_reporting_limit"] = pd.to_numeric(raw["gm_reporting_limit"], errors="coerce")
    detected = raw["gm_result_modifier"].isin({"=", "J", ">"})
    raw["value"] = np.where(detected, raw["gm_result"], raw["gm_reporting_limit"] / 2.0)
    raw["date"] = pd.to_datetime(raw["gm_samp_collection_date"], errors="coerce")
    raw = raw.dropna(subset=["gm_well_id", "date", "value"])

    # Pivot : max par (well, date, chemical)
    pivot = (
        raw.groupby(["gm_well_id", "date", "gm_chemical_vvl"])["value"]
        .max()
        .reset_index()
        .pivot_table(
            index=["gm_well_id", "date"],
            columns="gm_chemical_vvl",
            values="value",
        )
        .reset_index()
    )
    pivot.columns.name = None
    chem_cols = [c for c in CO_CONTAMINANTS if c in pivot.columns]
    pivot = pivot.rename(columns={c: f"cocontam_{c.lower()}" for c in chem_cols})
    cocontam_cols = [f"cocontam_{c.lower()}" for c in chem_cols]

    logger.info("  %d mesures, %d puits, %d chemicals", len(pivot), pivot["gm_well_id"].nunique(), len(chem_cols))

    # ── Étape A : merge_asof exact par puits ±365j ───────────────────────────
    df = df.copy()
    df["collection_date"] = pd.to_datetime(df["collection_date"])
    df["_sort_key"] = np.arange(len(df))
    df_sorted = df.sort_values("collection_date").reset_index(drop=True)
    pivot_sorted = pivot.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        df_sorted,
        pivot_sorted[["gm_well_id", "date"] + cocontam_cols],
        left_on="collection_date",
        right_on="date",
        by="gm_well_id",
        tolerance=pd.Timedelta(days=365),
        direction="nearest",
    ).drop(columns=["date"], errors="ignore")
    merged = merged.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    n_direct = merged[cocontam_cols].notna().any(axis=1).sum()
    logger.info("  Match direct (par puits) : %d/%d lignes", n_direct, len(merged))

    # ── Étape B : fallback spatial (article §2.1.2 : moyenne ≤50 km) ─────────
    # Moyenne historique par puits de mesure → KD-Tree → affecter aux puits sans données
    well_avg = (
        pivot.drop(columns=["date"])
        .groupby("gm_well_id")[cocontam_cols]
        .mean()
        .reset_index()
    )
    # Récupérer les coordonnées des puits de mesure depuis allwells
    aw_path = RAW_DIR / "gama" / "gama_allwells.csv"
    if aw_path.exists():
        aw = pd.read_csv(aw_path, encoding="latin-1", low_memory=False,
                         usecols=["gm_well_id", "gm_latitude", "gm_longitude"]
                         ).drop_duplicates("gm_well_id").dropna()
        well_avg = well_avg.merge(
            aw.rename(columns={"gm_latitude": "lat_src", "gm_longitude": "lon_src"}),
            on="gm_well_id", how="inner",
        )
        if len(well_avg) > 0:
            src_tree = build_kdtree(
                well_avg["lat_src"].values.astype(float),
                well_avg["lon_src"].values.astype(float),
            )
            needs_fallback = merged[cocontam_cols].isna().all(axis=1)
            if needs_fallback.any():
                fb_lat = merged.loc[needs_fallback, "latitude"].values
                fb_lon = merged.loc[needs_fallback, "longitude"].values
                dist_km_fb, idx_fb = nearest_neighbor(fb_lat, fb_lon, src_tree)
                # N'utiliser le fallback que si la source est dans 50 km
                for i, (orig_idx, d, src_i) in enumerate(
                    zip(merged.index[needs_fallback], dist_km_fb, idx_fb)
                ):
                    if d <= 50.0:
                        for col in cocontam_cols:
                            merged.at[orig_idx, col] = well_avg[col].iloc[src_i]
                n_fallback = merged[cocontam_cols].notna().any(axis=1).sum() - n_direct
                logger.info("  Fallback spatial (≤50 km)   : +%d lignes", n_fallback)

    any_enriched = merged[cocontam_cols].notna().any(axis=1).sum()
    logger.info("  Total co-contaminants : %d/%d lignes (%.1f%%)",
                any_enriched, len(merged), 100 * any_enriched / len(merged))
    return merged


# ══════════════════════════════════════════════════════════
# 4. Météo NOAA GHCND
# ══════════════════════════════════════════════════════════

def _make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"token": token, "User-Agent": "CA-PFAS-ASGWS/0.1"})
    return s


def _fetch_stations_ca(session: requests.Session, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / "stations_ca.json"
    if cache.exists():
        logger.info("  [cache] stations CA")
        return pd.DataFrame(json.loads(cache.read_text()))

    all_rows: list[dict] = []
    offset = 1
    while True:
        params = {
            "datasetid": DATASET_GHCND,
            "locationid": "FIPS:06",
            "startdate": "2015-01-01",
            "enddate": "2024-12-31",
            "limit": 1000,
            "offset": offset,
        }
        r = session.get(f"{API_BASE}/stations", params=params, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        all_rows.extend(results)
        total = data["metadata"]["resultset"]["count"]
        logger.info("  [stations] offset=%d +%d (total=%d)", offset, len(results), total)
        offset += 1000
        if offset > total:
            break
        time.sleep(0.25)

    df = pd.DataFrame(all_rows)
    df = df[df["datacoverage"] >= MIN_COVERAGE].copy()
    df = df[df["maxdate"] >= "2015-01-01"].copy()
    df = df.rename(columns={"id": "station_id"})
    cache.write_text(df.to_json(orient="records"))
    logger.info("  %d stations CA sauvegardées", len(df))
    return df


def _fetch_station_year(session: requests.Session, station_id: str,
                        year: int, datatype: str, cache_dir: Path) -> pd.DataFrame:
    fname = f"{station_id.replace(':', '_')}_{year}_{datatype}.json"
    cache = cache_dir / fname
    if cache.exists():
        return pd.DataFrame(json.loads(cache.read_text()))
    params = {
        "datasetid": DATASET_GHCND,
        "stationid": station_id,
        "datatypeid": datatype,
        "startdate": f"{year}-01-01",
        "enddate": f"{year}-12-31",
        "limit": 1000,
        "units": "metric",
    }
    for attempt in range(3):
        try:
            r = session.get(f"{API_BASE}/data", params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            results = r.json().get("results", [])
            df = pd.DataFrame(results)
            cache.write_text(df.to_json(orient="records"))
            return df
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    cache.write_text("[]")
    return pd.DataFrame()


def _aggregate_window(daily: pd.DataFrame, end_date: pd.Timestamp,
                      days: int, fn: str) -> float:
    if daily.empty:
        return np.nan
    start = end_date - pd.Timedelta(days=days)
    sub = daily.loc[(daily["date"] > start) & (daily["date"] <= end_date), "value"]
    return float(getattr(sub, fn)()) if not sub.empty else np.nan


def add_noaa_weather(df: pd.DataFrame, cache_dir: Path, token: str) -> pd.DataFrame:
    noaa_cache = cache_dir / "noaa"
    noaa_cache.mkdir(parents=True, exist_ok=True)
    session = _make_session(token)

    logger.info("NOAA — récupération des stations CA…")
    stations = _fetch_stations_ca(session, noaa_cache)
    if stations.empty:
        logger.warning("  aucune station — skip météo")
        return df
    logger.info("  %d stations disponibles", len(stations))

    tree = build_kdtree(stations["latitude"].values, stations["longitude"].values)
    dist_km, idx = nearest_neighbor(df["latitude"].values, df["longitude"].values, tree)
    df["weather_station_id"] = stations["station_id"].values[idx]
    df["weather_station_dist_km"] = dist_km.round(2)
    df["weather_data_coverage"] = stations["datacoverage"].values[idx]

    # Ensemble des (station_id, année) à télécharger
    needed: set[tuple[str, int, str]] = set()
    for _, row in df[["weather_station_id", "collection_date"]].dropna().iterrows():
        yr = int(row["collection_date"].year)
        for y in (yr - 1, yr):
            for dt in DATATYPES:
                needed.add((row["weather_station_id"], y, dt))

    logger.info("  %d (station, année, datatype) à récupérer…", len(needed))

    daily_store: dict[str, list[pd.DataFrame]] = {}
    done = 0
    t0 = time.time()

    def task(sid: str, yr: int, dt: str) -> tuple[str, str, pd.DataFrame]:
        return sid, dt, _fetch_station_year(session, sid, yr, dt, noaa_cache)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(task, sid, yr, dt) for sid, yr, dt in needed]
        for fut in as_completed(futures):
            sid, dt, sub = fut.result()
            daily_store.setdefault(sid, []).append(sub)
            done += 1
            if done % 200 == 0:
                rate = done / (time.time() - t0 + 1e-9)
                logger.info("    %d/%d (%.1f req/s)", done, len(needed), rate)

    logger.info("  Calcul des agrégats météo…")
    rows: list[dict[str, Any]] = []
    for _, row in df[["gm_well_id", "collection_date", "weather_station_id"]].iterrows():
        sid = row["weather_station_id"]
        d = row["collection_date"]
        if pd.isna(d) or not sid or sid not in daily_store:
            rows.append({"gm_well_id": row["gm_well_id"], "collection_date": d})
            continue
        parts = [p for p in daily_store[sid] if not p.empty]
        if not parts:
            rows.append({"gm_well_id": row["gm_well_id"], "collection_date": d})
            continue
        big = pd.concat(parts, ignore_index=True)
        big["date"] = pd.to_datetime(big["date"], errors="coerce")
        prcp = big[big["datatype"] == "PRCP"].dropna(subset=["date"])
        tmax = big[big["datatype"] == "TMAX"].dropna(subset=["date"])
        tmin = big[big["datatype"] == "TMIN"].dropna(subset=["date"])
        out: dict[str, Any] = {"gm_well_id": row["gm_well_id"], "collection_date": d}
        out["prcp_30d_sum_mm"] = _aggregate_window(prcp, d, 30, "sum")
        out["prcp_90d_sum_mm"] = _aggregate_window(prcp, d, 90, "sum")
        out["prcp_365d_sum_mm"] = _aggregate_window(prcp, d, 365, "sum")
        out["prcp_30d_max_daily_mm"] = _aggregate_window(prcp, d, 30, "max")
        if not prcp.empty:
            w = prcp[(prcp["date"] > d - pd.Timedelta(days=365)) & (prcp["date"] <= d)]
            out["prcp_365d_n_wet_days"] = int((w["value"] > 0).sum())
        else:
            out["prcp_365d_n_wet_days"] = np.nan
        out["tmax_365d_mean_c"] = _aggregate_window(tmax, d, 365, "mean")
        out["tmin_365d_mean_c"] = _aggregate_window(tmin, d, 365, "mean")
        rows.append(out)

    weather_df = pd.DataFrame(rows)
    df = df.merge(weather_df, on=["gm_well_id", "collection_date"], how="left")
    n_ok = df["prcp_365d_sum_mm"].notna().sum()
    logger.info("  météo enrichie : %d/%d lignes avec données", n_ok, len(df))
    return df


# ══════════════════════════════════════════════════════════
# 5. Sol SSURGO (USDA SDA)
# ══════════════════════════════════════════════════════════

_SSURGO_COLS = [
    # Fractions totales
    "soil_sand_pct", "soil_clay_pct", "soil_silt_pct", "soil_om_pct",
    "soil_ph", "soil_ksat_um_s", "soil_awc_cm_cm", "soil_bulk_density",
    # Sous-fractions sable (rangs 6, 36, 38, 41)
    "soil_sand_vfine_pct", "soil_sand_fine_pct", "soil_sand_medium_pct",
    "soil_sand_coarse_pct", "soil_sand_vcoarse_pct",
    # Limons (rangs 9, 46)
    "soil_silt_coarse_pct", "soil_silt_fine_pct",
    # Rétention eau (rang 14)
    "soil_water_1bar_pct", "soil_water_15bar_pct",
    # Texture USDA (rangs 22, 31)
    "soil_texture_class",
]


def _compute_gradation(s: pd.Series) -> tuple[float, float]:
    """
    Calcule Cu=D60/D10 et Cc=D30²/(D10×D60) par interpolation log-linéaire
    sur la distribution granulométrique SSURGO (0.002-2mm).
    """
    clay = s.get("soil_clay_pct", np.nan)
    silt_c = s.get("soil_silt_coarse_pct", np.nan)
    silt_f = s.get("soil_silt_fine_pct", np.nan)
    svf = s.get("soil_sand_vfine_pct", np.nan)
    sf = s.get("soil_sand_fine_pct", np.nan)
    sm = s.get("soil_sand_medium_pct", np.nan)
    sc = s.get("soil_sand_coarse_pct", np.nan)
    svc = s.get("soil_sand_vcoarse_pct", np.nan)

    fracs = [clay, silt_f, silt_c, svf, sf, sm, sc, svc]
    if any(np.isnan(f) for f in fracs):
        return np.nan, np.nan

    # Bornes (mm) et cumul passant
    sizes = np.array([0.002, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0])
    cumul = np.cumsum(fracs)

    def interp_d(pct: float) -> float:
        for k in range(len(cumul) - 1):
            if cumul[k] <= pct <= cumul[k + 1] and cumul[k + 1] > cumul[k]:
                t = (pct - cumul[k]) / (cumul[k + 1] - cumul[k])
                return float(np.exp(np.log(sizes[k]) + t * (np.log(sizes[k + 1]) - np.log(sizes[k]))))
        return np.nan

    d10 = interp_d(10.0)
    d30 = interp_d(30.0)
    d60 = interp_d(60.0)
    if any(np.isnan(x) or x <= 0 for x in [d10, d30, d60]):
        return np.nan, np.nan
    return d60 / d10, d30 ** 2 / (d10 * d60)


def add_ssurgo_soil(df: pd.DataFrame, ssurgo_parquet: Path) -> pd.DataFrame:
    """Joint les propriétés SSURGO et dérive les variables granulométriques du papier."""
    if not ssurgo_parquet.exists():
        logger.warning(
            "ssurgo_ca_points.parquet absent — skip sol (lance src.collect_ssurgo)"
        )
        return df

    logger.info("SSURGO sol — lecture %s…", ssurgo_parquet.name)
    soil = pd.read_parquet(ssurgo_parquet)

    available = [c for c in _SSURGO_COLS if c in soil.columns]
    df = df.copy()
    df["lat_r"] = df["latitude"].round(4)
    df["lon_r"] = df["longitude"].round(4)
    df = df.merge(soil[["lat_r", "lon_r"] + available], on=["lat_r", "lon_r"], how="left")
    df = df.drop(columns=["lat_r", "lon_r"])

    # Variables dérivées
    if "soil_water_15bar_pct" in df.columns and "soil_clay_pct" in df.columns:
        df["soil_ratio_water_clay"] = df["soil_water_15bar_pct"] / df["soil_clay_pct"].replace(0, np.nan)

    grad_cols = ["soil_clay_pct", "soil_silt_fine_pct", "soil_silt_coarse_pct",
                 "soil_sand_vfine_pct", "soil_sand_fine_pct", "soil_sand_medium_pct",
                 "soil_sand_coarse_pct", "soil_sand_vcoarse_pct"]
    if all(c in df.columns for c in grad_cols):
        grad = df[grad_cols].apply(_compute_gradation, axis=1, result_type="expand")
        df["soil_gradation_uniformity"] = grad[0]
        df["soil_gradation_curvature"] = grad[1]

    n_ok = df["soil_sand_pct"].notna().sum()
    logger.info("  SSURGO enrichi : %d/%d lignes (%.1f%%)", n_ok, len(df), 100 * n_ok / len(df))
    return df


# ══════════════════════════════════════════════════════════
# 7. Qualité de l'air EPA AQS
# ══════════════════════════════════════════════════════════

# Param → nom de colonne final
_AQS_PARAMS = {
    "88101": "aqs_pm25_ugm3",
    "81102": "aqs_pm10_ugm3",
    "42602": "aqs_no2_ppb",
    "42401": "aqs_so2_ppb",
    "61101": "aqs_wind_ms",
    "62201": "aqs_humidity_pct",
    "44201": "aqs_ozone_ppb",   # rang 24 Table S9
    "42101": "aqs_co_ppm",      # rang 50 Table S9
}


def add_aqs_air_quality(df: pd.DataFrame, aqs_dir: Path) -> pd.DataFrame:
    """
    Pour chaque puits × année, affecte les valeurs annuelles du moniteur AQS le plus proche
    (≤ 50 km). Variables : PM2.5, PM10, NO2, SO2, vent, humidité.
    """
    aqs_parquet = aqs_dir / "aqs_ca_annual.parquet"
    if not aqs_parquet.exists():
        logger.warning("aqs_ca_annual.parquet absent — skip AQS (lance src.collect_aqs)")
        return df

    logger.info("AQS air quality — lecture %s…", aqs_parquet.name)
    aqs = pd.read_parquet(aqs_parquet)
    aqs = aqs.dropna(subset=["latitude", "longitude", "annual_mean"])

    # Mapper param_code → col_name
    aqs["col_name"] = aqs["param_code"].map(_AQS_PARAMS)
    aqs = aqs.dropna(subset=["col_name"])

    # Initialiser les colonnes
    for col in _AQS_PARAMS.values():
        if col not in df.columns:
            df[col] = np.nan

    df["collection_date"] = pd.to_datetime(df["collection_date"])
    df["_year"] = df["collection_date"].dt.year

    years = df["_year"].dropna().unique().astype(int)

    for year in sorted(years):
        aqs_yr = aqs[aqs["year"] == year]
        if aqs_yr.empty:
            continue

        mask_yr = df["_year"] == year
        if not mask_yr.any():
            continue

        for param, col_name in _AQS_PARAMS.items():
            sub = aqs_yr[aqs_yr["param_code"] == param].drop_duplicates(
                subset=["latitude", "longitude"]
            )
            if sub.empty:
                continue

            mon_tree = build_kdtree(sub["latitude"].values, sub["longitude"].values)
            well_lat = df.loc[mask_yr, "latitude"].values
            well_lon = df.loc[mask_yr, "longitude"].values
            dist_km, idx = nearest_neighbor(well_lat, well_lon, mon_tree)

            vals = sub["annual_mean"].values[idx]
            vals[dist_km > 50.0] = np.nan  # ignorer si > 50 km
            df.loc[mask_yr, col_name] = vals

    df = df.drop(columns=["_year"])
    n_ok = df["aqs_pm25_ugm3"].notna().sum()
    logger.info("  AQS enrichi : %d/%d lignes avec PM2.5", n_ok, len(df))
    return df


# ══════════════════════════════════════════════════════════
# 8. Hydrologie NASA GLDAS
# ══════════════════════════════════════════════════════════

_GLDAS_COLS = [
    "rainfall_mm_month", "et_mm_month", "runoff_mm",
    "soil_moi_0_10_kg_m2", "soil_moi_10_40_kg_m2",
    "soil_moi_40_100_kg_m2", "soil_moi_100_200_kg_m2",  # NEW
    "root_zone_moist_kg_m2",
    "temp_c",       # NEW: température air (rang 25)
    "snowpack_mm",  # NEW: équivalent eau neige
]


def add_gldas_hydrology(df: pd.DataFrame, gldas_parquet: Path) -> pd.DataFrame:
    """
    Pour chaque puits × mois de collecte, extrait les valeurs GLDAS de la cellule 0.25°
    la plus proche. Variables : précip., ET, ruissellement, humidité du sol.
    """
    if not gldas_parquet.exists():
        logger.warning(
            "gldas_ca_monthly.parquet absent — skip GLDAS (lance src.process_gldas)"
        )
        return df

    logger.info("GLDAS hydrology — lecture %s…", gldas_parquet.name)
    gldas = pd.read_parquet(gldas_parquet)

    # Initialiser les colonnes
    for col in _GLDAS_COLS:
        if col not in df.columns:
            df[col] = np.nan
    if "gldas_dist_km" not in df.columns:
        df["gldas_dist_km"] = np.nan

    df["collection_date"] = pd.to_datetime(df["collection_date"])
    df["_ym"] = df["collection_date"].dt.to_period("M")

    # Construire un KD-Tree par (year, month)
    for ym, grp_df in df.groupby("_ym"):
        year, month = ym.year, ym.month
        gldas_ym = gldas[(gldas["year"] == year) & (gldas["month"] == month)]
        if gldas_ym.empty:
            continue

        g_tree = build_kdtree(gldas_ym["lat"].values, gldas_ym["lon"].values)
        well_lat = grp_df["latitude"].values
        well_lon = grp_df["longitude"].values
        dist_km, idx = nearest_neighbor(well_lat, well_lon, g_tree)

        for col in _GLDAS_COLS:
            if col not in gldas_ym.columns:
                continue
            vals = gldas_ym[col].values[idx]
            vals = vals.astype(float)
            df.loc[grp_df.index, col] = vals

        df.loc[grp_df.index, "gldas_dist_km"] = dist_km.round(3)

    df = df.drop(columns=["_ym"])

    # Humidité totale du sol 0-200cm (≈ Soil_Moisture_mm du papier, rang 18)
    moi_layers = ["soil_moi_0_10_kg_m2", "soil_moi_10_40_kg_m2",
                  "soil_moi_40_100_kg_m2", "soil_moi_100_200_kg_m2"]
    present_layers = [c for c in moi_layers if c in df.columns]
    if len(present_layers) == 4:
        df["soil_moisture_total_mm"] = df[present_layers].sum(axis=1, min_count=4)

    n_ok = df["rainfall_mm_month"].notna().sum()
    logger.info("  GLDAS enrichi : %d/%d lignes avec rainfall", n_ok, len(df))
    return df


# ══════════════════════════════════════════════════════════
# Export + data dictionnaire
# ══════════════════════════════════════════════════════════

def _build_data_dict(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        na_pct = round(100 * df[col].isna().mean(), 2)
        nuniq = df[col].nunique(dropna=True)
        vmin = vmax = ""
        if df[col].dtype.kind in ("f", "i"):
            vmin = round(float(df[col].min()), 4)
            vmax = round(float(df[col].max()), 4)
        rows.append({"column": col, "dtype": dtype, "na_pct": na_pct,
                     "nunique": nuniq, "min": vmin, "max": vmax})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main(with_weather: bool = True, with_aqs: bool = True,
         with_gldas: bool = True, with_ssurgo: bool = True) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement wells_pfas_clean.parquet…")
    df = pd.read_parquet(PROCESSED_DIR / "wells_pfas_clean.parquet")
    logger.info("  %d lignes × %d colonnes", *df.shape)

    # Filtrer les lignes sans coordonnées valides
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    logger.info("  après filtre lat/lon : %d lignes", len(df))

    # Étape 1 — Cibles
    df = compute_targets(df)

    # Étape 1b — Catégorie du puits
    allwells_path = RAW_DIR / "gama" / "gama_allwells.csv"
    if allwells_path.exists():
        df = add_well_category(df, allwells_path)
    else:
        logger.warning("gama_allwells.csv absent — skip well_category")

    # Étape 2 — SGMA bassins
    basins_path = RAW_DIR / "environment" / "sgma_basins.geojson"
    if basins_path.exists():
        df = add_sgma_basins(df, basins_path)
    else:
        logger.warning("sgma_basins.geojson absent — skip")

    # Étape 3 — Contamination GeoTracker (distance + type)
    gt_path = RAW_DIR / "contamination" / "geotracker_pfas_investigation_sites.csv"
    if gt_path.exists():
        df = add_contamination(df, gt_path)
    else:
        logger.warning("geotracker_pfas_investigation_sites.csv absent — skip")

    # Étape 3b — Co-contaminants GAMA
    gama_path = RAW_DIR / "gama" / "gama.csv"
    if gama_path.exists():
        df = add_co_contaminants(df, gama_path)
    else:
        logger.warning("gama.csv absent — skip co-contaminants")

    # Étape 5 — Sol SSURGO
    if with_ssurgo:
        ssurgo_parquet = RAW_DIR / "environment" / "ssurgo" / "ssurgo_ca_points.parquet"
        df = add_ssurgo_soil(df, ssurgo_parquet)
    else:
        logger.info("SSURGO désactivé (--no-ssurgo)")

    # Étape 4 — Météo NOAA
    if with_weather:
        token = _load_env_var("NOAA_CDO_TOKEN")
        if token:
            df = add_noaa_weather(df, CACHE_DIR, token)
        else:
            logger.warning("NOAA_CDO_TOKEN absent — skip météo")
    else:
        logger.info("Météo NOAA désactivée (--no-weather)")

    # Étape 5 — Qualité de l'air EPA AQS
    if with_aqs:
        aqs_dir = RAW_DIR / "environment" / "aqs"
        df = add_aqs_air_quality(df, aqs_dir)
    else:
        logger.info("AQS désactivé (--no-aqs)")

    # Étape 6 — Hydrologie NASA GLDAS
    if with_gldas:
        gldas_parquet = RAW_DIR / "environment" / "gldas_ca_monthly.parquet"
        df = add_gldas_hydrology(df, gldas_parquet)
    else:
        logger.info("GLDAS désactivé (--no-gldas)")

    # Export
    out_parquet = PROCESSED_DIR / "CA-PFAS-ASGWS.parquet"
    out_csv = PROCESSED_DIR / "CA-PFAS-ASGWS.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    logger.info("Sauvegardé : %s  (%d × %d)", out_parquet.name, *df.shape)

    data_dict = _build_data_dict(df)
    data_dict.to_csv(PROCESSED_DIR / "data_dict.csv", index=False)
    logger.info("data_dict.csv écrit (%d colonnes)", len(data_dict))

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-weather", action="store_true",
                        help="Sauter l'enrichissement NOAA")
    parser.add_argument("--no-aqs", action="store_true",
                        help="Sauter la qualité de l'air EPA AQS")
    parser.add_argument("--no-gldas", action="store_true",
                        help="Sauter l'hydrologie NASA GLDAS")
    parser.add_argument("--no-ssurgo", action="store_true",
                        help="Sauter le sol SSURGO")
    args = parser.parse_args()
    main(with_weather=not args.no_weather,
         with_aqs=not args.no_aqs,
         with_gldas=not args.no_gldas,
         with_ssurgo=not args.no_ssurgo)
