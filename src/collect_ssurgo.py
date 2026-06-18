"""
Collecte des propriétés de sol SSURGO (USDA) via l'API SDA pour tous les puits CA.

Pour chaque position unique (lat, lon) du dataset, interroge :
  https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest

Variables extraites (chorizon) :
  Textures globales : sandtotal_r, claytotal_r, silttotal_r
  Fractions sable (Dong 2024 Table S9 rangs 6, 36, 38, 41) :
    sandvf_r (0.05-0.10mm), sandf_r (0.10-0.25mm), sandm_r (0.25-0.50mm, rang 6!),
    sandc_r (0.50-1.0mm), sandvc_r (1.0-2.0mm)
  Limon grossier siltco_r (0.02-0.05mm, rangs 9, 46)
  Rétention eau : wthirdbar_r (1/3 bar), wfifteenbar_r (15 bar → Ratio_Water_Clay rang 14)
  Classe texturale : texcl (USDA, rangs 22, 31)
  Autres : om_r, ph1to1h2o_r, ksat_r, awc_r, dbthirdbar_r

Usage :
    python -m src.collect_ssurgo               # tous les puits (~9 800 points)
    python -m src.collect_ssurgo --workers 8   # parallélisme accru
    python -m src.collect_ssurgo --test        # 20 points test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "environment" / "ssurgo"
OUT_PARQUET = OUT_DIR / "ssurgo_ca_points.parquet"
CACHE_DIR = OUT_DIR / "cache"

SDA_URL = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest"
COORD_ROUND = 4        # 0.0001° ≈ 10 m
REQUEST_TIMEOUT = 30
# Sentinel fichier cache pour "aucune donnée à cette coordonnée" (zone non couverte)
_NO_DATA = "null"

# Fractions numériques de chorizon  (noms API SDA vérifiés)
SOIL_VARS = [
    "sandtotal_r",   # sable total (0.05-2mm) %
    "sandvf_r",      # sable très fin (0.05-0.10mm) %
    "sandfine_r",    # sable fin (0.10-0.25mm) %      ← SSURGO: sandfine_r
    "sandmed_r",     # sable moyen (0.25-0.50mm) %   ← rang 6 Table S9; SSURGO: sandmed_r
    "sandco_r",      # sable grossier (0.50-1.0mm) %  ← SSURGO: sandco_r
    "sandvc_r",      # sable très grossier (1.0-2.0mm) %
    "silttotal_r",   # limon total (0.002-0.05mm) %
    "siltco_r",      # limon grossier (0.02-0.05mm) %  ← rangs 9, 46
    "siltfine_r",    # limon fin (0.002-0.02mm) %       ← disponible directement
    "claytotal_r",   # argile (<0.002mm) %
    "om_r",          # matière organique %
    "ph1to1h2o_r",   # pH eau 1:1
    "ksat_r",        # conductivité hydraulique saturée μm/s
    "awc_r",         # eau disponible cm/cm
    "dbthirdbar_r",  # densité apparente g/cm³
    "wthirdbar_r",   # teneur en eau à 1/3 bar (% pds)
    "wfifteenbar_r", # teneur en eau à 15 bar (% pds) → Ratio_Water_Clay rang 14
]

# Champs texte (non convertis en float)
STRING_VARS = {"texturerv"}

SOIL_VARS_STR = ["texturerv"]  # classe texturale USDA ← rangs 22, 31

RENAME = {
    "sandtotal_r":   "soil_sand_pct",
    "sandvf_r":      "soil_sand_vfine_pct",
    "sandfine_r":    "soil_sand_fine_pct",
    "sandmed_r":     "soil_sand_medium_pct",
    "sandco_r":      "soil_sand_coarse_pct",
    "sandvc_r":      "soil_sand_vcoarse_pct",
    "silttotal_r":   "soil_silt_pct",
    "siltco_r":      "soil_silt_coarse_pct",
    "siltfine_r":    "soil_silt_fine_pct",
    "claytotal_r":   "soil_clay_pct",
    "om_r":          "soil_om_pct",
    "ph1to1h2o_r":   "soil_ph",
    "ksat_r":        "soil_ksat_um_s",
    "awc_r":         "soil_awc_cm_cm",
    "dbthirdbar_r":  "soil_bulk_density",
    "wthirdbar_r":   "soil_water_1bar_pct",
    "wfifteenbar_r": "soil_water_15bar_pct",
    "texturerv":     "soil_texture_class",
}

# Marqueur de version de schéma : présence de sandmed_r dans le cache
_SCHEMA_KEY = "sandmed_r"


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _cache_path(lat: float, lon: float) -> Path:
    key = f"{lat:.4f}_{lon:.4f}".replace("-", "m")
    return CACHE_DIR / f"{key}.json"


def _is_cached(lat: float, lon: float) -> bool:
    """Vrai si le point a une réponse définitive avec le schéma actuel (ou no-data)."""
    cp = _cache_path(lat, lon)
    if not cp.exists() or cp.stat().st_size == 0:
        return False
    text = cp.read_text().strip()
    if text == _NO_DATA:
        return True  # zone non couverte — toujours valide
    try:
        data = json.loads(text)
        return _SCHEMA_KEY in data  # re-interroger si ancien schéma
    except json.JSONDecodeError:
        return False


def _read_cache(lat: float, lon: float) -> dict | None:
    cp = _cache_path(lat, lon)
    if not cp.exists():
        return None
    text = cp.read_text().strip()
    if text == _NO_DATA or not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _query_point(session: requests.Session, lat: float, lon: float) -> dict | None:
    """Interroge SDA pour un point, cache le résultat, retourne le dict ou None."""
    # Lecture cache (ne pas re-interroger si déjà résolu)
    if _is_cached(lat, lon):
        return _read_cache(lat, lon)

    wkt = f"POINT({lon:.6f} {lat:.6f})"
    all_vars = SOIL_VARS + SOIL_VARS_STR
    vars_sql = ", ".join(f"ch.{v}" for v in all_vars)
    sql = (
        f"SELECT TOP 1 {vars_sql} "
        f"FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{wkt}') AS x "
        f"INNER JOIN component co ON co.mukey = x.mukey "
        f"INNER JOIN chorizon ch ON ch.cokey = co.cokey "
        f"WHERE co.majcompflag = 'Yes' "
        f"ORDER BY co.comppct_r DESC, ch.hzdept_r ASC"
    )

    for attempt in range(6):
        try:
            r = session.post(
                SDA_URL,
                json={"query": sql, "format": "json+columnname"},
                timeout=REQUEST_TIMEOUT,
            )
            # Maintenance en cours → attendre sans cacher
            if b"maintenance" in r.content.lower():
                wait = 90 * (attempt + 1)
                logger.warning("SDA en maintenance — attente %ds (tentative %d/6)", wait, attempt + 1)
                time.sleep(wait)
                continue

            if r.status_code != 200:
                time.sleep(4 * (attempt + 1))
                continue

            body = r.json()
            rows = body.get("Table", [])
            if not rows or len(rows) < 2:
                # Aucune donnée (zone urbaine, océan…) — cacher comme no-data
                _cache_path(lat, lon).write_text(_NO_DATA)
                return None

            headers, values = rows[0], rows[1]
            valid_keys = set(SOIL_VARS) | set(SOIL_VARS_STR)
            result = {h: v for h, v in zip(headers, values) if h in valid_keys}
            _cache_path(lat, lon).write_text(json.dumps(result))
            return result

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.debug("  erreur attempt %d (%s, %s): %s", attempt + 1, lat, lon, e)
            time.sleep(2 * (attempt + 1))

    # Après 6 tentatives d'erreur réseau (pas maintenance), ne pas cacher
    return None


def collect(n_workers: int = 6, test: bool = False) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    src = PROCESSED_DIR / "wells_pfas_clean.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Introuvable : {src}")

    coords = (
        pd.read_parquet(src, columns=["latitude", "longitude"])
        .dropna()
        .assign(lat_r=lambda d: d["latitude"].round(COORD_ROUND),
                lon_r=lambda d: d["longitude"].round(COORD_ROUND))
        [["lat_r", "lon_r"]]
        .drop_duplicates()
    )

    if test:
        coords = coords.head(20)

    pending = [
        (lat, lon)
        for lat, lon in coords.itertuples(index=False)
        if not _is_cached(lat, lon)
    ]
    already = len(coords) - len(pending)
    logger.info("%d points uniques — %d en cache, %d à interroger",
                len(coords), already, len(pending))

    if pending:
        session = _make_session()
        done = 0
        n_ok = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_query_point, session, lat, lon): (lat, lon)
                       for lat, lon in pending}
            for fut in as_completed(futures):
                done += 1
                lat, lon = futures[fut]
                result = fut.result()
                if result:
                    n_ok += 1
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                if done % 500 == 0 or done == len(pending):
                    logger.info("  [%d/%d] %.1f pts/min  (%d avec données)",
                                done, len(pending), rate, n_ok)

    # Construire le DataFrame final depuis le cache
    records = []
    for lat, lon in coords.itertuples(index=False):
        data = _read_cache(lat, lon)
        if data:
            row: dict = {"lat_r": lat, "lon_r": lon}
            for k, v in data.items():
                target = RENAME.get(k, k)
                if k in STRING_VARS:
                    row[target] = str(v) if v is not None else None
                else:
                    try:
                        row[target] = float(v) if v is not None else float("nan")
                    except (TypeError, ValueError):
                        row[target] = float("nan")
            records.append(row)

    if not records:
        logger.warning("Aucune donnée SSURGO collectée")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.to_parquet(OUT_PARQUET, index=False)
    key_cols = ["soil_sand_medium_pct", "soil_silt_coarse_pct", "soil_clay_pct",
                "soil_texture_class", "soil_water_15bar_pct"]
    logger.info("Sauvegardé : %s (%d points couverts / %d uniques)",
                OUT_PARQUET.name, len(df), len(coords))
    for c in key_cols:
        if c not in df.columns:
            continue
        n = df[c].notna().sum()
        logger.info("  %-28s: %d/%d (%.0f%%)", c, n, len(df), 100 * n / len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--test", action="store_true", help="20 points seulement")
    args = parser.parse_args()
    collect(n_workers=args.workers, test=args.test)
