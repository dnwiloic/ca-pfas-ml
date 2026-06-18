"""
Téléchargement des données annuelles EPA AQS (Air Quality System) pour la Californie.

Paramètres :
  PM2.5     88101 | PM10     81102 | NO2  42602
  SO2       42401 | Wind     61101 | Humidity 62201
  Ozone     44201 | CO       42101

Endpoint : https://aqs.epa.gov/data/api/annualData/byState
Credentials dans .env : AQS_EMAIL, AQS_KEY

Usage :
    python -m src.collect_aqs
    python -m src.collect_aqs --years 2019 2020 2021
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AQS_DIR = PROJECT_ROOT / "data" / "raw" / "environment" / "aqs"
AQS_BASE = "https://aqs.epa.gov/data/api/annualData/byState"

# Codes param AQS → nom de colonne dans le dataset final
PARAM_CODES: dict[str, str] = {
    "88101": "pm25",
    "81102": "pm10",
    "42602": "no2",
    "42401": "so2",
    "61101": "wind_speed",
    "62201": "humidity",
    "44201": "ozone",   # Ozone (ppb), rang 24 Table S9
    "42101": "co",      # CO (ppm), rang 50 Table S9
}

STATE_FIPS = "06"  # Californie
YEARS = list(range(2015, 2026))
REQUEST_TIMEOUT = 60


def _load_env(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.strip().startswith(f"{name}="):
                return line.split("=", 1)[1].strip()
    return ""


def _fetch_annual(session: requests.Session, param: str, year: int) -> list[dict]:
    cache = AQS_DIR / f"aqs_{param}_{year}.json"
    if cache.exists():
        logger.debug("  [cache] param=%s year=%d", param, year)
        return json.loads(cache.read_text())

    params = {
        "email": session.params["email"],
        "key": session.params["key"],
        "param": param,
        "bdate": f"{year}0101",
        "edate": f"{year}1231",
        "state": STATE_FIPS,
    }
    for attempt in range(4):
        try:
            r = session.get(AQS_BASE, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.warning("  rate-limit param=%s year=%d → attente %ds", param, year, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            body = r.json()
            rows = body.get("Data", [])
            cache.write_text(json.dumps(rows))
            logger.info("  param=%s year=%d → %d monitors", param, year, len(rows))
            return rows
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning("  erreur attempt %d : %s", attempt + 1, e)
            time.sleep(2 * (attempt + 1))
    cache.write_text("[]")
    return []


def collect(years: list[int] | None = None) -> pd.DataFrame:
    AQS_DIR.mkdir(parents=True, exist_ok=True)
    email = _load_env("AQS_EMAIL")
    key = _load_env("AQS_KEY")
    if not email or not key:
        raise SystemExit("AQS_EMAIL et AQS_KEY requis dans .env")

    session = requests.Session()
    session.params = {"email": email, "key": key}  # type: ignore[assignment]

    target_years = years or YEARS
    all_rows: list[dict] = []

    for year in target_years:
        for param, col_name in PARAM_CODES.items():
            rows = _fetch_annual(session, param, year)
            for r in rows:
                all_rows.append({
                    "year": year,
                    "param_code": param,
                    "param_name": col_name,
                    "monitor_id": r.get("state_code", "") + r.get("county_code", "") + r.get("site_number", ""),
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude"),
                    "annual_mean": r.get("arithmetic_mean"),
                    "sample_duration": r.get("sample_duration", ""),
                    "pollutant_standard": r.get("pollutant_standard", ""),
                    "completeness_pct": r.get("percent_complete"),
                })
            time.sleep(0.5)

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.warning("Aucune donnée AQS téléchargée")
        return df

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["annual_mean"] = pd.to_numeric(df["annual_mean"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "annual_mean"])

    # Garder uniquement la durée "Annual" (ou 24h pour PM2.5)
    annual_mask = df["sample_duration"].str.contains("24|Annual|annual|1 HOUR", na=False, case=False)
    df = df[annual_mask].copy()

    out = AQS_DIR / "aqs_ca_annual.parquet"
    df.to_parquet(out, index=False)
    logger.info("Sauvegardé : %s (%d lignes)", out.name, len(df))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Années à télécharger (défaut: 2015-2025)")
    args = parser.parse_args()
    collect(years=args.years)
