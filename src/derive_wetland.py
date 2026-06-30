"""Dérive la proximité de zones humides par puits (NLCD wetland 90/95).

Prédicteur PFAS-GW distinctif de **Li & MacDonald Gibson (2022)** : les zones
humides accumulent et relarguent les PFAS (source/voie diffuse vers la nappe).

On échantillonne NLCD sur une grille 3×3 (±1,5 km) autour de chaque puits et on
calcule la **fraction de zones humides au voisinage** (proxy de proximité), plus
le flag « zone humide au puits ».

Source : NLCD Land Cover ImageServer (public, getSamples)
         data/processed/well_hydraulic_gradient.csv (univers des 11 333 puits)
Sortie : data/processed/well_wetland.csv (jointure sur gm_well_id)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
WELLS = ROOT / "data" / "processed" / "well_hydraulic_gradient.csv"
OUT = ROOT / "data" / "processed" / "well_wetland.csv"
NLCD = ("https://sampleserver6.arcgisonline.com/arcgis/rest/services/"
        "NLCDLandCover2001/ImageServer")
BATCH = 900
GRID_M = 1500.0           # demi-pas de la grille 3×3 (boîte ~3 km)
WETLAND = {90, 95}        # woody + emergent herbaceous wetlands


def _get_samples(lonlat: np.ndarray) -> np.ndarray:
    out = np.full(len(lonlat), np.nan)
    for start in range(0, len(lonlat), BATCH):
        chunk = lonlat[start:start + BATCH]
        geom = json.dumps({"points": chunk.tolist(), "spatialReference": {"wkid": 4326}})
        for attempt in range(5):
            try:
                r = requests.post(f"{NLCD}/getSamples", data={
                    "geometry": geom, "geometryType": "esriGeometryMultipoint",
                    "returnFirstValueOnly": "true", "f": "json"}, timeout=120)
                js = r.json()
                if "samples" in js:
                    for s in js["samples"]:
                        try:
                            out[start + int(s["locationId"])] = float(s["value"])
                        except (ValueError, TypeError):
                            pass
                    break
                raise RuntimeError(js.get("error", js))
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    print(f"  ⚠️ lot {start}: {exc}")
                else:
                    time.sleep(1.5 * (attempt + 1))
        time.sleep(0.1)
        print(f"  {min(start + BATCH, len(lonlat))}/{len(lonlat)}", end="\r")
    print()
    return out


def main() -> None:
    w = pd.read_csv(WELLS)
    lat = w["latitude"].values
    lon = w["longitude"].values
    n = len(w)
    dlat = GRID_M / 110540.0
    dlon = GRID_M / (111320.0 * np.cos(np.radians(lat)))

    # grille 3×3 : 9 points / puits
    offs = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    pts = np.empty((9 * n, 2))
    for k, (i, j) in enumerate(offs):
        pts[k::9] = np.column_stack([lon + j * dlon, lat + i * dlat])
    center = offs.index((0, 0))

    print(f"NLCD — grille 3×3 : {len(pts)} points ({n} puits × 9)…")
    cls = _get_samples(pts).reshape(n, 9)   # (n, 9) — attention : reshape suit l'ordre k

    is_wet = np.isin(cls, list(WETLAND))
    valid = ~np.isnan(cls)
    frac = np.where(valid.sum(1) > 0, is_wet.sum(1) / np.maximum(valid.sum(1), 1), np.nan)

    w["wetland_frac_3km"] = np.round(frac, 4)
    w["wetland_at_well"] = is_wet[:, center].astype(int)
    w["wetland_any_3km"] = (is_wet.sum(1) > 0).astype(int)
    w["wetland_missing"] = (valid.sum(1) == 0).astype(int)

    cols = ["gm_well_id", "latitude", "longitude",
            "wetland_frac_3km", "wetland_at_well", "wetland_any_3km", "wetland_missing"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w[cols].to_csv(OUT, index=False)

    print(f"écrit {OUT} ({n} puits)")
    print(f"  wetland au puits     : {100*w['wetland_at_well'].mean():.1f}%")
    print(f"  ≥1 wetland dans 3 km : {100*w['wetland_any_3km'].mean():.1f}%")
    print(f"  fraction médiane     : {np.nanmedian(frac):.3f}  | missing {100*w['wetland_missing'].mean():.1f}%")


if __name__ == "__main__":
    main()
