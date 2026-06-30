"""Dérive l'élévation et la pente du terrain par puits (DEM USGS 3DEP).

Features topographiques absentes du pipeline (≈47 mentions dans le corpus PFAS/GW,
cf. docs/features_litterature.md) :
- `elevation_m` — position hydrogéologique (altitude du sol) ;
- `slope_deg` / `slope_pct` — la pente contrôle ruissellement vs infiltration
  (pente forte → ruissellement, recharge faible ; terrain plat → infiltration des
  PFAS de surface vers la nappe).

Échantillonnage `getSamples` (3DEP ElevationServer, public, sans token), en POST
par lots, mapping par `locationId`. Pente par différences finies sur 4 voisins
cardinaux à ±100 m.

Source : USGS 3DEP ElevationServer (ImageServer)
         data/processed/well_hydraulic_gradient.csv (univers des 11 333 puits)
Sortie : data/processed/well_topography.csv (jointure sur gm_well_id)
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
OUT = ROOT / "data" / "processed" / "well_topography.csv"
DEM = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
BATCH = 900
OFFSET_M = 100.0   # distance des voisins pour la pente


def _get_samples(lonlat: np.ndarray) -> np.ndarray:
    """Élévation (m) aux points (lon,lat WGS84) via getSamples. NaN si nodata."""
    out = np.full(len(lonlat), np.nan)
    for start in range(0, len(lonlat), BATCH):
        chunk = lonlat[start:start + BATCH]
        geom = json.dumps({"points": chunk.tolist(), "spatialReference": {"wkid": 4326}})
        for attempt in range(4):
            try:
                r = requests.post(f"{DEM}/getSamples", data={
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
                if attempt == 3:
                    print(f"  ⚠️ lot {start}: {exc}")
                else:
                    time.sleep(2.0 * (attempt + 1))
        print(f"  {min(start + BATCH, len(lonlat))}/{len(lonlat)}", end="\r")
    print()
    return out


def main() -> None:
    w = pd.read_csv(WELLS)
    lat = w["latitude"].values
    lon = w["longitude"].values
    n = len(w)

    dlat = OFFSET_M / 110540.0
    dlon = OFFSET_M / (111320.0 * np.cos(np.radians(lat)))

    # 5 points par puits : centre, N, S, E, O
    pts = np.empty((5 * n, 2))
    pts[0::5] = np.column_stack([lon, lat])               # centre
    pts[1::5] = np.column_stack([lon, lat + dlat])         # N
    pts[2::5] = np.column_stack([lon, lat - dlat])         # S
    pts[3::5] = np.column_stack([lon + dlon, lat])         # E
    pts[4::5] = np.column_stack([lon - dlon, lat])         # O

    print(f"3DEP — échantillonnage de {len(pts)} points ({n} puits × 5)…")
    z = _get_samples(pts)
    zc, zn, zs, ze, zo = z[0::5], z[1::5], z[2::5], z[3::5], z[4::5]

    dz_dy = (zn - zs) / (2 * OFFSET_M)
    dz_dx = (ze - zo) / (2 * OFFSET_M)
    grad = np.sqrt(dz_dx ** 2 + dz_dy ** 2)               # m/m

    w["elevation_m"] = np.round(zc, 1)
    w["slope_pct"] = np.round(grad * 100, 3)
    w["slope_deg"] = np.round(np.degrees(np.arctan(grad)), 3)
    w["topo_missing"] = (np.isnan(zc) | np.isnan(grad)).astype(int)

    cols = ["gm_well_id", "latitude", "longitude",
            "elevation_m", "slope_pct", "slope_deg", "topo_missing"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w[cols].to_csv(OUT, index=False)

    print(f"écrit {OUT} ({n} puits)")
    print(f"  elevation_m : médiane {np.nanmedian(zc):.0f} m  | NA {100*np.isnan(zc).mean():.1f}%")
    print(f"  slope_deg   : médiane {np.nanmedian(w['slope_deg']):.2f}°  | p90 {np.nanpercentile(w['slope_deg'],90):.2f}°")


if __name__ == "__main__":
    main()
