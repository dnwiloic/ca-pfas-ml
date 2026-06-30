"""Dérive l'occupation du sol et la densité de population par puits.

Deux features mécanistes **transférables** de charge anthropique diffuse, qui
visent à *remplacer* le proxy positionnel lat/lon :

- `pop_density_per_km2` (WorldPop 1 km, ~2020) — intensité urbaine lissée,
  proxy de charge PFAS « produits de consommation » ;
- classe NLCD groupée (`lc_developed` / `lc_cultivated` / `lc_natural` /
  `lc_water_wetland`) + `dev_intensity` (0–4) — distingue l'urbain de
  l'agricole (biosolides / irrigation par eaux recyclées), signal que la
  population seule ne capte pas.

Échantillonnage par `getSamples` (ImageServer ArcGIS), en POST par lots, mapping
par `locationId`. Aucune donnée massive téléchargée.

Source  : WorldPop 1km ImageServer ; NLCD Land Cover ImageServer (public).
          data/processed/well_hydraulic_gradient.csv (coords des 11 333 puits).
Sortie  : data/processed/well_landuse.csv (jointure sur gm_well_id).
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
OUT = ROOT / "data" / "processed" / "well_landuse.csv"

WORLDPOP = ("https://worldpop.arcgis.com/arcgis/rest/services/"
            "WorldPop_Population_Density_1km/ImageServer")
NLCD = ("https://sampleserver6.arcgisonline.com/arcgis/rest/services/"
        "NLCDLandCover2001/ImageServer")   # public, sans token (classes NLCD stables)
BATCH = 800

# regroupement des classes NLCD
NLCD_DEVELOPED = {21, 22, 23, 24}
NLCD_CULTIVATED = {81, 82}
NLCD_WATER_WETLAND = {11, 12, 90, 95}
DEV_INTENSITY = {21: 1, 22: 2, 23: 3, 24: 4}   # open / low / med / high


def _get_samples(svc: str, lonlat: np.ndarray) -> np.ndarray:
    """Échantillonne un ImageServer aux points (lon,lat WGS84). Renvoie un vecteur float (NaN si nodata)."""
    out = np.full(len(lonlat), np.nan)
    for start in range(0, len(lonlat), BATCH):
        chunk = lonlat[start:start + BATCH]
        geom = json.dumps({"points": chunk.tolist(), "spatialReference": {"wkid": 4326}})
        for attempt in range(4):
            try:
                r = requests.post(f"{svc}/getSamples", data={
                    "geometry": geom, "geometryType": "esriGeometryMultipoint",
                    "returnFirstValueOnly": "true", "f": "json"}, timeout=90)
                js = r.json()
                if "samples" in js:
                    for s in js["samples"]:
                        loc = int(s["locationId"])
                        try:
                            out[start + loc] = float(s["value"])
                        except (ValueError, TypeError):
                            pass
                    break
                raise RuntimeError(js.get("error", js))
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    print(f"  ⚠️ lot {start}: échec ({exc})")
                else:
                    time.sleep(1.5 * (attempt + 1))
        print(f"  {min(start + BATCH, len(lonlat))}/{len(lonlat)}", end="\r")
    print()
    return out


def main() -> None:
    w = pd.read_csv(WELLS)
    lonlat = w[["longitude", "latitude"]].values

    print("WorldPop — densité de population…")
    pop = _get_samples(WORLDPOP, lonlat)
    pop = np.where(pop < 0, np.nan, pop)        # nodata → NaN
    w["pop_density_per_km2"] = np.round(pop, 2)
    w["pop_density_log1p"] = np.round(np.log1p(np.clip(pop, 0, None)), 4)

    print("NLCD — classe d'occupation du sol…")
    lc = _get_samples(NLCD, lonlat)
    cls = pd.Series(lc, index=w.index).round().astype("Int64")
    w["nlcd_class"] = cls
    w["lc_developed"] = cls.isin(NLCD_DEVELOPED).astype("Int64")
    w["lc_cultivated"] = cls.isin(NLCD_CULTIVATED).astype("Int64")
    w["lc_water_wetland"] = cls.isin(NLCD_WATER_WETLAND).astype("Int64")
    w["lc_natural"] = (~cls.isin(NLCD_DEVELOPED | NLCD_CULTIVATED | NLCD_WATER_WETLAND)
                       & cls.notna()).astype("Int64")
    w["dev_intensity"] = cls.map(DEV_INTENSITY).fillna(0).astype("Int64")
    w["landuse_missing"] = (cls.isna() | pd.isna(pop)).astype(int)

    cols = ["gm_well_id", "latitude", "longitude",
            "pop_density_per_km2", "pop_density_log1p",
            "nlcd_class", "lc_developed", "lc_cultivated", "lc_natural",
            "lc_water_wetland", "dev_intensity", "landuse_missing"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w[cols].to_csv(OUT, index=False)

    print(f"\nécrit {OUT} ({len(w)} puits)")
    print(f"  pop_density médiane : {np.nanmedian(pop):.1f} hab/km²  | NA : {np.isnan(pop).mean()*100:.1f}%")
    print("  occupation du sol :")
    for c in ["lc_developed", "lc_cultivated", "lc_natural", "lc_water_wetland"]:
        print(f"    {c:18s}: {100*w[c].mean():.1f}%")
    print(f"  landuse_missing : {100*w['landuse_missing'].mean():.1f}%")


if __name__ == "__main__":
    main()
