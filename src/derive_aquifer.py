"""Dérive le type d'aquifère (unité hydrogéologique) par puits.

Comble l'écart le plus structurant vs la littérature (confinement ~60, lithologie
~90 mentions, cf. docs/features_litterature.md) : on n'avait que des bassins
**administratifs** SGMA, pas l'**unité hydrogéologique**. Source : USGS Principal
Aquifers (type de roche/matériau), jointure point-dans-polygone.

Mécanisme PFAS : un aquifère **non consolidé (sable/gravier)** — Central Valley,
bassins côtiers — est perméable, fortement pompé et **vulnérable** à l'infiltration
des PFAS de surface ; la **roche dure** (igné/métamorphique) est peu productive et
moins exposée.

Source : USGS Principal Aquifers FeatureServer (hébergé arcgis.water.nv.gov)
         data/processed/well_hydraulic_gradient.csv (univers des 11 333 puits)
Cache  : data/raw/hydro/usgs_principal_aquifers_ca.geojson
Sortie : data/processed/well_aquifer.csv (jointure sur gm_well_id)
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
WELLS = ROOT / "data" / "processed" / "well_hydraulic_gradient.csv"
CACHE = ROOT / "data" / "raw" / "hydro" / "usgs_principal_aquifers_ca.geojson"
OUT = ROOT / "data" / "processed" / "well_aquifer.csv"
SVC = ("https://arcgis.water.nv.gov/arcgis/rest/services/BaseLayers/"
       "USGS_Aquifers_Principal/FeatureServer/0")

# ROCK_TYPE USGS → matériau groupé
ROCK_GROUP = {100: "unconsolidated", 200: "semiconsolidated", 300: "sandstone",
              400: "carbonate", 500: "sandstone_carbonate", 600: "igneous_metamorphic",
              999: "other"}


def _fetch_ca() -> gpd.GeoDataFrame:
    if CACHE.exists():
        return gpd.read_file(CACHE)
    bbox = {"xmin": -124.6, "ymin": 32.4, "xmax": -114.0, "ymax": 42.1,
            "spatialReference": {"wkid": 4326}}
    params = {"where": "1=1", "geometry": json.dumps(bbox),
              "geometryType": "esriGeometryEnvelope", "spatialRel": "esriSpatialRelIntersects",
              "outFields": "ROCK_NAME,ROCK_TYPE,AQ_NAME,AQ_CODE", "outSR": "4326",
              "returnGeometry": "true", "f": "geojson"}
    r = requests.get(f"{SVC}/query", params=params, timeout=120)
    r.raise_for_status()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(r.text)
    return gpd.read_file(CACHE)


def main() -> None:
    poly = _fetch_ca().to_crs("EPSG:4326")
    poly = poly[poly.geometry.notna()].copy()
    print(f"polygones aquifères chargés : {len(poly)}")

    w = pd.read_csv(WELLS)
    gw = gpd.GeoDataFrame(
        w.copy(),
        geometry=[Point(lo, la) for lo, la in zip(w["longitude"], w["latitude"])],
        crs="EPSG:4326")
    joined = gpd.sjoin(gw, poly[["ROCK_TYPE", "AQ_NAME", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(gw.index)

    rt = pd.to_numeric(joined["ROCK_TYPE"], errors="coerce")
    w["aquifer_material"] = rt.map(ROCK_GROUP).fillna("other")
    w["aquifer_name"] = joined["AQ_NAME"].fillna("none").values
    w["aquifer_unconsolidated"] = (rt == 100).astype(int)          # le plus vulnérable
    w["aquifer_hardrock"] = rt.isin([600, 999]).astype(int)        # roche dure/autre
    w["aquifer_missing"] = rt.isna().astype(int)

    cols = ["gm_well_id", "latitude", "longitude", "aquifer_material",
            "aquifer_name", "aquifer_unconsolidated", "aquifer_hardrock", "aquifer_missing"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    w[cols].to_csv(OUT, index=False)

    print(f"écrit {OUT} ({len(w)} puits)")
    print("répartition aquifer_material :")
    print((100 * w["aquifer_material"].value_counts() / len(w)).round(1).astype(str).add(" %").to_string())
    print(f"  unconsolidated : {100*w['aquifer_unconsolidated'].mean():.1f}%  | missing : {100*w['aquifer_missing'].mean():.1f}%")


if __name__ == "__main__":
    main()
