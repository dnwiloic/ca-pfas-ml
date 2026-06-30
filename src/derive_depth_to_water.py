"""Dérive la profondeur de la nappe (épaisseur de la zone non-saturée) par puits.

profondeur = élévation du sol (`gse`) − élévation de la nappe (`gwe_mean`),
interpolée aux puits depuis les stations DWR Periodic GWL par IDW (k plus proches,
pondération 1/d², distance grand-cercle).

Feature mécaniste **transférable** : une nappe profonde filtre/ralentit la
percolation des PFAS de surface vers l'aquifère, indépendamment de la région —
contrairement à lat/lon. À ajouter au nœud `well` (cf. DATA_COLLECTION_RECOMMENDATIONS §2),
avec son indicateur de fiabilité `dtw_far`.

Source  : data/raw/hydro/dwr_periodic_gwl_recent.csv (gse, gwe_mean, lat, lon)
          data/processed/well_hydraulic_gradient.csv (coords des 11 333 puits)
Sortie  : data/processed/well_depth_to_water.csv (jointure sur gm_well_id)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DWR = ROOT / "data" / "raw" / "hydro" / "dwr_periodic_gwl_recent.csv"
WELLS = ROOT / "data" / "processed" / "well_hydraulic_gradient.csv"
OUT = ROOT / "data" / "processed" / "well_depth_to_water.csv"

FT_TO_M = 0.3048
EARTH_R_KM = 6371.0088
K_NEIGHBORS = 8
FAR_KM = 20.0          # au-delà de cette distance à la station la plus proche → feature peu fiable
DTW_MIN_FT = -50.0     # en-deçà : gse/gwe erroné (rejet) ; on garde l'artésien léger
DTW_MAX_FT = 2000.0    # au-delà : aberrant


def _latlon_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([
        EARTH_R_KM * np.cos(la) * np.cos(lo),
        EARTH_R_KM * np.cos(la) * np.sin(lo),
        EARTH_R_KM * np.sin(la),
    ])


def main() -> None:
    s = pd.read_csv(DWR)
    s["dtw_ft"] = s["gse"] - s["gwe_mean"]
    n0 = len(s)
    s = s[(s["dtw_ft"] > DTW_MIN_FT) & (s["dtw_ft"] < DTW_MAX_FT)].copy()
    print(f"stations DWR : {n0} → {len(s)} après filtre d'aberrations")
    s["dtw_m"] = s["dtw_ft"].clip(lower=0.0) * FT_TO_M   # clip à 0 = nappe affleurante

    stree = cKDTree(_latlon_xyz(s["latitude"].values, s["longitude"].values))
    s_dtw = s["dtw_m"].values

    w = pd.read_csv(WELLS)
    dist_xyz, idx = stree.query(_latlon_xyz(w["latitude"].values, w["longitude"].values),
                                k=K_NEIGHBORS)
    dkm = 2 * EARTH_R_KM * np.arcsin(np.clip(dist_xyz / (2 * EARTH_R_KM), 0, 1))

    # IDW pondéré 1/d² sur les K plus proches
    wts = 1.0 / np.maximum(dkm, 1e-3) ** 2
    dtw = (s_dtw[idx] * wts).sum(axis=1) / wts.sum(axis=1)

    w["depth_to_water_m"] = dtw.round(2)
    w["depth_to_water_log1p"] = np.log1p(dtw).round(4)
    w["dtw_nearest_station_km"] = dkm[:, 0].round(3)
    w["dtw_far"] = (dkm[:, 0] > FAR_KM).astype(int)

    out = w[["gm_well_id", "latitude", "longitude", "depth_to_water_m",
             "depth_to_water_log1p", "dtw_nearest_station_km", "dtw_far"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"écrit {OUT} ({len(out)} puits)")
    print("\ndepth_to_water_m :")
    print(out["depth_to_water_m"].describe(percentiles=[.1, .5, .9]).round(1).to_string())
    print(f"\n% puits éloignés (>{FAR_KM} km d'une station) : {100 * w['dtw_far'].mean():.1f}%")


if __name__ == "__main__":
    main()
