"""Dérive les indicateurs géochimiques redox par puits depuis GAMA.

Complète l'échelle redox déjà partiellement présente via les co-contaminants
(NO₃, Mn, Fe, SO₄, TDS) avec les paramètres manquants disponibles dans
`gama.csv` : oxygène dissous (DO), pH, ammonium (NH₃/NH₄ as N), TOC.

Mécanisme PFAS : les conditions redox contrôlent la transformation des précurseurs
en PFAA terminaux mobiles ; le pH module la sorption ; DO et NH₄ situent le puits
sur l'échelle oxique→anoxique. Signal *intrinsèque au puits* (pas un proxy
positionnel).

⚠️ Le redox est une propriété LOCALE (varie avec la profondeur/aquifère) → on
n'interpole PAS spatialement : médiane historique par puits + masque de manquant.
La haute proportion de NA est assumée (le HGT la gère par masque, XGBoost en natif).

Source : data/raw/gama/gama.csv (gm_chemical_vvl ∈ {DO, PH, NH3NH4N, TOCH})
         data/processed/well_hydraulic_gradient.csv (univers des 11 333 puits)
Sortie : data/processed/well_redox.csv (jointure sur gm_well_id)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GAMA = ROOT / "data" / "raw" / "gama" / "gama.csv"
WELLS = ROOT / "data" / "processed" / "well_hydraulic_gradient.csv"
OUT = ROOT / "data" / "processed" / "well_redox.csv"

# code VVL GAMA → nom de colonne de sortie
REDOX = {"DO": "redox_do_mgl", "PH": "redox_ph",
         "NH3NH4N": "redox_nh4n_mgl", "TOCH": "redox_toc_mgl"}
DETECTED = {"=", "J", ">"}


def main() -> None:
    raw = pd.read_csv(
        GAMA, encoding="latin-1", low_memory=False,
        usecols=["gm_well_id", "gm_chemical_vvl", "gm_result_modifier",
                 "gm_result", "gm_reporting_limit", "gm_latitude", "gm_longitude"],
        dtype={"gm_result": str, "gm_reporting_limit": str},
    )
    raw = raw[raw["gm_chemical_vvl"].isin(REDOX)].copy()
    result = pd.to_numeric(raw["gm_result"], errors="coerce")
    rl = pd.to_numeric(raw["gm_reporting_limit"], errors="coerce")
    det = raw["gm_result_modifier"].isin(DETECTED)
    raw["value"] = np.where(det, result, rl / 2.0)
    raw = raw.dropna(subset=["gm_well_id", "value"])

    # médiane historique par (puits, paramètre)
    piv = (raw.groupby(["gm_well_id", "gm_chemical_vvl"])["value"].median()
           .unstack("gm_chemical_vvl")).rename(columns=REDOX)

    w = pd.read_csv(WELLS)[["gm_well_id", "latitude", "longitude"]]
    out = w.merge(piv, on="gm_well_id", how="left")

    # flag = mesure DIRECTE au puits (avant tout fallback)
    for col in REDOX.values():
        if col not in out.columns:
            out[col] = np.nan
        out[f"{col}_observed"] = out[col].notna().astype(int)

    # --- fallback spatial IDW (comme les co-contaminants) : les paramètres redox
    #     ne sont quasi pas mesurés AUX puits PFAS → interpolation depuis les puits
    #     de mesure voisins. Rayon court (redox local) ; flag _observed conservé. ---
    src = (raw.groupby("gm_well_id")
           .agg(lat=("gm_latitude", "median"), lon=("gm_longitude", "median"))
           .join(raw.groupby("gm_well_id")["gm_chemical_vvl"].apply(set).rename("codes")))
    src_med = raw.groupby(["gm_well_id", "gm_chemical_vvl"])["value"].median().unstack("gm_chemical_vvl").rename(columns=REDOX)
    src = src.join(src_med).dropna(subset=["lat", "lon"])
    EARTH = 6371.0088
    def xyz(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.column_stack([EARTH*np.cos(la)*np.cos(lo), EARTH*np.cos(la)*np.sin(lo), EARTH*np.sin(la)])
    from scipy.spatial import cKDTree
    RADIUS_KM, K = 20.0, 5
    for col in REDOX.values():
        s = src.dropna(subset=[col])
        if len(s) < K:
            continue
        tree = cKDTree(xyz(s["lat"].values, s["lon"].values))
        need = out[col].isna().values
        if not need.any():
            continue
        dxyz, idx = tree.query(xyz(out.loc[need, "latitude"].values, out.loc[need, "longitude"].values), k=K)
        dkm = 2*EARTH*np.arcsin(np.clip(dxyz/(2*EARTH), 0, 1))
        wts = np.where(dkm <= RADIUS_KM, 1.0/np.maximum(dkm, 1e-3)**2, 0.0)
        vals = s[col].values[idx]
        num = (vals*wts).sum(1); den = wts.sum(1)
        filled = np.where(den > 0, num/np.where(den == 0, 1, den), np.nan)
        out.loc[need, col] = filled

    out["redox_ph"] = out["redox_ph"].clip(3, 12)            # pH plausible
    out["redox_do_mgl"] = out["redox_do_mgl"].clip(0, 20)    # DO plausible

    cols = (["gm_well_id", "latitude", "longitude"]
            + list(REDOX.values())
            + [f"{c}_observed" for c in REDOX.values()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(OUT, index=False)

    print(f"écrit {OUT} ({len(out)} puits)")
    print(f"{'paramètre':16s} {'direct':>7s} {'+fallback':>10s}  médiane")
    for col in REDOX.values():
        cov_d = 100 * out[f"{col}_observed"].mean()
        cov_f = 100 * out[col].notna().mean()
        print(f"  {col:14s} {cov_d:6.1f}% {cov_f:9.1f}%  {out[col].median():.2f}")


if __name__ == "__main__":
    main()
