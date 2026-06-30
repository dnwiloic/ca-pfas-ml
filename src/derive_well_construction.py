"""Dérive la profondeur d'extraction du puits (dimension verticale 3D).

Mécanisme PFAS : un puits peu profond capte une **nappe superficielle vulnérable**
à l'infiltration des PFAS de surface ; un puits profond capte un aquifère
**protégé**. C'est la dimension verticale absente du graphe 2D.

`well_depth_ft` existant (~19 % des puits) est déjà un fort prédicteur transférable ;
la **crépine** (`gm_top/bottom_depth_of_screen_ft`) de `gama_allwells.csv` couvre
~60 % des puits → on construit une profondeur effective à couverture maximale.

Source : data/raw/gama/gama_allwells.csv (depth + crépine, métadonnées puits)
         data/processed/well_hydraulic_gradient.csv (univers des 11 333 puits)
Sortie : data/processed/well_construction.csv (jointure sur gm_well_id)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALLWELLS = ROOT / "data" / "raw" / "gama" / "gama_allwells.csv"
WELLS = ROOT / "data" / "processed" / "well_hydraulic_gradient.csv"
OUT = ROOT / "data" / "processed" / "well_construction.csv"


def main() -> None:
    aw = pd.read_csv(
        ALLWELLS, encoding="latin-1", low_memory=False,
        usecols=["gm_well_id", "gm_well_depth_ft",
                 "gm_top_depth_of_screen_ft", "gm_bottom_depth_of_screen_ft"],
    ).drop_duplicates("gm_well_id")

    w = pd.read_csv(WELLS)[["gm_well_id", "latitude", "longitude"]]
    m = w.merge(aw, on="gm_well_id", how="left")

    top = m["gm_top_depth_of_screen_ft"]
    bot = m["gm_bottom_depth_of_screen_ft"]
    depth = m["gm_well_depth_ft"]

    # bornage : profondeurs plausibles (0–3000 ft), crépine bottom > top
    for s in (top, bot, depth):
        s.mask((s <= 0) | (s > 3000), inplace=True)
    bad = bot < top
    top, bot = top.mask(bad), bot.mask(bad)

    m["screen_top_ft"] = top.round(1)
    m["screen_bottom_ft"] = bot.round(1)
    m["screen_mid_ft"] = ((top + bot) / 2).round(1)
    m["screen_length_ft"] = (bot - top).round(1)
    m["well_depth_ft"] = depth.round(1)
    # profondeur effective : milieu de crépine, sinon profondeur totale (couverture max)
    m["depth_eff_ft"] = m["screen_mid_ft"].fillna(m["well_depth_ft"]).round(1)
    m["depth_eff_log1p"] = np.log1p(m["depth_eff_ft"].clip(lower=0)).round(4)
    m["depth_missing"] = m["depth_eff_ft"].isna().astype(int)

    cols = ["gm_well_id", "latitude", "longitude", "screen_top_ft", "screen_bottom_ft",
            "screen_mid_ft", "screen_length_ft", "well_depth_ft",
            "depth_eff_ft", "depth_eff_log1p", "depth_missing"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    m[cols].to_csv(OUT, index=False)

    print(f"écrit {OUT} ({len(m)} puits)")
    for c in ["screen_top_ft", "screen_bottom_ft", "screen_mid_ft",
              "screen_length_ft", "well_depth_ft", "depth_eff_ft"]:
        print(f"  {c:18s}: {100*m[c].notna().mean():5.1f}% non-NA  | médiane {m[c].median():.0f} ft")


if __name__ == "__main__":
    main()
