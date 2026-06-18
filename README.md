# CA-PFAS-ASGWS — Reconstruction du dataset

Reproduction du pipeline décrit dans :

- Dong et al., *ACS EST Water* **2024**, 4, 969–981 ([A3.pdf](../../articles_2/V_en/A3/A3.pdf))
- Supporting Information ([ew3c00134_si_001.pdf](../../articles_2/V_en/A3/ew3c00134_si_001.pdf))

## Arborescence

```
ca-pfas-ml/
├── data/raw/          # téléchargements bruts (23 sources)
├── data/processed/    # jeux nettoyés + CA-PFAS-ASGWS.csv
├── notebooks/         # exploration
├── src/
│   ├── collect.py     # Phase 1
│   ├── clean.py       # Phase 2
│   ├── merge.py       # Phase 2–3
│   ├── ml_binary.py   # Phase 4 — tâche 1
│   └── ml_multilabel.py  # Phase 4 — tâche 2
├── outputs/           # cartes, figures (Phases 5–6)
└── requirements.txt
```

## Phases

| Phase | Script / dossier | Objectif |
|-------|------------------|----------|
| 0 | (structure) | Environnement et arborescence |
| 1 | `collect.py` | 23 sources (GAMA, EPA, météo, sol…) |
| 2 | `clean.py`, `merge.py` | Nettoyage, fusion spatiale, imputation |
| 3 | `merge.py` | Dataset final (~26 901 × 157, 38 PFAS, 4 classes) |
| 4 | `ml_binary.py`, `ml_multilabel.py` | Présélection binaire + multilabel |
| 5–6 | `notebooks/`, `outputs/` | Spearman, cartes de risque |

## Installation

```bash
cd ca-pfas-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Cibles article

- **Binaire** : somme PFAS > 70 ng/L
- **Multilabel** : chaque PFAS > 2 ng/L (seuils UCMR5)
- **Classes d’observation** : 0 (>25 000 obs) … 3 (>100 obs)

## Phase 1 — Collecte (`src/collect.py`)

23 sources enregistrées dans `src/sources_registry.py` (Table S1, SI).

```bash
cd ca-pfas-ml
pip install -r requirements.txt

python3 -m src.collect --list
python3 -m src.collect --sync-cache    # indexer data/raw existant
python3 -m src.collect --status

# GAMA : ne retélécharge pas ce qui est déjà en cache
python3 -m src.collect --group gama --max-mb 0

# Reprendre seulement les échecs / .part (ex. usgsnwis.csv.part)
python3 -m src.collect --retry-errors --max-mb 0
```

Cache : `data/cache/index.json`. Reprise : `*.part` + `*.part.meta`. Runs : `data/raw/manifest.json`.

| Catégorie | Automatique | Manuel / token |
|-----------|-------------|----------------|
| GAMA (12) | 11 CSV via CKAN + copie `usgs2_GAMA_PBP/Table_*.txt` | DOI USGS si absent |
| Contamination (4) | GeoTracker ArcGIS, EPA industry XLSX | NPDES, TRI, CDR (ECHO) |
| Environnement (7) | Bassins SGMA (GeoJSON) | NOAA, AQS, PurpleAir, NCSS, air ZIP |

**Note DDW** : l’article cite DDW 2015–2019 ; le portail expose désormais `ddw2010-2019.csv`.

## Prochaine étape

**Phase 2** — `clean.py` + `merge.py` (nettoyage et fusion spatiale).
