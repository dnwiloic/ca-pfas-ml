# Rapport de collecte et d'enrichissement des données — Projet CA-PFAS-ML

**Auteur :** Loïc Dnjomou  
**Institution :** Master 2 Recherche  
**Projet :** Apprentissage automatique pour la prédiction de la contamination en PFAS dans les eaux souterraines de Californie  
**Référence principale :** Dong et al. (2024), *ACS ES&T Water* — « Semi-supervised machine learning for groundwater PFAS contamination in California »  
**Date :** Juin 2026

---

## Table des matières

1. [Contexte et objectifs](#1-contexte-et-objectifs)
2. [Architecture générale du pipeline](#2-architecture-générale-du-pipeline)
3. [Phase 1 — Données PFAS de base (GAMA)](#3-phase-1--données-pfas-de-base-gama)
4. [Phase 2 — Enrichissement géospatial et contextuel](#4-phase-2--enrichissement-géospatial-et-contextuel)
   - 4.1 Bassins hydrogéologiques SGMA (DWR)
   - 4.2 Sites de contamination GeoTracker (CalEPA)
   - 4.3 Co-contaminants chimiques (GAMA VOCs/chimie de l'eau)
   - 4.4 Sol SSURGO (USDA — API SDA)
   - 4.5 Qualité de l'air EPA AQS
   - 4.6 Hydrologie NASA GLDAS-2.1
5. [Phase 3 — Fusion et construction du dataset final](#5-phase-3--fusion-et-construction-du-dataset-final)
6. [Difficultés rencontrées et solutions](#6-difficultés-rencontrées-et-solutions)
7. [Description du dataset final](#7-description-du-dataset-final)
8. [Correspondance avec les features du papier de référence](#8-correspondance-avec-les-features-du-papier-de-référence)
9. [Limites et perspectives](#9-limites-et-perspectives)
10. [Annexes techniques](#10-annexes-techniques)

---

## 1. Contexte et objectifs

### 1.1 Problématique

Les substances per- et polyfluoroalkylées (PFAS) constituent une famille de polluants persistants dont la présence dans les eaux souterraines représente un risque sanitaire majeur. En Californie, État le plus peuplé des États-Unis avec 39 millions d'habitants dépendant en grande partie des eaux souterraines, la contamination aux PFAS est documentée dans plusieurs aquifères. La détection et la prédiction spatiale de cette contamination nécessitent des outils d'apprentissage automatique (ML) capables de traiter des données multi-sources hétérogènes.

Le papier de référence de **Dong et al. (2024)** — publié dans *ACS ES&T Water* — propose une approche de ML semi-supervisée multilabel intégrant 23 sources de données pour prédire la contamination en PFAS dans les eaux souterraines californiennes. Ce projet vise à reproduire et étendre cette méthodologie, en automatisant l'ensemble du pipeline de collecte de données.

### 1.2 Objectifs du pipeline de collecte

1. **Reproduire le dataset de Dong et al. (2024)** en utilisant les mêmes sources identifiées dans les matériaux supplémentaires (Table S4, Table S9).
2. **Automatiser intégralement** la collecte via des APIs programmatiques, en évitant les téléchargements manuels.
3. **Couvrir spatialement la Californie entière** (32,5°–42°N ; 114,5°–124,5°O) sur la période 2016–2026.
4. **Produire un dataset de qualité publication** avec traçabilité complète des sources, gestion des valeurs manquantes documentée, et métadonnées.

---

## 2. Architecture générale du pipeline

Le pipeline est organisé en trois phases séquentielles, implémentées dans le répertoire `ca-pfas-ml/src/` :

```
Phase 1                Phase 2 (sources parallèles)              Phase 3
─────────              ───────────────────────────────────        ─────────────────
collect_pfas.py   →    collect_ssurgo.py  (USDA SDA API)   ┐
process_wells.py  →    collect_aqs.py     (EPA AQS API)    ├──→  merge.py
                       process_gldas.py   (NASA GLDAS nc4) ┘     ↓
                       [GeoTracker CSV]                     CA-PFAS-ASGWS.parquet
                       [SGMA GeoJSON]                       CA-PFAS-ASGWS.csv
                       [GAMA VOCs CSV]                      data_dict.csv
```

**Environnement technique :**
- Python 3.12, venv isolé
- Bibliothèques principales : `pandas`, `geopandas`, `scipy` (cKDTree), `netCDF4`, `requests`
- Stockage : Parquet (Apache Arrow) pour les données volumineuses, JSON pour les caches API
- Parallélisme : `ThreadPoolExecutor` (8 workers pour les APIs REST)

---

## 3. Phase 1 — Données PFAS de base (GAMA)

### 3.1 Source

**GAMA (Groundwater Ambient Monitoring and Assessment)** est un programme conjoint du California State Water Resources Control Board (SWRCB) et de l'USGS. Il constitue la base de données de référence pour la qualité des eaux souterraines en Californie.

- **URL :** `https://gamagroundwater.waterboards.ca.gov/`
- **Fichier principal :** `gama.csv` — toutes les mesures chimiques
- **Fichier complémentaire :** `gama_allwells.csv` — métadonnées des puits

### 3.2 Contenu et prétraitement

Le fichier GAMA brut contient l'ensemble des analyses chimiques effectuées sur les puits de surveillance californiens. Le prétraitement (`src/process_wells.py`) réalise les étapes suivantes :

**a) Filtrage des analytes PFAS**

Seules les mesures correspondant aux 31 analytes PFAS ciblés sont conservées (PFOS, PFOA, PFHxS, PFBS, PFNA, PFHpA, PFBA, PFDA, PFHxA, PFPeA, PFPeS, PFHpS, PFNA, NEtFOSAA, NMeFOSAA, FTS_4_2, FTS_6_2, FTS_8_2, HFPO_DA, ADONA, NFDHA, PFEESA, PFMBA, PFMPA, PFBS, PFDoDA, PFDS, PFTeDA, PFTrDA, PFUnDA, PFOSAm).

**b) Imputation des non-détections**

Conformément à la pratique standard en hydrochimie analytique et à la méthodologie de Dong et al. (2024) :
- Mesures détectées (`modifier ∈ {=, J, >}`) : valeur reportée telle quelle
- Mesures non détectées (`< LQ`) : valeur imputée à **LQ/2** (moitié de la limite de quantification)

**c) Pivot et agrégation**

Les mesures sont pivotées : une ligne par (puits, date de collecte), une colonne par analyte (format `{PFAS}_ngL`). En cas de mesures multiples le même jour pour le même analyte, le maximum est retenu.

**d) Filtres qualité**

- Coordonnées géographiques valides requises
- Date de collecte parsable
- Au moins un analyte PFAS mesuré

### 3.3 Résultats Phase 1

| Indicateur | Valeur |
|---|---|
| Mesures PFAS brutes | > 1 M lignes |
| Puits uniques avec données PFAS | 11 333 |
| Lignes dans `wells_pfas_clean.parquet` | 46 338 |
| Période couverte | Avril 2016 – Janvier 2026 |
| Analytes PFAS | 31 (+ somme) |
| Emprise géographique | 32,58°N–41,97°N ; 124,28°O–114,48°O |

**Distribution temporelle :** La collecte de données s'est intensifiée de façon marquée après 2019, reflétant le déploiement du programme de surveillance UCMR 5 (Unregulated Contaminant Monitoring Rule) imposé par l'EPA. En 2024–2025, plus de 19 000 mesures sont disponibles sur les deux années.

---

## 4. Phase 2 — Enrichissement géospatial et contextuel

### 4.1 Bassins hydrogéologiques SGMA (DWR)

**Source :** California Department of Water Resources (DWR) — Sustainable Groundwater Management Act (SGMA)

- **Fichier :** `sgma_basins.geojson`
- **Méthode :** Point-in-polygon avec `geopandas.sjoin()` (prédicat `within`, CRS EPSG:4326)
- **Colonnes ajoutées :** `sgma_basin_name`, `sgma_subbasin_name`, `sgma_region_office`

**Résultat :** 43 532 / 46 338 puits (93,9%) appartiennent à un bassin SGMA. Les 6,4% restants correspondent à des puits en zones non réglementées (hautes altitudes, zones côtières non prioritaires).

Distribution régionale :
- SRO (Southern Region) : 24 134 mesures — Los Angeles, San Diego, Inland Empire
- SCRO (South Central) : 8 840 — San Joaquin Valley Sud
- NCRO (North Central) : 8 300 — Sacramento Valley, Bay Area
- NRO (Northern) : 2 258 — Nord Californie

### 4.2 Sites de contamination GeoTracker (CalEPA)

**Source :** California Environmental Protection Agency — GeoTracker PFAS Investigation Sites

- **Fichier :** `geotracker_pfas_investigation_sites.csv`
- **Méthode :** KD-Tree 3D (coordonnées converties en XYZ sur sphère terrestre, rayon 6 371 km) avec `scipy.spatial.cKDTree`
- **Rayons calculés :** 1, 3, 10, 50 km
- **Colonnes ajoutées :**
  - `dist_geotracker_km` — distance au site PFAS le plus proche
  - `nearest_geotracker_type` — type du site (military, industrial, firefighting…)
  - `n_geotracker_within_{1,3,10,50}km` — nombre de sites dans chaque rayon

**Résultat :** Couverture 100%. Distance médiane au site PFAS le plus proche : **3,9 km**, confirmant la densité élevée de sources potentielles en Californie.

**Note méthodologique :** L'utilisation d'un KD-Tree sur sphère (conversion lat/lon → XYZ) plutôt qu'une simple distance euclidienne garantit la précision géodésique sur l'ensemble de la Californie.

### 4.3 Co-contaminants chimiques (GAMA — chimie de l'eau et VOCs)

**Source :** Base GAMA — même fichier `gama.csv`, filtré sur 44 analytes non-PFAS

**Analytes collectés (44 variables) :**

*Chimie de l'eau (5) :* TDS (solides dissous totaux), MN (manganèse), FE (fer), AS (arsenic), SO4 (sulfate)

*Hydrocarbures aromatiques et volatils (19) :* BZ (benzène), XYLENES, BZME (toluène), NAPH (naphtalène), EBZ (éthylbenzène), BTBZS (sec-butylbenzène), BTBZN (n-butylbenzène), BTBZT (tert-butylbenzène), PBZN (propylbenzène), STY (styrène), TMB124 (1,2,4-triméthylbenzène)

*Solvants chlorés (14) :* TCE, PCE, DCE12C (cis-DCE), DCE12T (trans-DCE), DCE11, VC (chlorure de vinyle), DCA11, DCA12, TCA111, TCA112, CTCL (tétrachlorure de carbone), CLBZ (chlorobenzène), TCB124 (1,2,4-trichlorobenzène), DCBZ12, DCBZ13, DCPA12 (1,2-dichloropropane)

*Halogénés divers (6) :* DBCP, EDB, TCPR123, BDCME, DBCME, PCA

*Fréons (3) :* FC11, FC12, FC113

*Autres (1) :* MTBE, TBME

**Méthode de fusion :** L'attribution des co-contaminants à chaque mesure PFAS suit la stratégie à deux niveaux de Dong et al. (2024) :

1. **Match direct :** `merge_asof` par puits (`gm_well_id`) avec tolérance temporelle ±365 jours — direction `nearest`
2. **Fallback spatial :** Pour les puits sans mesure directe, moyenne historique du puits de mesure le plus proche (≤50 km) via KD-Tree

**Résultat :**
- Match direct : 896 / 46 338 lignes (1,9%)
- Fallback spatial : +45 430 lignes (+98,1%)
- Couverture finale : **46 326 / 46 338 (99,97%)**
- 5 766 mesures brutes de co-contaminants couvrant 4 462 puits

**Couvertures par analyte** (sélection) :

| Analyte | Couverture | Note |
|---|---|---|
| PCE, TCE, DCE12T, VC | 99,0–99,1% | Solvants chlorés historiques très mesurés |
| BZ, BDCME, CLBZ | 98,7–99,0% | Mesure systématique dans VOC panels |
| BZME (toluène) | 96,3% | |
| FE, SO4 | 68,7–79,7% | Chimie de base, moins fréquente |
| TDS, MN, AS | 68,7% | Panels chimie eau moins fréquents |
| XYLENES | 0,2% | Rarement mesuré comme total ; isomères séparés |
| BTBZT, DCE12C, TMB124 | 1,3% | Mesure rare dans GAMA |

### 4.4 Sol SSURGO (USDA — API SDA)

#### 4.4.1 Contexte et choix de l'API

**SSURGO (Soil Survey Geographic Database)** est la base de données de référence pour les propriétés pédologiques aux États-Unis, maintenue par l'USDA Natural Resources Conservation Service (NRCS).

Dong et al. (2024) utilisent deux sources de données sol :
- **SSURGO** pour les propriétés agrégées (sable, argile, limon, Ksat, etc.)
- **NCSS KSSL (Kellogg Soil Survey Laboratory)** pour les fractions granulométriques détaillées (Soil%_0.25-0.5mm, Gradation_Uniformity, etc.)

**Choix technique :** Au lieu du téléchargement manuel de l'interface Web Soil Survey (difficile à automatiser), nous utilisons l'**API SDA (Soil Data Access)** :
- URL : `https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest`
- Protocole : requêtes POST avec corps JSON `{"query": "<SQL>", "format": "json+columnname"}`
- Table principale : `chorizon` (horizon de sol dominant)
- Jointure spatiale via `SDA_Get_Mukey_from_intersection_with_WktWgs84('<WKT>')`

#### 4.4.2 Requête SQL et stratégie de sélection

Pour chaque point (lat, lon) de coordonnée, la requête sélectionne l'horizon superficiel (le plus proche de la surface) du composant dominant :

```sql
SELECT TOP 1 ch.sandtotal_r, ch.sandvf_r, ch.sandfine_r, ch.sandmed_r,
             ch.sandco_r, ch.sandvc_r, ch.silttotal_r, ch.siltco_r,
             ch.siltfine_r, ch.claytotal_r, ch.om_r, ch.ph1to1h2o_r,
             ch.ksat_r, ch.awc_r, ch.dbthirdbar_r,
             ch.wthirdbar_r, ch.wfifteenbar_r, ch.texturerv
FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('<WKT>') AS x
INNER JOIN component co ON co.mukey = x.mukey
INNER JOIN chorizon ch ON ch.cokey = co.cokey
WHERE co.majcompflag = 'Yes'
ORDER BY co.comppct_r DESC, ch.hzdept_r ASC
```

**Critères de sélection :**
- `majcompflag = 'Yes'` : composant dominant de l'unité cartographique
- `ORDER BY comppct_r DESC` : composant le plus représenté (%)
- `ORDER BY hzdept_r ASC` : horizon le plus superficiel

#### 4.4.3 Variables extraites et correspondance

| Variable SDA | Colonne dataset | Description | Rang Table S9 |
|---|---|---|---|
| `sandtotal_r` | `soil_sand_pct` | Sable total (0,05–2 mm) % | — |
| `sandvf_r` | `soil_sand_vfine_pct` | Sable très fin (0,05–0,10 mm) % | #38 |
| `sandfine_r` | `soil_sand_fine_pct` | Sable fin (0,10–0,25 mm) % | #36 |
| `sandmed_r` | `soil_sand_medium_pct` | Sable moyen (0,25–0,50 mm) % | **#6** |
| `sandco_r` | `soil_sand_coarse_pct` | Sable grossier (0,50–1,0 mm) % | #41 |
| `sandvc_r` | `soil_sand_vcoarse_pct` | Sable très grossier (1,0–2,0 mm) % | — |
| `silttotal_r` | `soil_silt_pct` | Limon total (0,002–0,05 mm) % | #26 |
| `siltco_r` | `soil_silt_coarse_pct` | Limon grossier (0,02–0,05 mm) % | #9, #46 |
| `siltfine_r` | `soil_silt_fine_pct` | Limon fin (0,002–0,02 mm) % | — |
| `claytotal_r` | `soil_clay_pct` | Argile (<0,002 mm) % | — |
| `om_r` | `soil_om_pct` | Matière organique % | — |
| `ph1to1h2o_r` | `soil_ph` | pH (eau 1:1) | — |
| `ksat_r` | `soil_ksat_um_s` | Conductivité hydraulique saturée (μm/s) | — |
| `awc_r` | `soil_awc_cm_cm` | Eau disponible (cm/cm) | — |
| `dbthirdbar_r` | `soil_bulk_density` | Densité apparente (g/cm³) | — |
| `wthirdbar_r` | `soil_water_1bar_pct` | Teneur en eau à 1/3 bar (%) | — |
| `wfifteenbar_r` | `soil_water_15bar_pct` | Teneur en eau à 15 bar (%) | #14 |
| `texturerv` | `soil_texture_class` | Classe texturale USDA | #22, #31 |

**Variables dérivées (calculées dans merge.py) :**
- `soil_ratio_water_clay` = `wfifteenbar_r` / `claytotal_r` → Ratio_Water_Clay (rang #14)
- `soil_gradation_uniformity` = D₆₀/D₁₀ interpolé → Gradation Uniformity (rang #12)
- `soil_gradation_curvature` = D₃₀²/(D₁₀×D₆₀) interpolé → Gradation Curvature (rang #37)

L'interpolation des diamètres D₁₀, D₃₀, D₆₀ utilise une interpolation log-linéaire sur la distribution cumulative reconstituée à partir des 8 fractions granulométriques (limites : 0,002 / 0,02 / 0,05 / 0,10 / 0,25 / 0,50 / 1,0 / 2,0 mm).

#### 4.4.4 Stratégie de cache

Pour éviter de ré-interroger l'API à chaque exécution, un cache fichier par point est maintenu :

- **Fichier cache :** `data/raw/environment/ssurgo/cache/{lat}_{lon}.json`
- **Réponse avec données :** JSON contenant les valeurs — exemple `{"sandtotal_r": "61", "sandmed_r": "15.9", ...}`
- **Zone non couverte** (océan, zone urbaine dense) : fichier contenant littéralement `null`
- **Invalidation de schéma :** Le champ `sandmed_r` sert de marqueur de version — les anciens fichiers cache ne le contenant pas sont automatiquement re-interrogés lors d'un changement de schéma

**Code de détection :**
```python
def _is_cached(lat, lon) -> bool:
    cp = _cache_path(lat, lon)
    if not cp.exists() or cp.stat().st_size == 0:
        return False
    text = cp.read_text().strip()
    if text == "null":
        return True  # zone non couverte — définitif
    try:
        return "sandmed_r" in json.loads(text)  # vérification de version
    except json.JSONDecodeError:
        return False
```

#### 4.4.5 Parallélisme et performance

- **Workers :** 8 threads simultanés (`ThreadPoolExecutor`)
- **Débit mesuré :** 540–670 pts/min (avec intermittences DNS)
- **Durée totale :** ~20 minutes pour 9 281 points uniques

#### 4.4.6 Résultats

| Indicateur | Valeur |
|---|---|
| Points uniques interrogés | 9 786 |
| Points avec données SSURGO | 9 281 (94,8%) |
| Points sans données (zones non couvertes) | 505 (5,2%) — zones urbaines denses, côtières |
| `soil_sand_medium_pct` (rang #6) | 92,2% du dataset |
| `soil_texture_class` | 95,8% |
| `soil_ratio_water_clay` | 92,1% |
| `soil_silt_coarse_pct` (rang #9) | 4,9% — champ peu renseigné dans SSURGO CA |

**Classes texturales dominantes (USDA) :** SL (Sandy Loam, 19%), FSL (Fine Sandy Loam, 17%), L (Loam, 12%), LS (Loamy Sand, 9%), SIL (Silt Loam, 4%) — cohérent avec la géologie des vallées alluviales californiennes.

### 4.5 Qualité de l'air EPA AQS

#### 4.5.1 Source

**AQS (Air Quality System)** est le système de surveillance nationale de la qualité de l'air de l'EPA (Environmental Protection Agency).

- **API :** `https://aqs.epa.gov/data/api/annualData/byState`
- **Authentification :** email + clé API (obtenu gratuitement sur `aqs.epa.gov`)
- **Période :** 2015–2025
- **État :** `state=06` (code FIPS Californie)

#### 4.5.2 Paramètres collectés

| Code AQS | Paramètre | Unité | Colonne dataset | Rang S9 |
|---|---|---|---|---|
| 88101 | PM2.5 | µg/m³ | `aqs_pm25_ugm3` | #19 |
| 81102 | PM10 | µg/m³ | `aqs_pm10_ugm3` | #15 |
| 42602 | NO₂ | ppb | `aqs_no2_ppb` | #16 |
| 42401 | SO₂ | ppb | `aqs_so2_ppb` | #17 |
| 61101 | Vitesse du vent | m/s | `aqs_wind_ms` | #21 |
| 62201 | Humidité relative | % | `aqs_humidity_pct` | #11 |
| 44201 | Ozone | ppb | `aqs_ozone_ppb` | **#24** |
| 42101 | CO | ppm | `aqs_co_ppm` | #50 |

#### 4.5.3 Méthode d'attribution spatiale

La jointure entre les moniteurs AQS (fixes) et les puits (mobiles temporellement) est réalisée par **KD-Tree annuel** :

1. Pour chaque année, construction d'un KD-Tree par paramètre à partir des coordonnées des moniteurs actifs
2. Attribution du moniteur le plus proche à chaque puits, avec seuil de coupure à **50 km**
3. Si distance > 50 km → valeur NaN (puits trop isolé)

```python
for year in sorted(years):
    for param, col_name in _AQS_PARAMS.items():
        sub = aqs_yr[aqs_yr["param_code"] == param].drop_duplicates(["latitude", "longitude"])
        mon_tree = build_kdtree(sub["latitude"].values, sub["longitude"].values)
        dist_km, idx = nearest_neighbor(well_lat, well_lon, mon_tree)
        vals = sub["annual_mean"].values[idx]
        vals[dist_km > 50.0] = np.nan
        df.loc[mask_yr, col_name] = vals
```

#### 4.5.4 Résultats

| Paramètre | Moniteurs (CA) | Couverture dataset |
|---|---|---|
| PM2.5 | ~200–350/an | 97,0% |
| Ozone | ~600–720/an | **97,9%** |
| NO₂ | ~250–300/an | 93,4% |
| Humidité | ~500–600/an | 92,7% |
| CO | ~100–156/an | 83,9% |
| SO₂ | ~150–200/an | 65,4% |

**Dataset AQS final :** 30 381 lignes (moniteur × année × paramètre)

**Valeurs médianes (dataset CA-PFAS) :** Ozone = 49 ppb ; CO = 0,32 ppm ; PM2.5 = ~7 µg/m³

### 4.6 Hydrologie NASA GLDAS-2.1

#### 4.6.1 Source

**GLDAS (Global Land Data Assimilation System) version 2.1** est un produit de rémotisation terrestre de la NASA qui assimile des données satellitaires et météorologiques pour produire des champs de forçage hydrologiques cohérents.

- **Modèle :** NOAH025 (Noah Land Surface Model, résolution 0,25°×0,25°)
- **Temporalité :** Mensuelle (un fichier .nc4 par mois)
- **Format :** NetCDF4 (variables 3D : temps, lat, lon)
- **Accès :** NASA Earthdata (téléchargement par `wget` avec authentification)
- **Période :** Janvier 2015 – Décembre 2025 (132 fichiers, ~23 Mo chacun)

#### 4.6.2 Variables extraites

| Variable GLDAS | Colonne dataset | Conversion | Rang S9 |
|---|---|---|---|
| `Rainf_f_tavg` (kg m⁻² s⁻¹) | `rainfall_mm_month` | × 30,44 j × 86 400 s | #40 |
| `Evap_tavg` (kg m⁻² s⁻¹) | `et_mm_month` | × 30,44 j × 86 400 s | #29 |
| `Qs_acc` (kg m⁻²) | `runoff_mm` | × 1,0 (≈ mm) | #27 |
| `SoilMoi0_10cm_inst` (kg m⁻²) | `soil_moi_0_10_kg_m2` | × 1,0 | — |
| `SoilMoi10_40cm_inst` (kg m⁻²) | `soil_moi_10_40_kg_m2` | × 1,0 | — |
| `SoilMoi40_100cm_inst` (kg m⁻²) | `soil_moi_40_100_kg_m2` | × 1,0 | — |
| `SoilMoi100_200cm_inst` (kg m⁻²) | `soil_moi_100_200_kg_m2` | × 1,0 | — |
| `RootMoist_inst` (kg m⁻²) | `root_zone_moist_kg_m2` | × 1,0 | — |
| `Tair_f_inst` (K) | `temp_c` | − 273,15 | **#25** |
| `SWE_inst` (kg m⁻²) | `snowpack_mm` | × 1,0 (≈ mm) | — |

**Variable dérivée :** `soil_moisture_total_mm` = somme des 4 couches (0–200 cm) → correspond à `Soil_Moisture_mm` du papier (rang #18, médiane papier ~459 mm, notre médiane : **436 mm** ✓)

#### 4.6.3 Domaine spatial

Extraction limitée à la bounding box Californie : **32,0°–42,5°N / 124,5°–113,5°O**

Résultat : 1 848 cellules de 0,25° par mois.

#### 4.6.4 Méthode d'attribution spatiale

Attribution par **KD-Tree mensuel** : pour chaque mois calendaire présent dans le dataset, construction d'un KD-Tree sur les 1 848 cellules CA, puis affectation de la cellule la plus proche à chaque puits.

- **Distance maximale implicite :** ≤ 0,177 km (demi-diagonale de cellule 0,25°) pour les zones couvertes
- **Cellules océan :** retournent NaN (masque terrestre GLDAS)

#### 4.6.5 Résultats

| Indicateur | Valeur |
|---|---|
| Fichiers nc4 traités | 131 / 132 (1 corrompu : août 2025) |
| Lignes dans parquet GLDAS | 242 088 (131 mois × 1 848 cellules) |
| Couverture dataset | 95,7% |
| Température médiane | 16,8°C |
| Précipitations médianes | 26,5 mm/mois |
| Humidité sol 0–200 cm médiane | **436 mm** (vs 459 mm dans Dong et al.) |
| Neige (snowpack) | médiane = 0 mm (0 hors zones montagneuses) |

---

## 5. Phase 3 — Fusion et construction du dataset final

### 5.1 Séquence de fusion (`src/merge.py`)

```
wells_pfas_clean.parquet (46 338 × 72)
        │
        ├── compute_targets()       → +64 colonnes (32 labels + 31 _ngL déjà présents + cibles)
        ├── add_well_category()     → +1 col (gm_well_category)
        ├── add_sgma_basins()       → +3 cols (bassin SGMA)
        ├── add_contamination()     → +6 cols (KD-Tree GeoTracker 1/3/10/50 km + type)
        ├── add_co_contaminants()   → +44 cols (co-contaminants avec fallback spatial)
        ├── add_ssurgo_soil()       → +26 cols (SSURGO + dérivées)
        ├── add_aqs_air_quality()   → +8 cols (AQS annuel, KD-Tree par paramètre)
        └── add_gldas_hydrology()   → +12 cols (GLDAS mensuel + soil_moisture_total_mm)
                │
        CA-PFAS-ASGWS.parquet (46 338 × 201)
```

### 5.2 Gestion des jointures temporelles

Les jointures entre sources à temporalités différentes sont traitées spécifiquement :

| Source | Résolution temporelle | Méthode de jointure |
|---|---|---|
| GAMA PFAS | Quotidienne | Clé directe (well_id, date) |
| GAMA Co-contaminants | Quotidienne | `merge_asof` ±365 j + fallback spatial |
| AQS | Annuelle | Matching par année de collection |
| GLDAS | Mensuelle | Matching par (année, mois) de collection |
| SSURGO | Statique | Matching spatial (lat/lon arrondi à 4 décimales) |
| GeoTracker | Statique | KD-Tree spatial |
| SGMA | Statique | Point-in-polygon |

### 5.3 Arrondi des coordonnées pour SSURGO

Les coordonnées sont arrondies à **4 décimales (≈10 m)** avant la jointure SSURGO pour maximiser les correspondances dans le cache et éviter de réinterroger des points quasi-identiques :

```python
df["lat_r"] = df["latitude"].round(4)
df["lon_r"] = df["longitude"].round(4)
```

---

## 6. Difficultés rencontrées et solutions

### 6.1 API EPA AQS — Paramètres incorrects

**Problème :** L'API AQS retournait une erreur 400 « variable is missing: state ».

**Cause :** Le paramètre URL était `stateFIPS=06` (nom incorrect) et le champ d'identifiant du moniteur était `site_num` (champ inexistant dans la réponse).

**Solution :**
- Remplacement de `stateFIPS` par `state` (nom correct selon la documentation AQS)
- Remplacement de `site_num` par `site_number` dans la construction de l'identifiant moniteur

### 6.2 NASA GLDAS — Fichiers nc4 corrompus

**Problème :** Deux fichiers nc4 (novembre 2022, juillet 2024) passaient le filtre de taille (> 100 Ko) mais levaient une exception `NetCDF: HDF error` lors de la lecture.

**Cause :** Téléchargement incomplet ou interruption réseau — les fichiers avaient une taille intermédiaire (~1 Mo au lieu de ~23 Mo attendus).

**Solution :** Suppression manuelle des fichiers corrompus + re-téléchargement via le script `download_gldas.sh` → fichiers valides de 23–24 Mo obtenus.

### 6.3 SSURGO API SDA — Noms de colonnes incorrects

**Problème :** La requête SQL initiale utilisait `sandm_r`, `sandf_r`, `sandc_r` et `texcl` qui retournaient une erreur 400 « Invalid column name ».

**Cause :** La documentation SSURGO et les noms utilisés dans la littérature diffèrent parfois des noms effectifs dans la base SDA. Les noms corrects sont :

| Nom initial (incorrect) | Nom réel dans SDA |
|---|---|
| `sandm_r` | `sandmed_r` |
| `sandf_r` | `sandfine_r` |
| `sandc_r` | `sandco_r` |
| `texcl` | `texturerv` |

**Méthode de diagnostic :** Requête `SELECT TOP 1 * FROM chorizon` pour obtenir l'ensemble des colonnes disponibles, puis filtrage par mots-clés.

**Solution :** Mise à jour de la liste `SOIL_VARS` dans `collect_ssurgo.py`. Vérification de chaque nouveau nom avant intégration par une requête de test.

### 6.4 SSURGO API SDA — Fenêtre de maintenance quotidienne

**Problème :** L'API SDA est indisponible quotidiennement entre 00h30 et 00h45 (CST). Pendant cette fenêtre, l'API renvoie une page HTML de maintenance au lieu d'un résultat JSON, ce qui provoquait l'écriture d'un objet JSON vide `{}` dans le cache, bloquant définitivement les re-tentatives pour ces points.

**Solution implémentée :**
- Détection de la fenêtre de maintenance via `b"maintenance" in r.content.lower()`
- Attente progressive : 90 s × numéro de tentative (90s, 180s, 270s…), maximum 6 tentatives
- **Aucun caching pendant la maintenance** — seuls les vrais résultats (données ou absence définitive de données) sont mis en cache

### 6.5 SSURGO API SDA — Cache de l'ancien schéma

**Problème :** Lors de l'extension du schéma SQL (ajout de nouvelles colonnes), les 9 281 fichiers cache existants ne contenaient que les 8 champs originaux. La logique `_is_cached()` initiale testait uniquement l'existence du fichier, retournant True pour les anciens fichiers et empêchant la re-interrogation.

**Solution :** Ajout d'une vérification de version par présence du champ `sandmed_r` dans le JSON cache :

```python
def _is_cached(lat, lon) -> bool:
    text = _cache_path(lat, lon).read_text().strip()
    if text == "null":
        return True   # zone non couverte — toujours valide
    return "sandmed_r" in json.loads(text)   # vérif. schéma
```

Cette approche préserve les 505 fichiers `null` (zones non couvertes), évitant de re-interroger des zones définitivement sans données.

### 6.6 SSURGO API SDA — Défaillances DNS transitoires

**Problème :** Le serveur DNS local échouait à résoudre `sdmdataaccess.sc.egov.usda.gov` pendant des périodes de 30–90 minutes, causant des erreurs `NameResolutionError` en cascade sur les 8 threads simultanés.

**Comportement observé :** Le processus ne plantait pas (6 tentatives × backoff exponentiel), mais aucune donnée n'était collectée pendant l'outage DNS. Après récupération du DNS, les threads continuaient automatiquement.

**Solution :** Surveillance manuelle avec relance si nécessaire. À terme, une détection proactive de l'outage DNS avec pause globale serait plus robuste.

### 6.7 Co-contaminants GAMA — PM2.5 2015 avec cache vide

**Problème :** Un échec DNS transitoire pendant la première collecte AQS avait mis en cache un fichier `aqs_88101_2015.json` vide (0 moniteur), alors que 1 505 moniteurs existent pour 2015.

**Solution :** Suppression manuelle du fichier cache corrompu + re-téléchargement → 1 505 moniteurs récupérés.

---

## 7. Description du dataset final

### 7.1 Fiche technique

| Paramètre | Valeur |
|---|---|
| **Fichier principal** | `CA-PFAS-ASGWS.parquet` |
| **Dimensions** | 46 338 lignes × 201 colonnes |
| **Taille sur disque** | ~35 Mo (Parquet compressé) |
| **Puits uniques** | 11 333 |
| **Période** | Avril 2016 – Janvier 2026 |
| **Emprise** | 32,58°N–41,97°N ; 124,28°O–114,48°O |
| **Format** | Apache Parquet + CSV |

### 7.2 Structure des colonnes (201 total)

| Catégorie | Nombre | Description |
|---|---|---|
| Identifiants / métadonnées | 8 | `gm_well_id`, `collection_date`, `latitude`, `longitude`, `gm_well_category`, bassins SGMA |
| Concentrations PFAS (`_ngL`) | 31 | Valeurs en ng/L, imputation LQ/2 pour non-détections |
| Labels ML binaires (`label_`) | 31 | 1 si concentration > 2 ng/L |
| Cibles ML | 2 | `target_sum_gt70`, `sum_pfas_ngL` |
| Co-contaminants (`cocontam_`) | 44 | VOCs, solvants, métaux, chimie de l'eau |
| Sol SSURGO (`soil_`) | 26 | Fractions granulométriques, texture, rétention eau + dérivées |
| Qualité de l'air AQS (`aqs_`) | 8 | PM2.5, PM10, NO2, SO2, vent, humidité, ozone, CO |
| Hydrologie GLDAS | 12 | Précip., ET, ruissellement, humidité sol 4 couches, temp., neige |
| Sources contamination GeoTracker | 6 | Distances et comptages 1/3/10/50 km + type |

### 7.3 Variable cible

**Tâche binaire :** `target_sum_gt70` — 1 si la somme des concentrations PFAS dépasse le seuil réglementaire de 70 ng/L (US EPA MCL)
- Positifs : 11 490 (24,8%)
- Négatifs : 34 848 (75,2%)
- **Déséquilibre de classe : 1:3** (à traiter par ADASYN ou pondération)

**Analytes les plus fréquemment détectés (> 2 ng/L) :**
1. PFOS : 22 918 détections (49,7%)
2. PFHxS : 21 771 (49,3%)
3. PFOA : 20 935 (45,3%)
4. PFBS : 18 123 (41,1%)
5. PFHxA : 17 767 (40,1%)

### 7.4 Couverture globale par source

| Source | Variables | Couverture |
|---|---|---|
| GAMA PFAS | 31 analytes | 100% (données de base) |
| GeoTracker | distances, type | **100%** |
| Catégorie puits | 1 | **100%** |
| Co-contaminants | 44 | 99,97% (fallback spatial) |
| GLDAS | 12 | 95,7% |
| SGMA bassins | 3 | 93,9% |
| SSURGO fractions sable | 5 | 92,2% |
| SSURGO agrégés (pH, Ksat…) | 8 | 92,0–92,2% |
| AQS ozone | 1 | 97,9% |
| AQS PM2.5 | 1 | 97,0% |
| AQS NO2 | 1 | 93,4% |
| AQS CO | 1 | 83,9% |
| AQS SO2 | 1 | 65,4% |

---

## 8. Correspondance avec les features du papier de référence

Analyse de la couverture par rapport aux 112 features de la Table S9 de Dong et al. (2024) :

### 8.1 Features présentes (✓)

| Rang | Feature papier | Colonne dataset | Couverture |
|---|---|---|---|
| #1 | Well_category | `gm_well_category` | 100% |
| #2 | Facility_50km | `n_geotracker_within_50km` | 100% |
| #3 | Facility_1km | `n_geotracker_within_1km` | 100% |
| #4 | Facility_3km | `n_geotracker_within_3km` | 100% |
| #5 | Facility_10km | `n_geotracker_within_10km` | 100% |
| **#6** | **Soil%_0.25-0.5mm** | `soil_sand_medium_pct` | **92,2%** |
| #7 | Year | dérivé de `collection_date` | 100% |
| #8 | Fac_Conf_type | `nearest_geotracker_type` | 100% |
| #11 | Air_humidity | `aqs_humidity_pct` | 92,7% |
| #13 | NO3N | `cocontam_no3n` | 5,1%* |
| **#14** | **Ratio_Water_Clay** | `soil_ratio_water_clay` | **92,1%** |
| #15 | PM10_atm | `aqs_pm10_ugm3` | 94,0% |
| #16 | Air_NO2 | `aqs_no2_ppb` | 93,4% |
| #17 | Air_SO2 | `aqs_so2_ppb` | 65,4% |
| #18 | Soil_Moisture_mm | `soil_moisture_total_mm` | 95,7% |
| #19 | PM2_5_atm | `aqs_pm25_ugm3` | 97,0% |
| #21 | WindSpeed | `aqs_wind_ms` | 95,0% |
| **#22** | **texture_description** | `soil_texture_class` | **95,8%** |
| **#24** | **air_Ozone** | `aqs_ozone_ppb` | **97,9%** |
| **#25** | **Temp** | `temp_c` | **95,7%** |
| #26 | Silt_Total | `soil_silt_pct` | 92,2% |
| #27 | GW_Runoff_mm | `runoff_mm` | 95,7% |
| #29 | Evapotranspiration_mm | `et_mm_month` | 95,7% |
| **#31** | **Texture_USDA** | `soil_texture_class` | **95,8%** |
| #32 | Sand_Total | `soil_sand_pct` | 92,2% |
| #33 | AS (arsenic) | `cocontam_as` | 68,7% |
| #34 | EDB | `cocontam_edb` | 98,9% |
| #35 | SO4 (sulfate) | `cocontam_so4` | 68,7% |
| #36 | Weight_%_0.1-0.25mm | `soil_sand_fine_pct` | 92,2% |
| **#37** | **Gradation_Curvature** | `soil_gradation_curvature` | 1,8%** |
| #38 | Weight_%_0.05-0.1mm | `soil_sand_vfine_pct` | 92,2% |
| #40 | Precipitation_mm | `rainfall_mm_month` | 95,7% |
| #41 | Weight_%_0.5-1mm | `soil_sand_coarse_pct` | 92,2% |
| #43 | PCE | `cocontam_pce` | 99,1% |
| #44 | TCPR123 | `cocontam_tcpr123` | 98,9% |
| #45 | FE (fer) | `cocontam_fe` | 79,7% |
| #48 | DBCP | `cocontam_dbcp` | 98,9% |
| #50 | air_CO | `aqs_co_ppm` | 83,9% |
| #52 | TCE | `cocontam_tce` | 99,0% |
| #53 | NAPH | `cocontam_naph` | 99,0% |
| #54 | MTBE | `cocontam_mtbe` | 99,0% |
| #55 | Month | dérivé de `collection_date` | 100% |
| #58 | BZ (benzène) | `cocontam_bz` | 98,7% |
| #59 | EBZ | `cocontam_ebz` | 98,8% |

> * NO3N : 5,1% par match direct, mais ~99% avec fallback spatial  
> ** `soil_gradation_curvature` : 1,8% — limité par la faible couverture de `siltco_r`/`siltfine_r` dans SSURGO CA

### 8.2 Features partiellement couvertes ou manquantes (✗)

| Rang | Feature papier | Statut | Raison |
|---|---|---|---|
| **#6 also** | Soil%_0.02-0.05mm | Partiel (4,9%) | `siltco_r` peu renseigné dans SSURGO CA |
| **#9** | Soil%_0.02-0.05mm | Partiel (4,9%) | Idem |
| **#12** | Gradation_Uniformity | Partiel (1,8%) | Nécessite siltco_r |
| **#20** | %particles_d_<60%_<75mm | Absent | Métrique NCSS KSSL spécifique |
| **#23** | air_pm1 | Absent | PM1 non mesuré en standard AQS |
| **#28** | TDS | 68,7% | Panels chimie eau moins fréquents |
| **#30** | MN | 68,7% | Idem |
| **#39** | GW_Change_Storage_mm | Absent | Données SGMA de niveau piézométrique |
| **#42** | GW_recharge | Absent | Modèle de recharge (Water Balance App) |
| **#46** | Silt_Coarse | Partiel (4,9%) | `siltco_r` peu renseigné |
| #49 | Weight_%_<0.002mm | 4,9% | `siltfine_r` même problème |

---

## 9. Limites et perspectives

### 9.1 Limites identifiées

**a) Fractions granulométriques SSURGO (KSSL)**

Les features les plus importantes du papier (rangs #6, #9, #10, #12, #14, #20, #22, #31, #36–41, #46) provenant de la base NCSS KSSL (*Kellogg Soil Survey Laboratory*) nécessitent des données de laboratoire granulométriques (tamis mécaniques), distinctes des données SSURGO composites. La base NCSS KSSL est disponible en téléchargement depuis `ncsslabdatamart.sc.egov.usda.gov` mais ne dispose pas d'une API point-par-point équivalente à SDA. La `siltco_r` dans SDA/SSURGO présente un taux de remplissage de seulement ~8% en Californie, rendant les coefficients de gradation (Cu, Cc) peu exploitables.

**b) Données de niveau piézométrique (GW_Change_Storage, GW_recharge)**

Les features hydrogéologiques `GW_Change_Storage_mm` (rang #39) et `GW_recharge` (rang #42) proviennent des données de gestion SGMA ou d'un modèle de bilan hydrique (ArcGIS Water Balance App). Ces données requièrent soit un accès aux rapports GSP (*Groundwater Sustainability Plans*) soit une modélisation complémentaire.

**c) PM1 (rang #23)**

La concentration en particules PM₁ (diamètre < 1 µm) n'est pas mesurée par le réseau standard AQS. Cette métrique est généralement disponible via les capteurs basse résolution PurpleAir, mais leur couverture spatiale et leur précision sont inférieures aux instruments de référence.

**d) Réseau de surveillance non aléatoire**

Le biais de sélection inhérent au réseau GAMA (surreprésentation des zones de densité de population, municipales) peut affecter les performances de généralisation du modèle ML. Les puits municipaux représentent 82,2% du dataset contre 1,5% pour les puits domestiques.

### 9.2 Perspectives d'amélioration

1. **NCSS KSSL** : Téléchargement de la base complète et interpolation spatiale (kriging ou IDW) sur les points PFAS — permettrait de couvrir les rangs #6, #9, #12, #37 avec une précision de laboratoire.
2. **Données SGMA** : Extraction des profils piézométriques depuis les DWRInfo pour `GW_Change_Storage_mm`.
3. **Données SWC (SNOTEL)** : Alternative à GLDAS pour le snowpack en zones montagneuses (meilleure résolution spatiale).
4. **Variables de land use** : NLCD (National Land Cover Database) ou CDFA pour la densité de culture — pertinent pour les PFAS agricoles.

---

## 10. Annexes techniques

### A. Structure des fichiers du projet

```
ca-pfas-ml/
├── src/
│   ├── process_wells.py      # Phase 1 : prétraitement GAMA PFAS
│   ├── collect_ssurgo.py     # Phase 2a : SSURGO via API SDA
│   ├── collect_aqs.py        # Phase 2b : EPA AQS air quality
│   ├── process_gldas.py      # Phase 2c : NASA GLDAS nc4
│   └── merge.py              # Phase 3 : fusion finale
├── data/
│   ├── raw/
│   │   ├── gama/
│   │   │   ├── gama.csv                    # GAMA mesures brutes
│   │   │   └── gama_allwells.csv           # Métadonnées puits
│   │   ├── contamination/
│   │   │   └── geotracker_pfas_*.csv       # Sites PFAS CalEPA
│   │   └── environment/
│   │       ├── sgma_basins.geojson         # Bassins SGMA (DWR)
│   │       ├── ssurgo/
│   │       │   ├── ssurgo_ca_points.parquet
│   │       │   └── cache/                  # ~9786 fichiers JSON
│   │       ├── aqs/
│   │       │   ├── aqs_ca_annual.parquet
│   │       │   └── aqs_{param}_{year}.json # Cache par paramètre/année
│   │       ├── nasa_gldas/
│   │       │   └── GLDAS_NOAH025_M.A*.nc4  # 132 fichiers mensuels
│   │       └── gldas_ca_monthly.parquet
│   └── processed/
│       ├── wells_pfas_clean.parquet        # Après Phase 1
│       ├── CA-PFAS-ASGWS.parquet          # Dataset final
│       ├── CA-PFAS-ASGWS.csv              # Export CSV
│       └── data_dict.csv                  # Dictionnaire de données
└── docs/
    └── rapport_collecte_donnees.md        # Ce rapport
```

### B. Commandes de reproduction

```bash
# Environnement
python3 -m venv .venv
source .venv/bin/activate
pip install pandas geopandas scipy netCDF4 requests pyarrow

# Phase 1 — PFAS
python -m src.process_wells

# Phase 2 — Sources parallèles (lancer simultanément si possible)
python -m src.collect_ssurgo --workers 8
python -m src.collect_aqs
python -m src.process_gldas

# Phase 3 — Fusion
python -m src.merge --no-weather   # Sans NOAA (optionnel)
python -m src.merge                # Avec météo NOAA complète
```

**Variables d'environnement requises** (fichier `.env`) :
```
AQS_EMAIL=your@email.com
AQS_KEY=your_aqs_api_key
NOAA_CDO_TOKEN=your_noaa_token   # optionnel pour météo
```

### C. Références des APIs utilisées

| API | Endpoint | Documentation |
|---|---|---|
| USDA SDA (SSURGO) | `https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest` | `sdmdataaccess.nrcs.usda.gov/docs` |
| EPA AQS | `https://aqs.epa.gov/data/api/annualData/byState` | `aqs.epa.gov/aqsweb/documents/data_api.html` |
| NASA GLDAS | Earthdata Search (`GLDAS_NOAH025_M_2.1`) | `disc.gsfc.nasa.gov` |
| GAMA | `https://gamagroundwater.waterboards.ca.gov/` | Téléchargement direct CSV |
| GeoTracker | `https://geotracker.waterboards.ca.gov/` | Export CSV |
| SGMA/DWR | `https://sgma.water.ca.gov/webapi/` | GeoJSON direct |

### D. Versions des bibliothèques clés

```
pandas        2.2.x
geopandas     0.14.x
scipy         1.12.x
netCDF4       1.7.x
requests      2.31.x
pyarrow       15.x
```

---

*Rapport généré automatiquement à partir du pipeline de collecte — Juin 2026*  
*Dataset disponible : `ca-pfas-ml/data/processed/CA-PFAS-ASGWS.parquet`*  
*Dictionnaire de variables : `ca-pfas-ml/data/processed/data_dict.csv`*
