# Limite majeure — couverture des sources PFAS (et extension via geoenrich)

## 1. Constat vérifié

Le seul proxy de source de contamination du pipeline d'origine est la couche
GeoTracker (`add_contamination`, [merge.py:266](../src/merge.py#L266)). Elle ne
contient que **4 types de sites** pour **463 sites** :

| Type | n sites | Affecté à (nearest) |
|---|---|---|
| Chrome Plater | 271 | 46,4 % des observations |
| Bulk Terminal | 122 | 28,1 % |
| Refinery | 40 | 9,2 % |
| Airport | 30 | 16,2 % |

**24,1 %** des observations n'ont **aucun** site GeoTracker dans 10 km (4,1 % dans
50 km). Le proxy `nearest_geotracker_type` est donc dominé par un seul type
(galvanoplastie) et muet pour un quart des puits.

### Nature de la limite (formulation correcte)

Les 4 types présents *sont* de vraies sources PFAS — galvanoplastie (PFOS comme
anti-brouillard), aéroports/raffineries/terminaux (AFFF). La limite n'est donc
**pas** un biais « vers des sources non-PFAS », mais un **sous-ensemble partiel
des sources PFAS ponctuelles** (sites visés par les ordres d'investigation du
SWRCB) qui **omet** :

1. **AFFF militaire / DoD et fire-training** — la source n°1 des panaches PFAS.
   Absence **structurelle** : GeoTracker est une base d'**État** ; les sites
   **fédéraux/DoD sont hors juridiction**.
2. **STEP et épandage de biosolides / eaux recyclées** — source diffuse majeure.
3. **Décharges (lixiviats)** — absentes de cette couche (aucun type « Landfill »).
4. **Fabrication/usage fluorochimique** (semi-conducteurs, textiles, papeterie).

Le fichier `epa_pfas_industry_sectors.xlsx` (présent mais **non utilisé** par le
pipeline) confirme cet univers : il liste explicitement *Sewage Treatment
Facilities*, *Solid Waste Landfills*, *Fire Training Facilities*, *National
Defense* comme secteurs PFAS — mais c'est une **taxonomie NAICS non
géolocalisée**, inutilisable telle quelle.

**Recommandation mémoire** : présenter ceci comme limite majeure ; ne jamais
affirmer que « le modèle identifie les sources ». Les comptes/types GeoTracker
sont un proxy partiel.

## 2. Extension réalisée — 3 couches de sources ajoutées (geoenrich)

Le moteur `geoenrich` rend l'ajout d'une source **trivial** : un bloc `proximity`
de plus dans la config. Trois couches géolocalisées et interrogeables ont été
ajoutées (téléchargeur reproductible : `geoenrich/examples/fetch_source_layers.py`) :

| Couche | Source | Sites CA | Préfixe |
|---|---|---|---|
| Stations d'épuration (effluents/biosolides) | EPA FRS / ICIS-NPDES | 350 (186 maj., 164 min.) | `wwtp_` |
| **Sites DoD/fédéraux PFAS (AFFF — source n°1)** | EPA PFAS Analytic Tools | 83 (33 détectés, 41 suspectés) | `dod_` |
| **Décharges / sites d'enfouissement (lixiviats)** | CalRecycle SWIS | 1 710 (211 landfills + 1 419 disposal) | `landfill_` |

Apport mesuré (18 nouvelles variables `*_dist_km`, `*_n_within_{1,3,10,50}km`,
`*_nearest_type`) :

| Métrique | STEP | DoD | Décharges |
|---|---|---|---|
| Plus proche, médiane | 7,1 km | 14,3 km | **2,9 km** |
| Obs. avec ≥1 site < 3 km | 18,6 % | 5,6 % | **52,2 %** |
| Obs. avec ≥1 site < 10 km | 66,3 % | 33,0 % | **97,0 %** |

Bilan de couverture des sources :
- Avant : **24,1 %** des observations sans aucun site GeoTracker < 10 km.
- Après ajout STEP : restait **13,7 %** sans aucune source (GeoTracker+STEP) < 10 km.
- **Ajout DoD + décharges → 12,3 % des observations totales gagnent une source
  PFAS < 10 km qu'aucune couche n'apportait.** Les décharges (médiane 2,9 km) sont
  même un proxy *plus dense* que GeoTracker (3,9 km).

Détail méthodologique à citer : les coordonnées DoD sont le **centroïde de
l'installation** (caveat EPA), non le point exact de rejet AFFF.

Les 4 couches de source vivent dans
[`geoenrich/configs/california.yaml`](../../geoenrich/configs/california.yaml)
(9 opérateurs au total).

## 3. Restant (impact secondaire)

1. **Fire-training areas** spécifiques (souvent agrégées dans l'installation DoD ;
   les aéroports Part 139 sont partiellement couverts par GeoTracker « Airport »).
2. **Industrie fluorochimique fine** : la couche EPA *PFAS Analytic Tools –
   Industry Sectors* expose **22 857 sites CA** géolocalisés (textile, papeterie,
   semi-conducteurs…), mais très dense → plutôt un proxy d'« industrialité » qu'une
   source spécifique ; à ajouter avec discernement (ex. filtrer par secteur).

Chaque ajout = un bloc `proximity` supplémentaire (~6 lignes) dans la config.
