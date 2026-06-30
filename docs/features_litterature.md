# Features de la littérature PFAS / eaux souterraines absentes de notre pipeline

Analyse du corpus local (≈ articles `article/A*.pdf`, `articles_2/`, revue
`Revised_Literature_Review_Report.markdown`) pour identifier des prédicteurs
**utilisés ailleurs mais pas dans notre dataset**. Méthode : (1) revue de
synthèse, (2) scan de fréquence d'un vocabulaire de prédicteurs sur ~149 k mots
du corpus GW/PFAS, (3) lecture ciblée des tables de prédicteurs.

> Les fréquences sont **indicatives** (signal d'usage dans le corpus), pas une
> méta-analyse rigoureuse. Le corpus mêle PFAS-GW (Dong, Li & MacDonald Gibson)
> et contamination-GW au sens large (Cr, nitrate, PCP) — qui partagent le même
> vocabulaire de features.

## Ce que nous avons déjà (couvre tout le papier de référence)

Dong/Olivares 2024 = sources, météo, qualité air, sol, hydrologie,
co-contaminants → **tous présents** + nos ajouts (gradient DWR, profondeur nappe,
profondeur/crépine, occupation du sol). Donc les écarts viennent des **autres**
articles, pas de la référence.

## Écarts prioritaires (feature ↔ littérature ↔ accessibilité)

| Feature manquante | Évidence corpus | Mécanisme PFAS | Source / accès | Tier |
|---|---|---|---|---|
| **Topographie : élévation + pente** | ~47 mentions | pente → ruissellement/infiltration ; élévation → position hydrogéo | DEM USGS 3DEP / SRTM, `getSamples` (comme NLCD) | **1** |
| **Type d'aquifère / confinement** | confinement ~60 ; lithologie/géologie ~90 | aquifère **non confiné = vulnérable** à l'infiltration de surface | USGS Principal Aquifers (polygones, point-in-polygon comme SGMA) | **1** |
| **Proximité/fraction de zones humides** | clé chez **Li & MacDonald Gibson 2022** (PFAS-GW) | les PFAS s'accumulent dans les wetlands → source/voie | NLCD classes 90/95 (déjà interrogeable) → fraction + distance | **1** |
| **Distance à la côte** | ~24 (coast/shoreline) | AFFF côtier, bases navales, intrusion saline | côte CA (calcul direct) | **1** |
| **Charge azotée agricole / biosolides** | agriculture ~52 ; intrants N/biosolides ~47 | irrigation eaux recyclées + épandage biosolides = PFAS diffus | USGS nitrogen loading ; sites d'épandage biosolides (Water Board) ; *déjà proxy via `cocontam_no3n` + NLCD cultivé* | 2 |
| **Base flow index / âge de l'eau** | ~33 (baseflow/residence/age) | eau ancienne = exposition PFAS différée ; BFI = réactivité aquifère | USGS BFI (raster point-sample) | 2 |
| **Transmissivité aquifère** | conductivité/transmissivité ~59 | vitesse de transport (≠ ksat de sol qu'on a en surface) | bases aquifères (plus difficile) | 2 |
| **Taux de recharge** | ~7 | vitesse d'arrivée des PFAS de surface | USGS recharge (raster) | 3 |

Déjà testés/écartés : densité de population (~1, inverse), redox/DO (~20, sparse+positionnel),
routes/trafic (~8, couvert par `dev_intensity`).

## Recommandation — 3 ajouts à fort rendement, faisables avec l'outillage existant

1. **Topographie (élévation + pente)** — `getSamples` sur un DEM (même mécanique
   que `derive_landuse.py`). Mécaniste, transférable a priori, très peu cher.
2. **Type d'aquifère / confinement** — jointure polygone USGS Principal Aquifers
   (même mécanique que SGMA). Comble le plus gros écart conceptuel : on n'a que
   des bassins **administratifs** (SGMA), pas des unités **hydrogéologiques**.
3. **Proximité de zones humides** — extension de `derive_landuse.py` (fraction
   NLCD wetland en voisinage + distance). C'est **le** prédicteur PFAS-GW
   distinctif (Li & MacDonald Gibson) qu'on n'a pas.

Chaque candidat passe le **protocole d'adoption** déjà établi : garder seulement
si AUC intra-bloc (CV spatiale k=8) ≈ AUC globale, et croiser avec les variables
de sélection. Vu le plafond mono-feature observé (~0,6), l'objectif est d'ajouter
des features **transférables et orthogonales**, pas un signal miracle.

### État d'avancement

- **#1 Topographie — FAIT** (`src/derive_topography.py` → `well_topography.csv`).
  Verdict : `elevation_m` ✅ mineure (~0,57 propre, intra-bloc ; 0,68 brut gonflé
  par le confondant puits-surveillance ; redondant +0,40 avec depth_to_water) ;
  `slope` ❌ rejeté (0,505, nul à l'échelle 100 m).
- **#2 Type d'aquifère / confinement — FAIT, REJETÉ** (`src/derive_aquifer.py` →
  `well_aquifer.csv`, USGS Principal Aquifers). AUC 0,510 (intra 0,523) → aléatoire.
  Cause : 83 % des puits CA dans le même matériau (sable/gravier non consolidé) →
  carte nationale à 5 classes trop grossière pour discriminer ; confinement absent
  de la couche. Leçon : feature importante en littérature ≠ utile sur ce dataset
  (résolution + homogénéité). Discrimination réelle = carte géologique CA plus fine
  (effort élevé, gain incertain).
- **#3 Proximité zones humides — FAIT, REJETÉ** (`src/derive_wetland.py` →
  `well_wetland.csv`, NLCD wetland 90/95, grille 3×3 ±1,5 km). AUC 0,506 → aléatoire.
  Cause : seulement 7,3 % des puits CA ont une zone humide < 3 km (population de
  vallées arides/urbaines). Importante chez Li & MacDonald Gibson (région riche en
  wetlands) mais **non transposable** à la population CA.

### Méta-conclusion

Les 3 candidats littérature de **vulnérabilité hydrogéo générique** se comportent
ainsi sur CA : topographie → `elevation` mineure (~0,57) ; aquifère → nul (0,51,
population homogène) ; wetlands → nul (0,51, trop rares). **Les features
importantes en littérature ne se transposent pas mécaniquement** : `aquifer` et
`wetland` échouent ici parce que la population de puits CA est hydrogéologiquement
homogène. Seules transfèrent les features mécanistes à portée large (profondeur de
puits/nappe, intensité urbaine, élévation), toutes ~0,55–0,61. L'audit local
(intra-bloc + croisement sélection) est donc indispensable avant tout emprunt à
la littérature.

## Note d'honnêteté

Le corpus est surtout du transport/occurrence GW généraliste ; les deux seules
références **PFAS-GW spatiales** sont Dong (couverte) et Li & MacDonald Gibson
(d'où : wetlands, distance industries/décharges — ces dernières désormais
intégrées comme nœuds `facility`). Les autres écarts (topo, aquifère, base flow)
sont des features **standard de vulnérabilité des nappes**, mécanistiquement
fondées et transférables, mais leur apport reste à valider empiriquement.
