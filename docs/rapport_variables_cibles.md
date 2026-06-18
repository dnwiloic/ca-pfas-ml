# Rapport d'analyse — Variables cibles pour la modélisation PFAS

**Dataset :** `CA-PFAS-ASGWS.parquet` — 46 338 échantillons × 201 colonnes
**Période :** avril 2016 → janvier 2026 (concentré sur 2019-2025)
**Couverture :** eaux souterraines de Californie (programme GAMA / SWRCB)

---

## 1. Synthèse et recommandation

Le dataset permet de construire **trois familles de cibles** : binaire réglementaire, binaire de détection, et continue (régression). Voici la recommandation hiérarchisée :

| Priorité | Cible | Type | Prévalence / nature | Usage recommandé |
|----------|-------|------|---------------------|------------------|
| **1 (principale)** | **EPA 2024 NPDWR** | Binaire | 45.7 % positifs (≈ 1:1.2) | Modèle principal — pertinence réglementaire forte, classes équilibrées |
| **2 (secondaire)** | **log₁₀(sum_pfas_ngL)** | Régression | continu, skew log = 1.56 | Modèle de contamination « dose », plus riche en information |
| **3 (alternative)** | **Détection ≥ 1 PFAS** | Binaire | 61.0 % positifs | Cible « screening », moins liée à la réglementation mais robuste |
| 4 (exploratoire) | Hazard Index continu | Régression | médiane 0.44, P95 5.5 | Sortie interprétable « risque sanitaire mélange » |

> **Recommandation :** conserver **EPA 2024 NPDWR** comme cible de classification principale (déjà en place), et ajouter **log₁₀(sum_pfas)** comme cible de régression complémentaire pour le mémoire — les deux racontent une histoire différente et complémentaire.

---

## 2. Comprendre la donnée brute : détection vs concentration

Point **critique** pour toute définition de cible. Le dataset contient deux familles de colonnes pour chaque composé :

- **`<composé>_detected`** (booléen) — détection analytique réelle (mesure > limite de détection du laboratoire)
- **`<composé>_ngL`** (float) — concentration rapportée, **avec substitution des non-détects**

### 2.1 Les colonnes `_ngL` contiennent des valeurs substituées

Pour les non-détects, la concentration a été remplacée par une valeur de substitution (≈ ½ LDM). Exemple sur PFOA non détecté :

| Valeur substituée | Occurrences |
|-------------------|-------------|
| 1.00 ng/L | 14 170 |
| 0.90 ng/L | 3 314 |
| 0.85 ng/L | 3 076 |
| 0.95 ng/L | 799 |

**Conséquence :** `PFOA_ngL > 0` dans 99.6 % des cas, alors que `PFOA_detected = True` dans seulement **45.1 %**. Les colonnes `_ngL` brutes **surestiment massivement** la présence — il faut raisonner sur `_detected` pour la détection, sur `_ngL` (avec seuil) pour le dépassement réglementaire.

| Composé | Détection réelle | `_ngL > 0` |
|---------|-----------------|------------|
| PFOA | 45.1 % | 99.6 % |
| PFOS | 49.3 % | 99.6 % |
| PFHxS | 47.4 % | 95.1 % |
| PFNA | 13.3 % | 97.0 % |
| HFPO-DA (GenX) | **0.1 %** | 87.5 % |
| ADONA | 0.0 % | 87.7 % |

> ⚠️ **HFPO-DA et ADONA ne sont quasiment jamais détectés** (≤ 0.1 %). Toute cible individuelle sur ces composés serait dégénérée. Leur valeur `_ngL` est presque entièrement de la substitution.

### 2.2 Incohérence `_detected` / `_ngL` (à signaler dans le mémoire)

| Composé | `detected=False` **&** `_ngL > MCL` |
|---------|--------------------------------------|
| PFOA | 515 |
| PFOS | 532 |
| PFNA | 278 |
| PFHxS | 112 |

~500 échantillons par composé ont un flag « non détecté » mais une concentration au-dessus du MCL. Cause probable : agrégation de plusieurs prélèvements par puits (flag d'un échantillon, concentration max d'un autre), ou limites de rapport différentes entre laboratoires. **Impact sur la cible EPA 2024 : marginal** (ces ~500/46 000 = 1 %), mais à documenter comme limite de qualité de données.

---

## 3. Catalogue des cibles binaires réglementaires

| Code | Définition | Positifs | % | Ratio |
|------|-----------|----------|------|-------|
| **A1** | **EPA 2024 NPDWR** (MCL indiv. + Hazard Index) | 21 154 | **45.7 %** | 1 : 1.2 |
| A2 | EPA 2016 Health Advisory (PFOA + PFOS > 70) | 3 203 | 6.9 % | 1 : 13 |
| A3 | Somme totale PFAS > 70 ng/L | 11 490 | 24.8 % | 1 : 3 |
| A4 | Californie Notification Levels (PFOA > 5.1 OU PFOS > 6.5) | 17 743 | 38.3 % | 1 : 1.6 |
| A5 | Californie Response Levels (PFOA > 10 OU PFOS > 40) | 9 073 | 19.6 % | 1 : 4 |

### Décomposition de la cible EPA 2024 (A1)

| Mécanisme de dépassement | Positifs | % |
|--------------------------|----------|------|
| Par MCL individuel | 20 975 | 45.3 % |
| Par Hazard Index seul (sans MCL) | 179 | 0.4 % |
| · PFOS > 4 ng/L | 18 758 | 40.5 % |
| · PFOA > 4 ng/L | 16 282 | 35.1 % |
| · PFHxS > 10 ng/L | 6 896 | 14.9 % |
| · PFNA > 10 ng/L | 1 434 | 3.1 % |
| · HFPO-DA > 10 ng/L | **0** | 0.0 % |

**Lecture :** la cible EPA 2024 est presque entièrement portée par **PFOS et PFOA** (les deux MCLs à 4 ng/L). Le Hazard Index n'ajoute que 0.4 % de positifs supplémentaires. HFPO-DA ne contribue jamais.

> **Pourquoi A1 est la meilleure cible binaire :** seule à offrir un **équilibre de classes quasi-parfait (1:1.2)**, ce qui simplifie l'apprentissage (ADASYN optionnel), tout en étant **la norme réglementaire en vigueur** (avril 2024). A4 (CA Notification, 1:1.6) est une alternative valable si l'on veut un cadrage californien.

---

## 4. Catalogue des cibles binaires de détection

Indépendantes des seuils réglementaires, basées sur les colonnes `_detected` :

| Code | Définition | Positifs | % |
|------|-----------|----------|------|
| B1 | Au moins un PFAS détecté (sur 31) | 28 268 | 61.0 % |
| B2 | ≥ 3 PFAS détectés simultanément | 22 037 | 47.6 % |
| B3 | ≥ 5 PFAS détectés simultanément | 16 468 | 35.5 % |
| B4 | Au moins un PFAS réglementé (PFOA/PFOS/PFNA/PFHxS) détecté | 27 022 | 58.3 % |

**Distribution du nombre de PFAS co-détectés** (médiane = 2, moyenne = 3.4, max = 25) :

| Nb PFAS détectés | Échantillons |
|------------------|--------------|
| 0 | 39.0 % |
| 1 | 7.2 % |
| 2 | 6.2 % |
| 3 | 6.5 % |
| ≥ 5 | 35.5 % |

> **Usage :** B1 est une cible « screening / présence » utile si l'objectif scientifique est *« où trouve-t-on des PFAS ? »* plutôt que *« où dépasse-t-on la norme ? »*. Elle est moins biaisée par les seuils mais capture un phénomène différent (présence vs risque).

---

## 5. Catalogue des cibles continues (régression)

| Code | Variable | Médiane | Moyenne | P95 | Max | Skew (log₁₀) |
|------|----------|---------|---------|-----|-----|--------------|
| **C1** | **sum_pfas_ngL** | 32.3 | 174.0 | 551.6 | 5 723 | **1.56** |
| C2a | PFOA_ngL | 1.48 | 11.9 | 37.0 | 228 | 1.56 |
| C2b | PFOS_ngL | 2.00 | 20.6 | 63.9 | 422 | 1.30 |
| C2c | PFHxS_ngL | 2.00 | 19.0 | 52.0 | 480 | 2.00 |
| C3 | Hazard Index (continu) | 0.437 | — | 5.51 | 51.3 | — |

**Toutes ces variables sont fortement asymétriques** (skew brut de `sum_pfas` = 6.1). Une **transformation log₁₀(x + 1)** est indispensable et ramène le skew à ~1.5 (acceptable pour une régression).

> **Recommandation régression :** `log₁₀(sum_pfas_ngL + 1)`. C'est la cible continue la plus informative — elle agrège les 31 composés et conserve toute la dynamique de contamination, là où la classification binaire « écrase » l'information en 0/1. Attention : `sum_pfas` inclut les valeurs substituées, donc le « plancher » (~32 ng/L médian) reflète en partie la substitution, pas une vraie contamination.

---

## 6. Considérations méthodologiques transversales

### 6.1 Forte hétérogénéité spatiale → risque de fuite spatiale

Prévalence EPA 2024 par comté (n > 500) :

| Comté | Prévalence | n |
|-------|-----------|------|
| Contra Costa | 84.9 % | 544 |
| Alameda | 83.9 % | 1 277 |
| Orange | 71.6 % | 3 477 |
| San Diego | 69.3 % | 1 386 |
| Los Angeles | 58.4 % | 12 191 |
| … | … | … |
| Tulare | 18.0 % | 1 147 |
| Monterey | **9.7 %** | 1 226 |

L'écart 9.7 % → 84.9 % signifie que **`county`, `latitude`, `longitude` sont quasi-suffisants pour prédire la cible** par simple mémorisation géographique. C'est exactement pourquoi le protocole Dong et al. (2024) — déjà appliqué dans `ml_binary.py` via `LOCATION_FEATURES` — **retire les identifiants de localisation pure** pour forcer l'apprentissage des variables environnementales. Cette précaution vaut pour **toutes** les cibles ci-dessus.

### 6.2 Dérive temporelle de la prévalence

| Année | Prévalence EPA 2024 | n |
|-------|---------------------|------|
| 2019 | 41.8 % | 3 683 |
| 2021 | 48.9 % | 5 697 |
| 2022 | 56.8 % | 4 921 |
| 2024 | 43.4 % | 9 331 |
| 2025 | 40.0 % | 10 025 |

La prévalence oscille entre 40 % et 57 % selon l'effort d'échantillonnage annuel (campagnes ciblées sur sites à risque certaines années). Les features temporelles (`year`, `month`, `season`) capturent partiellement ce signal — légitime, mais à surveiller pour ne pas confondre tendance d'échantillonnage et tendance environnementale.

### 6.3 Fuite de données — rappel

Pour **toute** cible dérivée des PFAS, les colonnes suivantes sont des **fuites** et doivent être exclues des features (déjà géré) :
- `*_ngL` (concentrations — source directe de la cible)
- `*_detected`, `label_*` (détection binaire — corrélation jusqu'à 0.81)
- `sum_pfas_ngL`, `pfas_class_assignment`, l'ancienne cible `target_sum_gt70`

---

## 7. Décision finale recommandée

```
CIBLE PRINCIPALE (classification)
└── target_epa2024  =  (PFOA>4) ∨ (PFOS>4) ∨ (PFNA>10) ∨ (PFHxS>10)
                       ∨ (HFPO_DA>10) ∨ (Hazard Index > 1)
    → 45.7 % positifs, équilibre 1:1.2, norme en vigueur

CIBLE COMPLÉMENTAIRE (régression, pour le mémoire)
└── log₁₀(sum_pfas_ngL + 1)
    → continu, capture l'intensité de contamination

CIBLE ALTERNATIVE (screening, robustesse)
└── any_pfas_detected  =  OR(31 colonnes _detected)
    → 61 % positifs, indépendant des seuils réglementaires
```

**Justification du choix principal :** la cible EPA 2024 est la seule à combiner (1) pertinence réglementaire directe (norme fédérale 2024), (2) équilibre de classes naturel évitant le sur-échantillonnage agressif, et (3) ancrage sur les deux composés les mieux mesurés du dataset (PFOA/PFOS, > 99 % de couverture analytique). Les cibles A2/A5 sont trop déséquilibrées (1:13, 1:4) ; les cibles de détection B sont scientifiquement valides mais s'éloignent de l'objectif réglementaire du projet.

---

*Rapport généré le 2026-06-14 — analyse exploratoire des cibles, projet ca-pfas-ml (Phase 4 ML).*
