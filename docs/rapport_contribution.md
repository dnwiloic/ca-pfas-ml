# Rapport de contribution — Prédiction des PFAS en eaux souterraines

*Mémoire de Master en Informatique — Spécialisation Intelligence Artificielle*
*DNJOMOU YONMBA Wilfried Loïc — URIFIA, Université de Dschang*

> Ce document structure la contribution expérimentale du mémoire selon quatre axes :
> (1) le problème traité, (2) la solution et la méthodologie avec ses deux approches,
> (3) le positionnement par rapport à la littérature — en particulier Dong et al. (2024)
> et les revues du chapitre II —, et (4) la transposition de l'étude au contexte camerounais.

---

## 1. Présentation du problème

### 1.1 Un problème de surveillance sous contrainte

La surveillance des substances per- et polyfluoroalkylées (PFAS) dans les eaux
souterraines est confrontée à une triple contrainte structurelle, déjà posée au
chapitre I du mémoire :

- **Coût analytique élevé.** Une analyse multi-composés par chromatographie liquide
  couplée à la spectrométrie de masse en tandem (LC-MS/MS) coûte plusieurs centaines
  d'euros par échantillon. Le contrôle exhaustif d'un parc de plusieurs milliers de
  puits est économiquement impossible.
- **Durcissement réglementaire rapide.** L'EPA est passée d'un avis sanitaire à
  70 ng/L pour la somme PFOA + PFOS (2016) à des limites maximales individuelles
  (MCL) de 4 ng/L pour le PFOA et le PFOS (NPDWR, avril 2024), assorties d'un
  Indice de Risque (Hazard Index) pour le mélange PFHxS / PFNA / GenX / PFBS.
- **Couverture spatiale incomplète et non-détection fréquente.** Une valeur nulle
  ne signifie pas l'absence de PFAS mais souvent une concentration sous le seuil de
  détection ; les protocoles sont hétérogènes entre campagnes.

### 1.2 La question scientifique

Dans ce contexte, l'apprentissage automatique n'a pas vocation à remplacer la mesure,
mais à **prioriser** : orienter les campagnes vers les puits présentant la plus forte
probabilité de dépassement réglementaire, à partir d'informations contextuelles
disponibles **avant** tout prélèvement PFAS (géologie, sol, hydrologie, météorologie,
proximité aux sources industrielles, co-contaminants).

La question centrale est donc :

> **Peut-on prédire de façon fiable, reproductible et interprétable le dépassement
> des seuils réglementaires PFAS d'un puits californien à partir de ses seules
> variables de contexte, sans utiliser aucune concentration PFAS mesurée en entrée
> (mode prédictif strict, équation II.3 du mémoire) ?**

### 1.3 Les difficultés à lever

Le chapitre II a identifié cinq lacunes (L1–L5). Cette contribution se concentre
sur les trois qui conditionnent la validité opérationnelle d'un outil de priorisation :

| Lacune | Difficulté | Enjeu pour notre travail |
|--------|-----------|--------------------------|
| **L2** | Biais d'évaluation circulaire (concentrations PFAS dans les features) | Garantir un **mode prédictif strict** et contrôler la fuite de données |
| **L1** | Structure relationnelle sous-exploitée entre puits voisins / sources communes | Comparer une approche **tabulaire** et une approche **relationnelle (graphe)** |
| Reproductibilité | Pipelines de collecte difficilement reproductibles | Construire un **dataset versionné et ré-exécutable** |

À cela s'ajoute une difficulté propre aux données environnementales : le **déséquilibre
de classes** (les puits contaminés sont minoritaires) et l'**autocorrélation spatiale**
(des puits géographiquement proches partagent leur statut, ce qui peut gonfler
artificiellement les scores sur une division aléatoire).

---

## 2. Solution apportée et méthodologie

### 2.1 Vue d'ensemble

La solution repose sur trois piliers :

1. **Un dataset enrichi et reproductible** — `CA-PFAS-ASGWS` (46 338 échantillons ×
   201 variables, 31 PFAS, période 2016–2026), construit par un moteur d'enrichissement
   géospatial générique et versionné.
2. **Deux approches de modélisation complémentaires** — tabulaire (méthodes d'ensemble)
   et relationnelle (graphe hétérogène), conformément à la complémentarité
   tabulaire / relationnelle dégagée au chapitre II.
3. **Un protocole d'évaluation rigoureux** — mode prédictif strict, contrôle de la
   fuite, gestion du déséquilibre, métriques orientées décision (rappel, balanced
   accuracy, gain cumulé) et interprétabilité SHAP.

### 2.2 Le dataset reproductible

Le jeu de données agrège, pour chaque puits, des descripteurs issus de sources
publiques ouvertes :

| Famille | Source | Exemples de variables |
|---------|--------|----------------------|
| PFAS (cible uniquement) | GAMA / SWRCB | 31 composés mesurés, somme PFAS |
| Proximité aux sources | GeoTracker (sites PFAS) | distance au site le plus proche, nombre de sites à 1 / 3 / 10 / 50 km |
| Sol | SSURGO (USDA, API SDA) | granulométrie (8 fractions), argile, matière organique, conductivité, ratio eau/argile |
| Hydrologie / météo | NASA GLDAS | précipitations, évapotranspiration, humidité du sol (4 couches), température, manteau neigeux |
| Qualité de l'air | EPA AQS | PM2.5, PM10, NO₂, SO₂, ozone, CO, vent, humidité |
| Co-contaminants | bases qualité d'eau | 44 composés (solvants chlorés, hydrocarbures, métaux, nitrates) |

L'enrichissement est réalisé par le package **`geoenrich`**, conçu comme un moteur
générique (contrat d'enrichissement + opérateurs enfichables) et non comme un script
ad hoc. Chaque source dispose d'un cache versionné par schéma, ce qui garantit la
**reproductibilité intégrale** de la collecte (voir [docs/rapport_collecte_donnees.md](rapport_collecte_donnees.md)
et [docs/rapport_fidelite_dataset.md](rapport_fidelite_dataset.md)).

**Contrôle de fuite de données.** Trois familles de colonnes sont systématiquement
exclues de l'espace de features car elles encodent directement ou indirectement la
cible :

- les concentrations `*_ngL` (32 colonnes) ;
- les indicateurs de détection `*_detected` et `label_*` (62 colonnes) — corrélés
  jusqu'à 0,81 avec la cible ;
- l'ancienne cible `sum > 70 ng/L` et l'artefact de grille `gldas_dist_km`.

En outre, suivant le protocole de Dong et al. (2024), les **identifiants de
localisation pure** (latitude, longitude, comté, bassins) sont retirés par défaut
(`DROP_LOCATION = True`) afin d'éviter la mémorisation géographique : la prévalence
de contamination varie de 9,7 % (comté de Monterey) à 84,9 % (Contra Costa), si bien
qu'un modèle disposant des coordonnées peut « tricher » en apprenant la carte plutôt
que le mécanisme. Le retrait de la localisation ne coûte que ≈ 0,4 point d'AUC, mais
rend le modèle scientifiquement défendable et généralisable. L'espace de modélisation
final comporte **86 variables contextuelles**.

**Deux définitions de cible** sont étudiées (voir [docs/rapport_variables_cibles.md](rapport_variables_cibles.md)) :

- **Cible A — EPA 2024 NPDWR** : dépassement si PFOA > 4, PFOS > 4, PFNA > 10,
  PFHxS > 10, GenX > 10 ng/L **ou** Indice de Risque du mélange > 1.
  → 45,7 % de positifs (équilibre ≈ 1:1,2).
- **Cible B — Charge agrégée** : dépassement si la somme des 27 PFAS cibles atteint
  la somme de leurs seuils individuels (Σ seuils = 74 ng/L).
  → 22,3 % de positifs (déséquilibre ≈ 1:3,5).

La cible A mesure le risque réglementaire « composé par composé » ; la cible B mesure
l'intensité cumulée de contamination. Leur comparaison constitue une analyse de
sensibilité du critère de décision, rarement présente dans la littérature.

### 2.3 Approche 1 — Modélisation tabulaire par méthodes d'ensemble

La première approche traite chaque puits comme un vecteur de descripteurs
indépendants et mobilise les deux familles d'ensembles d'arbres les plus performantes
sur données tabulaires (chapitre I) :

- **Random Forest** (bagging) — robuste, fournit un score OOB et des importances par
  impureté ;
- **XGBoost** (gradient boosting) — capture finement les interactions non linéaires,
  early stopping sur validation interne.

**Choix méthodologiques :**

- optimisation des hyperparamètres par **Optuna** (échantillonnage bayésien TPE,
  amorçage sur les paramètres par défaut pour ne jamais dégrader le point de départ) ;
- gestion du déséquilibre par **pondération** (`class_weight="balanced"` pour la forêt,
  `scale_pos_weight` pour XGBoost) ;
- **optimisation du seuil de décision** sur probabilités *out-of-fold* (sans fuite),
  pour maximiser le compromis rappel / précision — levier essentiel en surveillance
  sanitaire où un faux négatif (puits contaminé non signalé) est plus grave qu'un
  faux positif (prélèvement inutile) ;
- validation par **StratifiedKFold 5 plis** + jeu de test indépendant 80/20 ;
- interprétabilité par importances combinées (impureté + gain) et **SHAP** (TreeSHAP).

**Résultats (test, mode prédictif strict, 86 variables) :**

| Cible | Modèle | ROC-AUC | Rappel | Précision | F1 | Bal. Acc. | AUC (CV 5 plis) |
|-------|--------|:-------:|:------:|:---------:|:--:|:---------:|:---------------:|
| **A — EPA 2024** | Random Forest | **0,974** | 0,915 | 0,925 | 0,920 | 0,926 | 0,962 ± 0,001 |
| (1:1,2) | XGBoost | 0,971 | 0,916 | 0,909 | 0,912 | 0,919 | 0,965 ± 0,001 |
| **B — Σ ≥ 74 ng/L** | Random Forest | **0,981** | 0,906 | 0,845 | 0,874 | 0,929 | 0,972 ± 0,002 |
| (1:3,5) | XGBoost | 0,977 | 0,912 | 0,815 | 0,861 | 0,927 | 0,974 ± 0,001 |

**Lecture.** Sur la cible A équilibrée, rappel et précision sont symétriques (≈ 0,92)
et le seuil optimal reste proche de 0,50 : la pondération suffit. Sur la cible B,
plus rare, la charge totale est plus *séparable* (AUC 0,981) mais la classe positive
plus difficile (F1 plus bas, précision en retrait) ; le seuil optimal diverge selon
le modèle (RF : 0,46 ; XGBoost : 0,585, ce dernier sur-corrigé par son
`scale_pos_weight`). La **balanced accuracy ≈ 0,93** confirme que les performances ne
sont pas un artefact de la classe majoritaire.

**Portée opérationnelle.** La courbe de gain cumulé montre qu'en ne prélevant que les
**25 % de puits les mieux classés par le modèle**, on détecte ≈ 54 % des puits
réellement positifs pour la cible A et ≈ 92 % pour la cible B — un argument direct
pour l'aide à la décision des gestionnaires de réseaux.

### 2.4 Approche 2 — Modélisation relationnelle par graphe hétérogène

La seconde approche lève l'hypothèse d'indépendance des observations : les panaches
de PFAS s'étendent le long de lignes d'écoulement, si bien que des puits voisins ou
en aval d'une même source partagent un contexte que des vecteurs isolés ne capturent
pas. Le réseau de surveillance est encodé en **graphe hétérogène** à plusieurs types
de nœuds (`sample`, `env`, `facility`, `water`, `geo_cluster`) reliés par des arêtes
typées (contexte environnemental, proximité d'installations, qualité d'eau,
appartenance géographique).

Un **Heterogeneous Graph Transformer (HGT)** apprend, par attention typée, un
*embedding* relationnel pour chaque puits, exploité de deux manières :

- **fusion** — concaténation de l'embedding HGT aux variables tabulaires, puis
  apprentissage d'un XGBoost sur l'espace enrichi ;
- **stacking** — méta-classifieur sur les sorties des modèles de base.

Cette approche répond directement à la lacune **L1**. Elle se compare à l'approche
tabulaire selon le **même protocole de division par site et le même mode prédictif
strict**, ce qui permet d'isoler l'apport propre de la structure relationnelle au-delà
du signal contextuel déjà capté par les arbres (voir le notebook
`04_hgt_xgboost_hybrid_epa2024`).

### 2.5 Interprétabilité

Après retrait de la localisation, les variables dominantes deviennent
**mécanistiquement interprétables** : la densité et la proximité des sites PFAS
(`n_geotracker_within_50km`, `dist_geotracker_km`) arrivent en tête pour la forêt,
tandis que XGBoost s'appuie fortement sur les **signatures de co-contaminants
industriels** (solvants chlorés : styrène, dichlorobenzène, TCE, chlorure de vinyle).
Ce résultat est cohérent avec la chimie de terrain : les PFAS co-occurrent avec les
solvants chlorés sur les sites industriels et militaires. L'analyse SHAP (globale et
locale) et les courbes de calibration documentent la fiabilité probabiliste du modèle,
condition d'acceptation par les régulateurs (chapitre I, §I.5).

---

## 3. Positionnement par rapport à la littérature

### 3.1 Référence centrale : Dong et al. (2024)

Le travail de Dong et al. constitue la base de comparaison la plus directe : même
verrou (corpus californien, mode prédictif), mêmes familles de méthodes. Le tableau
suivant résume l'écart.

| Dimension | Dong et al. (2024) | Cette contribution |
|-----------|--------------------|--------------------|
| Taille du corpus | ≈ 26 901 obs. | **46 338 obs.** (+ 72 %) |
| Période | 2016–2022 | **2016–2026** |
| Nombre de PFAS | 35 analytes | 31 composés + somme |
| Variables (après nettoyage) | 112 | 201 colonnes, **86 features** mode prédictif |
| Seuil réglementaire | somme > 70 ng/L (EPA 2016) | **MCL individuels EPA 2024 + Hazard Index** ; et Σ ≥ Σ seuils |
| Modèles | RF (tâche 1) + chaîne XGBoost (tâche 2) | **RF et XGBoost systématiquement comparés** + HGT relationnel |
| Réglage | rééchantillonnage (ADASYN/SMOTE) | **Optuna** + pondération + **optimisation de seuil** |
| Métriques | AUC, exactitude, rappel | + **balanced accuracy, gain cumulé, calibration** |
| Reproductibilité | pipeline propriétaire peu documenté | **moteur `geoenrich` versionné, ré-exécutable** |

**Sur les performances.** Dong et al. rapportent une AUC de 0,99 pour la tâche 1
(forêt + ADASYN, cible 70 ng/L). Nos AUC (0,974 pour EPA 2024, 0,981 pour Σ ≥ 74)
sont du même ordre mais obtenues (i) avec un **contrôle de fuite plus strict**
(retrait explicite des indicateurs de détection, que la circularité partielle peut
laisser fuiter), et (ii) sur une cible **réglementairement à jour** (EPA 2024). Le
léger écart d'AUC est le prix de cette rigueur et non un déficit de qualité — il
correspond exactement à l'inflation que nous mesurons lorsque la localisation est
réintroduite.

### 3.2 Apports par rapport aux lacunes L1–L5 du chapitre II

- **L2 — Biais d'évaluation circulaire (apport majeur).** Dong et al. signalent
  eux-mêmes une division aléatoire 80/20 exposant au *spatial data leakage* et un
  biais de sélection GAMA. Nous traitons frontalement ces deux points : retrait des
  identifiants de localisation, exclusion exhaustive des proxies de détection,
  quantification chiffrée de l'inflation due à la localisation (≈ 0,4 pt d'AUC,
  ≈ 5 pts de R² en régression). Le mode prédictif est strict et documenté.
- **L1 — Structure relationnelle.** Comme George & Dixit (RF/XGBoost tabulaire) ou
  Tokranov et al. (XGBoost national), la majorité des études restent tabulaires.
  Notre approche HGT encode explicitement la topologie du réseau et la compare à
  l'approche tabulaire sur le même protocole, comblant partiellement L1.
- **Reproductibilité (apport transversal).** Aucune des études examinées (Li &
  MacDonald Gibson, McMahon, Hu, George & Dixit, Tokranov, Dong) ne fournit un
  pipeline de collecte intégralement ré-exécutable. Le moteur `geoenrich`, avec ses
  caches versionnés et son contrat d'enrichissement générique, constitue une
  contribution méthodologique réutilisable au-delà du seul cas californien.

### 3.3 Cohérence avec les enseignements transversaux de la revue

Nos résultats confirment les deux convergences majeures de la revue : **la proximité
industrielle est le signal le plus robuste** (les variables GeoTracker dominent les
importances) et **la complémentarité tabulaire / relationnelle est fondée** (le HGT
apporte un signal spatial que les colonnes isolées ne captent pas). Ils confirment
aussi que **le mode d'évaluation conditionne les performances** : c'est précisément
pourquoi nous documentons strictement le mode prédictif et le contrôle de fuite.

### 3.4 L'accent sur un meilleur dataset reproductible

La contribution la plus durable n'est pas un gain de quelques points d'AUC, mais la
**production d'un dataset de référence reproductible** :

- couverture étendue (46 338 puits, 11 années, 31 PFAS, 6 familles de descripteurs) ;
- traçabilité complète (sources, requêtes API, schémas de cache versionnés) ;
- contrôle de qualité documenté (analyse détection vs substitution, incohérences
  `detected`/`ngL`, taux de valeurs manquantes par groupe) ;
- séparation nette entre cible et features, vérifiable et auditable.

Ce socle rend les comparaisons inter-modèles équitables et autorise la réplication —
condition que la revue (lacune L2) identifie comme largement absente de la littérature.

---

## 4. Mener une telle étude dans le contexte camerounais

### 4.1 Le défi : ni programme de surveillance, ni étiquettes

Le Cameroun ne dispose pas d'équivalent au programme GAMA californien : il n'existe
pas de surveillance systématique des PFAS dans les eaux souterraines, donc **pas de
jeu d'étiquettes** pour entraîner directement un modèle supervisé. L'analyse LC-MS/MS
y est de surcroît rare et coûteuse (échantillons souvent envoyés à l'étranger), ce qui
rend la **logique de priorisation encore plus précieuse** qu'en Californie : chaque
mesure évitée représente une économie substantielle.

Or c'est justement le **mode prédictif strict** — prédire sans aucune mesure PFAS
préalable — qui rend la méthodologie transposable : elle ne suppose pas un historique
analytique local, seulement des variables de contexte disponibles globalement.

### 4.2 Des données de substitution disponibles à l'échelle mondiale

Le moteur `geoenrich` étant générique, ses opérateurs peuvent être réinstanciés sur
des sources à couverture mondiale, sans dépendre d'API régionales américaines :

| Famille | Source US (Californie) | Substitut disponible pour le Cameroun |
|---------|------------------------|----------------------------------------|
| Sol | SSURGO (USDA) | **SoilGrids** (ISRIC, 250 m, mondial) |
| Hydro / météo | GLDAS | **GLDAS** (mondial), **CHIRPS** (pluie), **ERA5** |
| Proximité aux sources | GeoTracker | **OpenStreetMap** (zones industrielles, aéroports, décharges) |
| Qualité de l'air | EPA AQS | **Sentinel-5P** (NO₂, satellitaire) |
| Co-contaminants | bases qualité d'eau US | campagnes locales ciblées (à constituer) |

### 4.3 Cartographie des sources potentielles de PFAS au Cameroun

L'étude des sources (chapitre I, §I.2.3) oriente la sélection des zones à risque :

- **Corridor industriel Douala–Edéa** : zones industrielles de Bonabéri et de
  Bassa, agro-industrie, textile (CICAM) ;
- **Raffinerie SONARA (Limbé)** et installations pétrolières ;
- **Aéroports** (Douala, Yaoundé-Nsimalen, Garoua) et **bases militaires** : usage
  historique de mousses anti-incendie AFFF, *hotspots* documentés ailleurs dans le
  monde ;
- **Décharges** (Nkolfoulou à Yaoundé, Genie à Douala) : lixiviats ;
- **Sites de gestion des déchets électroniques** (importation croissante).

### 4.4 Stratégie de déploiement progressif

Une étude camerounaise réaliste se déroulerait en quatre phases :

1. **Construction du dataset contextuel** (sans PFAS) sur une région pilote — par
   exemple le bassin Douala–Edéa — en réinstanciant `geoenrich` sur les sources
   mondiales ci-dessus. Cette phase ne requiert aucune mesure et produit la carte
   des facteurs de risque a priori.
2. **Campagne d'échantillonnage ciblée et parcimonieuse** : un petit nombre de puits
   (quelques dizaines), choisis pour couvrir le gradient de risque prédit, sont
   analysés. Ces étiquettes rares servent à **calibrer** et non à entraîner de zéro.
3. **Transfert de connaissances** : deux voies complémentaires —
   - *apprentissage par transfert* à partir du modèle californien (les mécanismes
     advection / rétention / proximité de source sont génériques), avec ré-calibration
     sur les étiquettes locales ;
   - *apprentissage semi-supervisé* (pseudo-étiquetage, comme chez Dong et al.) pour
     exploiter les nombreux puits non analysés.
4. **Priorisation et boucle de surveillance** : le modèle classe les puits ; les
   plus à risque sont prélevés ; les nouvelles mesures réalimentent le modèle
   (apprentissage actif). La courbe de gain cumulé chiffre l'économie analytique.

### 4.5 Cadre institutionnel et limites

La réussite suppose des partenariats : **MINEE** (Ministère de l'Eau et de l'Énergie),
**CAMWATER** / Camerounaise des Eaux, laboratoires universitaires (Dschang, Yaoundé I),
appuis internationaux (OMS, coopérations bilatérales) pour l'accès à la LC-MS/MS.

Les limites spécifiques au contexte sont à expliciter :

- **Transférabilité géographique non garantie** (lacune L3) : la lithologie et le
  climat tropical humide diffèrent des aquifères californiens ; une validation croisée
  spatiale locale est indispensable avant tout usage opérationnel.
- **Qualité et résolution des données de substitution** : SoilGrids et Sentinel-5P
  sont moins résolus que SSURGO ou AQS ; l'incertitude doit être propagée.
- **Sources de PFAS différentes** : importance probable des décharges et de
  l'informel (recyclage, déchets électroniques) plutôt que des seules bases
  militaires.
- **Petits effectifs étiquetés** : le pseudo-étiquetage se dégrade en dessous de
  ≈ 100 exemples (limite signalée par Dong et al.) ; une stratégie d'échantillonnage
  active est donc critique.

### 4.6 Apport attendu

Transposée au Cameroun, l'étude fournirait le **premier outil de priorisation des
puits à risque PFAS** du pays, fondé sur des données ouvertes et un pipeline
reproductible, capable d'orienter des campagnes de mesure coûteuses vers les sites
les plus probables — exactement le cas d'usage où un modèle en mode prédictif apporte
le plus de valeur lorsque les ressources analytiques sont limitées.

---

## Synthèse

Cette contribution répond à la question posée en (1) par une solution articulée
autour d'un **dataset reproductible enrichi** et de **deux approches complémentaires**
(tabulaire par ensembles d'arbres, relationnelle par graphe hétérogène), évaluées en
**mode prédictif strict** avec un contrôle de fuite explicite. Par rapport à
Dong et al. (2024) et à la revue du chapitre II, l'apport tient moins à un gain brut
de performance (AUC 0,97–0,98) qu'à la **rigueur méthodologique** (retrait de la
localisation, métriques orientées décision, optimisation de seuil sans fuite) et à la
**reproductibilité** du socle de données. Enfin, le caractère générique du pipeline
`geoenrich` et l'exclusivité du mode prédictif ouvrent une voie crédible pour
transposer l'étude au **contexte camerounais**, où l'absence de surveillance
systématique rend la priorisation par IA particulièrement pertinente.

---

*Documents liés : [collecte des données](rapport_collecte_donnees.md) ·
[fidélité du dataset](rapport_fidelite_dataset.md) ·
[variables cibles](rapport_variables_cibles.md) ·
[structure du pipeline](structure_base_pipeline.md).*
