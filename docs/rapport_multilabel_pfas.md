# Prédiction multilabel semi-supervisée des PFAS individuels en eaux souterraines
### Reproduction et extension du protocole de Dong et al. (2024) sur une base étendue et reproductible

*Rapport de contribution — composante « prédiction multi-analytes » (Tâche 2)*

---

## 1. Présentation du problème

### 1.1 Le verrou de la surveillance des PFAS

Les substances per- et polyfluoroalkylées (PFAS) forment une famille de plusieurs milliers de
composés de synthèse, persistants (« polluants éternels » en raison de la liaison C–F), mobiles et
toxiques à des concentrations de l'ordre du ng/L. Leur surveillance dans les eaux souterraines — qui
alimentent une part majeure de l'eau potable — se heurte à trois contraintes structurelles :

1. **Coût analytique élevé** : chaque mesure exige une chromatographie LC-MS/MS, ce qui interdit
   un contrôle exhaustif du parc de puits.
2. **Couverture incomplète et hétérogène** : les programmes de surveillance ne mesurent pas tous le
   même panel d'analytes. Un même puits peut être renseigné pour le PFOS mais pas pour le PFPeS.
3. **Multiplicité des analytes** : la réglementation et les méthodes évoluent (UCMR-5 cible 35 PFAS),
   si bien que de nombreuses concentrations individuelles manquent dans les campagnes passées.

La conséquence est une **matrice de mesures lacunaire** : pour reconstruire un profil PFAS complet,
il faudrait analyser chaque puits pour chaque analyte, ce qui est économiquement impossible.

### 1.2 La tâche : prédire les PFAS individuels sans les mesurer

Notre objectif est la **Tâche 2** du pipeline de Dong et al. (2024) : prédire, pour chaque puits et
**sans aucune mesure PFAS en entrée**, quels PFAS individuels dépassent leur seuil réglementaire, à
partir des seules variables environnementales (hydrogéologie, sol, climat, qualité de l'air,
proximité aux sources de contamination).

Au sens de la distinction fondamentale entre **mode prédictif** et **mode confirmatoire**, ce travail
se place exclusivement en **mode prédictif** : la cible est prédite à partir d'un vecteur de variables
contextuelles `x_c`, sans concentration PFAS `x_p` parmi les entrées :

> Mode prédictif : ŷ = g(x_c) — *« dois-je envoyer un préleveur ? »*

C'est le régime opérationnellement pertinent : orienter les futures campagnes de prélèvement et
**reconstruire les profils manquants** des années passées, plutôt que compléter un profil déjà
partiellement mesuré (mode confirmatoire).

### 1.3 Nature du problème d'apprentissage

La tâche cumule trois difficultés qui en font un problème **multilabel et semi-supervisé** :

| Particularité | Conséquence méthodologique |
|---|---|
| 27 PFAS prédits simultanément | Classification **multi-étiquettes** (≠ 27 problèmes binaires indépendants) |
| Les PFAS **co-occurrent** (sources communes) | **Chaîne de classifieurs** : chaque PFAS exploite les prédictions des précédents |
| Mesures manquantes par programme (NaN) | **Semi-supervision** : pseudo-étiquetage des échantillons non mesurés |
| Forte rareté de certains analytes | **Rééchantillonnage SMOTE** par label |

---

## 2. Solution, méthodologie et résultats

### 2.1 Vue d'ensemble du pipeline

Le pipeline reproduit la logique de Dong et al. (2024) — chaîne de classifieurs XGBoost +
semi-supervision par pseudo-étiquetage + SMOTE — en l'adaptant à la réglementation 2024 et à une
base de données étendue. Quatre choix méthodologiques structurent la solution.

**(a) Cible : seuils réglementaires EPA 2024 NPDWR.** Plutôt que le seuil analytique uniforme de
2 ng/L (Dong et al.) ou l'ancien avis sanitaire de 70 ng/L (EPA 2016), chaque PFAS utilise son
**Maximum Contaminant Level (MCL) individuel** quand il existe, sinon 2 ng/L :

| PFAS | Seuil | Source |
|---|---|---|
| PFOA, PFOS | **4 ng/L** | MCL EPA 2024 NPDWR |
| PFHxS, PFNA | **10 ng/L** | MCL EPA 2024 NPDWR |
| 23 autres | 2 ng/L | Limite analytique (UCMR-5) |

Quatre composés sont exclus car jamais au-dessus de leur seuil sur la base : PFEESA, PFMBA, PFMPA
(max ≤ 2 ng/L) et HFPO-DA/GenX (max = 4,5 ng/L < MCL 10 ng/L). Il reste **27 PFAS cibles**.

**(b) Retrait des identifiants de localisation.** Comme Dong et al., les variables géographiques
pures (latitude, longitude, comté, codes de bassins) sont **retirées** pour forcer l'apprentissage
des mécanismes environnementaux et éviter la mémorisation spatiale. Les variables de **proximité aux
sources** (distance et densité de sites PFAS) sont **conservées** : elles sont mécanistiques, non
positionnelles. Une étude d'ablation confirme l'intérêt : retirer la localisation ne coûte que
−0,45 pt de macro-AUROC alors que ces variables représentaient ~16 % de l'importance du modèle —
preuve que le modèle s'appuyait sur une « béquille spatiale » sans valeur de généralisation.

**(c) Paliers de couverture (les « classes » de Dong et al.).** Point souvent mal compris : les
quatre classes de Dong et al. ne sont **pas des familles chimiques** mais des **paliers de
disponibilité des données**. Nos 27 PFAS se répartissent naturellement en quatre paliers de
couverture analytique quasi identiques :

| Palier | Couverture mesurée | PFAS | n |
|---|---|---|---|
| **0** | ~94–99,8 % | PFOS, PFOA, PFNA, PFHpA, PFUnDA, PFDA, PFHxA, PFHxS, PFBS, PFDoDA | 10 |
| **1** | ~87 % | ADONA, F53B-major, F53B-minor | 3 |
| **2** | ~55–59 % | PFTeDA, PFTrDA, NEtFOSAA, NMeFOSAA, PFPeS, FTS(4:2/6:2/8:2), PFBA, PFPeA, PFHpS | 11 |
| **3** | ~14–45 % | NFDHA, PFOSAm, PFDS | 3 |

Une propriété remarquable de notre base est l'**emboîtement parfait** de la couverture : lorsqu'un
PFAS rare est mesuré, PFOS l'est aussi dans ~100 % des cas. Cela ouvre une variante d'architecture
inaccessible à Dong et al. (voir §2.3).

**(d) Semi-supervision calibrée par palier.** Le pseudo-étiquetage (réintégration des prédictions de
confiance > 0,95 sur les échantillons non mesurés) n'apporte un gain que dans un régime
intermédiaire de données. La politique est donc calibrée par palier : désactivée aux paliers 0/1
(saturés en données), active au palier 2 (régime favorable), active mais conservatrice (confiance
≥ 0,99) au palier 3 pour limiter l'injection de bruit.

### 2.2 Protocole expérimental

- **Base** : CA-PFAS-ASGWS étendue — **46 338 observations × 201 colonnes**, période 2016–2026.
- **Features** : 97 variables après retrait de la localisation (93 numériques + 4 catégorielles),
  réparties en proximité aux sources (Geotracker), sol (SSURGO), hydro-climat (GLDAS), qualité de
  l'air (AQS) et co-contaminants.
- **Découpage** : 80 / 20 stratifié sur PFOS — **37 070 en entraînement, 9 268 en test**.
- **Prétraitement** : imputation médiane (numériques), encodage ordinal (catégorielles).
- **Classifieur** : chaîne XGBoost par palier, SMOTE par label, semi-supervision calibrée.
- **Métriques** : macro-AUROC, micro-F1, Hamming loss, Exact Match Ratio (EMR), + AUROC par PFAS et
  par palier.

### 2.3 Choix de l'architecture — benchmark de trois approches

Trois stratégies de division en classes ont été comparées sur le **même** jeu de test :

| Approche | Principe | Macro-AUROC |
|---|---|---|
| `global` | Une chaîne unique ordonnée par prévalence | 0,9589 |
| `nested` | Une chaîne ordonnée par palier + semi-sup calibrée (exploite l'emboîtement) | 0,9604 |
| **`class`** | **4 chaînes indépendantes, une par palier** (réplication Dong et al.) | **0,9661** |

L'approche **`class`** l'emporte, et particulièrement sur les **PFAS rares** (palier 3) : isoler
une chaîne courte par palier évite qu'une longue chaîne amont peu fiable ne propage du bruit sur les
analytes peu mesurés — exactement l'argument de Dong et al., confirmé sur nos données. Une étude
d'amélioration a ensuite montré que le seul levier réellement utile est le **renforcement des
hyperparamètres XGBoost** (400 arbres, profondeur 7), retenu pour le modèle final ; l'ordre par
sous-famille chimique n'apporte rien (chaînes déjà courtes par palier).

### 2.4 Résultats généraux

Modèle final (`class` + XGBoost renforcé), sur le jeu de test (9 268 puits) :

| Métrique | Valeur |
|---|---|
| **Macro-AUROC** (moyenne sur 27 PFAS) | **0,967** |
| **Micro-F1** | **0,862** |
| Hamming loss (↓ meilleur) | 0,157 |
| Exact Match Ratio | 0,215 |
| AUROC par PFAS | **0,887 (PFOSAm) → 0,993 (ADONA)**, médiane 0,968 |

L'apport du renforcement des hyperparamètres sur la baseline : micro-F1 +0,58 pt, EMR +2,9 pt.

### 2.5 Résultats par palier (« par classe »)

| Palier | n PFAS | Macro-AUROC | Hamming ↓ | Exact Match |
|---|---|---|---|---|
| **0** (les plus mesurés) | 10 | 0,9675 | 0,0655 | 0,6724 |
| **1** | 3 | **0,9872** | 0,0144 | 0,9783 |
| **2** | 11 | 0,9721 | 0,1114 | 0,5270 |
| **3** (les plus rares) | 3 | 0,9268 | 0,1457 | 0,7488 |

**Lecture.** La performance reste élevée sur tous les paliers, y compris pour les PFAS rares (palier 3,
AUROC ≈ 0,93). La dégradation attendue sur les analytes peu mesurés est nettement atténuée par
l'isolement des chaînes et la semi-supervision conservatrice. Une démonstration d'inférence sur des
puits « aveugles » (mesures masquées) atteint **97,3 % d'exactitude par-PFAS** sur les analytes
réellement mesurés, et le modèle identifie correctement les puits contaminés.

---

## 3. Positionnement par rapport à la littérature

### 3.1 Comparaison directe avec Dong et al. (2024)

Dong et al. (2024) constitue la référence directe (même pipeline conceptuel, même région). Le
tableau ci-dessous confronte les deux travaux.

| Élément | Dong et al. (2024) | Notre contribution |
|---|---|---|
| Observations | 26 901 | **46 338** |
| Colonnes | 157 | **201** |
| Période | 2016–2022 | **2016–2026** |
| Features d'entrée | 112 | 97 (après retrait localisation) |
| Cible (Tâche 2) | détection > 2 ng/L (seuil uniforme) | **MCL EPA 2024 par PFAS** (4/10 ng/L) ou 2 ng/L |
| PFAS prédits | 35 (4 classes par n d'obs.) | 27 (4 paliers de couverture) |
| Architecture | chaîne XGBoost + semi-sup + SMOTE | identique, + benchmark de 3 variantes |
| Macro-AUROC par classe | 0,938 / 0,965 / 0,942 / 0,985 | **0,968 / 0,987 / 0,972 / 0,927** |
| AUROC minimal individuel | 0,729 (PFBA) | **0,887 (PFOSAm)** |

**Convergences.** Nos résultats confirment les conclusions de Dong et al. : (i) la chaîne de
classifieurs surpasse les classifieurs indépendants ; (ii) les 4 chaînes par classe l'emportent ;
(iii) la proximité aux sources et les variables de sol dominent l'importance ; (iv) la
semi-supervision aide surtout le régime intermédiaire de données.

**Apports différenciants.**
- **AUROC minimal nettement relevé** (0,887 vs 0,729) : le palier des analytes les plus rares est
  mieux traité grâce au calibrage de la semi-supervision par palier et à des seuils EPA 2024 qui
  rééquilibrent partiellement les classes.
- **Adaptation réglementaire** : passage de l'ancien seuil 70 ng/L / 2 ng/L uniforme aux MCL
  individuels EPA 2024, qui est l'horizon normatif actuel.
- **Variante `nested`** : exploitant l'emboîtement de la couverture (rare mesuré ⇒ fréquent mesuré),
  elle conserve les corrélations inter-paliers que les 4 chaînes isolées perdent — option non
  disponible dans le protocole original.

### 3.2 Lecture au regard des revues de littérature

La revue de l'état de l'art (chapitre II du mémoire) dégage des constantes que nos résultats
recoupent :

- **La proximité industrielle est le signal le plus robuste** (Li et al. 2022 ; George & Dixit 2021 ;
  Tokranov et al. 2024) : nos features de proximité aux sources (`n_geotracker_within_*`,
  `dist_geotracker_km`) ressortent en tête de l'importance, ce qui valide la cohérence mécaniste
  (advection depuis les sources, déposition atmosphérique).
- **Le mode d'évaluation conditionne les performances** : les études affichant F1 > 0,90
  (George & Dixit, Kibbey) intègrent souvent des concentrations PFAS dans leurs features (mode
  **confirmatoire** ou mixte), ce qui crée une circularité partielle. Notre travail est en **mode
  prédictif strict**, sans aucune mesure PFAS en entrée : nos scores (macro-AUROC 0,967) sont donc
  directement comparables aux performances *opérationnelles*, pas à des scores confirmatoires gonflés.
- **Les méthodes d'ensemble à base d'arbres dominent les réseaux profonds génériques sur ce régime
  tabulaire** (Borisov et al. 2024) : le choix d'XGBoost comme classifieur de chaîne est conforme à
  ce constat, d'autant que notre corpus reste de taille modérée avec une fraction étiquetée limitée.

### 3.3 Le gap principal : la construction d'un meilleur jeu de données reproductible

Au-delà du modèle, la contribution majeure de ce travail est la **construction d'une base étendue,
documentée et reproductible**, qui répond directement à la lacune **L2 (biais d'évaluation
circulaire)** identifiée dans la revue, et prépare l'adressage de **L3 (transférabilité)**.

1. **Extension substantielle du corpus.** De 26 901 à **46 338 observations** (+72 %) et de 157 à
   **201 colonnes**, avec une période allongée jusqu'en 2026. L'enrichissement des features suit la
   table S9 de Dong et al. : ajout de champs granulométriques fins du sol (sables/limons par classe
   de taille, texture USDA), de l'ozone et du CO (qualité de l'air), de la température et du manteau
   neigeux GLDAS, d'une vingtaine de co-contaminants supplémentaires, et de variables de sol dérivées
   (coefficients de gradation, ratio eau/argile).

2. **Reproductibilité de bout en bout.** La collecte est entièrement scriptée (modules `collect_*`,
   `process_gldas`, `merge`) et documentée (rapport de collecte). Les sources (GAMA, Geotracker,
   USDA-SSURGO, NASA-GLDAS, EPA-AQS) sont interrogées par API publiques avec mise en cache, de sorte
   que la base peut être **régénérée et étendue** sans intervention manuelle. Le moteur d'enrichissement
   géospatial (générique, à opérateurs enfichables) a été validé par parité contre la base de
   référence, ce qui le rend portable à d'autres datasets et régions.

3. **Cible alignée sur la réglementation en vigueur.** En adoptant les MCL EPA 2024, la base produit
   des étiquettes directement exploitables pour la priorisation actuelle, là où l'ancien seuil 70 ng/L
   sous-estimait fortement le risque (≈ 5 % de dépassements en Californie).

4. **Vérification de l'absence de fuite confirmatoire.** Aucune concentration PFAS, ni indicateur de
   détection dérivé, ne figure parmi les features ; la cible est strictement séparée des entrées.
   C'est la condition pour que les performances reflètent un usage réel sur des puits jamais analysés.

**Limites assumées**, communes à Dong et al. : biais de sélection du programme GAMA
(sur-représentation des zones suspectes), découpage 80/20 aléatoire exposant à une fuite spatiale
résiduelle (une **validation croisée spatiale** par blocs géographiques est la prochaine étape de
rigueur), et dépendance du pseudo-étiquetage à un volume minimal d'exemples étiquetés.

---

## 4. Transposition au contexte camerounais

La question centrale est : **une telle étude est-elle réalisable au Cameroun, où l'infrastructure de
surveillance et les bases publiques n'ont pas l'ampleur californienne ?** La réponse est *oui, par
adaptation*, à condition de réorganiser la chaîne de valeur autour de la rareté des données — ce qui
est précisément le terrain de force de l'approche semi-supervisée.

### 4.1 Le verrou : l'absence de données PFAS et de programmes équivalents

Le Cameroun ne dispose ni d'un équivalent du programme GAMA, ni de Geotracker, ni de bases EPA
ouvertes. Les mesures PFAS y sont quasi inexistantes et la réglementation nationale sur ces composés
est embryonnaire. La capacité analytique LC-MS/MS locale est limitée : les échantillons doivent
souvent être envoyés vers des laboratoires régionaux ou internationaux, à coût élevé. C'est donc le
**régime extrême de rareté d'étiquettes** — celui pour lequel la semi-supervision et les chaînes de
classifieurs sont les plus pertinentes.

### 4.2 Une stratégie en trois temps

**Temps 1 — Construire les variables contextuelles à partir de données ouvertes mondiales.** La
quasi-totalité des features de notre pipeline a un équivalent global, mobilisable sans campagne de
terrain :

| Famille de variables | Source californienne | Substitut global mobilisable au Cameroun |
|---|---|---|
| Hydro-climat | GLDAS (NASA) | **GLDAS / ERA5** — couverture mondiale, directement applicable |
| Sol | SSURGO (USDA) | **SoilGrids (ISRIC)** — granulométrie, carbone organique, argile, à 250 m |
| Proximité aux sources | Geotracker (EPA) | **OpenStreetMap / inventaires industriels** — usines, aéroports, casernes, décharges |
| Qualité de l'air | AQS (EPA) | **Sentinel-5P / CAMS** — NO₂, SO₂, aérosols par télédétection |
| Hydrogéologie | SGMA | Cartes hydrogéologiques nationales (MINEE), BGR, cartes des aquifères africains |
| Population / usage du sol | — | **WorldPop, ESA WorldCover** |

Le moteur d'enrichissement géospatial développé ici, étant générique et à opérateurs enfichables, peut
être **reciblé sur ces sources** pour produire une base camerounaise de features contextuelles sur un
maillage de puits, **avant toute mesure PFAS**.

**Temps 2 — Échantillonnage ciblé minimal pour amorcer l'étiquetage.** Plutôt qu'une couverture
exhaustive, on constitue un petit jeu étiqueté (quelques centaines de puits) par prélèvement **priorisé
sur les zones à plus fort risque** : pôle industriel de Douala-Bonabéri, aéroports internationaux
(Douala, Yaoundé-Nsimalen, Garoua) où des mousses anti-incendie AFFF ont pu être utilisées, sites
militaires, grandes décharges et zones sans traitement des eaux usées. Ce ciblage maximise le nombre
de détections positives par échantillon analysé, donc la valeur informative de chaque mesure coûteuse.

**Temps 3 — Apprentissage semi-supervisé et transfert.** Avec ~quelques centaines d'étiquettes et un
large socle de puits non étiquetés, l'architecture exacte de ce travail s'applique :
pseudo-étiquetage des puits non mesurés, chaînes de classifieurs exploitant la co-occurrence des PFAS,
SMOTE pour le déséquilibre. Le retrait de la localisation favorise la généralisation aux zones non
échantillonnées. À court terme, le **transfert de connaissances** depuis le modèle californien
(features partagées, importances de la proximité aux sources) peut servir d'amorçage avant
réentraînement local.

### 4.3 Adaptations spécifiques et précautions

- **Réglementation cible** : en l'absence de MCL nationaux, adopter les seuils EPA 2024 ou les valeurs
  de la directive européenne (somme de PFAS) comme référence, en documentant le choix.
- **Sources de contamination locales** : intégrer des proxys adaptés (zones franches industrielles,
  ports, raffinage, agro-industrie, sites d'incendie) plutôt que la typologie US.
- **Hétérogénéité hydrogéologique** : les aquifères de socle et sédimentaires camerounais diffèrent des
  bassins californiens ; une **validation croisée spatiale** (par région) est ici indispensable, et non
  optionnelle, pour estimer honnêtement la généralisation.
- **Incertitude et usage** : communiquer des probabilités calibrées et restituer les résultats
  conjointement avec hydrogéologues et autorités de l'eau (MINEE, CAMWATER) ; l'outil **priorise et
  explique** (analyse SHAP des drivers), la décision reste argumentée au-delà du score.

### 4.4 Valeur attendue

Dans un pays où le contrôle exhaustif est hors de portée, un tel système offre exactement ce dont les
gestionnaires ont besoin : une **carte de priorisation des puits** construite sans analyse PFAS
préalable, permettant de concentrer un budget de prélèvement limité là où le risque prédit est le plus
élevé. La méthodologie reproductible (collecte scriptée, features ouvertes mondiales, pipeline
semi-supervisé) en fait un cadre **transférable** à d'autres pays à données rares, ce qui constitue la
réponse concrète à la lacune de **transférabilité géographique** soulignée dans la littérature.

---

## Synthèse

Ce travail reproduit et étend la Tâche 2 de Dong et al. (2024) pour la prédiction multilabel
semi-supervisée de 27 PFAS individuels en mode prédictif strict. Sur une base **étendue et
reproductible** (46 338 × 201, seuils EPA 2024), l'approche des 4 chaînes par palier de couverture
atteint une **macro-AUROC de 0,967** (AUROC par PFAS de 0,887 à 0,993), avec un AUROC minimal
nettement supérieur à la référence. La contribution principale tient autant au modèle qu'à la
**construction d'un meilleur jeu de données reproductible**, qui répond au biais d'évaluation
circulaire de la littérature et fournit une méthodologie directement **transposable au contexte
camerounais** par substitution de sources ouvertes mondiales et échantillonnage ciblé.
