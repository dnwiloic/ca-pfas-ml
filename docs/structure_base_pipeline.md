# Structure de base et modèle d'enrichissement du pipeline

**Thèse défendue.** Le projet ne se limite pas à reconstruire le dataset de
Dong et al. (2024). Il en **généralise la logique** : un *moteur d'enrichissement
géospatial réutilisable* qui prend n'importe quel jeu de données respectant une
**structure de base minimale**, et le ressort augmenté de dizaines de variables
contextuelles (sol, air, hydrologie, contamination de voisinage, co-polluants…).

Le dataset PFAS californien est simplement le **premier « client »** de ce
moteur. La contribution est l'abstraction, pas le cas particulier.

---

## 1. Le retournement de perspective

| Vision « reproduction » | Vision « moteur générique » (celle du mémoire) |
|---|---|
| « J'ai refait le dataset de l'article » | « J'ai conçu un système qui *fabrique* ce type de dataset » |
| Couplé aux PFAS et à GAMA | Découplé : la mesure étudiée est une charge utile interchangeable |
| Un livrable figé | Un pipeline adaptable à d'autres polluants / régions / sources |
| Contribution = données | Contribution = méthode + outil réutilisable |

Le principe central : **séparer le *quoi* (ce qu'on mesure) du *où/quand*
(les clés de jointure).** Le *quoi* change d'une étude à l'autre ; le *où/quand*
est universel. Tout l'enrichissement ne dépend que du *où/quand*.

---

## 2. La structure de base — contrat d'entrée minimal

Tout dataset qui veut « passer par le pipeline et ressortir enrichi » doit
fournir ces colonnes. C'est le **contrat d'entrée**.

| Champ | Type | Rôle | Obligatoire |
|---|---|---|---|
| `entity_id` | str / int | identifiant unique du point d'observation (puits, site, station, parcelle…) | **oui** |
| `latitude` | float (WGS84) | clé de **jointure spatiale** | **oui** |
| `longitude` | float (WGS84) | clé de **jointure spatiale** | **oui** |
| `observation_date` | date | clé de **jointure temporelle** | oui *si* enrichissements temporels |
| `measurement_*` | numérique | **charge utile** — ce qu'on étudie (PFAS, nitrate, métaux, n'importe quoi) | optionnel |

> Dans l'implémentation actuelle, ces champs portent les noms GAMA
> (`gm_well_id`, `latitude`, `longitude`, `collection_date`). Une **couche
> d'adaptation au nommage** (renommage à l'ingestion) suffit à brancher
> n'importe quelle source — voir §5.

**Invariant fondamental.** Si une ligne possède `(latitude, longitude,
observation_date)`, le pipeline peut lui attacher *toutes* les variables
contextuelles, **quelle que soit la nature de la mesure étudiée**.

---

## 3. Catalogue des enrichissements (la « sortie augmentée »)

Chaque enrichissement est un **opérateur enfichable** classé par la clé qu'il
exige. C'est ce qui rend le système modulaire : on ajoute/retire un opérateur
sans toucher aux autres.

### 3a. Enrichissements purement **spatiaux** (besoin : `latitude`, `longitude`)

| Opérateur | Mécanisme | Variables produites (exemples) |
|---|---|---|
| Bassins hydrogéologiques | `sjoin` polygone « within » | `sgma_basin_name`, `dwr_basin`, `regional_board` |
| Contamination de voisinage | `cKDTree` distance + comptage par rayon | `dist_geotracker_km`, `n_geotracker_within_{1,3,10,50}km`, `nearest_*_type` |
| Sol (SSURGO) | plus proche point (lat/lon arrondis) | `soil_sand_pct`, `soil_clay_pct`, `soil_ph`, `soil_ksat`, … (26 variables) |

### 3b. Enrichissements **spatio-temporels** (besoin : `lat`, `lon`, `date`)

| Opérateur | Mécanisme | Variables produites (exemples) |
|---|---|---|
| Co-contaminants | `merge_asof` ±365 j par entité | `cocontam_no3n`, `cocontam_tce`, `cocontam_as`, … (44 variables) |
| Météo (NOAA) | station la plus proche + fenêtre temporelle agrégée | température, précipitations, … |
| Qualité de l'air (AQS) | station la plus proche + fenêtre temporelle | `aqs_pm25`, `aqs_no2`, `aqs_ozone`, … (8 variables) |
| Hydrologie (GLDAS) | maille raster mensuelle la plus proche | `runoff_mm`, `soil_moisture_total_mm`, `snowpack_mm`, … |

**Bilan reconstruction PFAS :** à partir d'un socle de ~5 colonnes
`(id, lat, lon, date, mesures)`, le pipeline produit **201 colonnes** dont
~150 variables contextuelles. C'est la démonstration quantifiée du gain.

---

## 4. Le modèle d'opérateur enfichable

Tous les enrichissements partagent la **même signature** — c'est la propriété
qui fonde la réutilisabilité :

```
add_<source>(df: contrat, source: chemin|API) -> df + nouvelles colonnes
```

Propriétés :
- **Composable** : `df |> add_A |> add_B |> add_C` (ordre indifférent).
- **Optionnel** : chaque opérateur s'active/désactive (`--no-aqs`, `--no-gldas`…
  déjà présents dans `merge.py:main`).
- **Idempotent côté clés** : n'altère jamais le contrat, n'ajoute que des
  colonnes.
- **Indépendant de la charge utile** : ne lit jamais les colonnes `measurement_*`.

C'est une architecture de type *feature store géospatial* : un registre
d'opérateurs, un contrat d'entrée, une sortie enrichie.

---

## 5. Ce qu'il reste à faire pour que la généricité soit *prouvée*

La logique est déjà générique ; il faut la **rendre explicite et démontrable** :

1. **Couche d'adaptation au nommage** : une fonction `to_contract(df, mapping)`
   qui renomme les colonnes source vers `entity_id, latitude, longitude,
   observation_date`. ~30 lignes.
2. **Paramétrer la charge utile** : remplacer la liste d'analytes PFAS codée en
   dur par un argument (liste de colonnes `measurement_*` ou « tout le reste »).
3. **Preuve par l'exemple** : faire passer **un second dataset hétérogène**
   (ex. nitrate seul, ou un jeu d'une autre région) par le pipeline et montrer
   qu'il ressort enrichi des mêmes ~150 variables. **C'est l'expérience qui
   transforme la revendication en résultat** — sans elle, « adaptable » reste
   une affirmation.

---

## 6. Formulation pour le mémoire

> *« Nous proposons un moteur d'enrichissement géospatial générique. À partir
> d'un contrat d'entrée minimal — un identifiant, des coordonnées et une date —
> tout jeu de données d'observations ponctuelles peut être augmenté d'environ
> 150 variables environnementales et contextuelles, via une bibliothèque
> d'opérateurs enfichables fondés sur deux primitives de jointure (spatiale par
> k-d tree, temporelle par appariement asof). Le dataset PFAS de Dong et al.
> (2024), que nous reconstruisons et étendons, en constitue la première
> instanciation et la validation. »*

Trois arguments de défense :
1. **Découplage prouvé** : la mesure étudiée n'intervient jamais dans
   l'enrichissement (§4).
2. **Gain quantifié** : 5 → 201 colonnes (§3).
3. **Généralité démontrée** : un second dataset passé avec succès (§5.3).

---

*Voir aussi [rapport_fidelite_dataset.md](rapport_fidelite_dataset.md) (validation
de fidélité) et [rapport_collecte_donnees.md](rapport_collecte_donnees.md)
(détail des sources).*
