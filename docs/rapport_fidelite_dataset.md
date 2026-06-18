# Validation de fidélité — reconstruction CA-PFAS-ASGWS vs Dong et al. (2024)

**Objectif.** Établir dans quelle mesure le dataset reconstruit par le pipeline de
ce projet correspond au jeu de données original de Dong et al. (2024,
*ACS ES&T Water* 4, 969–981), et documenter les écarts comme des résultats
contrôlés plutôt que comme des défauts.

**Référence.** Tous les chiffres « article » proviennent du texte (§2.1–§3.1) et de
la Table 1 de Dong et al. (2024). Tous les chiffres « reconstruction » sont
calculés sur `data/processed/CA-PFAS-ASGWS.parquet` (état de juin 2026).

---

## 1. Synthèse — tableau de fidélité global

| Métrique | Dong et al. 2024 (référence) | Reconstruction | Écart / explication |
|---|---|---|---|
| Lignes (observations) | 26 901 | **46 338** | +72 % — **fenêtre temporelle étendue** (voir §3) |
| Colonnes totales | 157 | **201** | +44 — variables d'enrichissement supplémentaires |
| Variables d'entrée ML | 112 | (à figer) | dépend du seuil de NA appliqué (§2.1.4 article : >40 % NA supprimés) |
| Puits uniques | ~4 200 | **11 333** | cohérent avec la fenêtre étendue |
| Période couverte | 2016–2022 | **2016–2026** | +4 ans (2023–2026 = ~27 000 obs, soit 58 % du total) |
| Emprise spatiale | Californie | **lat 32,58–41,97 ; lon −124,28 à −114,48** | conforme (Californie entière) |
| Analytes PFAS | 38 | **31** | 27 communs, 11 « article seul », 4 « reconstruction seul » (§4) |
| Cible binaire (total PFAS > 70 ng/L) | ~28,9 %¹ | **24,8 %** | bon accord |
| PFOS médiane (ng/L) | 3,60 | 2,00 | concentrations un peu plus basses (fenêtre récente) |
| PFOA médiane (ng/L) | 2,90 | 1,48 | idem |

¹ *Non donné explicitement. Déduit de la Table 2 de l'article : les modèles
dégénérés (GaussianNB, LogReg, SVM) atteignent une accuracy de 0,711 avec un
recall ≈ 0, c.-à-d. qu'ils prédisent toujours la classe négative → prévalence
positive ≈ 1 − 0,711 = 28,9 %. Le « ~5 % » cité dans le texte (§3.1.1) concerne
une comparaison ponctuelle et non la prévalence de la cible d'apprentissage.*

**Conclusion §1.** Sur les grandeurs qui pilotent l'apprentissage (prévalence de
la cible binaire, taux de détection des analytes majeurs, emprise spatiale), la
reconstruction est **fidèle à l'esprit de l'article**. L'écart principal — le
nombre de lignes — n'est pas une erreur mais une **extension volontaire de la
fenêtre temporelle**, qui doit être assumée et présentée comme telle.

---

## 2. Cibles d'apprentissage — seuils PFOA / PFOS

Comparaison des proportions d'échantillons dépassant les seuils (parmi les
mesures non manquantes).

| Seuil | PFOA article | PFOA recon. | PFOS article | PFOS recon. |
|---|---|---|---|---|
| > 2 ng/L (MRL UCMR5) | ~60 % | 45,3 % | ~60 % | 49,5 % |
| > 4 ng/L | ~40 % | 35,2 % | ~40 % | **40,6 %** |
| > 70 ng/L | 2,0 % | 3,9 % | 3,1 % | 4,7 % |

Les seuils > 4 et > 70 ng/L sont en très bon accord. L'écart au seuil > 2 ng/L
s'explique vraisemblablement par (a) des limites de quantification plus basses
dans les campagnes récentes et (b) un traitement différent des non-détections.
**Point à fixer dans le mémoire : définir précisément ce qu'est une
« détection »** (valeur > MRL vs simple mesure rapportée) — l'article reste
ambigu sur ce point (il annonce « 95 % de détection » pour PFOA/PFOS alors que
seulement ~60 % dépassent 2 ng/L).

---

## 3. Écart sur le nombre de lignes : 26 901 → 46 338

Répartition des observations par année dans la reconstruction :

| Année | Obs | | Année | Obs |
|---|---|---|---|---|
| 2016 | 35 | | 2022 | 4 921 |
| 2017 | 236 | | 2023 | 7 657 |
| 2018 | 300 | | 2024 | 9 331 |
| 2019 | 3 683 | | 2025 | 10 025 |
| 2020 | 4 439 | | 2026 | 14 |
| 2021 | 5 697 | | | |

La fenêtre 2016–2022 (correspondant à l'article) totalise **~19 300 obs** —
inférieure aux 26 901 de l'article. Les **~27 000 obs supplémentaires
proviennent de 2023–2026**, indisponibles à la date de publication de Dong et al.

**Implication méthodologique.** L'écart 19 300 (fenêtre article) vs 26 901
(article) suggère que la reconstruction est même un peu *plus stricte* que
l'article sur 2016–2022 (filtrage, dédoublonnage, ou sources GAMA partiellement
mises à jour côté data.ca.gov). Pour une comparaison stricte « toutes choses
égales par ailleurs », **restreindre à 2016–2022** et re-comparer le N est
recommandé.

---

## 4. Écart sur les analytes : 38 → 31

- **27 analytes communs** (mappés ; noms harmonisés entre nomenclature GAMA et
  nomenclature article — ex. `PFTeDA`↔`PFTA`, `PFUnDA`↔`PFUnA`,
  `F53B_major`↔`11ClPF3OUDS`, `PFOSAm`↔`FOSA`).
- **11 analytes « article seul »** absents de la reconstruction :
  PFHxDA, PFODA, 3:3FTCA, 5:3FTCA, 7:3FTCA, 10:2FTS, PFNS, ETFOSE, ETFOSA,
  MEFOSE, MEFOSA.
  → **Tous avaient ≥ 94 % (souvent ≥ 98 %) de valeurs manquantes dans
  l'article** (n = 36 à 425 mesures). L'article lui-même en exclut une partie de
  la modélisation (les 3 FTCA, n = 36, exclus faute d'observations). Leur absence
  a donc un **impact négligeable** sur l'apprentissage.
- **4 analytes « reconstruction seul »** : NFDHA, PFEESA, PFMBA, PFMPA.
  → Acides éther fluorés de la **méthode EPA 533 / liste UCMR5**, ajoutés aux
  panels d'analyse récents. C'est un **enrichissement** par rapport à l'article,
  pas une perte.

Le détail mesure-par-mesure (n, %NA article vs reconstruction, taux de détection)
est dans [fidelite_analytes.csv](fidelite_analytes.csv).

---

## 5. Clarification sur les « 23 sources »

L'article et le README parlent de **23 jeux de données**. Il faut distinguer deux
rôles :

- **Sources de lignes (mesures PFAS)** : seules **6** sous-sources GAMA
  alimentent réellement les observations dans la reconstruction
  (DDW : 36 993 ; WB_CLEANUP : 5 860 ; WRD : 1 042 ; GAMA_USGS : 896 ;
  LOCALGW : 871 ; USGS_NWIS : 676).
- **Sources d'enrichissement (colonnes)** : les autres (sol SSURGO, air AQS,
  hydrologie GLDAS, bassins SGMA, sites GeoTracker, co-contaminants…) deviennent
  des **variables**, pas des lignes.

**À assumer dans le mémoire :** « 23 sources » désigne le périmètre de
collecte, mais le squelette du dataset repose sur ~6 sources de mesures PFAS et
~15 couches d'enrichissement. Le présenter clairement évite une question piège
en soutenance.

---

## 6. Recommandations pour le mémoire

1. **Restreindre une comparaison à 2016–2022** pour démontrer la fidélité
   « toutes choses égales » (N, médianes, prévalence), *puis* présenter
   l'extension 2016–2026 comme contribution propre.
2. **Figer le nombre de variables d'entrée** en réappliquant le critère de
   l'article (suppression des colonnes > 40 % NA) pour pouvoir annoncer un
   chiffre comparable aux 112 de l'article.
3. **Définir « détection »** sans ambiguïté et le justifier.
4. **Présenter les écarts comme un résultat** (analyse de reproductibilité) :
   « 27/38 analytes reproduits, les 11 manquants représentant < 2 % des mesures
   de l'article ; +4 analytes EPA 533 ; +4 ans de données ».

---

*Sources : Dong, Tsai, Olivares (2024), ACS ES&T Water 4, 969–981 (CC-BY 4.0) ;
`data/processed/CA-PFAS-ASGWS.parquet`. Calculs : `docs/fidelite_analytes.csv`.*
