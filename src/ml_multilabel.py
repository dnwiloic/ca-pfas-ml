"""
Phase 4 — Tâche 2 : Classification multilabel semi-supervisée (PFAS individuels)

Protocole Dong et al. (2024) :
  Label  : dépassement du seuil réglementaire individuel (EPA 2024 NPDWR si MCL
           existant, sinon 2 ng/L ≈ limite analytique de détection).
           Les non-détects sont stockés à MDL/2 ≈ 1 ng/L → label 0.
           NaN = analyte non mesuré dans ce programme → échantillon non étiqueté.
           27 PFAS cibles — 4 exclus car jamais > leur seuil :
             PFEESA, PFMBA, PFMPA (max ≤ 2 ng/L)
             HFPO_DA (max = 4.5 ng/L < MCL EPA 2024 de 10 ng/L)

  Semi-supervision (pseudo-étiquetage) :
    1. Modèles XGBoost initiaux entraînés sur les échantillons étiquetés
    2. Prédiction sur les échantillons non mesurés (NaN)
    3. Confiance > 0.95 → pseudo-label 1 ; < 0.05 → pseudo-label 0
    4. Réentraînement sur données étiquetées + pseudo-étiquetées (1 round)

  Chaîne de classifieurs (ClassifierChain) :
    Ordre : du PFAS le plus fréquent au plus rare
    Chaque modèle k utilise [X, p̂₁, …, p̂ₖ₋₁] comme features
    Les probabilités de chaîne proviennent des modèles entraînés avant k

  Rééquilibrage : SMOTE appliqué au fold d'entraînement de chaque label
  Validation    : StratifiedKFold 5-fold (stratifié sur PFOS, label pivot)
  Métriques     :
    - Par label  : AUROC, F1, Precision, Recall (sur les échantillons étiquetés)
    - Global     : Hamming loss, Exact Match Ratio, macro-AUROC, micro-F1

Usage :
    python -m src.ml_multilabel               # entraînement complet (~30 min)
    python -m src.ml_multilabel --no-pseudo   # sans pseudo-étiquetage
    python -m src.ml_multilabel --no-chain    # classifieurs indépendants
    python -m src.ml_multilabel --shap        # SHAP top-5 labels (~15 min)
    python -m src.ml_multilabel --test-run    # 5 000 lignes, rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Seuils de détection par PFAS (ng/L)
# MCLs individuels EPA 2024 NPDWR pour les 5 composés réglementés ;
# 2 ng/L (≈ MDL analytique) pour tous les autres sans MCL individuel.
# Exclus automatiquement : PFAS dont max(dataset) ≤ seuil.
DETECTION_THRESHOLDS: dict[str, float] = {
    # ── EPA 2024 NPDWR — MCLs individuels ──────────────────────────────
    "PFOA_ngL":      4.0,   # MCL 4 ng/L  → 35.1 % des éch. étiquetés
    "PFOS_ngL":      4.0,   # MCL 4 ng/L  → 40.5 %
    "PFNA_ngL":     10.0,   # MCL 10 ng/L →  3.1 %
    "PFHxS_ngL":    10.0,   # MCL 10 ng/L → 14.9 %
    # HFPO_DA_ngL : MCL 10 ng/L mais max dataset = 4.5 ng/L → 0 détection → exclu
    # ── Pas de MCL individuel : seuil analytique par défaut ────────────
    "PFBS_ngL":      2.0,   # HI seulement (réf. 2000 ng/L) → 39.1 %
    "PFHxA_ngL":     2.0,   # 38.3 %
    "FTS_6_2_ngL":   2.0,   # 27.9 %
    "PFHpA_ngL":     2.0,   # 27.8 %
    "PFBA_ngL":      2.0,   # 27.4 %
    "FTS_8_2_ngL":   2.0,   # 24.4 %
    "PFPeA_ngL":     2.0,   # 23.5 %
    "PFPeS_ngL":     2.0,   # 10.2 %
    "NEtFOSAA_ngL":  2.0,   #  9.5 %
    "NMeFOSAA_ngL":  2.0,   #  9.2 %
    "PFDA_ngL":      2.0,   #  8.2 %
    "FTS_4_2_ngL":   2.0,   #  7.9 %
    "PFHpS_ngL":     2.0,   #  4.7 %
    "PFUnDA_ngL":    2.0,   #  4.7 %
    "PFDoDA_ngL":    2.0,   #  4.2 %
    "F53B_minor_ngL":2.0,   #  4.1 %
    "ADONA_ngL":     2.0,   #  3.8 %
    "PFTeDA_ngL":    2.0,   #  3.8 %
    "PFTrDA_ngL":    2.0,   #  3.7 %
    "PFOSAm_ngL":    2.0,   #  3.6 %
    "F53B_major_ngL":2.0,   #  2.5 %
    "PFDS_ngL":      2.0,   #  2.3 %
    "NFDHA_ngL":     2.0,   #  1.7 %
}
DEFAULT_THRESHOLD: float = 2.0  # ng/L — valeur par défaut (MDL analytique)

PSEUDO_HIGH:  float = 0.95
PSEUDO_LOW:   float = 0.05
CV_FOLDS:     int   = 5
TEST_SIZE:    float = 0.20
RANDOM_STATE: int   = 42

# 27 PFAS cibles — ordonnés par prévalence décroissante (avec seuils EPA 2024)
# Exclus (0 dépassement de leur seuil) : PFEESA, PFMBA, PFMPA, HFPO_DA
PFAS_TARGET_COLS: list[str] = [
    "PFOS_ngL",       # 40.5 % (MCL EPA 2024 : 4 ng/L)
    "PFBS_ngL",       # 39.1 % (2 ng/L)
    "PFHxA_ngL",      # 38.3 % (2 ng/L)
    "PFOA_ngL",       # 35.1 % (MCL EPA 2024 : 4 ng/L)
    "FTS_6_2_ngL",    # 27.9 % (2 ng/L)
    "PFHpA_ngL",      # 27.8 % (2 ng/L)
    "PFBA_ngL",       # 27.4 % (2 ng/L)
    "FTS_8_2_ngL",    # 24.4 % (2 ng/L)
    "PFPeA_ngL",      # 23.5 % (2 ng/L)
    "PFHxS_ngL",      # 14.9 % (MCL EPA 2024 : 10 ng/L)
    "PFPeS_ngL",      # 10.2 % (2 ng/L)
    "NEtFOSAA_ngL",   #  9.5 % (2 ng/L)
    "NMeFOSAA_ngL",   #  9.2 % (2 ng/L)
    "PFDA_ngL",       #  8.2 % (2 ng/L)
    "FTS_4_2_ngL",    #  7.9 % (2 ng/L)
    "PFHpS_ngL",      #  4.7 % (2 ng/L)
    "PFUnDA_ngL",     #  4.7 % (2 ng/L)
    "PFDoDA_ngL",     #  4.2 % (2 ng/L)
    "F53B_minor_ngL", #  4.1 % (2 ng/L)
    "ADONA_ngL",      #  3.8 % (2 ng/L)
    "PFTeDA_ngL",     #  3.8 % (2 ng/L)
    "PFTrDA_ngL",     #  3.7 % (2 ng/L)
    "PFOSAm_ngL",     #  3.6 % (2 ng/L)
    "PFNA_ngL",       #  3.1 % (MCL EPA 2024 : 10 ng/L)
    "F53B_major_ngL", #  2.5 % (2 ng/L)
    "PFDS_ngL",       #  2.3 % (2 ng/L)
    "NFDHA_ngL",      #  1.7 % (2 ng/L)
]

# Index colonne → position dans PFAS_TARGET_COLS (référence d'alignement des proba)
_TARGET_IDX = {c: i for i, c in enumerate(PFAS_TARGET_COLS)}

# ──────────────────────────────────────────────────────────────
# Paliers de couverture analytique (≈ "Classes" de Dong et al. 2024)
# ──────────────────────────────────────────────────────────────
# ⚠️ Ce ne sont PAS des familles chimiques, mais des regroupements par
# DISPONIBILITÉ des mesures (nb d'échantillons étiquetés), comme dans le papier.
# La couverture est parfaitement EMBOÎTÉE : si un PFAS rare est mesuré, les PFAS
# fréquents le sont aussi (~100 %). Deux usages possibles de ces paliers :
#   • approche "class"  : 4 chaînes indépendantes (réplication Dong et al.)
#   • approche "nested" : 1 chaîne unique ordonnée par palier (exploite les
#                         corrélations inter-paliers grâce à l'emboîtement)
COVERAGE_TIERS: dict[int, list[str]] = {
    0: [  # ~94–99.8 % mesurés
        "PFOS_ngL", "PFOA_ngL", "PFNA_ngL", "PFHpA_ngL", "PFUnDA_ngL",
        "PFDA_ngL", "PFHxA_ngL", "PFHxS_ngL", "PFBS_ngL", "PFDoDA_ngL",
    ],
    1: [  # ~87 % mesurés
        "ADONA_ngL", "F53B_major_ngL", "F53B_minor_ngL",
    ],
    2: [  # ~55–59 % mesurés
        "PFTeDA_ngL", "PFTrDA_ngL", "NEtFOSAA_ngL", "NMeFOSAA_ngL", "PFPeS_ngL",
        "FTS_8_2_ngL", "FTS_6_2_ngL", "FTS_4_2_ngL", "PFBA_ngL", "PFPeA_ngL",
        "PFHpS_ngL",
    ],
    3: [  # ~14–45 % mesurés
        "NFDHA_ngL", "PFOSAm_ngL", "PFDS_ngL",
    ],
}

# Politique de semi-supervision PAR PALIER (calibrage du pseudo-étiquetage).
# Justification (Dong et al. + nos ratios étiquetés/non-étiquetés) :
#   Palier 0/1 : ~40–46k étiquetés / peu de non-étiquetés → rien à gagner (off)
#   Palier 2   : ~26k étiq. / ~20k non-étiq. → sweet spot, +2.9 % attendu (on)
#   Palier 3   : ~6–20k étiq. / ~26–40k non-étiq. → on mais CONSERVATEUR (conf
#                relevée à 0.99) pour limiter l'injection de pseudo-labels bruités
TIER_PSEUDO_POLICY: dict[int, dict] = {
    0: {"use_pseudo": False, "conf_high": 0.95, "conf_low": 0.05},
    1: {"use_pseudo": False, "conf_high": 0.95, "conf_low": 0.05},
    2: {"use_pseudo": True,  "conf_high": 0.95, "conf_low": 0.05},
    3: {"use_pseudo": True,  "conf_high": 0.99, "conf_low": 0.01},
}

# Colonnes à exclure de la matrice de features (fuite, identifiants)
_LEAK_SUFFIXES = ("_ngL",)
_LEAK_PREFIXES = ("label_",)
_LEAK_EXACT = {
    "sum_pfas_ngL", "pfas_class_assignment", "target_sum_gt70",
    "gm_well_id", "collection_date", "target_epa2024",
}

CATEGORICAL_FEATURES = [
    "gm_well_category",
    "nearest_geotracker_type",   # proximité source — mécanistique, CONSERVÉE
    "soil_texture_class",
    "sgma_region_office",        # localisation administrative
    "gm_dataset_name",
    "county",                    # localisation administrative
    "regional_board",            # localisation administrative
    "dwr_region",                # localisation administrative
]

# Attributs de LOCALISATION PURE — retirés par défaut (protocole Dong et al. 2024).
# Objectif : forcer le modèle à apprendre des variables environnementales et
# éviter la mémorisation spatiale (autocorrélation gonflant artificiellement les
# scores sur un split aléatoire). Les features de PROXIMITÉ AUX SOURCES
# (dist_geotracker_km, n_geotracker_within_*, nearest_geotracker_type) sont
# CONSERVÉES : elles sont mécanistiques, pas des identifiants de position.
LOCATION_FEATURES = {
    "latitude", "longitude",
    "county", "regional_board", "dwr_region",
    "sgma_region_office", "sgma_basin_name", "sgma_subbasin_name", "dwr_basin",
}

XGB_PARAMS: dict = {
    "n_estimators":      300,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "objective":         "binary:logistic",
    "eval_metric":       "auc",
    "n_jobs":            4,
    "random_state":      RANDOM_STATE,
    "verbosity":         0,
}

XGB_PARAMS_FAST: dict = {
    **XGB_PARAMS,
    "n_estimators": 100,
    "n_jobs": 2,
}


# ══════════════════════════════════════════════════════════════
# 1. Matrice de labels
# ══════════════════════════════════════════════════════════════

def build_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Y[i, k] = 1   si l'échantillon i dépasse le seuil réglementaire du PFAS k
                   (MCL EPA 2024 NPDWR si disponible, sinon 2 ng/L ≈ MDL)
    Y[i, k] = 0   si mesuré et ≤ seuil (non-détect ou inférieur au MCL)
    Y[i, k] = NaN si PFAS k non mesuré dans ce programme (échantillon non étiqueté)
    """
    Y = pd.DataFrame(index=df.index, dtype=float)
    for col in PFAS_TARGET_COLS:
        thr = DETECTION_THRESHOLDS.get(col, DEFAULT_THRESHOLD)
        if col not in df.columns:
            logger.warning("Label manquant dans le dataset : %s", col)
            Y[col] = np.nan
            continue
        raw = df[col]
        Y[col] = np.where(raw.isna(), np.nan, (raw > thr).astype(float))

    logger.info("Matrice de labels : %d × %d", *Y.shape)
    logger.info("  %-25s  %8s  %8s  %10s  %10s", "PFAS", "MCL(ng/L)", "Tested", "Positifs", "Prev%")
    for col in PFAS_TARGET_COLS:
        thr = DETECTION_THRESHOLDS.get(col, DEFAULT_THRESHOLD)
        n_lbl = Y[col].notna().sum()
        prev  = Y[col].mean()
        src   = "EPA 2024" if thr != DEFAULT_THRESHOLD else "2 ng/L"
        logger.info("  %-25s  %8.1f  %8d  %10d  %9.1f%%  [%s]",
                    col, thr, n_lbl, int(Y[col].sum()), 100 * prev, src)
    return Y


# ══════════════════════════════════════════════════════════════
# 2. Matrice de features
# ══════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame, drop_location: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    drop_location : si True (défaut, protocole Dong et al. 2024), retire les
                    identifiants géographiques purs (lat/lon, county, bassins…)
                    pour forcer l'apprentissage environnemental. Les features de
                    proximité aux sources restent toujours présentes.
    """
    df = df.copy()
    dt = pd.to_datetime(df["collection_date"])
    df["year"]   = dt.dt.year
    df["month"]  = dt.dt.month
    df["season"] = ((dt.dt.month % 12) // 3 + 1).astype(int)

    drop: set[str] = set()
    for c in df.columns:
        if any(c.endswith(s) for s in _LEAK_SUFFIXES):
            drop.add(c)
        if any(c.startswith(p) for p in _LEAK_PREFIXES):
            drop.add(c)
    drop |= _LEAK_EXACT
    drop |= {"sgma_basin_name", "sgma_subbasin_name", "dwr_basin"}

    if drop_location:
        loc_present = LOCATION_FEATURES & set(df.columns)
        drop |= loc_present
        logger.info("Localisation retirée (protocole Dong et al.) : %s",
                    ", ".join(sorted(loc_present)) or "aucune")
    else:
        logger.info("Localisation CONSERVÉE (--keep-location)")

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns and c not in drop]
    num_cols = [
        c for c in df.columns
        if c not in drop and c not in cat_cols
        and df[c].dtype.kind in ("f", "i", "u")
    ]

    logger.info("Features : %d numériques + %d catégorielles = %d total",
                len(num_cols), len(cat_cols), len(num_cols) + len(cat_cols))
    return df[num_cols + cat_cols].copy(), num_cols, cat_cols


# ══════════════════════════════════════════════════════════════
# 3. Prétraitement
# ══════════════════════════════════════════════════════════════

def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )),
        ]), cat_cols),
    ], remainder="drop")


# ══════════════════════════════════════════════════════════════
# 4. Entraînement XGB avec SMOTE pour un label binaire
# ══════════════════════════════════════════════════════════════

def _train_xgb(X: np.ndarray, y: np.ndarray,
               use_smote: bool = True, fast: bool = False,
               xgb_params: dict | None = None) -> XGBClassifier:
    """
    Entraîne un XGBClassifier binaire avec SMOTE optionnel.
    xgb_params : surcharge des hyperparamètres (sinon XGB_PARAMS[_FAST]).
    """
    params = dict(xgb_params) if xgb_params is not None else (XGB_PARAMS_FAST if fast else XGB_PARAMS)
    pos = int(y.sum())
    neg = int(len(y) - pos)

    if use_smote and pos >= 2 and neg >= 2:
        k = min(5, pos - 1)
        try:
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
            X, y = sm.fit_resample(X, y)
        except Exception:
            pass
    elif not use_smote:
        # Compenser le déséquilibre via scale_pos_weight
        params = {**params, "scale_pos_weight": neg / max(pos, 1)}

    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════════════════════
# 5. Pseudo-étiquetage
# ══════════════════════════════════════════════════════════════

def _pseudo_label(
    model: XGBClassifier, X_unlabeled: np.ndarray,
    conf_high: float = PSEUDO_HIGH, conf_low: float = PSEUDO_LOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (pseudo_labels, indices_retenus) pour les échantillons
    dont la confiance dépasse les seuils (conf_high / conf_low).
    """
    if len(X_unlabeled) == 0:
        return np.array([]), np.array([], dtype=int)
    proba = model.predict_proba(X_unlabeled)[:, 1]
    mask  = (proba >= conf_high) | (proba <= conf_low)
    idx   = np.where(mask)[0]
    labels = (proba[idx] >= conf_high).astype(float)
    return labels, idx


# ══════════════════════════════════════════════════════════════
# 6. Chaîne de classifieurs semi-supervisée
# ══════════════════════════════════════════════════════════════

def _build_chain(
    X_proc: np.ndarray,
    Y_train: pd.DataFrame,
    order: list[str],
    use_chain: bool,
    use_smote: bool,
    fast: bool,
    pseudo_for,
    xgb_params: dict | None = None,
) -> tuple[list[tuple[XGBClassifier, str]], np.ndarray]:
    """
    Construit UNE chaîne de classifieurs sur les colonnes `order`.

    pseudo_for : callable col → (use_pseudo: bool, conf_high: float, conf_low: float)
                 permet une politique de pseudo-étiquetage par label/palier.

    Returns
    -------
    models : list de (XGBClassifier|None, col) dans l'ordre `order`.
    train_proba : ndarray (n_train, len(order)) — proba sur le train (ordre chaîne).
    """
    n = len(X_proc)
    train_proba = np.full((n, len(order)), np.nan)
    models: list[tuple[XGBClassifier, str]] = []

    for k, col in enumerate(order):
        y_full  = Y_train[col].values          # float NaN/0/1
        lbl_idx = np.where(~np.isnan(y_full))[0]
        unl_idx = np.where(np.isnan(y_full))[0]
        y_lbl   = y_full[lbl_idx].astype(int)

        if len(lbl_idx) == 0:
            logger.warning("  Label %s : aucun échantillon étiqueté.", col)
            models.append((None, col))
            continue

        # Construire les features du label k (X + chaîne si use_chain)
        X_lbl = X_proc[lbl_idx]
        X_unl = X_proc[unl_idx] if len(unl_idx) > 0 else None

        if use_chain and k > 0:
            X_lbl = np.hstack([X_lbl, train_proba[lbl_idx, :k]])
            if X_unl is not None:
                X_unl = np.hstack([X_unl, train_proba[unl_idx, :k]])

        # Phase 1 — modèle initial sur échantillons étiquetés
        m0 = _train_xgb(X_lbl, y_lbl, use_smote=use_smote, fast=fast, xgb_params=xgb_params)

        # Phase 2 — pseudo-étiquetage + réentraînement (politique par label)
        use_pseudo, conf_high, conf_low = pseudo_for(col)
        n_pseudo = 0
        if use_pseudo and X_unl is not None:
            pseudo_lbl, pseudo_sel = _pseudo_label(m0, X_unl, conf_high, conf_low)
            n_pseudo = len(pseudo_sel)
            if n_pseudo > 0:
                X_aug = np.vstack([X_lbl, X_unl[pseudo_sel]])
                y_aug = np.concatenate([y_lbl, pseudo_lbl])
                m_final = _train_xgb(X_aug, y_aug, use_smote=use_smote, fast=fast, xgb_params=xgb_params)
            else:
                m_final = m0
        else:
            m_final = m0

        # Stocker les probabilités sur train (pour les labels suivants dans la chaîne)
        all_proba = m_final.predict_proba(X_proc if not (use_chain and k > 0)
                                          else np.hstack([X_proc, train_proba[:, :k]]))[:, 1]
        train_proba[:, k] = all_proba

        models.append((m_final, col))
        logger.info(
            "  [%02d/%02d] %-25s  étiq.=%6d  pos=%.1f%%  pseudo=%4d",
            k + 1, len(order), col, len(lbl_idx), 100 * y_lbl.mean(), n_pseudo,
        )

    return models, train_proba


def fit_chain(
    X_proc: np.ndarray,
    Y_train: pd.DataFrame,
    use_pseudo: bool = True,
    use_chain: bool = True,
    use_smote: bool = True,
    fast: bool = False,
) -> tuple[list[tuple[XGBClassifier, str]], np.ndarray]:
    """
    Chaîne GLOBALE (approche "global") — ordre PFAS_TARGET_COLS (prévalence),
    politique de pseudo-étiquetage uniforme. Comportement historique.
    """
    return _build_chain(
        X_proc, Y_train, PFAS_TARGET_COLS,
        use_chain=use_chain, use_smote=use_smote, fast=fast,
        pseudo_for=lambda col: (use_pseudo, PSEUDO_HIGH, PSEUDO_LOW),
    )


def _coverage_order() -> list[str]:
    """PFAS ordonnés par palier de couverture (0 → 3), du plus au moins mesuré."""
    return [c for tier in sorted(COVERAGE_TIERS) for c in COVERAGE_TIERS[tier]]


def _tier_of(col: str) -> int | None:
    for tier, cols in COVERAGE_TIERS.items():
        if col in cols:
            return tier
    return None


def fit_nested_chain(
    X_proc: np.ndarray,
    Y_train: pd.DataFrame,
    use_chain: bool = True,
    use_smote: bool = True,
    fast: bool = False,
    tier_policy: dict = TIER_PSEUDO_POLICY,
) -> tuple[list[tuple[XGBClassifier, str]], np.ndarray]:
    """
    Chaîne EMBOÎTÉE (approche "nested") — UNE chaîne unique ordonnée par palier
    de couverture, exploitant les corrélations inter-paliers (valide car la
    couverture est emboîtée), avec pseudo-étiquetage calibré PAR PALIER.
    """
    def pseudo_for(col: str) -> tuple[bool, float, float]:
        pol = tier_policy[_tier_of(col)]
        return pol["use_pseudo"], pol["conf_high"], pol["conf_low"]

    return _build_chain(
        X_proc, Y_train, _coverage_order(),
        use_chain=use_chain, use_smote=use_smote, fast=fast, pseudo_for=pseudo_for,
    )


def fit_class_chains(
    X_proc: np.ndarray,
    Y_train: pd.DataFrame,
    use_chain: bool = True,
    use_smote: bool = True,
    fast: bool = False,
    tier_policy: dict = TIER_PSEUDO_POLICY,
    tiers: dict[int, list[str]] | None = None,
    xgb_params: dict | None = None,
) -> dict[int, list[tuple[XGBClassifier, str]]]:
    """
    4 chaînes INDÉPENDANTES (approche "class", réplication Dong et al.) — une par
    palier de couverture, chacune avec son SMOTE et sa semi-supervision propres.

    tiers      : surcharge la composition/ordre des paliers (défaut COVERAGE_TIERS).
                 Permet d'expérimenter un ordre intra-palier (ex. par sous-famille
                 chimique, comme Dong et al.).
    xgb_params : surcharge des hyperparamètres XGBoost.
    """
    tiers = tiers if tiers is not None else COVERAGE_TIERS
    class_models: dict[int, list] = {}
    for tier in sorted(tiers):
        cols = tiers[tier]
        pol  = tier_policy[tier]
        logger.info("── Classe %d (%d PFAS, pseudo=%s) ──",
                    tier, len(cols), pol["use_pseudo"])
        models, _ = _build_chain(
            X_proc, Y_train, cols,
            use_chain=use_chain, use_smote=use_smote, fast=fast,
            pseudo_for=lambda col, pol=pol: (pol["use_pseudo"], pol["conf_high"], pol["conf_low"]),
            xgb_params=xgb_params,
        )
        class_models[tier] = models
    return class_models


def _predict_single_chain(
    models: list[tuple[XGBClassifier, str]],
    X_proc: np.ndarray,
    use_chain: bool = True,
) -> np.ndarray:
    """Prédictions d'UNE chaîne. Returns ndarray (n, len(models)) en ordre chaîne."""
    n = len(X_proc)
    proba = np.full((n, len(models)), np.nan)
    for k, (model, _) in enumerate(models):
        if model is None:
            continue
        X_k = np.hstack([X_proc, proba[:, :k]]) if (use_chain and k > 0) else X_proc
        proba[:, k] = model.predict_proba(X_k)[:, 1]
    return proba


def predict_chain(
    models: list[tuple[XGBClassifier, str]],
    X_proc: np.ndarray,
    use_chain: bool = True,
) -> np.ndarray:
    """
    Prédictions d'une chaîne unique, RÉALIGNÉES sur l'ordre PFAS_TARGET_COLS
    (n_samples × 27), quel que soit l'ordre interne de la chaîne. Compatible
    avec evaluate_multilabel.
    """
    chain_proba = _predict_single_chain(models, X_proc, use_chain)
    out = np.full((len(X_proc), len(PFAS_TARGET_COLS)), np.nan)
    for k, (_, col) in enumerate(models):
        if col in _TARGET_IDX:
            out[:, _TARGET_IDX[col]] = chain_proba[:, k]
    return out


def predict_class_chains(
    class_models: dict[int, list[tuple[XGBClassifier, str]]],
    X_proc: np.ndarray,
    use_chain: bool = True,
) -> np.ndarray:
    """Prédictions des 4 chaînes, fusionnées et alignées sur PFAS_TARGET_COLS."""
    out = np.full((len(X_proc), len(PFAS_TARGET_COLS)), np.nan)
    for models in class_models.values():
        chain_proba = _predict_single_chain(models, X_proc, use_chain)
        for k, (_, col) in enumerate(models):
            if col in _TARGET_IDX:
                out[:, _TARGET_IDX[col]] = chain_proba[:, k]
    return out


# ══════════════════════════════════════════════════════════════
# 7. Métriques multilabel
# ══════════════════════════════════════════════════════════════

def _label_name(col: str) -> str:
    return col.replace("_ngL", "")


def evaluate_multilabel(
    Y_true: pd.DataFrame,
    Y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    targets: list[str] | None = None,
) -> dict:
    """
    Calcule les métriques par label et globales.
    Seules les lignes avec le label étiqueté (notna) entrent dans le calcul.

    Y_pred_proba : ndarray (n, 27) aligné sur PFAS_TARGET_COLS.
    targets      : sous-ensemble de colonnes à évaluer (défaut : toutes). Permet
                   de calculer des métriques restreintes à un palier de couverture.
    """
    targets = targets if targets is not None else PFAS_TARGET_COLS
    per_label: list[dict] = []

    y_true_bin_list: list[np.ndarray] = []
    y_pred_bin_list: list[np.ndarray] = []

    for col in targets:
        k = _TARGET_IDX[col]
        y_full = Y_true[col].values
        mask   = ~np.isnan(y_full)
        if mask.sum() < 10:
            continue

        yt = y_full[mask].astype(int)
        yp = Y_pred_proba[mask, k]
        yb = (yp >= threshold).astype(int)

        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")
        ap  = average_precision_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")
        f1  = f1_score(yt, yb, zero_division=0)
        prec_val = f1_score(yt, yb, zero_division=0, average="binary",
                            pos_label=1) if yt.sum() > 0 else float("nan")

        per_label.append({
            "label":         _label_name(col),
            "n_labeled":     int(mask.sum()),
            "prevalence":    round(float(yt.mean()), 4),
            "roc_auc":       round(auc, 4) if not np.isnan(auc) else None,
            "avg_precision": round(ap, 4)  if not np.isnan(ap)  else None,
            "f1":            round(f1, 4),
        })

        y_true_bin_list.append(yt)
        y_pred_bin_list.append(yb)

    # Hamming loss et Exact Match sur les échantillons ayant TOUS les labels ciblés
    target_idx = [_TARGET_IDX[c] for c in targets]
    all_labeled_mask = ~np.any(np.isnan(Y_true[targets].values), axis=1)
    n_full = all_labeled_mask.sum()
    global_metrics: dict = {"n_full_panel": int(n_full)}

    if n_full > 0:
        Y_t = Y_true[targets].values[all_labeled_mask].astype(int)
        Y_p = (Y_pred_proba[np.ix_(all_labeled_mask, target_idx)] >= threshold).astype(int)
        global_metrics["hamming_loss"]       = round(float(hamming_loss(Y_t, Y_p)), 4)
        global_metrics["exact_match_ratio"]  = round(float((Y_t == Y_p).all(axis=1).mean()), 4)

    # Macro-AUROC (labels avec au moins 2 classes)
    valid_aucs = [r["roc_auc"] for r in per_label if r["roc_auc"] is not None]
    global_metrics["macro_roc_auc"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None

    micro_f1 = f1_score(
        np.concatenate(y_true_bin_list),
        np.concatenate(y_pred_bin_list),
        zero_division=0,
    ) if y_true_bin_list else None
    global_metrics["micro_f1"] = round(float(micro_f1), 4) if micro_f1 is not None else None

    return {"per_label": per_label, "global": global_metrics}


def evaluate_by_tier(Y_true: pd.DataFrame, Y_pred_proba: np.ndarray,
                     threshold: float = 0.5) -> dict[int, dict]:
    """Métriques globales restreintes à chaque palier de couverture (≈ Table 3 Dong)."""
    out: dict[int, dict] = {}
    for tier, cols in COVERAGE_TIERS.items():
        present = [c for c in cols if c in Y_true.columns]
        if not present:
            continue
        m = evaluate_multilabel(Y_true, Y_pred_proba, threshold, targets=present)
        out[tier] = m["global"]
    return out


# ══════════════════════════════════════════════════════════════
# 8. Cross-validation
# ══════════════════════════════════════════════════════════════

def cross_validate_chain(
    X_proc: np.ndarray,
    Y: pd.DataFrame,
    use_pseudo: bool,
    use_chain: bool,
    use_smote: bool,
    fast: bool,
) -> list[dict]:
    """5-fold stratified CV, stratification sur PFOS (label pivot)."""
    pivot_col = "PFOS_ngL"
    pivot_y = Y[pivot_col].fillna(-1).values

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results: list[dict] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_proc, pivot_y)):
        logger.info("── CV fold %d/%d ──", fold + 1, CV_FOLDS)
        X_tr, X_val = X_proc[tr_idx], X_proc[val_idx]
        Y_tr = Y.iloc[tr_idx].reset_index(drop=True)
        Y_val = Y.iloc[val_idx].reset_index(drop=True)

        models_fold, _ = fit_chain(
            X_tr, Y_tr,
            use_pseudo=use_pseudo,
            use_chain=use_chain,
            use_smote=use_smote,
            fast=fast,
        )
        proba_val = predict_chain(models_fold, X_val, use_chain=use_chain)
        metrics_fold = evaluate_multilabel(Y_val, proba_val)

        fold_summary = {
            "fold":          fold + 1,
            "hamming_loss":  metrics_fold["global"].get("hamming_loss"),
            "emr":           metrics_fold["global"].get("exact_match_ratio"),
            "macro_auc":     metrics_fold["global"].get("macro_roc_auc"),
            "micro_f1":      metrics_fold["global"].get("micro_f1"),
        }
        fold_results.append(fold_summary)
        logger.info(
            "  Fold %d : Hamming=%.4f  EMR=%.4f  macro-AUC=%.4f  micro-F1=%.4f",
            fold + 1,
            fold_summary["hamming_loss"] or float("nan"),
            fold_summary["emr"]          or float("nan"),
            fold_summary["macro_auc"]    or float("nan"),
            fold_summary["micro_f1"]     or float("nan"),
        )
    return fold_results


# ══════════════════════════════════════════════════════════════
# 9. Visualisations
# ══════════════════════════════════════════════════════════════

def plot_per_label_auc(per_label: list[dict], out_path: Path) -> None:
    labels  = [r["label"] for r in per_label if r["roc_auc"] is not None]
    aucs    = [r["roc_auc"] for r in per_label if r["roc_auc"] is not None]
    # Tri par AUC décroissant
    order   = sorted(range(len(aucs)), key=lambda i: aucs[i], reverse=True)
    labels  = [labels[i] for i in order]
    aucs    = [aucs[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, len(labels) * 0.36 + 1.5))
    colors  = plt.cm.RdYlGn(np.array(aucs))
    ax.barh(labels[::-1], aucs[::-1], color=colors[::-1])
    ax.axvline(0.5,  color="gray",      lw=1, ls="--", label="aléatoire")
    ax.axvline(0.75, color="steelblue", lw=1, ls=":",  label="0.75")
    ax.set_xlabel("AUROC (test set)")
    ax.set_title("AUROC par PFAS — Classification multilabel (Dong et al. 2024)")
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-label AUROC → %s", out_path)


def plot_cv_scores(fold_results: list[dict], out_path: Path) -> None:
    folds    = [r["fold"] for r in fold_results]
    metrics  = ["hamming_loss", "emr", "macro_auc", "micro_f1"]
    labels   = ["Hamming loss", "Exact Match", "Macro-AUROC", "Micro-F1"]
    colors   = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 4))
    for m, lbl, col in zip(metrics, labels, colors):
        vals = [r.get(m) or float("nan") for r in fold_results]
        ax.plot(folds, vals, "o-", label=lbl, color=col)
        ax.axhline(np.nanmean(vals), ls="--", color=col, alpha=0.5, lw=1)
    ax.set_xticks(folds)
    ax.set_xlabel("Fold CV")
    ax.set_title(f"Validation croisée {CV_FOLDS}-fold — Multilabel PFAS")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("CV scores → %s", out_path)


def plot_label_heatmap(Y: pd.DataFrame, out_path: Path) -> None:
    """Matrice de co-occurrence des labels détectés (sur panel complet)."""
    full_mask = ~np.any(np.isnan(Y[PFAS_TARGET_COLS].values), axis=1)
    Y_full = Y[PFAS_TARGET_COLS][full_mask].astype(float)
    if len(Y_full) < 100:
        return

    corr = Y_full.corr(method="pearson")
    short = [_label_name(c) for c in PFAS_TARGET_COLS if c in corr.columns]
    corr.columns = short
    corr.index   = short

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title("Corrélation de Pearson entre labels PFAS (panel complet)")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Label heatmap → %s", out_path)


# ══════════════════════════════════════════════════════════════
# 10. SHAP (optionnel)
# ══════════════════════════════════════════════════════════════

def compute_shap_top(
    models: list[tuple[XGBClassifier, str]],
    X_proc: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    top_n_labels: int = 5,
    n_sample: int = 1000,
) -> None:
    import shap

    top_labels = PFAS_TARGET_COLS[:top_n_labels]  # labels les plus fréquents
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_proc), min(n_sample, len(X_proc)), replace=False)

    for k, (model, col) in enumerate(models):
        if col not in top_labels or model is None:
            continue
        X_s = X_proc[idx]
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_s)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        if sv.ndim > 1 and sv.shape[1] > len(feature_names):
            # chain features appended — truncate to base features
            sv = sv[:, :len(feature_names)]

        mean_abs = np.abs(sv).mean(axis=0)
        n_feats  = min(len(feature_names), len(mean_abs))
        shap_df  = pd.DataFrame({
            "feature":      feature_names[:n_feats],
            "mean_abs_shap": mean_abs[:n_feats],
        }).sort_values("mean_abs_shap", ascending=False)

        label_short = _label_name(col)
        shap_df.to_csv(out_dir / f"ml_multilabel_shap_{label_short}.csv", index=False)

        shap.summary_plot(sv[:, :n_feats], X_s[:, :n_feats],
                          feature_names=feature_names[:n_feats],
                          max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"ml_multilabel_shap_{label_short}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("SHAP %s → %s", label_short, FIGURES_DIR)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run(
    use_pseudo: bool = True,
    use_chain:  bool = True,
    use_smote:  bool = True,
    drop_location: bool = True,
    compute_shap: bool = False,
    test_run: bool = False,
) -> dict:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ───────────────────────────────────────────
    logger.info("Chargement du dataset…")
    df = pd.read_parquet(PROCESSED_DIR / "CA-PFAS-ASGWS.parquet")
    if test_run:
        df = df.sample(5_000, random_state=RANDOM_STATE)
        logger.info("Mode test-run : 5 000 lignes")

    fast = test_run

    # ── Matrice de labels ─────────────────────────────────────
    logger.info("Construction de la matrice de labels…")
    Y = build_label_matrix(df)

    # ── Matrice de features ───────────────────────────────────
    X, num_cols, cat_cols = build_feature_matrix(df, drop_location=drop_location)
    feature_names = num_cols + cat_cols
    logger.info("Dataset : %d lignes × %d features", *X.shape)

    # ── Split train / test ────────────────────────────────────
    pivot_y = Y["PFOS_ngL"].fillna(-1).values
    X_train_df, X_test_df, Y_train, Y_test, idx_tr, idx_te = train_test_split(
        X, Y, np.arange(len(df)),
        test_size=TEST_SIZE,
        stratify=pivot_y,
        random_state=RANDOM_STATE,
    )
    logger.info("Train : %d  |  Test : %d", len(Y_train), len(Y_test))

    # ── Prétraitement ─────────────────────────────────────────
    preprocessor  = build_preprocessor(num_cols, cat_cols)
    X_train_proc  = preprocessor.fit_transform(X_train_df)
    X_test_proc   = preprocessor.transform(X_test_df)

    # ── Heatmap de co-occurrence ──────────────────────────────
    plot_label_heatmap(Y, FIGURES_DIR / "ml_multilabel_label_corr.png")

    # ── Validation croisée ────────────────────────────────────
    logger.info("Validation croisée %d-fold…", CV_FOLDS)
    fold_results = cross_validate_chain(
        X_train_proc, Y_train.reset_index(drop=True),
        use_pseudo=use_pseudo,
        use_chain=use_chain,
        use_smote=use_smote,
        fast=fast,
    )
    cv_df = pd.DataFrame(fold_results)
    cv_df.to_csv(PROCESSED_DIR / "ml_multilabel_cv_scores.csv", index=False)

    mean_cv = {
        "hamming_loss_mean": round(float(cv_df["hamming_loss"].dropna().mean()), 4),
        "hamming_loss_std":  round(float(cv_df["hamming_loss"].dropna().std()), 4),
        "emr_mean":          round(float(cv_df["emr"].dropna().mean()), 4),
        "emr_std":           round(float(cv_df["emr"].dropna().std()), 4),
        "macro_auc_mean":    round(float(cv_df["macro_auc"].dropna().mean()), 4),
        "macro_auc_std":     round(float(cv_df["macro_auc"].dropna().std()), 4),
        "micro_f1_mean":     round(float(cv_df["micro_f1"].dropna().mean()), 4),
        "micro_f1_std":      round(float(cv_df["micro_f1"].dropna().std()), 4),
    }
    logger.info(
        "CV moyen : Hamming=%.4f±%.4f  EMR=%.4f±%.4f  macro-AUC=%.4f±%.4f  micro-F1=%.4f±%.4f",
        mean_cv["hamming_loss_mean"], mean_cv["hamming_loss_std"],
        mean_cv["emr_mean"],          mean_cv["emr_std"],
        mean_cv["macro_auc_mean"],    mean_cv["macro_auc_std"],
        mean_cv["micro_f1_mean"],     mean_cv["micro_f1_std"],
    )

    # ── Entraînement final (tout le train) ────────────────────
    logger.info("Entraînement final de la chaîne de classifieurs…")
    models_final, _ = fit_chain(
        X_train_proc,
        Y_train.reset_index(drop=True),
        use_pseudo=use_pseudo,
        use_chain=use_chain,
        use_smote=use_smote,
        fast=fast,
    )

    # ── Évaluation sur le test set ────────────────────────────
    logger.info("Évaluation sur le test set…")
    proba_test = predict_chain(models_final, X_test_proc, use_chain=use_chain)
    Y_test_reset = Y_test.reset_index(drop=True)
    test_metrics = evaluate_multilabel(Y_test_reset, proba_test)

    per_label_df = pd.DataFrame(test_metrics["per_label"])
    per_label_df.to_csv(PROCESSED_DIR / "ml_multilabel_per_label_metrics.csv", index=False)

    # ── Rapport console ───────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════════════════")
    logger.info("RÉSULTATS TEST SET — Multilabel PFAS (Dong et al. 2024)")
    logger.info("══════════════════════════════════════════════════════")
    g = test_metrics["global"]
    logger.info("  Hamming loss      : %.4f", g.get("hamming_loss") or float("nan"))
    logger.info("  Exact Match Ratio : %.4f", g.get("exact_match_ratio") or float("nan"))
    logger.info("  Macro-AUROC       : %.4f", g.get("macro_roc_auc") or float("nan"))
    logger.info("  Micro-F1          : %.4f", g.get("micro_f1") or float("nan"))
    logger.info("")
    logger.info("  %-20s  %8s  %8s  %8s", "Label", "AUROC", "F1", "Prevalence")
    for r in sorted(test_metrics["per_label"], key=lambda x: -(x["roc_auc"] or 0)):
        logger.info("  %-20s  %8s  %8.4f  %7.1f%%",
                    r["label"],
                    f"{r['roc_auc']:.4f}" if r["roc_auc"] else "   N/A",
                    r["f1"],
                    100 * r["prevalence"])
    logger.info("══════════════════════════════════════════════════════")

    # ── Sauvegarde modèle ─────────────────────────────────────
    artifact = {
        "models":        models_final,
        "preprocessor":  preprocessor,
        "feature_names": feature_names,
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "pfas_targets":  PFAS_TARGET_COLS,
        "detection_thresholds": DETECTION_THRESHOLDS,
        "use_chain":     use_chain,
        "use_pseudo":    use_pseudo,
        "drop_location": drop_location,
    }
    model_path = MODELS_DIR / "chain_multilabel.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Modèle sauvegardé → %s", model_path)

    # ── Rapport JSON ──────────────────────────────────────────
    results = {
        "task":               "multilabel_pfas_detection",
        "protocol":           "Dong et al. 2024 — Classifier Chain + XGBoost + SMOTE + Semi-supervised",
        "detection_thresholds_ngL": DETECTION_THRESHOLDS,
        "default_threshold_ngL":   DEFAULT_THRESHOLD,
        "n_pfas_targets":     len(PFAS_TARGET_COLS),
        "pfas_targets":       [_label_name(c) for c in PFAS_TARGET_COLS],
        "n_train":            int(len(Y_train)),
        "n_test":             int(len(Y_test)),
        "n_features":         len(feature_names),
        "use_chain":          use_chain,
        "use_pseudo":         use_pseudo,
        "use_smote":          use_smote,
        "drop_location":      drop_location,
        "location_removed":   sorted(LOCATION_FEATURES) if drop_location else [],
        "pseudo_high_thresh": PSEUDO_HIGH,
        "pseudo_low_thresh":  PSEUDO_LOW,
        "cv":                 mean_cv,
        "test":               test_metrics["global"],
        "per_label_test":     test_metrics["per_label"],
    }
    with open(PROCESSED_DIR / "ml_multilabel_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Graphiques ────────────────────────────────────────────
    plot_per_label_auc(test_metrics["per_label"], FIGURES_DIR / "ml_multilabel_per_label_auc.png")
    plot_cv_scores(fold_results, FIGURES_DIR / "ml_multilabel_cv_scores.png")

    if compute_shap:
        compute_shap_top(models_final, X_test_proc, feature_names, PROCESSED_DIR)

    return results


# ══════════════════════════════════════════════════════════════
# Comparaison des approches de division en classes
# ══════════════════════════════════════════════════════════════

def _plot_approaches(summary_df: pd.DataFrame, per_tier_df: pd.DataFrame,
                     out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"global": "#7f8c8d", "nested": "#2980b9", "class": "#e67e22"}

    # (1) Métriques globales par approche
    metrics = ["macro_auc", "micro_f1", "exact_match"]
    labels  = ["Macro-AUROC", "Micro-F1", "Exact Match"]
    x = np.arange(len(metrics))
    w = 0.8 / max(len(summary_df), 1)
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] if row[m] is not None else 0 for m in metrics]
        ax1.bar(x + i * w, vals, w, label=row["approche"],
                color=palette.get(row["approche"], None))
    ax1.set_xticks(x + w * (len(summary_df) - 1) / 2)
    ax1.set_xticklabels(labels)
    ax1.set_title("Métriques globales (test set)")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3); ax1.set_ylim(0, 1)

    # (2) Macro-AUROC par palier
    tiers = sorted(per_tier_df["palier"].unique())
    x2 = np.arange(len(tiers))
    approaches = per_tier_df["approche"].unique()
    w2 = 0.8 / max(len(approaches), 1)
    for i, app in enumerate(approaches):
        sub = per_tier_df[per_tier_df["approche"] == app].set_index("palier")
        vals = [sub.loc[t, "macro_auc"] if t in sub.index and sub.loc[t, "macro_auc"] is not None
                else 0 for t in tiers]
        ax2.bar(x2 + i * w2, vals, w2, label=app, color=palette.get(app, None))
    ax2.set_xticks(x2 + w2 * (len(approaches) - 1) / 2)
    ax2.set_xticklabels([f"Palier {t}" for t in tiers])
    ax2.set_title("Macro-AUROC par palier de couverture")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3); ax2.set_ylim(0.5, 1)

    fig.suptitle("Comparaison des approches de division en classes", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Comparaison approches → %s", out_path)


def run_comparison(
    drop_location: bool = True,
    use_smote: bool = True,
    test_run: bool = False,
    approaches: tuple[str, ...] = ("global", "nested", "class"),
) -> dict:
    """
    Entraîne et compare plusieurs approches de division en classes sur le MÊME
    split train/test :
      - "global" : chaîne unique ordonnée par prévalence (baseline historique)
      - "nested" : chaîne unique ordonnée par palier + semi-sup calibrée par palier
      - "class"  : 4 chaînes indépendantes, une par palier (réplication Dong et al.)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Données (même prétraitement que run()) ───────────────────
    logger.info("Chargement du dataset…")
    df = pd.read_parquet(PROCESSED_DIR / "CA-PFAS-ASGWS.parquet")
    if test_run:
        df = df.sample(8_000, random_state=RANDOM_STATE)
        logger.info("Mode test-run : 8 000 lignes")
    fast = test_run

    Y = build_label_matrix(df)
    X, num_cols, cat_cols = build_feature_matrix(df, drop_location=drop_location)
    pivot_y = Y["PFOS_ngL"].fillna(-1).values
    X_tr_df, X_te_df, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=pivot_y, random_state=RANDOM_STATE,
    )
    Y_train = Y_train.reset_index(drop=True)
    Y_test  = Y_test.reset_index(drop=True)
    pre  = build_preprocessor(num_cols, cat_cols)
    X_tr = pre.fit_transform(X_tr_df)
    X_te = pre.transform(X_te_df)
    logger.info("Train : %d  |  Test : %d", len(Y_train), len(Y_test))

    # ── Entraînement des approches demandées ─────────────────────
    proba_by_approach: dict[str, np.ndarray] = {}

    if "global" in approaches:
        logger.info("═══ GLOBAL (baseline, chaîne prévalence, pseudo uniforme) ═══")
        m, _ = fit_chain(X_tr, Y_train, use_pseudo=True, use_chain=True,
                         use_smote=use_smote, fast=fast)
        proba_by_approach["global"] = predict_chain(m, X_te, use_chain=True)

    if "nested" in approaches:
        logger.info("═══ NESTED (chaîne emboîtée par palier + semi-sup calibrée) ═══")
        m, _ = fit_nested_chain(X_tr, Y_train, use_chain=True,
                                use_smote=use_smote, fast=fast)
        proba_by_approach["nested"] = predict_chain(m, X_te, use_chain=True)

    if "class" in approaches:
        logger.info("═══ CLASS (4 chaînes indépendantes, réplication Dong et al.) ═══")
        cm = fit_class_chains(X_tr, Y_train, use_chain=True,
                              use_smote=use_smote, fast=fast)
        proba_by_approach["class"] = predict_class_chains(cm, X_te, use_chain=True)

    # ── Évaluation comparée ──────────────────────────────────────
    summary_rows, per_tier_rows = [], []
    for name, proba in proba_by_approach.items():
        g = evaluate_multilabel(Y_test, proba)["global"]
        summary_rows.append({
            "approche":     name,
            "macro_auc":    g.get("macro_roc_auc"),
            "micro_f1":     g.get("micro_f1"),
            "hamming_loss": g.get("hamming_loss"),
            "exact_match":  g.get("exact_match_ratio"),
        })
        for tier, tm in evaluate_by_tier(Y_test, proba).items():
            per_tier_rows.append({
                "approche":     name,
                "palier":       tier,
                "macro_auc":    tm.get("macro_roc_auc"),
                "hamming_loss": tm.get("hamming_loss"),
                "exact_match":  tm.get("exact_match_ratio"),
            })

    summary_df  = pd.DataFrame(summary_rows)
    per_tier_df = pd.DataFrame(per_tier_rows)
    summary_df.to_csv(PROCESSED_DIR / "ml_multilabel_approaches_comparison.csv", index=False)
    per_tier_df.to_csv(PROCESSED_DIR / "ml_multilabel_approaches_by_tier.csv", index=False)
    _plot_approaches(summary_df, per_tier_df,
                     FIGURES_DIR / "ml_multilabel_approaches.png")

    # ── Console ──────────────────────────────────────────────────
    logger.info("")
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("COMPARAISON DES APPROCHES — Test set")
    logger.info("══════════════════════════════════════════════════════════")
    logger.info("  %-8s  %9s  %8s  %8s  %8s", "approche", "macro-AUC",
                "micro-F1", "Hamming", "ExMatch")
    for r in summary_rows:
        logger.info("  %-8s  %9.4f  %8.4f  %8.4f  %8.4f",
                    r["approche"], r["macro_auc"] or 0, r["micro_f1"] or 0,
                    r["hamming_loss"] or 0, r["exact_match"] or 0)
    logger.info("──────────────────────────────────────────────────────────")
    logger.info("  Macro-AUROC par palier :")
    for tier in sorted(COVERAGE_TIERS):
        line = f"    Palier {tier} : "
        for name in proba_by_approach:
            v = next((x["macro_auc"] for x in per_tier_rows
                      if x["approche"] == name and x["palier"] == tier), None)
            line += f"{name}={v:.4f}  " if v is not None else f"{name}=N/A  "
        logger.info(line)
    logger.info("══════════════════════════════════════════════════════════")

    comparison = {
        "approaches":         list(proba_by_approach.keys()),
        "n_train":            int(len(Y_train)),
        "n_test":             int(len(Y_test)),
        "drop_location":      drop_location,
        "coverage_tiers":     {t: [_label_name(c) for c in cols]
                               for t, cols in COVERAGE_TIERS.items()},
        "tier_pseudo_policy": TIER_PSEUDO_POLICY,
        "summary":            summary_rows,
        "by_tier":            per_tier_rows,
    }
    with open(PROCESSED_DIR / "ml_multilabel_approaches_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparaison sauvegardée → ml_multilabel_approaches_comparison.json")
    return comparison


# ══════════════════════════════════════════════════════════════
# Modèle final figé — approche "class" (meilleure du benchmark)
# ══════════════════════════════════════════════════════════════

def _prepare_data(test_run: bool = False, drop_location: bool = True,
                  sample: int | None = None) -> dict:
    """Charge, étiquette, split (stratifié PFOS) et prétraite. Réutilisable."""
    df = pd.read_parquet(PROCESSED_DIR / "CA-PFAS-ASGWS.parquet")
    if test_run:
        df = df.sample(8_000, random_state=RANDOM_STATE)
    elif sample is not None and sample < len(df):
        df = df.sample(sample, random_state=RANDOM_STATE)

    Y = build_label_matrix(df)
    X, num_cols, cat_cols = build_feature_matrix(df, drop_location=drop_location)
    pivot_y = Y["PFOS_ngL"].fillna(-1).values
    X_tr_df, X_te_df, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, stratify=pivot_y, random_state=RANDOM_STATE,
    )
    Y_train = Y_train.reset_index(drop=True)
    Y_test  = Y_test.reset_index(drop=True)
    pre  = build_preprocessor(num_cols, cat_cols)
    X_tr = pre.fit_transform(X_tr_df)
    X_te = pre.transform(X_te_df)
    return {
        "df": df, "Y": Y, "X_tr": X_tr, "X_te": X_te,
        "Y_train": Y_train, "Y_test": Y_test, "preprocessor": pre,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "feature_names": num_cols + cat_cols, "fast": test_run,
    }


# Hyperparamètres XGBoost renforcés ("hp+") — meilleure config du notebook 06.
# Gains sur données complètes vs baseline : micro-F1 +0.58 pt, EMR +2.9 pt.
XGB_STRONG: dict = {
    **XGB_PARAMS,
    "n_estimators":     400,
    "max_depth":        7,
    "learning_rate":    0.03,
    "subsample":        0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
}


def freeze_class_model(
    drop_location: bool = True,
    use_smote: bool = True,
    test_run: bool = False,
    xgb_params: dict | None = None,
    out_name: str = "class_multilabel.pkl",
) -> dict:
    """
    Entraîne l'approche "class" (4 chaînes indépendantes, meilleure du benchmark)
    sur le jeu complet et sauvegarde l'artefact final + ses métriques.

    xgb_params : hyperparamètres XGBoost (défaut : XGB_STRONG = config "hp+").
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if xgb_params is None:
        xgb_params = XGB_STRONG
    logger.info("Chargement & préparation des données…")
    data = _prepare_data(test_run=test_run, drop_location=drop_location)
    logger.info("Train : %d  |  Test : %d", len(data["Y_train"]), len(data["Y_test"]))

    logger.info("Entraînement CLASS (4 chaînes indépendantes, hp=%s)…",
                {k: xgb_params[k] for k in ("n_estimators", "max_depth", "learning_rate")})
    class_models = fit_class_chains(
        data["X_tr"], data["Y_train"], use_chain=True,
        use_smote=use_smote, fast=data["fast"], xgb_params=xgb_params,
    )

    proba_test = predict_class_chains(class_models, data["X_te"], use_chain=True)
    metrics  = evaluate_multilabel(data["Y_test"], proba_test)
    by_tier  = evaluate_by_tier(data["Y_test"], proba_test)

    artifact = {
        "approach":             "class",
        "class_models":         class_models,
        "preprocessor":         data["preprocessor"],
        "feature_names":        data["feature_names"],
        "num_cols":             data["num_cols"],
        "cat_cols":             data["cat_cols"],
        "pfas_targets":         PFAS_TARGET_COLS,
        "coverage_tiers":       COVERAGE_TIERS,
        "detection_thresholds": DETECTION_THRESHOLDS,
        "tier_pseudo_policy":   TIER_PSEUDO_POLICY,
        "xgb_params":           xgb_params,
        "use_chain":            True,
        "drop_location":        drop_location,
    }
    out_path = MODELS_DIR / out_name
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Modèle CLASS figé → %s", out_path)

    results = {
        "approach":   "class",
        "n_train":    int(len(data["Y_train"])),
        "n_test":     int(len(data["Y_test"])),
        "test":       metrics["global"],
        "by_tier":    {str(t): m for t, m in by_tier.items()},
        "per_label":  metrics["per_label"],
    }
    with open(PROCESSED_DIR / "ml_multilabel_class_final.json", "w") as f:
        json.dump(results, f, indent=2)

    g = metrics["global"]
    logger.info("════════════════════════════════════════════════")
    logger.info("MODÈLE CLASS FIGÉ — Test set")
    logger.info("  Macro-AUROC : %.4f", g.get("macro_roc_auc") or float("nan"))
    logger.info("  Micro-F1    : %.4f", g.get("micro_f1") or float("nan"))
    logger.info("  Hamming     : %.4f", g.get("hamming_loss") or float("nan"))
    logger.info("════════════════════════════════════════════════")
    return results


# ══════════════════════════════════════════════════════════════
# Inférence — nouveau puits sans mesure PFAS
# ══════════════════════════════════════════════════════════════

def predict_exceedances(artifact_path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Prédit les dépassements réglementaires pour de nouveaux puits.

    Paramètres
    ----------
    artifact_path : chemin vers chain_multilabel.pkl
    df_new        : DataFrame avec les features environnementales
                    (pas besoin de colonnes _ngL, elles seront ignorées)

    Retourne
    --------
    DataFrame (n_puits × n_pfas) avec :
      - colonnes suffixées _proba  : P(dépassement du MCL)
      - colonnes suffixées _exceed : 1 si P ≥ 0.5 (dépassement probable)
      - colonne 'n_exceedances'    : nombre de PFAS dépassant leur MCL
    """
    with open(artifact_path, "rb") as f:
        art = pickle.load(f)

    preprocessor  = art["preprocessor"]
    num_cols      = art["num_cols"]
    cat_cols      = art["cat_cols"]
    pfas_targets  = art["pfas_targets"]
    thresholds    = art.get("detection_thresholds", {})
    use_chain     = art.get("use_chain", True)

    # Construire la matrice de features (même pipeline qu'à l'entraînement)
    dt = pd.to_datetime(df_new["collection_date"])
    df_tmp = df_new.copy()
    df_tmp["year"]   = dt.dt.year
    df_tmp["month"]  = dt.dt.month
    df_tmp["season"] = ((dt.dt.month % 12) // 3 + 1).astype(int)

    available_num = [c for c in num_cols if c in df_tmp.columns]
    available_cat = [c for c in cat_cols if c in df_tmp.columns]
    X_new = preprocessor.transform(df_tmp[available_num + available_cat])

    # Compatible avec les deux types d'artefact : chaîne unique ("models")
    # ou 4 chaînes indépendantes ("class_models")
    if "class_models" in art:
        proba_matrix = predict_class_chains(art["class_models"], X_new, use_chain=use_chain)
    else:
        proba_matrix = predict_chain(art["models"], X_new, use_chain=use_chain)

    # Construire le DataFrame de résultats annoté avec les MCL
    result = pd.DataFrame(index=df_new.index)
    for k, col in enumerate(pfas_targets):
        thr = thresholds.get(col, DEFAULT_THRESHOLD)
        short = _label_name(col)
        result[f"{short}_proba"]  = proba_matrix[:, k]
        result[f"{short}_exceed"] = (proba_matrix[:, k] >= 0.5).astype(int)
        result[f"{short}_mcl_ngL"] = thr

    exceed_cols = [c for c in result.columns if c.endswith("_exceed")]
    result["n_exceedances"] = result[exceed_cols].sum(axis=1)

    # Signaler les PFAS avec un MCL EPA 2024 explicite
    epa_labels = [_label_name(c) for c, v in thresholds.items()
                  if v != DEFAULT_THRESHOLD]
    result.attrs["epa2024_mcl_labels"] = epa_labels
    result.attrs["detection_thresholds"] = {
        _label_name(c): v for c, v in thresholds.items()
    }

    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Multilabel semi-supervised PFAS — Dong et al. (2024)"
    )
    parser.add_argument("--no-pseudo",  action="store_true",
                        help="Désactiver le pseudo-étiquetage")
    parser.add_argument("--no-chain",   action="store_true",
                        help="Classifieurs indépendants (sans chaîne)")
    parser.add_argument("--no-smote",   action="store_true",
                        help="Désactiver SMOTE (utilise scale_pos_weight)")
    parser.add_argument("--keep-location", action="store_true",
                        help="Conserver lat/lon/county/bassins (désactive le "
                             "protocole Dong et al. de retrait de la localisation)")
    parser.add_argument("--approach", choices=["global", "compare", "class"], default="global",
                        help="'global' : chaîne unique (défaut) ; "
                             "'compare' : compare global vs nested vs class ; "
                             "'class' : entraîne et FIGE le modèle final (4 chaînes)")
    parser.add_argument("--shap",       action="store_true",
                        help="Calculer les SHAP values (top-5 labels)")
    parser.add_argument("--test-run",   action="store_true",
                        help="Sous-échantillon pour vérification rapide")
    args = parser.parse_args()

    if args.approach == "compare":
        run_comparison(
            drop_location=not args.keep_location,
            use_smote=not args.no_smote,
            test_run=args.test_run,
        )
    elif args.approach == "class":
        freeze_class_model(
            drop_location=not args.keep_location,
            use_smote=not args.no_smote,
            test_run=args.test_run,
        )
    else:
        run(
            use_pseudo=not args.no_pseudo,
            use_chain=not args.no_chain,
            use_smote=not args.no_smote,
            drop_location=not args.keep_location,
            compute_shap=args.shap,
            test_run=args.test_run,
        )
