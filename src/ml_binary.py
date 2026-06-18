"""
Classification binaire PFAS — Seuils EPA 2024 NPDWR

Cible (target_epa2024) :
  1  si l'échantillon dépasse au moins un MCL individuel EPA 2024
     OU si l'Indice de Risque du mélange > 1

  MCLs individuels (EPA, avril 2024) :
    PFOA  : 4 ng/L       PFOS  : 4 ng/L
    PFNA  : 10 ng/L      PFHxS : 10 ng/L      GenX (HFPO-DA) : 10 ng/L

  Indice de Risque mélange (PFHxS + PFNA + GenX + PFBS) :
    HI = PFNA/10 + PFHxS/10 + HFPO_DA/10 + PFBS/2000  > 1

Modèle    : RandomForestClassifier (sklearn)
Rééquil.  : ADASYN (imbalanced-learn) — optionnel, données ~1:1.2
Validation: StratifiedKFold 5-fold + split 80/20 test final
Sortie    : models/rf_binary_epa2024.pkl
            data/processed/ml_binary_results.json
            data/processed/ml_binary_feature_importance.csv
            data/processed/ml_binary_cv_scores.csv
            reports/figures/ml_binary_*.png

Usage :
    python -m src.ml_binary               # entraînement complet
    python -m src.ml_binary --no-adasyn   # sans rééquilibrage
    python -m src.ml_binary --tune        # GridSearchCV (~20 min)
    python -m src.ml_binary --shap        # SHAP values (lent)
    python -m src.ml_binary --test-run    # 5 000 lignes, rapide
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
FIGURES_DIR   = PROJECT_ROOT / "reports" / "figures"

# ──────────────────────────────────────────────────────────────
# Seuils réglementaires EPA 2024 NPDWR
# ──────────────────────────────────────────────────────────────
EPA2024_MCLS: dict[str, float] = {
    "PFOA_ngL":     4.0,
    "PFOS_ngL":     4.0,
    "PFNA_ngL":    10.0,
    "PFHxS_ngL":   10.0,
    "HFPO_DA_ngL": 10.0,
}

# Valeurs de référence pour l'Indice de Risque mélange (EPA 2024)
HI_REFS: dict[str, float] = {
    "PFNA_ngL":    10.0,
    "PFHxS_ngL":   10.0,
    "HFPO_DA_ngL": 10.0,
    "PFBS_ngL":  2000.0,
}

# Colonnes de fuite absolue (jamais incluses comme features)
_LEAK_SUFFIXES = ("_ngL",)
_LEAK_PREFIXES = ("label_",)
_LEAK_EXACT = {
    "sum_pfas_ngL", "pfas_class_assignment", "target_sum_gt70",
    "gm_well_id", "collection_date", "target_epa2024",
}

# Catégorielles à encoder — sélectionnées pour pertinence métier
CATEGORICAL_FEATURES = [
    "gm_well_category",        # Municipal / Monitoring / Domestic…      rang S9 #1
    "nearest_geotracker_type", # type du site PFAS le plus proche — CONSERVÉE
    "soil_texture_class",      # classe texturale USDA (SL, FSL, L…)     rang S9 #22/#31
    "sgma_region_office",      # bureau régional SGMA — localisation
    "gm_dataset_name",         # programme de surveillance GAMA
    "county",                  # comté — localisation
    "regional_board",          # tableau régional de l'eau — localisation
    "dwr_region",              # région DWR — localisation
]

# Attributs de LOCALISATION PURE — retirés par défaut (protocole Dong et al. 2024).
# Force l'apprentissage des variables environnementales et évite la mémorisation
# spatiale (autocorrélation gonflant les scores sur un split aléatoire). Les
# features de PROXIMITÉ AUX SOURCES (dist_geotracker_km, n_geotracker_within_*,
# nearest_geotracker_type) sont CONSERVÉES : mécanistiques, pas positionnelles.
LOCATION_FEATURES = {
    "latitude", "longitude",
    "county", "regional_board", "dwr_region",
    "sgma_region_office", "sgma_basin_name", "sgma_subbasin_name", "dwr_basin",
}

RF_PARAMS: dict = {
    "n_estimators":    500,
    "max_features":    "sqrt",
    "min_samples_leaf": 2,
    "max_depth":       None,
    "oob_score":       True,
    "class_weight":    "balanced",
    "n_jobs":          -1,
    "random_state":    42,
}

GRIDSEARCH_PARAMS = {
    "n_estimators":     [300, 500],
    "max_depth":        [None, 30],
    "min_samples_leaf": [1, 2, 5],
    "max_features":     ["sqrt", 0.3],
}

TEST_SIZE     = 0.20
CV_FOLDS      = 5
RANDOM_STATE  = 42


# ══════════════════════════════════════════════════════════════
# 1. Cible EPA 2024
# ══════════════════════════════════════════════════════════════

def compute_target_epa2024(df: pd.DataFrame) -> pd.Series:
    """
    Retourne une série binaire : 1 si l'échantillon dépasse au moins un MCL
    individuel EPA 2024 ou si l'Indice de Risque du mélange > 1.
    """
    exceeded = pd.Series(False, index=df.index)
    for col, mcl in EPA2024_MCLS.items():
        if col in df.columns:
            exceeded |= df[col].fillna(0) > mcl

    hi = pd.Series(0.0, index=df.index)
    for col, ref in HI_REFS.items():
        if col in df.columns:
            hi += df[col].fillna(0) / ref
    exceeded |= hi > 1.0

    pos = exceeded.sum()
    logger.info(
        "Cible EPA 2024 : %d positifs (%.1f%%)  |  %d négatifs (%.1f%%)",
        pos, 100 * pos / len(df), len(df) - pos, 100 * (1 - pos / len(df)),
    )
    return exceeded.astype(int)


# ══════════════════════════════════════════════════════════════
# 2. Sélection et ingénierie des features
# ══════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame, drop_location: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Construit la matrice de features X :
      - ajoute Year, Month, Season depuis collection_date
      - exclut les colonnes de fuite
      - sépare numériques et catégorielles

    drop_location : si True (défaut, protocole Dong et al. 2024), retire les
                    identifiants géographiques purs (lat/lon, county, bassins…)
                    pour forcer l'apprentissage environnemental. Les features de
                    proximité aux sources restent toujours présentes.
    """
    df = df.copy()

    dt = pd.to_datetime(df["collection_date"])
    df["year"]   = dt.dt.year
    df["month"]  = dt.dt.month
    df["season"] = ((dt.dt.month % 12) // 3 + 1).astype(int)  # 1=hiver…4=automne

    # Colonnes à exclure
    drop: set[str] = set()
    for c in df.columns:
        if any(c.endswith(s) for s in _LEAK_SUFFIXES):
            drop.add(c)
        if any(c.startswith(p) for p in _LEAK_PREFIXES):
            drop.add(c)
    drop |= _LEAK_EXACT
    # Colonnes à haute cardinalité et redondantes avec d'autres features géo
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
        if c not in drop
        and c not in cat_cols
        and df[c].dtype.kind in ("f", "i", "u")
    ]

    logger.info(
        "Features : %d numériques + %d catégorielles = %d total",
        len(num_cols), len(cat_cols), len(num_cols) + len(cat_cols),
    )
    return df[num_cols + cat_cols].copy(), num_cols, cat_cols


# ══════════════════════════════════════════════════════════════
# 3. Prétraitement
# ══════════════════════════════════════════════════════════════

def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ], remainder="drop")


# ══════════════════════════════════════════════════════════════
# 4. Métriques et graphiques
# ══════════════════════════════════════════════════════════════

def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             y_proba: np.ndarray, label: str = "") -> dict:
    auc    = roc_auc_score(y_true, y_proba)
    ap     = average_precision_score(y_true, y_proba)
    f1     = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "set":           label,
        "roc_auc":       round(auc, 4),
        "avg_precision": round(ap, 4),
        "f1":            round(f1, 4),
        "precision_pos": round(report["1"]["precision"], 4),
        "recall_pos":    round(report["1"]["recall"], 4),
        "precision_neg": round(report["0"]["precision"], 4),
        "recall_neg":    round(report["0"]["recall"], 4),
        "accuracy":      round(report["accuracy"], 4),
        "n_pos":         int(y_true.sum()),
        "n_neg":         int(len(y_true) - y_true.sum()),
    }
    logger.info(
        "[%s] AUC=%.4f  AP=%.4f  F1=%.4f  Prec=%.4f  Recall=%.4f",
        label, auc, ap, f1,
        report["1"]["precision"], report["1"]["recall"],
    )
    return metrics


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   auc: float, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="steelblue", label=f"RF (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC — Classification binaire PFAS (EPA 2024)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curve → %s", out_path)


def plot_feature_importance(fi: pd.DataFrame, top_n: int, out_path: Path) -> None:
    top = fi.head(top_n)
    fig, ax = plt.subplots(figsize=(8, top_n * 0.36 + 1))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))[::-1]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Importance (MDI)")
    ax.set_title(f"Top {top_n} features — Random Forest PFAS (EPA 2024)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance → %s", out_path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=["< MCL (0)", "≥ MCL (1)"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Matrice de confusion — Test set (EPA 2024)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix → %s", out_path)


# ══════════════════════════════════════════════════════════════
# 5. SHAP
# ══════════════════════════════════════════════════════════════

def compute_shap(model: RandomForestClassifier,
                 X_proc: np.ndarray, feature_names: list[str],
                 out_dir: Path, n_sample: int = 1000) -> None:
    import shap
    logger.info("Calcul SHAP sur %d échantillons…", n_sample)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_proc), min(n_sample, len(X_proc)), replace=False)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_proc[idx])
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    mean_abs   = np.abs(sv).mean(axis=0)
    shap_df    = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    shap_df    = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(out_dir / "ml_binary_shap_importance.csv", index=False)

    shap.summary_plot(sv, X_proc[idx], feature_names=feature_names,
                      max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_binary_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP → %s", FIGURES_DIR / "ml_binary_shap_summary.png")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def run(use_adasyn: bool = True, tune: bool = False,
        drop_location: bool = True,
        compute_shap_vals: bool = False, test_run: bool = False) -> dict:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Chargement ───────────────────────────────────────────
    logger.info("Chargement du dataset…")
    df = pd.read_parquet(PROCESSED_DIR / "CA-PFAS-ASGWS.parquet")
    if test_run:
        df = df.sample(5_000, random_state=RANDOM_STATE)
        logger.info("Mode test-run : 5 000 lignes")

    # ── Cible EPA 2024 ────────────────────────────────────────
    df["target_epa2024"] = compute_target_epa2024(df)
    y = df["target_epa2024"].values

    # ── Features ──────────────────────────────────────────────
    X, num_cols, cat_cols = build_feature_matrix(df, drop_location=drop_location)
    feature_names = num_cols + cat_cols
    logger.info("Dataset : %d lignes × %d features", *X.shape)

    # ── Split train / test ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )
    logger.info(
        "Train : %d (%.1f%% pos)  |  Test : %d (%.1f%% pos)",
        len(y_train), 100 * y_train.mean(),
        len(y_test),  100 * y_test.mean(),
    )

    # ── Prétraitement ─────────────────────────────────────────
    preprocessor  = build_preprocessor(num_cols, cat_cols)
    X_train_proc  = preprocessor.fit_transform(X_train)
    X_test_proc   = preprocessor.transform(X_test)

    # ── ADASYN ────────────────────────────────────────────────
    if use_adasyn:
        logger.info("ADASYN rééquilibrage…")
        ada = ADASYN(random_state=RANDOM_STATE, n_jobs=-1)
        X_train_res, y_train_res = ada.fit_resample(X_train_proc, y_train)
        logger.info(
            "  Après ADASYN : %d échantillons (%.1f%% pos)",
            len(y_train_res), 100 * y_train_res.mean(),
        )
    else:
        X_train_res, y_train_res = X_train_proc, y_train
        logger.info("ADASYN désactivé")

    # ── Validation croisée (sur train brut, non biaisée) ──────
    logger.info("Validation croisée %d-fold…", CV_FOLDS)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores: list[dict] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_proc, y_train)):
        X_tr, X_val = X_train_proc[tr_idx], X_train_proc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        if use_adasyn:
            try:
                X_tr, y_tr = ADASYN(random_state=RANDOM_STATE, n_jobs=-1).fit_resample(X_tr, y_tr)
            except Exception:
                pass

        rf_fold = RandomForestClassifier(**{**RF_PARAMS, "oob_score": False})
        rf_fold.fit(X_tr, y_tr)
        fold_m = evaluate(y_val, rf_fold.predict(X_val),
                          rf_fold.predict_proba(X_val)[:, 1], f"CV fold {fold+1}")
        fold_m["fold"] = fold + 1
        cv_scores.append(fold_m)

    cv_df = pd.DataFrame(cv_scores)
    logger.info(
        "CV moyen : AUC=%.4f±%.4f  F1=%.4f±%.4f  AP=%.4f±%.4f",
        cv_df["roc_auc"].mean(), cv_df["roc_auc"].std(),
        cv_df["f1"].mean(),      cv_df["f1"].std(),
        cv_df["avg_precision"].mean(), cv_df["avg_precision"].std(),
    )
    cv_df.to_csv(PROCESSED_DIR / "ml_binary_cv_scores.csv", index=False)

    # ── GridSearch optionnel ──────────────────────────────────
    if tune:
        logger.info("GridSearchCV…")
        gs = GridSearchCV(
            RandomForestClassifier(oob_score=False, n_jobs=-1, random_state=42,
                                   class_weight="balanced"),
            GRIDSEARCH_PARAMS,
            cv=StratifiedKFold(3, shuffle=True, random_state=42),
            scoring="roc_auc", n_jobs=-1, verbose=1,
        )
        gs.fit(X_train_res, y_train_res)
        logger.info("Meilleurs params : %s  (AUC=%.4f)", gs.best_params_, gs.best_score_)
        RF_PARAMS.update(gs.best_params_)

    # ── Entraînement final ────────────────────────────────────
    logger.info("Entraînement final sur %d échantillons…", len(y_train_res))
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_res, y_train_res)
    if hasattr(rf, "oob_score_"):
        logger.info("  OOB score : %.4f", rf.oob_score_)

    # ── Évaluation test ───────────────────────────────────────
    y_pred_test  = rf.predict(X_test_proc)
    y_proba_test = rf.predict_proba(X_test_proc)[:, 1]
    test_metrics = evaluate(y_test, y_pred_test, y_proba_test, "Test")

    # ── Feature importance MDI ────────────────────────────────
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi["rank"] = fi.index + 1
    fi.to_csv(PROCESSED_DIR / "ml_binary_feature_importance.csv", index=False)

    # ── SHAP optionnel ────────────────────────────────────────
    if compute_shap_vals:
        compute_shap(rf, X_test_proc, feature_names, PROCESSED_DIR)

    # ── Sauvegarde artefact ───────────────────────────────────
    artifact = {
        "preprocessor":  preprocessor,
        "model":         rf,
        "feature_names": feature_names,
        "num_cols":      num_cols,
        "cat_cols":      cat_cols,
        "target":        "target_epa2024",
        "epa2024_mcls":  EPA2024_MCLS,
        "hi_refs":       HI_REFS,
    }
    model_path = MODELS_DIR / "rf_binary_epa2024.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Modèle sauvegardé → %s", model_path)

    # ── Rapport JSON ──────────────────────────────────────────
    results = {
        "target":       "target_epa2024",
        "regulation":   "EPA 2024 NPDWR (MCLs individuels + HI mélange)",
        "mcls":         EPA2024_MCLS,
        "n_train":      int(len(y_train)),
        "n_test":       int(len(y_test)),
        "n_features":   len(feature_names),
        "adasyn":       use_adasyn,
        "drop_location": drop_location,
        "location_removed": sorted(LOCATION_FEATURES) if drop_location else [],
        "cv": {
            "folds":        CV_FOLDS,
            "roc_auc_mean": round(float(cv_df["roc_auc"].mean()), 4),
            "roc_auc_std":  round(float(cv_df["roc_auc"].std()), 4),
            "f1_mean":      round(float(cv_df["f1"].mean()), 4),
            "f1_std":       round(float(cv_df["f1"].std()), 4),
            "ap_mean":      round(float(cv_df["avg_precision"].mean()), 4),
            "ap_std":       round(float(cv_df["avg_precision"].std()), 4),
        },
        "test":         test_metrics,
        "top10_features": fi.head(10)[["rank", "feature", "importance"]].to_dict("records"),
        "rf_params":    {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
    }
    with open(PROCESSED_DIR / "ml_binary_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Graphiques ────────────────────────────────────────────
    plot_roc_curve(y_test, y_proba_test, test_metrics["roc_auc"],
                   FIGURES_DIR / "ml_binary_roc.png")
    plot_feature_importance(fi, top_n=30,
                            out_path=FIGURES_DIR / "ml_binary_feature_importance.png")
    plot_confusion_matrix(y_test, y_pred_test,
                          FIGURES_DIR / "ml_binary_confusion_matrix.png")

    # ── Résumé console ────────────────────────────────────────
    logger.info("")
    logger.info("════════════════════════════════════════════════════")
    logger.info("RÉSULTATS FINAUX — Test set (EPA 2024 NPDWR)")
    logger.info("════════════════════════════════════════════════════")
    logger.info("  ROC-AUC           : %.4f", test_metrics["roc_auc"])
    logger.info("  Average Precision : %.4f", test_metrics["avg_precision"])
    logger.info("  F1 (classe +1)    : %.4f", test_metrics["f1"])
    logger.info("  Précision         : %.4f", test_metrics["precision_pos"])
    logger.info("  Rappel            : %.4f", test_metrics["recall_pos"])
    logger.info("  Accuracy          : %.4f", test_metrics["accuracy"])
    logger.info("")
    logger.info("  CV %d-fold : AUC=%.4f±%.4f  F1=%.4f±%.4f",
                CV_FOLDS,
                results["cv"]["roc_auc_mean"], results["cv"]["roc_auc_std"],
                results["cv"]["f1_mean"],      results["cv"]["f1_std"])
    logger.info("")
    logger.info("  Top 10 features (MDI) :")
    for row in results["top10_features"]:
        logger.info("    #%-3d  %-38s %.4f",
                    row["rank"], row["feature"], row["importance"])
    logger.info("════════════════════════════════════════════════════")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="RF binaire PFAS — EPA 2024 NPDWR")
    parser.add_argument("--no-adasyn",  action="store_true",
                        help="Désactiver ADASYN (données déjà ~équilibrées 1:1.2)")
    parser.add_argument("--tune",       action="store_true",
                        help="GridSearchCV sur les hyperparamètres (~20 min)")
    parser.add_argument("--keep-location", action="store_true",
                        help="Conserver lat/lon/county/bassins (désactive le "
                             "protocole Dong et al. de retrait de la localisation)")
    parser.add_argument("--shap",       action="store_true",
                        help="Calculer les SHAP values (~5 min)")
    parser.add_argument("--test-run",   action="store_true",
                        help="Sous-échantillon 5 000 lignes pour vérification rapide")
    args = parser.parse_args()

    run(
        use_adasyn=not args.no_adasyn,
        tune=args.tune,
        drop_location=not args.keep_location,
        compute_shap_vals=args.shap,
        test_run=args.test_run,
    )
