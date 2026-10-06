"""
generate_multidomain_stability_figures.py

Probabilistic Classifier Decision Stability:
Multi-Dataset Probabilistic Classifier Study

Author:
    Nicholas Christakis

Repository purpose:
    This script generates the computational results, tables and figures for a
    multi-domain binary-classification study of decision-stability diagnostics.

    The study evaluates how threshold decisions from probabilistic classifiers
    change when the training sample grows progressively. For each dataset, each
    Monte Carlo repetition uses a fixed held-out validation set and nested
    training prefixes. Predicted probabilities at consecutive training sizes
    are compared through Bernoulli KL displacement.

Datasets:
    The script expects five preprocessed CSV files in the working directory.
    Each file must contain a header row, and the final column must be the binary
    target variable.

    1. pdac.csv
       Target: cancer = 1, no cancer = 0

    2. credit_card.csv
       Target: default payment yes = 1, no = 0

    3. electrical_grid.csv
       Target: stable = 0, unstable = 1

    4. machine_failure.csv
       Target: failure = 1, no failure = 0

    5. banknotes.csv
       Target: forged = 1, not forged = 0

Models:
    The script evaluates four probabilistic classifiers:

    1. Logistic regression
    2. Random forest
    3. XGBoost
    4. Artificial neural network, implemented as a one-hidden-layer MLPClassifier

Main diagnostic quantities:
    For a validation instance x and consecutive training sizes n and n_next,

        p_prev(x) = predicted probability after training on n samples
        p_curr(x) = predicted probability after training on n_next samples

    The script computes the Bernoulli KL displacement

        D_step(x) = KL(Bern(p_curr(x)) || Bern(p_prev(x)))

    and the Pinsker threshold certificate

        C_Pinsker(x; tau) = 2 * |p_prev(x) - tau|^2.

    It also computes the exact Bernoulli threshold boundary

        C_exact(x; tau) = KL(Bern(tau) || Bern(p_prev(x))).

    If a threshold decision flips, then both inequalities hold up to numerical
    tolerance:

        D_step(x) >= C_Pinsker(x; tau)
        D_step(x) >= C_exact(x; tau)

    These are deterministic checks for Bernoulli threshold crossings. The
    substantive empirical quantities are flip incidence, KL displacement,
    exact-boundary ratio, Pinsker-boundary ratio, and their relationship to
    predictive performance.

Thresholds:
    By default, the script evaluates:

        tau in {0.3, 0.5, 0.7}

Figure colour-scale convention:
    Figures 2 and 3 use common colour scales across all threshold panels.

    Figure 2:
        The flip-incidence heatmaps use one common linear colour range across
        tau = 0.3, 0.5 and 0.7. Thus, equal colours represent equal
        percentages in all three panels.

    Figure 3:
        The median rho_exact heatmaps use one common logarithmic colour
        normalization across tau = 0.3, 0.5 and 0.7. The annotations show the
        original rho_exact values. Thus, equal colours represent equal
        rho_exact values across all three panels.

Outputs:
    All outputs are written to:

        outputs/<run_id>/

    Main files:
        outputs/<run_id>/tables/dataset_summary.csv
        outputs/<run_id>/tables/progressive_mc_results_summary.csv
        outputs/<run_id>/tables/theorem_points_long.csv
        outputs/<run_id>/tables/progressive_summary_by_dataset_model_tau.csv
        outputs/<run_id>/tables/overall_stability_by_dataset_model_tau.csv
        outputs/<run_id>/tables/final_performance_by_dataset_model_tau.csv

    Main manuscript figures:
        outputs/<run_id>/figures/fig1_exact_vs_pinsker_boundaries.png
        outputs/<run_id>/figures/fig2_flip_incidence_heatmaps_all_thresholds.png
        outputs/<run_id>/figures/fig3_median_rho_exact_heatmaps_all_thresholds.png
        outputs/<run_id>/figures/fig4_stability_performance_tradeoff_tau0p5.png
        outputs/<run_id>/figures/fig5_exact_to_pinsker_boundary_ratio_tau0p5.png
        outputs/<run_id>/figures/fig6_progressive_flip_rate_tau0p5.png
        outputs/<run_id>/figures/fig7_progressive_kl_displacement_tau0p5.png
        outputs/<run_id>/figures/fig8_final_auroc_brier_tau0p5.png

    Supplementary per-dataset/per-model figures:
        outputs/<run_id>/figures/supplementary/

Reproducibility notes:
    - The global seed is fixed at RANDOM_SEED_GLOBAL = 42.
    - Each Monte Carlo repetition uses the repetition index as the random state.
    - Feature scaling is fitted on the training pool only and applied to the
      validation set.
    - The same fitted scaler is used for all nested prefixes within a repetition.
      Therefore, validation leakage is avoided, but the preprocessing is not a
      fully sequential prefix-by-prefix pipeline.
    - Predicted probabilities are clipped to [1e-12, 1 - 1e-12] before KL
      calculations.
    - Inequality checks use tolerance 1e-10.

Usage:
    python generate_multidomain_stability_figures.py
"""

from __future__ import annotations

import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LogNorm

from xgboost import XGBClassifier

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_SEED_GLOBAL = 42
N_REPETITIONS = 20
VAL_SIZE = 0.30

TAUS = [0.3, 0.5, 0.7]
MAIN_TAU = 0.5

EPS = 1e-12
VIOL_TOL = 1e-10

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sns.set_style("white")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASETS = {
    "pdac": {
        "file": "pdac.csv",
        "label": "PDAC",
        "sector": "Medical diagnostics",
        "target_description": "cancer = 1, no cancer = 0",
        "train_sizes": [100, 110, 120, 130, 140],
    },
    "credit_card": {
        "file": "credit_card.csv",
        "label": "Credit card default",
        "sector": "Finance",
        "target_description": "default payment yes = 1, no = 0",
        "train_sizes": [100, 250, 500, 1_000, 2_000],
    },
    "electrical_grid": {
        "file": "electrical_grid.csv",
        "label": "Electrical grid",
        "sector": "Energy systems",
        "target_description": "stable = 0, unstable = 1",
        "train_sizes": [100, 250, 500, 1_000, 2_000],
    },
    "machine_failure": {
        "file": "machine_failure.csv",
        "label": "Machine failure",
        "sector": "Predictive maintenance",
        "target_description": "failure = 1, no failure = 0",
        "train_sizes": [100, 250, 500, 1_000, 2_000],
    },
    "banknotes": {
        "file": "banknotes.csv",
        "label": "Banknote authentication",
        "sector": "Security",
        "target_description": "forged = 1, not forged = 0",
        "train_sizes": [100, 150, 250, 400, 700],
    },
}

DATASET_ORDER = list(DATASETS.keys())

DATASET_LABELS = [
    DATASETS[k]["label"]
    for k in DATASET_ORDER
]


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_ORDER = [
    "logreg",
    "rf",
    "xgb",
    "ann",
]

MODEL_LABELS = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "ann": "ANN",
}


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

RUN_ID = time.strftime("%Y%m%d_%H%M%S")

BASE_DIR = Path("outputs") / RUN_ID
FIGURES_DIR = BASE_DIR / "figures"
SUPP_FIGURES_DIR = FIGURES_DIR / "supplementary"
TABLES_DIR = BASE_DIR / "tables"

for directory in [
    BASE_DIR,
    FIGURES_DIR,
    SUPP_FIGURES_DIR,
    TABLES_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def tau_tag(tau: float) -> str:
    return f"tau{str(tau).replace('.', 'p')}"


def coerce_binary_target(
    y: pd.Series,
) -> np.ndarray:

    """
    Convert the target column to a numpy array of binary 0/1 labels.
    """

    y_raw = y.copy()

    if y_raw.dtype == bool:
        return y_raw.astype(int).to_numpy()

    if np.issubdtype(
        y_raw.dtype,
        np.number,
    ):

        y_num = pd.to_numeric(
            y_raw,
            errors="coerce",
        )

        if y_num.isna().any():
            raise ValueError(
                "Target contains NaNs after numeric coercion."
            )

        unique_values = set(
            np.unique(y_num)
        )

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                "Numeric target values are not binary 0/1: "
                f"{sorted(list(unique_values))[:20]}"
            )

        return y_num.astype(int).to_numpy()

    y_str = (
        y_raw
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "0": 0,
        "1": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "neg": 0,
        "pos": 1,
        "negative": 0,
        "positive": 1,
        "nocancer": 0,
        "no-cancer": 0,
        "no cancer": 0,
        "cancer": 1,
        "pdac": 1,
        "stable": 0,
        "unstable": 1,
        "failure": 1,
        "machine failure": 1,
        "no failure": 0,
        "nofailure": 0,
        "forged": 1,
        "not forged": 0,
        "not_forged": 0,
        "genuine": 0,
        "authentic": 0,
    }

    observed = set(
        y_str.unique()
    )

    if not observed.issubset(
        set(mapping.keys())
    ):
        raise ValueError(
            "Unrecognized target labels: "
            f"{sorted(list(observed))[:30]}"
        )

    return (
        y_str
        .map(mapping)
        .astype(int)
        .to_numpy()
    )


def bern_kl(
    p: np.ndarray,
    q: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:

    """
    Compute KL(Bern(p) || Bern(q)) elementwise.
    """

    p = np.clip(
        np.asarray(p, dtype=float),
        eps,
        1.0 - eps,
    )

    q = np.clip(
        np.asarray(q, dtype=float),
        eps,
        1.0 - eps,
    )

    return (
        p * np.log(p / q)
        + (1.0 - p)
        * np.log(
            (1.0 - p)
            / (1.0 - q)
        )
    )


def stratified_progressive_order(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    """
    Create a class-aware ordering of training-pool indices.

    The ordering interleaves shuffled class-0 and class-1 indices so that
    early prefixes stay close to the full training-pool class proportion.
    """

    y = y.astype(int)

    idx0 = np.where(
        y == 0
    )[0]

    idx1 = np.where(
        y == 1
    )[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n = len(y)

    p1 = (
        len(idx1)
        / max(n, 1)
    )

    order = []

    i0 = 0
    i1 = 0

    for k in range(n):

        desired_ones = int(
            round(
                p1 * (k + 1)
            )
        )

        take_one = (
            i1 < desired_ones
            and i1 < len(idx1)
        )

        if i0 >= len(idx0):
            take_one = True

        if i1 >= len(idx1):
            take_one = False

        if take_one:
            order.append(
                idx1[i1]
            )
            i1 += 1
        else:
            order.append(
                idx0[i0]
            )
            i0 += 1

    return np.asarray(
        order,
        dtype=int,
    )


def get_training_sizes(
    pool_size: int,
    requested_sizes: list[int],
) -> np.ndarray:

    """
    Keep requested training sizes that fit inside the training pool.
    If too few are valid, fall back to five increasing sizes.
    """

    valid = sorted({
        int(s)
        for s in requested_sizes
        if 2 <= int(s) < pool_size
    })

    if len(valid) >= 2:
        return np.asarray(
            valid,
            dtype=int,
        )

    lower = max(
        20,
        int(0.20 * pool_size),
    )

    upper = max(
        lower + 1,
        int(0.95 * pool_size),
    )

    fallback = np.linspace(
        lower,
        upper,
        num=5,
    ).astype(int)

    fallback = sorted({
        int(s)
        for s in fallback
        if 2 <= int(s) < pool_size
    })

    if len(fallback) < 2:
        raise ValueError(
            "Training pool too small for progressive study: "
            f"{pool_size}"
        )

    return np.asarray(
        fallback,
        dtype=int,
    )


def build_models(
    random_state: int,
) -> dict[str, object]:

    """
    Build the four probabilistic classifiers used in the study.
    """

    return {

        "logreg": LogisticRegression(
            max_iter=5_000,
            solver="lbfgs",
            random_state=random_state,
        ),

        "rf": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),

        "xgb": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0,
        ),

        "ann": MLPClassifier(
            hidden_layer_sizes=(20,),
            activation="relu",
            solver="adam",
            max_iter=2_000,
            learning_rate_init=0.01,
            alpha=1e-4,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.15,
        ),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:

    """
    Compute threshold-dependent and threshold-free metrics.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "auc": float(
            roc_auc_score(
                y_true,
                y_proba,
            )
        )
        if len(np.unique(y_true)) == 2
        else np.nan,

        "auprc": float(
            average_precision_score(
                y_true,
                y_proba,
            )
        )
        if len(np.unique(y_true)) == 2
        else np.nan,

        "brier": float(
            brier_score_loss(
                y_true,
                y_proba,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_dataset(
    dataset_key: str,
    config: dict,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
]:

    """
    Load a preprocessed dataset.

    The final column is treated as the binary target. Nonnumeric feature
    columns are one-hot encoded to make the script robust to categorical
    fields that may remain in a preprocessed file.
    """

    path = Path(
        config["file"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: "
            f"{path.resolve()}"
        )

    df = (
        pd.read_csv(path)
        .dropna()
        .reset_index(drop=True)
    )

    if df.shape[1] < 2:
        raise ValueError(
            f"{path} must contain at least "
            "one feature and one target column."
        )

    target_series = df.iloc[:, -1]

    feature_df = (
        df.iloc[:, :-1]
        .copy()
    )

    feature_df = pd.get_dummies(
        feature_df,
        drop_first=False,
    )

    feature_df = feature_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if feature_df.isna().any().any():

        bad_cols = (
            feature_df.columns[
                feature_df.isna().any()
            ].tolist()
        )

        raise ValueError(
            "Nonnumeric or NaN feature values remain "
            f"after preprocessing in {path.name}. "
            f"Problem columns: {bad_cols[:20]}"
        )

    y = coerce_binary_target(
        target_series
    )

    unique_y = set(
        np.unique(y)
    )

    if unique_y != {0, 1}:
        raise ValueError(
            f"{path.name} target must contain both "
            f"classes 0 and 1. Found: {unique_y}"
        )

    return (
        feature_df.astype(float),
        y,
        df,
    )


# =============================================================================
# AGGREGATION HELPERS
# =============================================================================

def aggregate_progressive_summary(
    df_summary: pd.DataFrame,
) -> pd.DataFrame:

    """
    Aggregate repetition-level results by dataset, model,
    threshold and training size.
    """

    df_ok = df_summary[
        df_summary[
            "skipped_single_class_prefix"
        ] == 0
    ].copy()

    agg = (
        df_ok
        .groupby(
            [
                "dataset",
                "dataset_label",
                "model",
                "model_label",
                "tau",
                "train_size",
            ],
            as_index=False,
        )
        .agg(

            accuracy_mean=(
                "accuracy",
                "mean",
            ),

            accuracy_std=(
                "accuracy",
                "std",
            ),

            precision_mean=(
                "precision",
                "mean",
            ),

            precision_std=(
                "precision",
                "std",
            ),

            recall_mean=(
                "recall",
                "mean",
            ),

            recall_std=(
                "recall",
                "std",
            ),

            specificity_mean=(
                "specificity",
                "mean",
            ),

            specificity_std=(
                "specificity",
                "std",
            ),

            f1_mean=(
                "f1",
                "mean",
            ),

            f1_std=(
                "f1",
                "std",
            ),

            auc_mean=(
                "auc",
                "mean",
            ),

            auc_std=(
                "auc",
                "std",
            ),

            auprc_mean=(
                "auprc",
                "mean",
            ),

            auprc_std=(
                "auprc",
                "std",
            ),

            brier_mean=(
                "brier",
                "mean",
            ),

            brier_std=(
                "brier",
                "std",
            ),

            flip_rate_mean=(
                "flip_rate_step",
                "mean",
            ),

            flip_rate_std=(
                "flip_rate_step",
                "std",
            ),

            d_step_mean=(
                "d_step_mean",
                "mean",
            ),

            d_step_std=(
                "d_step_mean",
                "std",
            ),

            rho_pinsker_median_mean=(
                "rho_pinsker_median",
                "mean",
            ),

            rho_pinsker_median_std=(
                "rho_pinsker_median",
                "std",
            ),

            rho_exact_median_mean=(
                "rho_exact_median",
                "mean",
            ),

            rho_exact_median_std=(
                "rho_exact_median",
                "std",
            ),

            flips_sum=(
                "flips_step",
                "sum",
            ),

            violations_pinsker_sum=(
                "violations_pinsker",
                "sum",
            ),

            violations_exact_sum=(
                "violations_exact",
                "sum",
            ),
        )
    )

    return agg


def make_overall_stability_summary(
    df_points: pd.DataFrame,
) -> pd.DataFrame:

    """
    Summarize all validation-instance transition diagnostics
    by dataset, model and threshold.
    """

    rows = []

    for dataset_key in DATASET_ORDER:

        for model_key in MODEL_ORDER:

            for tau in TAUS:

                df_all = df_points[
                    (df_points["dataset"] == dataset_key)
                    &
                    (df_points["model"] == model_key)
                    &
                    (
                        np.isclose(
                            df_points["tau"],
                            tau,
                        )
                    )
                ].copy()

                df_flip = df_all[
                    df_all["flip"] == 1
                ].copy()

                n_all = len(df_all)
                n_flip = len(df_flip)

                row = {

                    "dataset": dataset_key,

                    "dataset_label":
                        DATASETS[
                            dataset_key
                        ]["label"],

                    "model": model_key,

                    "model_label":
                        MODEL_LABELS[
                            model_key
                        ],

                    "tau": tau,

                    "total_comparisons":
                        n_all,

                    "total_flips":
                        n_flip,

                    "flip_incidence":
                        (
                            n_flip / n_all
                            if n_all > 0
                            else np.nan
                        ),

                    "violations_pinsker":
                        (
                            int(
                                df_flip[
                                    "violates_pinsker"
                                ].sum()
                            )
                            if n_flip > 0
                            else 0
                        ),

                    "violations_exact":
                        (
                            int(
                                df_flip[
                                    "violates_exact"
                                ].sum()
                            )
                            if n_flip > 0
                            else 0
                        ),

                    "d_step_median_flips":
                        (
                            float(
                                df_flip[
                                    "d_step"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "c_pinsker_median_flips":
                        (
                            float(
                                df_flip[
                                    "c_pinsker"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "c_exact_median_flips":
                        (
                            float(
                                df_flip[
                                    "c_exact"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_pinsker_min":
                        (
                            float(
                                df_flip[
                                    "rho_pinsker"
                                ].min()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_pinsker_median":
                        (
                            float(
                                df_flip[
                                    "rho_pinsker"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_pinsker_p10":
                        (
                            float(
                                df_flip[
                                    "rho_pinsker"
                                ].quantile(0.10)
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_pinsker_p90":
                        (
                            float(
                                df_flip[
                                    "rho_pinsker"
                                ].quantile(0.90)
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_exact_min":
                        (
                            float(
                                df_flip[
                                    "rho_exact"
                                ].min()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_exact_median":
                        (
                            float(
                                df_flip[
                                    "rho_exact"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_exact_p10":
                        (
                            float(
                                df_flip[
                                    "rho_exact"
                                ].quantile(0.10)
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "rho_exact_p90":
                        (
                            float(
                                df_flip[
                                    "rho_exact"
                                ].quantile(0.90)
                            )
                            if n_flip > 0
                            else np.nan
                        ),

                    "boundary_ratio_median":
                        (
                            float(
                                df_flip[
                                    "boundary_ratio"
                                ].median()
                            )
                            if n_flip > 0
                            else np.nan
                        ),
                }

                rows.append(row)

    return pd.DataFrame(rows)


def make_final_performance_table(
    agg: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:

    """
    Extract final-training-size performance for each dataset/model/tau
    and merge with overall flip incidence.
    """

    rows = []

    for dataset_key in DATASET_ORDER:

        for model_key in MODEL_ORDER:

            for tau in TAUS:

                df = agg[
                    (agg["dataset"] == dataset_key)
                    &
                    (agg["model"] == model_key)
                    &
                    (
                        np.isclose(
                            agg["tau"],
                            tau,
                        )
                    )
                ].copy()

                if df.empty:
                    continue

                final_row = (
                    df.loc[
                        df["train_size"].idxmax()
                    ]
                    .to_dict()
                )

                ov = overall[
                    (overall["dataset"] == dataset_key)
                    &
                    (overall["model"] == model_key)
                    &
                    (
                        np.isclose(
                            overall["tau"],
                            tau,
                        )
                    )
                ]

                if not ov.empty:

                    final_row[
                        "flip_incidence"
                    ] = float(
                        ov[
                            "flip_incidence"
                        ].iloc[0]
                    )

                    final_row[
                        "rho_exact_median_overall"
                    ] = float(
                        ov[
                            "rho_exact_median"
                        ].iloc[0]
                    )

                    final_row[
                        "rho_pinsker_median_overall"
                    ] = float(
                        ov[
                            "rho_pinsker_median"
                        ].iloc[0]
                    )

                else:

                    final_row[
                        "flip_incidence"
                    ] = np.nan

                    final_row[
                        "rho_exact_median_overall"
                    ] = np.nan

                    final_row[
                        "rho_pinsker_median_overall"
                    ] = np.nan

                rows.append(final_row)

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def plot_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:

    """
    Plot mean +/- std on an axis, with optional clipping of the ribbon.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y_mean = np.asarray(
        y_mean,
        dtype=float,
    )

    y_std = np.asarray(
        y_std,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y_mean)
    )

    x = x[mask]
    y_mean = y_mean[mask]
    y_std = y_std[mask]

    lower = y_mean - y_std
    upper = y_mean + y_std

    if y_min is not None:
        lower = np.maximum(
            lower,
            y_min,
        )

    if y_max is not None:
        upper = np.minimum(
            upper,
            y_max,
        )

    ax.plot(
        x,
        y_mean,
        marker="o",
        linewidth=1.8,
        label=label,
    )

    ax.fill_between(
        x,
        lower,
        upper,
        alpha=0.15,
    )


def plot_boundary_figure() -> None:

    """
    Main Figure 1:
    exact Bernoulli threshold boundary versus Pinsker certificate.
    """

    p_grid = np.linspace(
        1e-4,
        1.0 - 1e-4,
        2_000,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
    )

    for tau in TAUS:

        tau_arr = np.full_like(
            p_grid,
            tau,
        )

        c_exact = bern_kl(
            tau_arr,
            p_grid,
        )

        c_pinsker = (
            2.0
            * (
                np.abs(
                    p_grid - tau
                ) ** 2
            )
        )

        axes[0].plot(
            p_grid,
            c_exact,
            linewidth=2,
            label=rf"Exact, $\tau={tau}$",
        )

        axes[0].plot(
            p_grid,
            c_pinsker,
            linestyle="--",
            linewidth=2,
            label=rf"Pinsker, $\tau={tau}$",
        )

        mask = (
            np.abs(
                p_grid - tau
            )
            > 1e-3
        )

        ratio = (
            c_exact[mask]
            /
            np.maximum(
                c_pinsker[mask],
                EPS,
            )
        )

        axes[1].plot(
            p_grid[mask],
            ratio,
            linewidth=2,
            label=rf"$\tau={tau}$",
        )

    axes[0].set_xlabel(
        r"Reference probability $p_0$"
    )

    axes[0].set_ylabel(
        "Boundary value"
    )

    axes[0].legend()

    axes[1].set_xlabel(
        r"Reference probability $p_0$"
    )

    axes[1].set_ylabel(
        r"$C_{\mathrm{exact}}/C_{\mathrm{Pinsker}}$"
    )

    axes[1].set_yscale("log")

    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig1_exact_vs_pinsker_boundaries.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_flip_incidence_heatmaps(
    overall: pd.DataFrame,
) -> None:

    """
    Main Figure 2:
    flip incidence heatmaps for all thresholds.

    A single common linear colour scale is used across all threshold panels.
    """

    matrices = []

    # -------------------------------------------------------------
    # First construct all three matrices.
    # -------------------------------------------------------------

    for tau in TAUS:

        df_tau = overall[
            np.isclose(
                overall["tau"],
                tau,
            )
        ].copy()

        pivot = df_tau.pivot(
            index="dataset_label",
            columns="model_label",
            values="flip_incidence",
        )

        pivot = pivot.reindex(
            index=DATASET_LABELS,
            columns=[
                MODEL_LABELS[m]
                for m in MODEL_ORDER
            ],
        )

        matrices.append(
            100.0 * pivot
        )

    # -------------------------------------------------------------
    # Determine ONE colour scale from ALL panels.
    # -------------------------------------------------------------

    all_values = np.concatenate([
        matrix.to_numpy(
            dtype=float
        ).ravel()
        for matrix in matrices
    ])

    finite_values = all_values[
        np.isfinite(all_values)
    ]

    if finite_values.size == 0:

        global_vmax = 1.0

    else:

        global_vmax = float(
            np.max(finite_values)
        )

        if global_vmax <= 0:
            global_vmax = 1.0

    # -------------------------------------------------------------
    # Plot the three panels using the SAME vmin/vmax.
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(TAUS),
        figsize=(17, 5),
        sharey=True,
    )

    for ax, tau, data in zip(
        axes,
        TAUS,
        matrices,
    ):

        sns.heatmap(
            data,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="Blues",

            # IMPORTANT:
            # identical colour range in all panels
            vmin=0.0,
            vmax=global_vmax,

            cbar=ax is axes[-1],

            linewidths=0.5,
            linecolor="white",
        )

        ax.set_title(
            rf"Flip incidence (%), $\tau={tau}$"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig2_flip_incidence_heatmaps_all_thresholds.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rho_exact_heatmaps(
    overall: pd.DataFrame,
) -> None:

    """
    Main Figure 3:
    median exact-boundary ratio heatmaps for all thresholds.

    The original rho_exact values are displayed as annotations.

    A single common logarithmic colour normalization is used across all
    threshold panels so that equal colours correspond to equal rho_exact
    values across tau = 0.3, 0.5 and 0.7.
    """

    matrices = []

    # -------------------------------------------------------------
    # First construct all three matrices.
    # -------------------------------------------------------------

    for tau in TAUS:

        df_tau = overall[
            np.isclose(
                overall["tau"],
                tau,
            )
        ].copy()

        pivot = df_tau.pivot(
            index="dataset_label",
            columns="model_label",
            values="rho_exact_median",
        )

        pivot = pivot.reindex(
            index=DATASET_LABELS,
            columns=[
                MODEL_LABELS[m]
                for m in MODEL_ORDER
            ],
        )

        matrices.append(pivot)

    # -------------------------------------------------------------
    # Determine ONE logarithmic colour range from ALL panels.
    # -------------------------------------------------------------

    all_values = np.concatenate([
        matrix.to_numpy(
            dtype=float
        ).ravel()
        for matrix in matrices
    ])

    finite_positive_values = all_values[
        np.isfinite(all_values)
        &
        (all_values > 0)
    ]

    if finite_positive_values.size == 0:

        rho_vmin = 1.0
        rho_vmax = 2.0

    else:

        rho_vmin = float(
            np.min(
                finite_positive_values
            )
        )

        rho_vmax = float(
            np.max(
                finite_positive_values
            )
        )

        # LogNorm requires a strictly positive range.
        rho_vmin = max(
            rho_vmin,
            EPS,
        )

        if rho_vmax <= rho_vmin:
            rho_vmax = rho_vmin * 10.0

    # One common logarithmic normalization for every panel.
    common_norm = LogNorm(
        vmin=rho_vmin,
        vmax=rho_vmax,
    )

    # -------------------------------------------------------------
    # Plot all panels with the SAME LogNorm.
    # -------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        len(TAUS),
        figsize=(17, 5),
        sharey=True,
    )

    for ax, tau, pivot in zip(
        axes,
        TAUS,
        matrices,
    ):

        annot = pivot.map(
            lambda x:
                ""
                if pd.isna(x)
                else f"{x:.2g}"
        )

        sns.heatmap(
            pivot,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="viridis",

            # IMPORTANT:
            # the SAME logarithmic normalization is used in
            # every threshold panel.
            norm=common_norm,

            cbar=ax is axes[-1],

            linewidths=0.5,
            linecolor="white",
        )

        ax.set_title(
            rf"Median $\rho_{{\mathrm{{exact}}}}$, $\tau={tau}$"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig3_median_rho_exact_heatmaps_all_thresholds.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_stability_performance_tradeoff(
    final_perf: pd.DataFrame,
) -> None:

    """
    Main Figure 4:
    flip incidence versus AUROC at tau = 0.5.
    """

    df = final_perf[
        np.isclose(
            final_perf["tau"],
            MAIN_TAU,
        )
    ].copy()

    df["flip_percent"] = (
        100.0
        * df["flip_incidence"]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6.2)
    )

    sns.scatterplot(
        data=df,
        x="flip_percent",
        y="auc_mean",
        hue="model_label",
        style="dataset_label",
        size="brier_mean",
        sizes=(60, 220),
        alpha=0.85,
        ax=ax,
    )

    ax.set_xlabel(
        "Flip incidence (%)"
    )

    ax.set_ylabel(
        "Final AUROC"
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig4_stability_performance_tradeoff_tau0p5.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_boundary_ratio_boxplot(
    df_points: pd.DataFrame,
) -> None:

    """
    Main Figure 5:
    exact-to-Pinsker boundary ratio for flipped points at tau = 0.5.
    """

    df = df_points[
        (
            np.isclose(
                df_points["tau"],
                MAIN_TAU,
            )
        )
        &
        (
            df_points["flip"] == 1
        )
        &
        np.isfinite(
            df_points["boundary_ratio"]
        )
        &
        (
            df_points["boundary_ratio"] > 0
        )
    ].copy()

    if df.empty:
        print(
            "No flipped points available "
            "for boundary-ratio boxplot."
        )
        return

    df["dataset_label"] = pd.Categorical(
        df["dataset_label"],
        categories=DATASET_LABELS,
        ordered=True,
    )

    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=[
            MODEL_LABELS[m]
            for m in MODEL_ORDER
        ],
        ordered=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6.2)
    )

    sns.boxplot(
        data=df,
        x="dataset_label",
        y="boundary_ratio",
        hue="model_label",
        ax=ax,
        showfliers=False,
    )

    ax.set_yscale("log")

    ax.set_xlabel("")

    ax.set_ylabel(
        r"$C_{\mathrm{exact}}/C_{\mathrm{Pinsker}}$"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig5_exact_to_pinsker_boundary_ratio_tau0p5.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_progressive_metric(
    agg: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    ylabel: str,
    title: str,
    filename: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:

    """
    Multi-panel progressive plot by dataset for tau = 0.5.
    """

    df = agg[
        np.isclose(
            agg["tau"],
            MAIN_TAU,
        )
    ].copy()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        sharey=False,
    )

    axes_flat = axes.ravel()

    for idx, dataset_key in enumerate(
        DATASET_ORDER
    ):

        ax = axes_flat[idx]

        df_d = df[
            df["dataset"] == dataset_key
        ].copy()

        for model_key in MODEL_ORDER:

            df_m = df_d[
                df_d["model"] == model_key
            ].sort_values(
                "train_size"
            )

            if df_m.empty:
                continue

            plot_mean_std(
                ax=ax,
                x=df_m[
                    "train_size"
                ].to_numpy(),

                y_mean=df_m[
                    metric_mean
                ].to_numpy(),

                y_std=df_m[
                    metric_std
                ].fillna(
                    0.0
                ).to_numpy(),

                label=MODEL_LABELS[
                    model_key
                ],

                y_min=y_min,
                y_max=y_max,
            )

        ax.set_title(
            DATASETS[
                dataset_key
            ]["label"]
        )

        ax.set_xlabel(
            "Training size"
        )

        ax.set_ylabel(
            ylabel
        )

        if (
            y_min is not None
            or
            y_max is not None
        ):
            ax.set_ylim(
                y_min,
                y_max,
            )

    axes_flat[-1].axis(
        "off"
    )

    handles, labels = (
        axes_flat[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="lower right",
        bbox_to_anchor=(
            0.97,
            0.08,
        ),
    )

    fig.suptitle(
        title,
        y=1.02,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_auroc_brier(
    final_perf: pd.DataFrame,
) -> None:

    """
    Main Figure 8:
    final AUROC and Brier score by dataset and model at tau = 0.5.
    """

    df = final_perf[
        np.isclose(
            final_perf["tau"],
            MAIN_TAU,
        )
    ].copy()

    df["dataset_label"] = pd.Categorical(
        df["dataset_label"],
        categories=DATASET_LABELS,
        ordered=True,
    )

    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=[
            MODEL_LABELS[m]
            for m in MODEL_ORDER
        ],
        ordered=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5.8),
    )

    sns.pointplot(
        data=df,
        x="dataset_label",
        y="auc_mean",
        hue="model_label",
        dodge=0.35,
        markers="o",
        errorbar=None,
        ax=axes[0],
    )

    axes[0].set_xlabel("")

    axes[0].set_ylabel(
        "AUROC"
    )

    axes[0].set_ylim(
        0.0,
        1.02,
    )

    axes[0].tick_params(
        axis="x",
        rotation=25,
    )

    axes[0].legend_.remove()

    sns.pointplot(
        data=df,
        x="dataset_label",
        y="brier_mean",
        hue="model_label",
        dodge=0.35,
        markers="o",
        errorbar=None,
        ax=axes[1],
    )

    axes[1].set_xlabel("")

    axes[1].set_ylabel(
        "Brier score"
    )

    axes[1].tick_params(
        axis="x",
        rotation=25,
    )

    axes[1].legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "fig8_final_auroc_brier_tau0p5.png",
        bbox_inches="tight",
    )

    plt.close(fig)


def aggregate_curves(
    curves: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    """
    Interpolate ROC curves to a common FPR grid while preserving
    the mandatory ROC origin (0, 0).
    """

    interpolated = []

    for x, y in curves:

        x = np.asarray(
            x,
            dtype=float,
        )

        y = np.asarray(
            y,
            dtype=float,
        )

        order = np.argsort(x)

        x = x[order]
        y = y[order]

        # Explicitly preserve ROC origin.
        x[0] = 0.0
        y[0] = 0.0

        # Remove repeated FPR values.
        unique_x = [x[0]]
        unique_y = [y[0]]

        for xi, yi in zip(
            x[1:],
            y[1:],
        ):

            if xi == unique_x[-1]:

                unique_y[-1] = max(
                    unique_y[-1],
                    yi,
                )

            else:

                unique_x.append(xi)
                unique_y.append(yi)

        x_unique = np.asarray(
            unique_x
        )

        y_unique = np.asarray(
            unique_y
        )

        y_interp = np.interp(
            grid,
            x_unique,
            y_unique,
            left=0.0,
            right=1.0,
        )

        y_interp[
            grid == 0.0
        ] = 0.0

        interpolated.append(
            y_interp
        )

    values = np.vstack(
        interpolated
    )

    return (
        values.mean(axis=0),
        values.std(axis=0),
    )


# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

def save_detail_theorem_scatter(
    df_points: pd.DataFrame,
    dataset_key: str,
    model_key: str,
    tau: float,
) -> None:

    """
    Supplementary theorem scatter for a dataset/model/tau.

    The x-axis is the Pinsker threshold certificate

        C_Pinsker = 2 |p_prev - tau|^2

    and the y-axis is the KL step displacement

        D_step = KL(Bern(p_curr) || Bern(p_prev)).

    Only threshold-flipped validation instances are shown.
    """

    df = df_points[
        (df_points["dataset"] == dataset_key)
        &
        (df_points["model"] == model_key)
        &
        (
            np.isclose(
                df_points["tau"],
                tau,
            )
        )
        &
        (
            df_points["flip"] == 1
        )
    ].copy()

    if df.empty:
        return

    x = df[
        "c_pinsker"
    ].to_numpy()

    y = df[
        "d_step"
    ].to_numpy()

    maxv = float(
        np.nanmax(
            [
                np.nanmax(x),
                np.nanmax(y),
            ]
        )
    )

    if (
        not np.isfinite(maxv)
        or maxv <= 0
    ):
        maxv = 1.0

    fig, ax = plt.subplots(
        figsize=(6.5, 5.2)
    )

    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.35,
    )

    ax.plot(
        [0, maxv],
        [0, maxv],
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    ax.set_xlabel(
        r"$C_{\mathrm{Pinsker}}=2|p_{\mathrm{prev}}-\tau|^2$"
    )

    ax.set_ylabel(
        r"$D_{\mathrm{step}}$"
    )

    fig.tight_layout()

    outpath = (
        SUPP_FIGURES_DIR
        /
        (
            f"supp_theorem_"
            f"{dataset_key}_"
            f"{model_key}_"
            f"{tau_tag(tau)}.png"
        )
    )

    fig.savefig(
        outpath,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_detail_roc(
    curves: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    dataset_key: str,
    model_key: str,
) -> None:

    """
    Supplementary final ROC curve for a dataset/model pair.
    """

    if not curves:
        return

    fpr_grid = np.linspace(
        0.0,
        1.0,
        201,
    )

    tpr_mean, tpr_std = aggregate_curves(
        curves,
        fpr_grid,
    )

    lower = np.clip(
        tpr_mean - tpr_std,
        0.0,
        1.0,
    )

    upper = np.clip(
        tpr_mean + tpr_std,
        0.0,
        1.0,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.2)
    )

    ax.plot(
        fpr_grid,
        tpr_mean,
        linewidth=2,
        label="Mean ROC",
    )

    ax.fill_between(
        fpr_grid,
        lower,
        upper,
        alpha=0.25,
        label="Std band",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        color="gray",
    )

    ax.set_xlabel(
        "False positive rate"
    )

    ax.set_ylabel(
        "True positive rate"
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    outpath = (
        SUPP_FIGURES_DIR
        /
        f"supp_roc_{dataset_key}_{model_key}.png"
    )

    fig.savefig(
        outpath,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_all_figures(
    agg: pd.DataFrame,
    overall: pd.DataFrame,
    final_perf: pd.DataFrame,
    df_points: pd.DataFrame,
    final_roc_curves: dict[
        tuple[str, str],
        list[
            tuple[
                np.ndarray,
                np.ndarray,
            ]
        ],
    ],
) -> None:

    """
    Generate all main and supplementary figures.
    """

    plot_boundary_figure()

    plot_flip_incidence_heatmaps(
        overall
    )

    plot_rho_exact_heatmaps(
        overall
    )

    plot_stability_performance_tradeoff(
        final_perf
    )

    plot_boundary_ratio_boxplot(
        df_points
    )

    plot_progressive_metric(
        agg=agg,
        metric_mean="flip_rate_mean",
        metric_std="flip_rate_std",
        ylabel="Flip rate",
        title=r" ",
        filename="fig6_progressive_flip_rate_tau0p5.png",
        y_min=0.0,
        y_max=None,
    )

    plot_progressive_metric(
        agg=agg,
        metric_mean="d_step_mean",
        metric_std="d_step_std",
        ylabel=r"Mean $D_{\mathrm{step}}$",
        title=r" ",
        filename="fig7_progressive_kl_displacement_tau0p5.png",
        y_min=0.0,
        y_max=None,
    )

    plot_final_auroc_brier(
        final_perf
    )

    for dataset_key in DATASET_ORDER:

        for model_key in MODEL_ORDER:

            save_detail_roc(
                curves=final_roc_curves.get(
                    (
                        dataset_key,
                        model_key,
                    ),
                    [],
                ),
                dataset_key=dataset_key,
                model_key=model_key,
            )

            for tau in TAUS:

                save_detail_theorem_scatter(
                    df_points=df_points,
                    dataset_key=dataset_key,
                    model_key=model_key,
                    tau=tau,
                )


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main() -> None:

    dataset_summary_rows = []
    summary_rows = []
    theorem_points_rows = []

    final_roc_curves: dict[
        tuple[str, str],
        list[
            tuple[
                np.ndarray,
                np.ndarray,
            ]
        ],
    ] = defaultdict(list)

    print("=" * 90)
    print(
        "Multi-domain Bayes-decision stability experiment"
    )
    print("=" * 90)

    print(
        "Outputs will be written to: "
        f"{BASE_DIR.resolve()}"
    )

    print("=" * 90)

    for dataset_index, dataset_key in enumerate(
        DATASET_ORDER
    ):

        config = DATASETS[
            dataset_key
        ]

        print(
            f"Loading dataset: "
            f"{config['label']} "
            f"({config['file']})"
        )

        X_full_raw, y_full, df_original = (
            load_dataset(
                dataset_key,
                config,
            )
        )

        rng_global = np.random.default_rng(
            RANDOM_SEED_GLOBAL
            + dataset_index
        )

        perm = rng_global.permutation(
            len(y_full)
        )

        X_full_raw = (
            X_full_raw
            .iloc[perm]
            .reset_index(drop=True)
        )

        y_full = y_full[
            perm
        ]

        n_total = len(
            y_full
        )

        n_features = (
            X_full_raw.shape[1]
        )

        n_positive = int(
            y_full.sum()
        )

        n_negative = int(
            n_total
            - n_positive
        )

        dataset_summary_rows.append({

            "dataset":
                dataset_key,

            "dataset_label":
                config["label"],

            "sector":
                config["sector"],

            "file":
                config["file"],

            "n_instances":
                n_total,

            "n_features_after_encoding":
                n_features,

            "n_positive":
                n_positive,

            "n_negative":
                n_negative,

            "positive_rate":
                n_positive / n_total,

            "target_description":
                config[
                    "target_description"
                ],
        })

        print(
            f"  n={n_total:,}, "
            f"features={n_features:,}, "
            f"positive={n_positive:,}, "
            f"negative={n_negative:,}"
        )

        for rep in range(
            N_REPETITIONS
        ):

            print(
                f"  Dataset={dataset_key}, "
                f"repetition "
                f"{rep + 1}/{N_REPETITIONS}"
            )

            idx_all = np.arange(
                n_total
            )

            idx_pool, idx_val = (
                train_test_split(
                    idx_all,
                    test_size=VAL_SIZE,
                    stratify=y_full,
                    random_state=rep,
                )
            )

            X_pool_raw = (
                X_full_raw
                .iloc[idx_pool]
                .reset_index(drop=True)
            )

            y_pool = y_full[
                idx_pool
            ]

            X_val_raw = (
                X_full_raw
                .iloc[idx_val]
                .reset_index(drop=True)
            )

            y_val = y_full[
                idx_val
            ]

            scaler = MinMaxScaler()

            X_pool = scaler.fit_transform(
                X_pool_raw.values
            )

            X_val = scaler.transform(
                X_val_raw.values
            )

            rng_rep = np.random.default_rng(
                rep
            )

            order = stratified_progressive_order(
                y_pool,
                rng_rep,
            )

            X_pool = X_pool[
                order
            ]

            y_pool = y_pool[
                order
            ]

            train_sizes = get_training_sizes(
                pool_size=len(y_pool),
                requested_sizes=config[
                    "train_sizes"
                ],
            )

            models = build_models(
                random_state=rep
            )

            for model_key, model in models.items():

                prev_proba = None
                prev_n = None
                last_y_proba = None

                for n in train_sizes:

                    X_train = X_pool[
                        :n
                    ]

                    y_train = y_pool[
                        :n
                    ]

                    if len(
                        np.unique(y_train)
                    ) < 2:

                        for tau in TAUS:

                            summary_rows.append({

                                "dataset":
                                    dataset_key,

                                "dataset_label":
                                    config["label"],

                                "model":
                                    model_key,

                                "model_label":
                                    MODEL_LABELS[
                                        model_key
                                    ],

                                "tau":
                                    tau,

                                "rep":
                                    rep,

                                "train_size":
                                    int(n),

                                "skipped_single_class_prefix":
                                    1,
                            })

                        continue

                    model.fit(
                        X_train,
                        y_train,
                    )

                    y_proba = (
                        model
                        .predict_proba(
                            X_val
                        )[:, 1]
                        .astype(float)
                    )

                    y_proba = np.clip(
                        y_proba,
                        EPS,
                        1.0 - EPS,
                    )

                    auc_value = float(
                        roc_auc_score(
                            y_val,
                            y_proba,
                        )
                    )

                    auprc_value = float(
                        average_precision_score(
                            y_val,
                            y_proba,
                        )
                    )

                    brier_value = float(
                        brier_score_loss(
                            y_val,
                            y_proba,
                        )
                    )

                    if prev_proba is not None:

                        p_prev = np.clip(
                            prev_proba,
                            EPS,
                            1.0 - EPS,
                        )

                        p_curr = np.clip(
                            y_proba,
                            EPS,
                            1.0 - EPS,
                        )

                        d_step = bern_kl(
                            p_curr,
                            p_prev,
                            eps=EPS,
                        )

                    else:

                        p_prev = None
                        p_curr = None
                        d_step = None

                    for tau in TAUS:

                        y_pred = (
                            y_proba >= tau
                        ).astype(int)

                        metrics = compute_metrics(
                            y_true=y_val,
                            y_pred=y_pred,
                            y_proba=y_proba,
                        )

                        # Threshold-free metrics are identical for all tau.
                        metrics["auc"] = auc_value
                        metrics["auprc"] = auprc_value
                        metrics["brier"] = brier_value

                        if prev_proba is None:

                            transition_values = {

                                "train_size_prev":
                                    np.nan,

                                "flips_step":
                                    np.nan,

                                "flip_rate_step":
                                    np.nan,

                                "violations_pinsker":
                                    np.nan,

                                "violations_exact":
                                    np.nan,

                                "d_step_mean":
                                    np.nan,

                                "d_step_median":
                                    np.nan,

                                "d_step_flips_mean":
                                    np.nan,

                                "c_pinsker_flips_mean":
                                    np.nan,

                                "c_exact_flips_mean":
                                    np.nan,

                                "rho_pinsker_min":
                                    np.nan,

                                "rho_pinsker_median":
                                    np.nan,

                                "rho_pinsker_p10":
                                    np.nan,

                                "rho_pinsker_p90":
                                    np.nan,

                                "rho_exact_min":
                                    np.nan,

                                "rho_exact_median":
                                    np.nan,

                                "rho_exact_p10":
                                    np.nan,

                                "rho_exact_p90":
                                    np.nan,
                            }

                        else:

                            c_pinsker = (
                                2.0
                                * (
                                    np.abs(
                                        p_prev - tau
                                    ) ** 2
                                )
                            )

                            c_exact = bern_kl(
                                np.full_like(
                                    p_prev,
                                    fill_value=tau,
                                ),
                                p_prev,
                                eps=EPS,
                            )

                            u_prev = (
                                p_prev >= tau
                            ).astype(int)

                            u_curr = (
                                p_curr >= tau
                            ).astype(int)

                            flip = (
                                u_prev
                                !=
                                u_curr
                            )

                            violations_pinsker = int(
                                np.sum(
                                    flip
                                    &
                                    (
                                        d_step
                                        + VIOL_TOL
                                        < c_pinsker
                                    )
                                )
                            )

                            violations_exact = int(
                                np.sum(
                                    flip
                                    &
                                    (
                                        d_step
                                        + VIOL_TOL
                                        < c_exact
                                    )
                                )
                            )

                            flips_step = int(
                                np.sum(flip)
                            )

                            if flips_step > 0:

                                rho_pinsker = (
                                    d_step[flip]
                                    /
                                    np.maximum(
                                        c_pinsker[flip],
                                        EPS,
                                    )
                                )

                                rho_exact = (
                                    d_step[flip]
                                    /
                                    np.maximum(
                                        c_exact[flip],
                                        EPS,
                                    )
                                )

                                transition_values = {

                                    "train_size_prev":
                                        int(prev_n),

                                    "flips_step":
                                        flips_step,

                                    "flip_rate_step":
                                        float(
                                            flips_step
                                            /
                                            len(y_val)
                                        ),

                                    "violations_pinsker":
                                        violations_pinsker,

                                    "violations_exact":
                                        violations_exact,

                                    "d_step_mean":
                                        float(
                                            np.mean(
                                                d_step
                                            )
                                        ),

                                    "d_step_median":
                                        float(
                                            np.median(
                                                d_step
                                            )
                                        ),

                                    "d_step_flips_mean":
                                        float(
                                            np.mean(
                                                d_step[flip]
                                            )
                                        ),

                                    "c_pinsker_flips_mean":
                                        float(
                                            np.mean(
                                                c_pinsker[
                                                    flip
                                                ]
                                            )
                                        ),

                                    "c_exact_flips_mean":
                                        float(
                                            np.mean(
                                                c_exact[
                                                    flip
                                                ]
                                            )
                                        ),

                                    "rho_pinsker_min":
                                        float(
                                            np.min(
                                                rho_pinsker
                                            )
                                        ),

                                    "rho_pinsker_median":
                                        float(
                                            np.median(
                                                rho_pinsker
                                            )
                                        ),

                                    "rho_pinsker_p10":
                                        float(
                                            np.percentile(
                                                rho_pinsker,
                                                10,
                                            )
                                        ),

                                    "rho_pinsker_p90":
                                        float(
                                            np.percentile(
                                                rho_pinsker,
                                                90,
                                            )
                                        ),

                                    "rho_exact_min":
                                        float(
                                            np.min(
                                                rho_exact
                                            )
                                        ),

                                    "rho_exact_median":
                                        float(
                                            np.median(
                                                rho_exact
                                            )
                                        ),

                                    "rho_exact_p10":
                                        float(
                                            np.percentile(
                                                rho_exact,
                                                10,
                                            )
                                        ),

                                    "rho_exact_p90":
                                        float(
                                            np.percentile(
                                                rho_exact,
                                                90,
                                            )
                                        ),
                                }

                            else:

                                transition_values = {

                                    "train_size_prev":
                                        int(prev_n),

                                    "flips_step":
                                        0,

                                    "flip_rate_step":
                                        0.0,

                                    "violations_pinsker":
                                        violations_pinsker,

                                    "violations_exact":
                                        violations_exact,

                                    "d_step_mean":
                                        float(
                                            np.mean(
                                                d_step
                                            )
                                        ),

                                    "d_step_median":
                                        float(
                                            np.median(
                                                d_step
                                            )
                                        ),

                                    "d_step_flips_mean":
                                        np.nan,

                                    "c_pinsker_flips_mean":
                                        np.nan,

                                    "c_exact_flips_mean":
                                        np.nan,

                                    "rho_pinsker_min":
                                        np.nan,

                                    "rho_pinsker_median":
                                        np.nan,

                                    "rho_pinsker_p10":
                                        np.nan,

                                    "rho_pinsker_p90":
                                        np.nan,

                                    "rho_exact_min":
                                        np.nan,

                                    "rho_exact_median":
                                        np.nan,

                                    "rho_exact_p10":
                                        np.nan,

                                    "rho_exact_p90":
                                        np.nan,
                                }

                            # -------------------------------------------------
                            # Store validation-instance theorem quantities.
                            # -------------------------------------------------

                            for i in range(
                                len(y_val)
                            ):

                                theorem_points_rows.append({

                                    "dataset":
                                        dataset_key,

                                    "dataset_label":
                                        config["label"],

                                    "sector":
                                        config["sector"],

                                    "model":
                                        model_key,

                                    "model_label":
                                        MODEL_LABELS[
                                            model_key
                                        ],

                                    "tau":
                                        tau,

                                    "rep":
                                        rep,

                                    "train_size_prev":
                                        int(prev_n),

                                    "train_size_curr":
                                        int(n),

                                    "val_row":
                                        int(i),

                                    "y_true":
                                        int(y_val[i]),

                                    "p_prev":
                                        float(
                                            p_prev[i]
                                        ),

                                    "p_curr":
                                        float(
                                            p_curr[i]
                                        ),

                                    "margin_prev":
                                        float(
                                            abs(
                                                p_prev[i]
                                                - tau
                                            )
                                        ),

                                    "flip":
                                        int(
                                            flip[i]
                                        ),

                                    "d_step":
                                        float(
                                            d_step[i]
                                        ),

                                    "c_pinsker":
                                        float(
                                            c_pinsker[i]
                                        ),

                                    "c_exact":
                                        float(
                                            c_exact[i]
                                        ),

                                    "rho_pinsker":
                                        float(
                                            d_step[i]
                                            /
                                            max(
                                                c_pinsker[i],
                                                EPS,
                                            )
                                        ),

                                    "rho_exact":
                                        float(
                                            d_step[i]
                                            /
                                            max(
                                                c_exact[i],
                                                EPS,
                                            )
                                        ),

                                    "boundary_ratio":
                                        float(
                                            c_exact[i]
                                            /
                                            max(
                                                c_pinsker[i],
                                                EPS,
                                            )
                                        ),

                                    "violates_pinsker":
                                        int(
                                            (
                                                d_step[i]
                                                + VIOL_TOL
                                                < c_pinsker[i]
                                            )
                                            and flip[i]
                                        ),

                                    "violates_exact":
                                        int(
                                            (
                                                d_step[i]
                                                + VIOL_TOL
                                                < c_exact[i]
                                            )
                                            and flip[i]
                                        ),
                                })

                        summary_rows.append({

                            "dataset":
                                dataset_key,

                            "dataset_label":
                                config["label"],

                            "sector":
                                config["sector"],

                            "model":
                                model_key,

                            "model_label":
                                MODEL_LABELS[
                                    model_key
                                ],

                            "tau":
                                tau,

                            "rep":
                                rep,

                            "train_size":
                                int(n),

                            "skipped_single_class_prefix":
                                0,

                            **metrics,
                            **transition_values,
                        })

                    prev_proba = (
                        y_proba.copy()
                    )

                    prev_n = int(n)

                    last_y_proba = (
                        y_proba.copy()
                    )

                if (
                    last_y_proba is not None
                    and
                    len(
                        np.unique(y_val)
                    ) == 2
                ):

                    fpr, tpr, _ = roc_curve(
                        y_val,
                        last_y_proba,
                    )

                    final_roc_curves[
                        (
                            dataset_key,
                            model_key,
                        )
                    ].append(
                        (
                            fpr,
                            tpr,
                        )
                    )

    # =========================================================================
    # SAVE TABLES
    # =========================================================================

    df_dataset_summary = pd.DataFrame(
        dataset_summary_rows
    )

    df_summary = pd.DataFrame(
        summary_rows
    )

    df_points = pd.DataFrame(
        theorem_points_rows
    )

    dataset_summary_path = (
        TABLES_DIR
        / "dataset_summary.csv"
    )

    summary_path = (
        TABLES_DIR
        / "progressive_mc_results_summary.csv"
    )

    theorem_path = (
        TABLES_DIR
        / "theorem_points_long.csv"
    )

    df_dataset_summary.to_csv(
        dataset_summary_path,
        index=False,
    )

    df_summary.to_csv(
        summary_path,
        index=False,
    )

    df_points.to_csv(
        theorem_path,
        index=False,
    )

    agg = aggregate_progressive_summary(
        df_summary
    )

    overall = make_overall_stability_summary(
        df_points
    )

    final_perf = make_final_performance_table(
        agg,
        overall,
    )

    agg_path = (
        TABLES_DIR
        / "progressive_summary_by_dataset_model_tau.csv"
    )

    overall_path = (
        TABLES_DIR
        / "overall_stability_by_dataset_model_tau.csv"
    )

    final_perf_path = (
        TABLES_DIR
        / "final_performance_by_dataset_model_tau.csv"
    )

    agg.to_csv(
        agg_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    final_perf.to_csv(
        final_perf_path,
        index=False,
    )

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================

    make_all_figures(
        agg=agg,
        overall=overall,
        final_perf=final_perf,
        df_points=df_points,
        final_roc_curves=final_roc_curves,
    )

    # =========================================================================
    # COMPLETION MESSAGE
    # =========================================================================

    print("=" * 90)
    print("All runs complete.")
    print("=" * 90)

    print(
        f"Output directory: "
        f"{BASE_DIR.resolve()}"
    )

    print(
        f"Dataset summary: "
        f"{dataset_summary_path}"
    )

    print(
        f"Progressive summary: "
        f"{summary_path}"
    )

    print(
        f"Validation-instance stability data: "
        f"{theorem_path}"
    )

    print(
        f"Aggregated progressive summary: "
        f"{agg_path}"
    )

    print(
        f"Overall stability summary: "
        f"{overall_path}"
    )

    print(
        f"Final performance summary: "
        f"{final_perf_path}"
    )

    print(
        f"Figures directory: "
        f"{FIGURES_DIR}"
    )

    print("=" * 90)

    print("Main manuscript figures:")

    for path in sorted(
        FIGURES_DIR.glob(
            "fig*.png"
        )
    ):
        print(
            f"  {path.name}"
        )

    print("Supplementary figures:")

    for path in sorted(
        SUPP_FIGURES_DIR.glob(
            "*.png"
        )
    ):
        print(
            f"  {path.name}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()
