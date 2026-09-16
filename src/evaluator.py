"""
Reproducible ML evaluation pipeline.

Input CSV columns:
    prediction : predicted positive-class probability [0, 1]
    label      : binary label 0/1
    subgroup   : subgroup identifier
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("prediction", "label", "subgroup")
N_BOOTSTRAPS = 1000
RANDOM_SEED = 42


def load_and_prepare(csv_files):
    """Load CSVs in filename order and remove rows with missing values."""
    paths = sorted(
        [Path(p) for p in csv_files],
        key=lambda p: p.name
    )

    frames = []

    for path in paths:
        df = pd.read_csv(path)

        missing = [
            column for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{path}: missing required columns: {missing}"
            )

        # Preserve original row order within each CSV.
        frames.append(
            df.loc[:, REQUIRED_COLUMNS].copy()
        )

    if not frames:
        raise ValueError("No CSV files supplied.")

    data = pd.concat(
        frames,
        ignore_index=True
    )

    # Drop missing rows before metrics, bootstrap, or thresholds.
    data = data.dropna(
        subset=list(REQUIRED_COLUMNS)
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid rows remain after removing missing values."
        )

    data["prediction"] = pd.to_numeric(
        data["prediction"],
        errors="raise"
    )

    data["label"] = pd.to_numeric(
        data["label"],
        errors="raise"
    ).astype(int)

    if (
        (data["prediction"] < 0)
        | (data["prediction"] > 1)
    ).any():
        raise ValueError(
            "prediction values must be between 0 and 1."
        )

    if (~data["label"].isin([0, 1])).any():
        raise ValueError(
            "label values must be binary 0/1."
        )

    return data


def average_ranks(values):
    """Return 1-based average ranks for tied values."""
    values = np.asarray(values)

    order = np.argsort(
        values,
        kind="mergesort"
    )

    sorted_values = values[order]
    ranks_sorted = np.empty(
        len(values),
        dtype=float
    )

    i = 0

    while i < len(values):
        j = i + 1

        while (
            j < len(values)
            and sorted_values[j] == sorted_values[i]
        ):
            j += 1

        # Average of ranks i+1 through j.
        ranks_sorted[i:j] = (
            (i + 1) + j
        ) / 2.0

        i = j

    ranks = np.empty(
        len(values),
        dtype=float
    )

    ranks[order] = ranks_sorted

    return ranks


def auroc(y_true, prediction):
    """
    AUROC using the Mann-Whitney rank formulation.
    Tied prediction scores receive average ranks.
    """
    y_true = np.asarray(y_true)
    prediction = np.asarray(prediction)

    positive = y_true == 1
    negative = y_true == 0

    n_positive = int(positive.sum())
    n_negative = int(negative.sum())

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = average_ranks(prediction)

    positive_rank_sum = float(
        ranks[positive].sum()
    )

    return (
        positive_rank_sum
        - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def macro_group_auroc(
    y_true,
    prediction,
    subgroup
):
    """Average AUROC across mixed-class subgroups."""
    values = []

    for group in pd.unique(subgroup):
        mask = subgroup == group

        value = auroc(
            y_true[mask],
            prediction[mask]
        )

        # Single-class groups are excluded.
        if not math.isnan(value):
            values.append(value)

    if not values:
        return float("nan")

    return float(np.mean(values))


def brier_score(y_true, prediction):
    """Mean squared difference between prediction and binary label."""
    return float(
        np.mean(
            (prediction - y_true) ** 2
        )
    )


def ece_10_bins(y_true, prediction):
    """
    Calculate 10-bin Expected Calibration Error.

    Bins:
    [0.0,0.1), [0.1,0.2), ... [0.8,0.9), [0.9,1.0]

    The final bin includes 1.0.

    ECE =
        sum((n_bin / N) * abs(accuracy - confidence))
    """
    edges = np.arange(
        0.0,
        1.0000001,
        0.1
    )

    n = len(prediction)
    ece = 0.0

    for i in range(10):
        left = edges[i]
        right = edges[i + 1]

        if i == 9:
            mask = (
                (prediction >= left)
                & (prediction <= 1.0)
            )
        else:
            mask = (
                (prediction >= left)
                & (prediction < right)
            )

        count = int(mask.sum())

        # Empty bins are omitted.
        if count == 0:
            continue

        confidence = float(
            np.mean(prediction[mask])
        )

        accuracy = float(
            np.mean(y_true[mask])
        )

        ece += (
            count / n
        ) * abs(
            accuracy - confidence
        )

    return float(ece)


def compute_metrics(data):
    """Calculate all point-estimate metrics."""
    y = data["label"].to_numpy(dtype=int)
    p = data["prediction"].to_numpy(dtype=float)
    g = data["subgroup"].to_numpy()

    return {
        "AUROC": auroc(y, p),
        "macro_group_AUROC": macro_group_auroc(
            y, p, g
        ),
        "Brier": brier_score(y, p),
        "ECE": ece_10_bins(y, p),
    }


def bootstrap_cis(
    data,
    n_bootstrap=N_BOOTSTRAPS
):
    """
    Calculate 95% percentile bootstrap CIs.

    One RNG is created before the bootstrap loop.
    One resample is generated per iteration and reused
    for all metrics.
    """
    y = data["label"].to_numpy(dtype=int)
    p = data["prediction"].to_numpy(dtype=float)
    g = data["subgroup"].to_numpy()

    n = len(data)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    bootstrap_values = {
        "AUROC": [],
        "macro_group_AUROC": [],
        "Brier": [],
        "ECE": [],
    }

    for _ in range(n_bootstrap):

        # Exactly one index array per replicate.
        indices = rng.choice(
            n,
            size=n,
            replace=True
        )

        y_sample = y[indices]
        p_sample = p[indices]
        g_sample = g[indices]

        values = {
            "AUROC": auroc(
                y_sample,
                p_sample
            ),

            "macro_group_AUROC":
                macro_group_auroc(
                    y_sample,
                    p_sample,
                    g_sample
                ),

            "Brier": brier_score(
                y_sample,
                p_sample
            ),

            "ECE": ece_10_bins(
                y_sample,
                p_sample
            ),
        }

        for name, value in values.items():
            if not math.isnan(value):
                bootstrap_values[name].append(
                    value
                )

    result = {}

    for name, values in bootstrap_values.items():

        if not values:
            result[name] = [
                float("nan"),
                float("nan")
            ]
            continue

        quantiles = np.percentile(
            np.asarray(values),
            [2.5, 97.5],
            method="linear"
        )

        result[name] = [
            float(quantiles[0]),
            float(quantiles[1])
        ]

    return result


def select_threshold(data):
    """
    Select threshold for equalized-odds gap.

    Thresholds:
        0.00, 0.01, ..., 1.00

    Positive classification:
        prediction >= threshold

    Eligibility:
        overall recall >= 0.80

    Valid subgroups:
        mixed-class subgroups only.

    Gap:
        max(
            max(TPR) - min(TPR),
            max(FPR) - min(FPR)
        )

    If gaps tie, choose the smallest threshold.
    """
    y = data["label"].to_numpy(dtype=int)
    p = data["prediction"].to_numpy(dtype=float)
    g = data["subgroup"].to_numpy()

    positives = y == 1
    negatives = y == 0

    total_positive = int(
        positives.sum()
    )

    if total_positive == 0:
        raise ValueError(
            "Overall recall cannot be calculated "
            "without positive labels."
        )

    # Only mixed-class subgroups are valid.
    valid_groups = []

    for group in pd.unique(g):
        mask = g == group
        group_labels = y[mask]

        if (
            np.any(group_labels == 1)
            and np.any(group_labels == 0)
        ):
            valid_groups.append(group)

    if not valid_groups:
        raise ValueError(
            "No mixed-class subgroups available."
        )

    candidates = []

    # Exactly 0.00 through 1.00 by 0.01.
    for i in range(101):

        threshold = i / 100.0

        predicted_positive = (
            p >= threshold
        )

        true_positive = int(
            np.sum(
                predicted_positive
                & positives
            )
        )

        overall_recall = (
            true_positive
            / total_positive
        )

        # Only thresholds with recall >= 0.80.
        if overall_recall < 0.80:
            continue

        tprs = []
        fprs = []

        for group in valid_groups:

            mask = g == group

            group_y = y[mask]
            group_predicted = (
                predicted_positive[mask]
            )

            group_positive = group_y == 1
            group_negative = group_y == 0

            tpr = (
                np.sum(
                    group_predicted
                    & group_positive
                )
                / np.sum(group_positive)
            )

            fpr = (
                np.sum(
                    group_predicted
                    & group_negative
                )
                / np.sum(group_negative)
            )

            tprs.append(float(tpr))
            fprs.append(float(fpr))

        tpr_gap = (
            max(tprs)
            - min(tprs)
        )

        fpr_gap = (
            max(fprs)
            - min(fprs)
        )

        equalized_odds_gap = max(
            tpr_gap,
            fpr_gap
        )

        candidates.append(
            (
                equalized_odds_gap,
                threshold
            )
        )

    if not candidates:
        raise ValueError(
            "No threshold satisfies recall >= 0.80."
        )

    # Smallest gap first.
    # If tied, smallest threshold first.
    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    return float(
        candidates[0][1]
    )


def evaluate(csv_files):
    """Run the complete evaluation pipeline."""
    data = load_and_prepare(
        csv_files
    )

    metrics = compute_metrics(data)
    confidence_intervals = bootstrap_cis(
        data
    )

    selected_threshold = select_threshold(
        data
    )

    return {
        "AUROC": metrics["AUROC"],
        "macro_group_AUROC":
            metrics["macro_group_AUROC"],
        "Brier": metrics["Brier"],
        "ECE": metrics["ECE"],

        "AUROC_95_CI":
            confidence_intervals["AUROC"],

        "macro_group_AUROC_95_CI":
            confidence_intervals[
                "macro_group_AUROC"
            ],

        "Brier_95_CI":
            confidence_intervals["Brier"],

        "ECE_95_CI":
            confidence_intervals["ECE"],

        "selected_threshold":
            selected_threshold,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible ML evaluation pipeline"
        )
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Input CSV files"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="evaluation_results.json"
    )

    args = parser.parse_args()

    result = evaluate(
        args.csv_files
    )

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            allow_nan=False
        )

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
