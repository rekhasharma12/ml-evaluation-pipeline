import math

import numpy as np
import pandas as pd

from src.evaluator import (
    auroc,
    brier_score,
    ece_10_bins,
    macro_group_auroc,
    select_threshold,
)


def test_auroc_with_ties():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.8, 0.8])

    result = auroc(y, p)

    assert abs(result - 0.75) < 1e-12


def test_single_class_auroc_is_nan():
    y = np.array([1, 1, 1])
    p = np.array([0.1, 0.5, 0.9])

    result = auroc(y, p)

    assert math.isnan(result)


def test_macro_group_auroc_excludes_single_class_group():
    y = np.array([
        0, 1,
        0, 1,
        1
    ])

    p = np.array([
        0.1, 0.9,
        0.2, 0.8,
        0.7
    ])

    subgroup = np.array([
        "A", "A",
        "B", "B",
        "C"
    ])

    result = macro_group_auroc(
        y,
        p,
        subgroup
    )

    assert abs(result - 1.0) < 1e-12


def test_brier_score():
    y = np.array([0, 1])
    p = np.array([0.25, 0.75])

    result = brier_score(
        y,
        p
    )

    assert abs(
        result - 0.0625
    ) < 1e-12


def test_ece_uses_10_bins():
    y = np.array([
        0,
        1,
        1,
        0
    ])

    p = np.array([
        0.05,
        0.15,
        0.85,
        0.95
    ])

    result = ece_10_bins(
        y,
        p
    )

    assert abs(
        result - 0.275
    ) < 1e-12


def test_threshold_uses_required_grid():
    data = pd.DataFrame({
        "prediction": [
            0.90,
            0.80,
            0.60,
            0.40,
            0.30,
            0.20
        ],

        "label": [
            1,
            1,
            0,
            0,
            1,
            0
        ],

        "subgroup": [
            "A",
            "A",
            "A",
            "A",
            "B",
            "B"
        ]
    })

    threshold = select_threshold(
        data
    )

    assert 0.00 <= threshold <= 1.00

    # Threshold must be a multiple of 0.01.
    assert abs(
        threshold * 100
        - round(threshold * 100)
    ) < 1e-12
