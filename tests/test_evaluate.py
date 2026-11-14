import numpy as np
import pytest

from xsell.evaluate import capture_at, lift_table, ranking_metrics

# 10 clientes, 2 interesados.
Y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
PERFECT_SCORE = np.array([0.9, 0.8, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1])


def test_perfect_ranking_has_metrics_equal_to_one():
    metrics = ranking_metrics(Y, PERFECT_SCORE)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_capture_at():
    assert capture_at(Y, PERFECT_SCORE, 0.1) == pytest.approx(0.5)  # 1 de 2 interesados
    assert capture_at(Y, PERFECT_SCORE, 0.2) == pytest.approx(1.0)
    assert capture_at(Y, PERFECT_SCORE, 1.0) == pytest.approx(1.0)


def test_lift_table():
    table = lift_table(Y, PERFECT_SCORE, n_bins=5)
    assert table["clientes"].sum() == len(Y)
    assert table["interesados"].sum() == Y.sum()
    assert table["captura_acumulada"].iloc[-1] == pytest.approx(1.0)
    # El primer tramo (2 clientes) tiene a los 2 interesados: tasa 100 %, lift 1 / 0,2 = 5.
    assert table.loc[1, "tasa_interes"] == pytest.approx(1.0)
    assert table.loc[1, "lift"] == pytest.approx(5.0)
