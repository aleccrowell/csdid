import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from csdid.att_gt import ATTgt

DATA = Path(__file__).parent.parent / "data" / "mpdta.csv"

# Deterministic point estimate (no bootstrap randomness affects ATT)
EXPECTED_ATT = -0.03995127515517632


@pytest.fixture(scope="module")
def model():
    np.random.seed(0)
    data = pd.read_csv(DATA)
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=data,
        biters=1000,
    ).fit(est_method="dr")


def test_simple_aggte_att(model):
    res = model.aggte("simple")
    att = float(res.atte["overall_att"])
    assert abs(att - EXPECTED_ATT) < 1e-8, f"ATT {att!r} != expected {EXPECTED_ATT!r}"


def test_simple_aggte_se_positive(model):
    res = model.aggte("simple")
    se = float(np.ravel(res.atte["overall_se"])[0])
    assert se > 0
    assert se < 0.1


def test_dynamic_aggte_structure(model):
    res = model.aggte("dynamic")
    for key in ("overall_att", "overall_se", "egt", "att_egt", "se_egt"):
        assert key in res.atte, f"missing key {key!r} in dynamic aggte output"
