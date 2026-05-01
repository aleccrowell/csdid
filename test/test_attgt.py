import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from csdid.att_gt import ATTgt

DATA = Path(__file__).parent.parent / "data" / "mpdta.csv"

# Deterministic point estimate (no bootstrap randomness affects ATT)
EXPECTED_ATT = -0.03995127515517632

AGGTE_KEYS = ("overall_att", "overall_se", "egt", "att_egt", "se_egt", "crit_val_egt")


def _load_data():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def model():
    np.random.seed(0)
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=_load_data(),
        biters=1000,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_ipw():
    np.random.seed(1)
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=_load_data(),
        biters=100,
    ).fit(est_method="ipw")


@pytest.fixture(scope="module")
def model_reg():
    np.random.seed(2)
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=_load_data(),
        biters=100,
    ).fit(est_method="reg")


@pytest.fixture(scope="module")
def model_no_bstrap():
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=_load_data(),
        biters=100,
    ).fit(est_method="dr", bstrap=False)


@pytest.fixture(scope="module")
def model_notyettreated():
    np.random.seed(3)
    return ATTgt(
        yname="lemp",
        gname="first.treat",
        idname="countyreal",
        tname="year",
        data=_load_data(),
        control_group="notyettreated",
        biters=100,
    ).fit(est_method="dr")


# ---------------------------------------------------------------------------
# Original regression tests (preserved exactly)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Aggregation types
# ---------------------------------------------------------------------------

def test_group_aggte_structure(model):
    res = model.aggte("group")
    for key in AGGTE_KEYS:
        assert key in res.atte, f"missing key {key!r} in group aggte output"


def test_group_aggte_egt_are_group_values(model):
    res = model.aggte("group")
    # egt should be the group (first-treatment-year) values, not event times
    egt = np.asarray(res.atte["egt"])
    # mpdta has treatment groups 2004, 2006, 2007
    assert all(e > 2000 for e in egt)


def test_group_aggte_se_all_positive(model):
    res = model.aggte("group")
    se_egt = np.ravel(res.atte["se_egt"])
    assert all(s > 0 for s in se_egt)


def test_calendar_aggte_structure(model):
    res = model.aggte("calendar")
    for key in AGGTE_KEYS:
        assert key in res.atte, f"missing key {key!r} in calendar aggte output"


def test_calendar_aggte_overall_att_is_scalar(model):
    res = model.aggte("calendar")
    att = float(res.atte["overall_att"])
    assert np.isfinite(att)


def test_calendar_aggte_egt_are_time_periods(model):
    res = model.aggte("calendar")
    egt = np.asarray(res.atte["egt"])
    # calendar egt should be actual years >= first treatment year
    assert all(e >= 2004 for e in egt)


def test_dynamic_aggte_egt_includes_negative_event_times(model):
    res = model.aggte("dynamic")
    egt = np.asarray(res.atte["egt"])
    assert any(e < 0 for e in egt), "dynamic egt should contain pre-treatment periods"


def test_dynamic_aggte_overall_att_negative(model):
    # Known from the mpdta data: overall effect on employment is negative
    res = model.aggte("dynamic")
    att = float(res.atte["overall_att"])
    assert att < 0


def test_invalid_aggte_type_raises(model):
    with pytest.raises(ValueError):
        model.aggte("nonexistent_type")


# ---------------------------------------------------------------------------
# Estimation methods
# ---------------------------------------------------------------------------

def test_ipw_simple_aggte_att_finite(model_ipw):
    res = model_ipw.aggte("simple")
    att = float(res.atte["overall_att"])
    assert np.isfinite(att)


def test_ipw_simple_aggte_se_positive(model_ipw):
    res = model_ipw.aggte("simple")
    se = float(np.ravel(res.atte["overall_se"])[0])
    assert se > 0


def test_reg_simple_aggte_att_finite(model_reg):
    res = model_reg.aggte("simple")
    att = float(res.atte["overall_att"])
    assert np.isfinite(att)


def test_reg_simple_aggte_se_positive(model_reg):
    res = model_reg.aggte("simple")
    se = float(np.ravel(res.atte["overall_se"])[0])
    assert se > 0


def test_all_methods_give_negative_overall_att(model, model_ipw, model_reg):
    for m in (model, model_ipw, model_reg):
        res = m.aggte("simple")
        att = float(res.atte["overall_att"])
        assert att < 0, f"Expected negative ATT for mpdta, got {att}"


# ---------------------------------------------------------------------------
# Bootstrap vs analytical SEs
# ---------------------------------------------------------------------------

def test_no_bstrap_simple_aggte_att_finite(model_no_bstrap):
    res = model_no_bstrap.aggte("simple")
    att = float(res.atte["overall_att"])
    assert np.isfinite(att)


def test_no_bstrap_simple_aggte_se_positive(model_no_bstrap):
    res = model_no_bstrap.aggte("simple")
    se = float(np.ravel(res.atte["overall_se"])[0])
    assert se > 0


def test_no_bstrap_att_matches_bootstrap_att_closely(model, model_no_bstrap):
    att_bstrap = float(model.aggte("simple").atte["overall_att"])
    att_analytic = float(model_no_bstrap.aggte("simple").atte["overall_att"])
    # Both estimate the same parameter; results should be close (same seed data)
    assert abs(att_bstrap - att_analytic) < 1e-6


# ---------------------------------------------------------------------------
# Control group: notyettreated
# ---------------------------------------------------------------------------

def test_notyettreated_simple_aggte_att_finite(model_notyettreated):
    res = model_notyettreated.aggte("simple")
    att = float(res.atte["overall_att"])
    assert np.isfinite(att)


def test_notyettreated_simple_aggte_se_positive(model_notyettreated):
    res = model_notyettreated.aggte("simple")
    se = float(np.ravel(res.atte["overall_se"])[0])
    assert se > 0


def test_notyettreated_dynamic_structure(model_notyettreated):
    res = model_notyettreated.aggte("dynamic")
    for key in ("overall_att", "overall_se", "egt", "att_egt", "se_egt"):
        assert key in res.atte


# ---------------------------------------------------------------------------
# ATTgt result structure
# ---------------------------------------------------------------------------

def test_fit_stores_results_dict(model):
    assert hasattr(model, "results")
    for key in ("group", "year", "att", "se"):
        assert key in model.results, f"missing key {key!r} in results"


def test_fit_results_att_array_length(model):
    att = np.asarray(model.results["att"])
    group = np.asarray(model.results["group"])
    assert len(att) == len(group)


def test_fit_results_se_positive(model):
    se = np.asarray(model.results["se"])
    assert all(s > 0 for s in se)


def test_fit_stores_mp_object(model):
    assert hasattr(model, "MP")
    for key in ("group", "att", "t", "DIDparams", "inffunc", "n"):
        assert key in model.MP, f"missing key {key!r} in MP"


# ---------------------------------------------------------------------------
# summ_attgt
# ---------------------------------------------------------------------------

def test_summ_attgt_returns_self(model):
    result = model.summ_attgt()
    assert result is model


def test_summ_attgt_creates_dataframe(model):
    model.summ_attgt()
    assert hasattr(model, "summary2")
    assert isinstance(model.summary2, pd.DataFrame)


def test_summ_attgt_has_expected_columns(model):
    model.summ_attgt()
    expected = ["Group", "Time", "ATT(g, t)", "Post", "Std. Error",
                "[95% Pointwise", "Conf. Band]", ""]
    assert list(model.summary2.columns) == expected


def test_summ_attgt_rounding(model):
    model.summ_attgt(n=2)
    att_col = model.summary2["ATT(g, t)"]
    # With n=2 decimal places, values should be rounded
    for val in att_col:
        if isinstance(val, float):
            assert round(val, 2) == val
