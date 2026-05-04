"""
Tests targeting the coverage gaps identified in the R did faithfulness audit.

Each section states which source lines it exercises.
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from csdid.att_gt import ATTgt
from csdid.attgt_fnc.drdid_trim import drdid_panel

DATA = Path(__file__).parent.parent / "data" / "mpdta.csv"
AGGTE_KEYS = ("overall_att", "overall_se", "egt", "att_egt", "se_egt", "crit_val_egt")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load():
    return pd.read_csv(DATA)


def _make_sparse_rc_frame():
    """RC dataset designed to produce NaN ATT(g,t) estimates.

    - Cohort 2004: units in all periods → finite ATTs.
    - Cohort 2005: units only in 2005 (no pre-period observations) → every
      ATT(g=2005, t) triggers the skip-condition and returns NaN.
    """
    np.random.seed(77)
    rows, uid = [], 0
    for period in [2003, 2004, 2005]:
        for _ in range(50):
            rows.append({"id": uid, "year": period, "g": 0, "y": np.random.randn()})
            uid += 1
    for period in [2003, 2004, 2005]:
        for _ in range(50):
            eff = 1.0 if period >= 2004 else 0.0
            rows.append({"id": uid, "year": period, "g": 2004, "y": eff + np.random.randn()})
            uid += 1
    for _ in range(50):
        rows.append({"id": uid, "year": 2005, "g": 2005, "y": 1.5 + np.random.randn()})
        uid += 1
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures (each built once for the entire test session)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Baseline DR model used as a reference in several tests."""
    np.random.seed(0)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), biters=200,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_rc():
    """panel=False — RC data path in compute_att_gt and preprocess_did."""
    np.random.seed(10)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), panel=False, biters=100,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_cband():
    """cband=True — bootstrap simultaneous confidence bands in all aggte types."""
    np.random.seed(11)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), cband=True, biters=200,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_universal():
    """base_period='universal' — exercises the universal-base skip logic."""
    np.random.seed(12)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), biters=100,
    ).fit(est_method="dr", base_period="universal")


@pytest.fixture(scope="module")
def model_balanced():
    """allow_unbalanced_panel=False — exercises makeBalancedPanel in preprocess."""
    np.random.seed(13)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), allow_unbalanced_panel=False, biters=100,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_clustered():
    """clustervar — exercises the clustered bootstrap branch in mboot."""
    np.random.seed(14)
    df = _load()
    df["state"] = df["countyreal"] % 10   # time-invariant; 10 synthetic clusters
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=df, clustervar="state", biters=100,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_covariates():
    """xformla with lpop covariate — exercises the patsy formula path."""
    np.random.seed(15)
    return ATTgt(
        yname="lemp", gname="first.treat",
        idname="countyreal", tname="year",
        data=_load(), xformla="lemp ~ lpop", biters=100,
    ).fit(est_method="dr")


@pytest.fixture(scope="module")
def model_nan():
    """RC model on sparse data — some ATT(g,t) values are NaN.

    Used to test na_rm=True and the ValueError raised when na_rm=False.
    """
    np.random.seed(78)
    df = _make_sparse_rc_frame()
    return ATTgt(
        yname="y", gname="g", idname="id", tname="year",
        data=df, panel=False, biters=100,
    ).fit(est_method="dr")


# ─────────────────────────────────────────────────────────────────────────────
# RC (panel=False) path
# compute_att_gt.py:159-224  preprocess_did.py:129-141  mboot.py:27
# ─────────────────────────────────────────────────────────────────────────────

class TestRCPath:
    def test_rc_model_fits_without_error(self, model_rc):
        assert hasattr(model_rc, "results")

    def test_rc_att_is_finite(self, model_rc):
        att = float(model_rc.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_rc_se_is_positive(self, model_rc):
        se = float(np.ravel(model_rc.aggte("simple").atte["overall_se"])[0])
        assert se > 0

    def test_rc_dynamic_aggte_structure(self, model_rc):
        res = model_rc.aggte("dynamic")
        for key in AGGTE_KEYS:
            assert key in res.atte

    def test_rc_group_aggte_structure(self, model_rc):
        res = model_rc.aggte("group")
        for key in AGGTE_KEYS:
            assert key in res.atte

    def test_rc_calendar_aggte_structure(self, model_rc):
        res = model_rc.aggte("calendar")
        for key in AGGTE_KEYS:
            assert key in res.atte

    def test_rc_ipw_method(self):
        np.random.seed(20)
        m = ATTgt(
            yname="lemp", gname="first.treat",
            idname="countyreal", tname="year",
            data=_load(), panel=False, biters=100,
        ).fit(est_method="ipw")
        att = float(m.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_rc_reg_method(self):
        np.random.seed(21)
        m = ATTgt(
            yname="lemp", gname="first.treat",
            idname="countyreal", tname="year",
            data=_load(), panel=False, biters=100,
        ).fit(est_method="reg")
        att = float(m.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_rc_overall_att_negative(self, model_rc):
        att = float(model_rc.aggte("simple").atte["overall_att"])
        assert att < 0


# ─────────────────────────────────────────────────────────────────────────────
# cband=True — bootstrap simultaneous confidence bands
# compute_aggte.py:230-246  compute_aggte.py:346-361  compute_aggte.py:446-463
# ─────────────────────────────────────────────────────────────────────────────

class TestCband:
    def test_group_cband_crit_val_at_least_1p96(self, model_cband):
        res = model_cband.aggte("group")
        cv = float(res.atte["crit_val_egt"])
        assert cv >= 1.96

    def test_dynamic_cband_crit_val_finite(self, model_cband):
        res = model_cband.aggte("dynamic")
        cv = float(res.atte["crit_val_egt"])
        assert np.isfinite(cv)

    def test_calendar_cband_crit_val_finite(self, model_cband):
        res = model_cband.aggte("calendar")
        cv = float(res.atte["crit_val_egt"])
        assert np.isfinite(cv)

    def test_cband_simple_aggte_structure(self, model_cband):
        res = model_cband.aggte("simple")
        assert "overall_att" in res.atte
        assert np.isfinite(float(res.atte["overall_att"]))

    def test_cband_crit_val_strictly_larger_than_pointwise(self, model_cband):
        """Simultaneous band should (virtually always) be wider than 1.96."""
        res_group = model_cband.aggte("group")
        assert float(res_group.atte["crit_val_egt"]) >= 1.96
        res_dyn = model_cband.aggte("dynamic")
        assert float(res_dyn.atte["crit_val_egt"]) >= 1.96


# ─────────────────────────────────────────────────────────────────────────────
# NaN ATT handling — na_rm and ValueError
# compute_aggte.py:70-91  compute_aggte.py:95
# ─────────────────────────────────────────────────────────────────────────────

class TestNanAtts:
    def test_nan_att_model_has_nan_atts(self, model_nan):
        att = np.asarray(model_nan.MP["att"])
        assert np.any(np.isnan(att)), "fixture should produce at least one NaN ATT"

    def test_group_aggte_raises_without_na_rm(self, model_nan):
        with pytest.raises(ValueError, match="Missing values"):
            model_nan.aggte("group")

    def test_simple_aggte_raises_without_na_rm(self, model_nan):
        with pytest.raises(ValueError, match="Missing values"):
            model_nan.aggte("simple")

    def test_group_aggte_with_na_rm_succeeds(self, model_nan):
        res = model_nan.aggte("group", na_rm=True)
        assert np.isfinite(float(res.atte["overall_att"]))

    def test_simple_aggte_with_na_rm_succeeds(self, model_nan):
        res = model_nan.aggte("simple", na_rm=True)
        assert np.isfinite(float(res.atte["overall_att"]))

    def test_dynamic_aggte_with_na_rm_succeeds(self, model_nan):
        res = model_nan.aggte("dynamic", na_rm=True)
        assert "egt" in res.atte

    def test_calendar_aggte_with_na_rm_succeeds(self, model_nan):
        res = model_nan.aggte("calendar", na_rm=True)
        assert "egt" in res.atte


# ─────────────────────────────────────────────────────────────────────────────
# Universal base period
# compute_att_gt.py:110-111
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalBase:
    def test_universal_model_fits(self, model_universal):
        assert hasattr(model_universal, "results")

    def test_universal_att_is_finite(self, model_universal):
        att = float(model_universal.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_universal_dynamic_includes_pretreatment(self, model_universal):
        res = model_universal.aggte("dynamic")
        egt = np.asarray(res.atte["egt"])
        assert any(e < 0 for e in egt)

    def test_universal_group_aggte_structure(self, model_universal):
        res = model_universal.aggte("group")
        for key in AGGTE_KEYS:
            assert key in res.atte

    def test_universal_has_more_att_entries_than_varying(self, model_universal, model):
        """Universal base includes an extra comparison per group per period."""
        n_universal = len(model_universal.MP["att"])
        n_varying = len(model.MP["att"])
        # Universal loops over all tlist; varying loops over tlist[1:]
        assert n_universal > n_varying


# ─────────────────────────────────────────────────────────────────────────────
# Callable est_method
# compute_att_gt.py:138
# ─────────────────────────────────────────────────────────────────────────────

class TestCallableEstMethod:
    def test_callable_drdid_panel_on_panel_data(self):
        np.random.seed(16)
        m = ATTgt(
            yname="lemp", gname="first.treat",
            idname="countyreal", tname="year",
            data=_load(), biters=100,
        ).fit(est_method=drdid_panel)
        att = float(m.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_callable_gives_same_att_as_named_dr(self):
        np.random.seed(17)
        m_callable = ATTgt(
            yname="lemp", gname="first.treat",
            idname="countyreal", tname="year",
            data=_load(), biters=100,
        ).fit(est_method=drdid_panel)
        np.random.seed(17)
        m_named = ATTgt(
            yname="lemp", gname="first.treat",
            idname="countyreal", tname="year",
            data=_load(), biters=100,
        ).fit(est_method="dr")
        att_callable = float(m_callable.aggte("simple").atte["overall_att"])
        att_named = float(m_named.aggte("simple").atte["overall_att"])
        # Point estimates are deterministic regardless of bootstrap
        assert abs(att_callable - att_named) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# Clustered bootstrap
# mboot.py:49-51, 58-62
# ─────────────────────────────────────────────────────────────────────────────

class TestClustering:
    def test_clustered_model_fits(self, model_clustered):
        assert hasattr(model_clustered, "results")

    def test_clustered_att_is_finite(self, model_clustered):
        att = float(model_clustered.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_clustered_se_is_positive(self, model_clustered):
        se = float(np.ravel(model_clustered.aggte("simple").atte["overall_se"])[0])
        assert se > 0

    def test_clustered_dynamic_aggte_structure(self, model_clustered):
        res = model_clustered.aggte("dynamic")
        for key in AGGTE_KEYS:
            assert key in res.atte

    def test_time_varying_clustervar_raises(self):
        df = _load()
        # Make 'lpop' vary over time (it's actually constant per county in mpdta,
        # but we perturb it so it varies per year for each unit)
        df = df.copy()
        df["tv_cluster"] = df["lpop"] + df["year"] * 0.001
        with pytest.raises(ValueError, match="time-varying"):
            ATTgt(
                yname="lemp", gname="first.treat",
                idname="countyreal", tname="year",
                data=df, clustervar="tv_cluster", biters=50,
            ).fit(est_method="dr")


# ─────────────────────────────────────────────────────────────────────────────
# allow_unbalanced_panel=False — makeBalancedPanel
# preprocess_did.py:108-126
# ─────────────────────────────────────────────────────────────────────────────

class TestBalancedPanel:
    def test_balanced_model_fits(self, model_balanced):
        assert hasattr(model_balanced, "results")

    def test_balanced_att_is_finite(self, model_balanced):
        att = float(model_balanced.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_balanced_att_negative(self, model_balanced):
        att = float(model_balanced.aggte("simple").atte["overall_att"])
        assert att < 0

    def test_balanced_n_equals_unbalanced_n_for_mpdta(self, model_balanced, model):
        """mpdta is already balanced; n should be the same either way."""
        assert model_balanced.dp["n"] == model.dp["n"]


# ─────────────────────────────────────────────────────────────────────────────
# balance_e and min_e/max_e in dynamic aggte
# compute_aggte.py:308-311
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicFilters:
    def test_balance_e_limits_event_times(self, model):
        res = model.aggte("dynamic", balance_e=1)
        egt = np.asarray(res.atte["egt"])
        assert all(e <= 1 for e in egt)

    def test_min_e_max_e_filters_event_times(self, model):
        res = model.aggte("dynamic", min_e=-1, max_e=2)
        egt = np.asarray(res.atte["egt"])
        assert all(-1 <= e <= 2 for e in egt)

    def test_min_e_only_positive_excludes_pretreatment(self, model):
        res = model.aggte("dynamic", min_e=0)
        egt = np.asarray(res.atte["egt"])
        assert all(e >= 0 for e in egt)

    def test_overall_att_with_balance_e_is_finite(self, model):
        res = model.aggte("dynamic", balance_e=2)
        att = float(res.atte["overall_att"])
        assert np.isfinite(att)


# ─────────────────────────────────────────────────────────────────────────────
# Covariates via xformla
# preprocess_did.py:36-37  compute_att_gt.py: covariates branch
# ─────────────────────────────────────────────────────────────────────────────

class TestCovariates:
    def test_covariate_model_fits(self, model_covariates):
        assert hasattr(model_covariates, "results")

    def test_covariate_att_is_finite(self, model_covariates):
        att = float(model_covariates.aggte("simple").atte["overall_att"])
        assert np.isfinite(att)

    def test_covariate_se_is_positive(self, model_covariates):
        se = float(np.ravel(model_covariates.aggte("simple").atte["overall_se"])[0])
        assert se > 0

    def test_covariate_att_near_no_covariate_att(self, model_covariates, model):
        """Adding a covariate shouldn't drastically change the estimate."""
        att_cov = float(model_covariates.aggte("simple").atte["overall_att"])
        att_base = float(model.aggte("simple").atte["overall_att"])
        assert abs(att_cov - att_base) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# att_gt.py:153-199  att_gt.py:211-248
# ─────────────────────────────────────────────────────────────────────────────

class TestPlots:
    def test_plot_attgt_returns_figure(self, model):
        fig = model.plot_attgt()
        assert fig is not None
        plt.close("all")

    def test_plot_attgt_specific_group(self, model):
        groups = sorted(set(int(g) for g in model.MP["group"] if g > 0))
        fig = model.plot_attgt(group=[groups[0]])
        assert fig is not None
        plt.close("all")

    def test_plot_attgt_invalid_group_raises(self, model):
        with pytest.raises(ValueError):
            model.plot_attgt(group=[99999])

    def test_plot_aggte_dynamic(self, model):
        model.aggte("dynamic")
        result = model.plot_aggte()
        assert result is not None
        plt.close("all")

    def test_plot_aggte_group(self, model):
        model.aggte("group")
        result = model.plot_aggte()
        assert result is not None
        plt.close("all")

    def test_plot_aggte_calendar(self, model):
        model.aggte("calendar")
        result = model.plot_aggte()
        assert result is not None
        plt.close("all")
