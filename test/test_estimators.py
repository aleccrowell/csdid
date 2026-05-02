"""
Unit tests for drdid_trim.py estimators.

These tests use synthetic datasets with known ATTs to verify that each
estimator function produces correct output shapes, near-correct point
estimates, and proper error handling.
"""
import numpy as np
import pytest

from csdid.attgt_fnc.drdid_trim import (
    _add_intercept,
    _trim_pscore,
    drdid_panel,
    drdid_rc,
    std_ipw_did_panel,
    std_ipw_did_rc,
)

TRUE_ATT_PANEL = 1.5
TRUE_ATT_RC = 2.0
ATT_TOL = 0.5  # generous tolerance for finite-sample estimates


@pytest.fixture(scope="module")
def panel_data():
    """300 observations: 100 treated, 200 control; true ATT = 1.5."""
    np.random.seed(42)
    n = 300
    D = np.zeros(n)
    D[:100] = 1.0
    y0 = np.random.randn(n)
    y1 = y0.copy()
    y1[:100] += TRUE_ATT_PANEL
    return y0, y1, D


@pytest.fixture(scope="module")
def rc_data():
    """400 obs across 4 cells (D x post); true ATT = 2.0."""
    np.random.seed(99)
    n_per_cell = 100
    n = 4 * n_per_cell
    D = np.zeros(n)
    post = np.zeros(n)
    # treated pre   [0:100]:   D=1, post=0
    D[:100] = 1.0
    # treated post  [100:200]: D=1, post=1
    D[100:200] = 1.0
    post[100:200] = 1.0
    # control pre   [200:300]: D=0, post=0 (defaults)
    # control post  [300:400]: D=0, post=1
    post[300:400] = 1.0

    y = np.random.randn(n)
    y[(D == 1) & (post == 1)] += TRUE_ATT_RC
    return y, post, D


# ---------------------------------------------------------------------------
# _add_intercept
# ---------------------------------------------------------------------------

class TestAddIntercept:
    def test_none_covariates_returns_ones_column(self):
        result = _add_intercept(None, 5)
        assert result.shape == (5, 1)
        np.testing.assert_array_equal(result, np.ones((5, 1)))

    def test_no_intercept_added_when_first_column_is_ones(self):
        cov = np.column_stack([np.ones(10), np.arange(10, dtype=float)])
        result = _add_intercept(cov, 10)
        assert result.shape == (10, 2)

    def test_intercept_prepended_when_missing(self):
        cov = np.arange(10, dtype=float).reshape(5, 2)
        result = _add_intercept(cov, 5)
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[:, 0], np.ones(5))

    def test_1d_covariate_becomes_column_vector(self):
        cov = np.array([2.0, 3.0, 4.0])
        result = _add_intercept(cov, 3)
        assert result.ndim == 2
        assert result.shape[1] == 2  # intercept + original column


# ---------------------------------------------------------------------------
# _trim_pscore
# ---------------------------------------------------------------------------

class TestTrimPscore:
    def test_treated_units_never_trimmed_below_1(self):
        ps = np.array([0.3, 0.6, 0.99, 0.5])
        D = np.array([1, 1, 1, 1])
        ps_fit, trim_ps = _trim_pscore(ps, D, 0.995)
        assert all(trim_ps)

    def test_controls_with_high_pscore_are_trimmed(self):
        ps = np.array([0.998, 0.5])
        D = np.array([0, 0])
        ps_fit, trim_ps = _trim_pscore(ps, D, 0.995)
        assert not trim_ps[0]  # 0.998 >= 0.995 → trimmed
        assert trim_ps[1]      # 0.5 < 0.995 → kept

    def test_pscore_capped_at_1_minus_epsilon(self):
        ps = np.array([1.0, 0.5])
        D = np.array([1, 0])
        ps_fit, _ = _trim_pscore(ps, D, 0.995)
        assert ps_fit[0] < 1.0

    def test_returns_two_arrays_of_correct_length(self):
        n = 20
        ps = np.random.rand(n)
        D = (np.arange(n) % 2).astype(float)
        ps_fit, trim_ps = _trim_pscore(ps, D, 0.995)
        assert len(ps_fit) == n
        assert len(trim_ps) == n


# ---------------------------------------------------------------------------
# std_ipw_did_panel
# ---------------------------------------------------------------------------

class TestStdIpwDidPanel:
    def test_returns_scalar_att_and_array_inffunc(self, panel_data):
        y0, y1, D = panel_data
        att, inf = std_ipw_did_panel(y1, y0, D)
        assert np.isscalar(att) or np.asarray(att).shape == ()
        assert inf.shape == (len(D),)

    def test_att_near_true_value(self, panel_data):
        y0, y1, D = panel_data
        att, _ = std_ipw_did_panel(y1, y0, D)
        assert abs(att - TRUE_ATT_PANEL) < ATT_TOL

    def test_zero_treatment_effect(self):
        np.random.seed(5)
        n = 400
        D = np.zeros(n)
        D[:200] = 1.0
        y0 = np.random.randn(n)
        att, _ = std_ipw_did_panel(y0, y0, D)  # y1 == y0 → ATT = 0
        assert abs(att) < 0.2

    def test_negative_weights_raise(self, panel_data):
        y0, y1, D = panel_data
        with pytest.raises(ValueError, match="non-negative"):
            std_ipw_did_panel(y1, y0, D, i_weights=np.full(len(D), -1.0))

    def test_with_positive_weights(self, panel_data):
        y0, y1, D = panel_data
        weights = np.abs(np.random.randn(len(D))) + 0.1
        att, inf = std_ipw_did_panel(y1, y0, D, i_weights=weights)
        assert np.isfinite(att)
        assert inf.shape == (len(D),)

    def test_influence_function_mean_near_zero(self, panel_data):
        y0, y1, D = panel_data
        _, inf = std_ipw_did_panel(y1, y0, D)
        assert abs(np.mean(inf)) < 0.5


# ---------------------------------------------------------------------------
# drdid_panel
# ---------------------------------------------------------------------------

class TestDrdidPanel:
    def test_returns_scalar_att_and_array_inffunc(self, panel_data):
        y0, y1, D = panel_data
        att, inf = drdid_panel(y1, y0, D)
        assert np.isscalar(att) or np.asarray(att).shape == ()
        assert inf.shape == (len(D),)

    def test_att_near_true_value(self, panel_data):
        y0, y1, D = panel_data
        att, _ = drdid_panel(y1, y0, D)
        assert abs(att - TRUE_ATT_PANEL) < ATT_TOL

    def test_zero_treatment_effect(self):
        np.random.seed(6)
        n = 400
        D = np.zeros(n)
        D[:200] = 1.0
        y0 = np.random.randn(n)
        att, _ = drdid_panel(y0, y0, D)
        assert abs(att) < 0.2

    def test_negative_weights_raise(self, panel_data):
        y0, y1, D = panel_data
        with pytest.raises(ValueError, match="non-negative"):
            drdid_panel(y1, y0, D, i_weights=np.full(len(D), -1.0))

    def test_with_covariates(self, panel_data):
        np.random.seed(10)
        y0, y1, D = panel_data
        cov = np.random.randn(len(D), 2)
        att, inf = drdid_panel(y1, y0, D, covariates=cov)
        assert np.isfinite(att)
        assert inf.shape == (len(D),)


# ---------------------------------------------------------------------------
# std_ipw_did_rc
# ---------------------------------------------------------------------------

class TestStdIpwDidRc:
    def test_returns_scalar_att_and_array_inffunc(self, rc_data):
        y, post, D = rc_data
        att, inf = std_ipw_did_rc(y, post, D)
        assert np.isscalar(att) or np.asarray(att).shape == ()
        assert inf.shape == (len(D),)

    def test_att_near_true_value(self, rc_data):
        y, post, D = rc_data
        att, _ = std_ipw_did_rc(y, post, D)
        assert abs(att - TRUE_ATT_RC) < ATT_TOL

    def test_zero_treatment_effect(self):
        np.random.seed(20)
        n_per_cell = 100
        n = 4 * n_per_cell
        D = np.zeros(n)
        D[:200] = 1.0
        post = np.zeros(n)
        post[100:200] = 1.0
        post[300:400] = 1.0
        y = np.random.randn(n)  # no treatment effect added
        att, _ = std_ipw_did_rc(y, post, D)
        assert abs(att) < 0.3

    def test_negative_weights_raise(self, rc_data):
        y, post, D = rc_data
        with pytest.raises(ValueError, match="non-negative"):
            std_ipw_did_rc(y, post, D, i_weights=np.full(len(D), -1.0))


# ---------------------------------------------------------------------------
# drdid_rc
# ---------------------------------------------------------------------------

class TestDrdidRc:
    def test_returns_scalar_att_and_array_inffunc(self, rc_data):
        y, post, D = rc_data
        att, inf = drdid_rc(y, post, D)
        assert np.isscalar(att) or np.asarray(att).shape == ()
        assert inf.shape == (len(D),)

    def test_att_near_true_value(self, rc_data):
        y, post, D = rc_data
        att, _ = drdid_rc(y, post, D)
        assert abs(att - TRUE_ATT_RC) < ATT_TOL

    def test_zero_treatment_effect(self):
        np.random.seed(30)
        n_per_cell = 200  # larger sample; drdid_rc has higher variance than IPW
        n = 4 * n_per_cell
        D = np.zeros(n)
        D[:400] = 1.0
        post = np.zeros(n)
        post[200:400] = 1.0
        post[600:800] = 1.0
        y = np.random.randn(n)
        att, _ = drdid_rc(y, post, D)
        assert abs(att) < 0.4

    def test_negative_weights_raise(self, rc_data):
        y, post, D = rc_data
        with pytest.raises(ValueError, match="non-negative"):
            drdid_rc(y, post, D, i_weights=np.full(len(D), -1.0))

    def test_with_covariates(self, rc_data):
        np.random.seed(40)
        y, post, D = rc_data
        cov = np.random.randn(len(D), 2)
        att, inf = drdid_rc(y, post, D, covariates=cov)
        assert np.isfinite(att)
        assert inf.shape == (len(D),)
