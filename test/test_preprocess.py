"""
Tests for attgt_fnc/preprocess_did.py.

Covers: normal operation, control group handling, anticipation filtering,
missing data warnings, and error conditions.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from csdid.attgt_fnc.preprocess_did import pre_process_did


def make_data(n_per_group=10, periods=(2000, 2001, 2002, 2003), groups=(2001, 2002, 0)):
    """Build a balanced panel with specified groups and time periods."""
    np.random.seed(0)
    rows = []
    unit_id = 1
    for g in groups:
        for _ in range(n_per_group):
            for year in periods:
                rows.append({"id": unit_id, "year": year, "y": np.random.randn(), "g": g})
            unit_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def basic_data():
    return make_data(n_per_group=10, periods=(2000, 2001, 2002, 2003), groups=(2001, 2002, 0))


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

def test_basic_preprocessing_returns_dict(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    assert isinstance(dp, dict)


def test_output_contains_required_keys(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    required = ["yname", "tname", "idname", "gname", "data", "tlist", "glist",
                "n", "nG", "nT", "control_group", "anticipation", "panel"]
    for key in required:
        assert key in dp, f"Missing key: {key!r}"


def test_glist_excludes_never_treated_group(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    assert 0 not in dp["glist"]


def test_glist_contains_treatment_cohorts(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    assert 2001 in dp["glist"]
    assert 2002 in dp["glist"]


def test_tlist_covers_all_periods(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    assert set(dp["tlist"]) == {2000, 2001, 2002, 2003}


def test_n_equals_number_of_unique_units(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    # 10 per group x 3 groups = 30 units
    assert dp["n"] == 30


def test_data_sorted_by_id_then_time(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    data = dp["data"]
    id_col, t_col = dp["idname"], dp["tname"]
    expected = data.sort_values([id_col, t_col]).reset_index(drop=True)
    pd.testing.assert_frame_equal(data.reset_index(drop=True), expected)


# ---------------------------------------------------------------------------
# Control group options
# ---------------------------------------------------------------------------

def test_nevertreated_control_group_stored(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data, control_group="nevertreated")
    assert dp["control_group"] == "nevertreated"


def test_notyettreated_control_group_stored(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data, control_group="notyettreated")
    assert dp["control_group"] == "notyettreated"


def test_nevertreated_raises_when_no_never_treated_units():
    # All units are treated (no 0 group in gname)
    data = make_data(n_per_group=10, periods=(2000, 2001, 2002), groups=(2001, 2002))
    with pytest.raises(ValueError, match="never-treated"):
        pre_process_did("y", "year", "id", "g", data, control_group="nevertreated")


def test_notyettreated_works_without_never_treated_units():
    # notyettreated should handle no 0 group by using later-treated cohorts as controls
    data = make_data(n_per_group=10, periods=(2000, 2001, 2002), groups=(2001, 2002))
    # This should not raise; it trims the sample instead
    dp = pre_process_did("y", "year", "id", "g", data, control_group="notyettreated")
    assert dp is not None


# ---------------------------------------------------------------------------
# Anticipation parameter
# ---------------------------------------------------------------------------

def test_anticipation_zero_keeps_all_cohorts(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data, anticipation=0)
    assert 2001 in dp["glist"]
    assert 2002 in dp["glist"]


def test_anticipation_one_drops_earliest_cohort(basic_data):
    # fp=2000; with anticipation=1, drop groups where g <= 2000+1=2001
    dp = pre_process_did("y", "year", "id", "g", basic_data, anticipation=1)
    assert 2001 not in dp["glist"]
    assert 2002 in dp["glist"]


def test_anticipation_too_large_raises():
    # anticipation=3 with fp=2000 drops groups <= 2003, leaving none
    data = make_data(n_per_group=10, periods=(2000, 2001, 2002, 2003), groups=(2001, 2002, 0))
    with pytest.raises(ValueError):
        pre_process_did("y", "year", "id", "g", data, anticipation=3)


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------

def test_missing_outcome_dropped_with_warning():
    data = make_data(n_per_group=10, periods=(2000, 2001, 2002), groups=(2001, 0))
    data.loc[data.index[0], "y"] = np.nan  # introduce one missing value
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = pre_process_did("y", "year", "id", "g", data)
    messages = " ".join(str(warning.message) for warning in w)
    assert "Dropped" in messages or "missing" in messages.lower()


def test_no_warnings_for_complete_data(basic_data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pre_process_did("y", "year", "id", "g", basic_data)
    # filter out unrelated warnings
    relevant = [x for x in w if "Dropped" in str(x.message)]
    assert len(relevant) == 0


# ---------------------------------------------------------------------------
# Early-treated units warning
# ---------------------------------------------------------------------------

def test_early_treated_units_warned_and_dropped():
    data = make_data(n_per_group=10, periods=(2000, 2001, 2002), groups=(2001, 0))
    # Add a unit treated at the first period (g == fp == 2000)
    early_unit = pd.DataFrame({
        "id": [999, 999, 999],
        "year": [2000, 2001, 2002],
        "y": [1.0, 2.0, 3.0],
        "g": [2000, 2000, 2000],
    })
    data = pd.concat([data, early_unit], ignore_index=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = pre_process_did("y", "year", "id", "g", data)
    messages = " ".join(str(warning.message) for warning in w)
    assert "treated" in messages.lower() or "Dropped" in messages


# ---------------------------------------------------------------------------
# Clustering and weights pass-through
# ---------------------------------------------------------------------------

def test_clustervars_stored_in_output(basic_data):
    data = basic_data.copy()
    data["cluster"] = data["id"] % 5  # create cluster variable
    dp = pre_process_did("y", "year", "id", "g", data, clustervar="cluster")
    assert dp["clustervars"] == ["cluster"]


def test_no_cluster_stored_as_none(basic_data):
    dp = pre_process_did("y", "year", "id", "g", basic_data)
    assert dp["clustervars"] is None


def test_weights_name_passed_through(basic_data):
    data = basic_data.copy()
    data["wt"] = 1.0
    dp = pre_process_did("y", "year", "id", "g", data, weights_name="wt")
    assert dp["weights_name"] == "wt"
