import numpy as np
import pandas as pd
import pytest

from csdid.utils.bmisc import TorF, makeBalancedPanel, multiplier_bootstrap, panel2cs2


class TestMakeBalancedPanel:
    def test_keeps_only_units_present_all_periods(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 2, 3],
            "year": [2000, 2001, 2000, 2001, 2000],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = makeBalancedPanel(df, "id", "year")
        assert set(result["id"].unique()) == {1, 2}
        assert 3 not in result["id"].values

    def test_returns_empty_when_no_unit_is_balanced(self):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "year": [2000, 2001, 2002],
            "y": [1.0, 2.0, 3.0],
        })
        result = makeBalancedPanel(df, "id", "year")
        assert len(result) == 0

    def test_sorted_by_id_then_year(self):
        df = pd.DataFrame({
            "id": [2, 1, 2, 1],
            "year": [2001, 2000, 2000, 2001],
            "y": [4.0, 1.0, 3.0, 2.0],
        })
        result = makeBalancedPanel(df, "id", "year")
        assert list(result["id"]) == [1, 1, 2, 2]
        assert list(result["year"]) == [2000, 2001, 2000, 2001]

    def test_preserves_all_rows_when_all_balanced(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "y": [1.0, 2.0, 3.0, 4.0],
        })
        result = makeBalancedPanel(df, "id", "year")
        assert len(result) == 4


class TestPanel2cs2:
    def test_computes_first_difference_correctly(self):
        df = pd.DataFrame({"id": [1, 1], "year": [2000, 2001], "y": [5.0, 8.0]})
        result = panel2cs2(df, "y", "id", "year")
        # dy should be 3.0 for the earlier row
        dy_vals = result["dy"].dropna().values
        assert len(dy_vals) == 1
        assert float(dy_vals[0]) == pytest.approx(3.0)

    def test_raises_for_more_than_two_periods(self):
        df = pd.DataFrame({"id": [1, 1, 1], "year": [2000, 2001, 2002], "y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="2 periods"):
            panel2cs2(df, "y", "id", "year")

    def test_raises_for_one_period(self):
        df = pd.DataFrame({"id": [1, 2], "year": [2000, 2000], "y": [1.0, 2.0]})
        with pytest.raises(ValueError):
            panel2cs2(df, "y", "id", "year")

    def test_multiple_units(self):
        df = pd.DataFrame({
            "id": [1, 1, 2, 2],
            "year": [2000, 2001, 2000, 2001],
            "y": [10.0, 12.0, 5.0, 6.0],
        })
        result = panel2cs2(df, "y", "id", "year")
        result = result.dropna()
        diffs = set(result["dy"].values)
        assert 2.0 in diffs
        assert 1.0 in diffs

    def test_dy_column_added(self):
        df = pd.DataFrame({"id": [1, 1], "year": [2000, 2001], "y": [3.0, 7.0]})
        result = panel2cs2(df, "y", "id", "year")
        assert "dy" in result.columns
        assert "y0" in result.columns
        assert "y1" in result.columns


class TestTorF:
    def test_passes_through_bool_array(self):
        arr = np.array([True, False, True])
        result = TorF(arr)
        np.testing.assert_array_equal(result, arr)

    def test_raises_for_int_numpy_array(self):
        arr = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            TorF(arr)

    def test_raises_for_float_numpy_array(self):
        arr = np.array([1.0, 0.0])
        with pytest.raises(ValueError):
            TorF(arr)

    def test_raises_for_plain_list(self):
        with pytest.raises((ValueError, AttributeError)):
            TorF([True, False])

    def test_all_true(self):
        arr = np.array([True, True, True])
        result = TorF(arr)
        assert result.all()

    def test_all_false(self):
        arr = np.array([False, False, False])
        result = TorF(arr)
        assert not result.any()


class TestMultiplierBootstrap:
    def test_output_shape(self):
        np.random.seed(42)
        inf_func = np.random.randn(100, 5)
        result = multiplier_bootstrap(inf_func, 200)
        assert result.shape == (200, 5)

    def test_zero_influence_func_gives_zero_output(self):
        inf_func = np.zeros((50, 3))
        result = multiplier_bootstrap(inf_func, 100)
        np.testing.assert_array_equal(result, np.zeros((100, 3)))

    def test_single_column(self):
        np.random.seed(0)
        inf_func = np.random.randn(200, 1)
        result = multiplier_bootstrap(inf_func, 50)
        assert result.shape == (50, 1)

    def test_output_mean_near_zero_for_zero_mean_input(self):
        np.random.seed(7)
        n, K, biters = 1000, 4, 500
        inf_func = np.random.randn(n, K)  # zero-mean columns
        result = multiplier_bootstrap(inf_func, biters)
        # bootstrap means should be close to zero
        assert np.abs(result.mean(axis=0)).max() < 0.15

    def test_respects_biters_parameter(self):
        np.random.seed(1)
        inf_func = np.random.randn(50, 2)
        for biters in [10, 100, 500]:
            result = multiplier_bootstrap(inf_func, biters)
            assert result.shape[0] == biters
