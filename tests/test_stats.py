import math

import pytest

from servebench import stats


def test_percentile_matches_linear_interpolation():
    data = [10, 20, 30, 40, 50]
    assert stats.percentile(data, 0) == 10
    assert stats.percentile(data, 50) == 30
    assert stats.percentile(data, 100) == 50
    assert stats.percentile(data, 25) == 20
    assert stats.percentile(data, 90) == pytest.approx(46.0)


def test_percentile_of_unsorted_and_single_and_empty():
    assert stats.percentile([5, 1, 3], 50) == 3
    assert stats.percentile([7], 99) == 7
    assert stats.percentile([], 50) is None


def test_percentile_rejects_out_of_range_q():
    with pytest.raises(ValueError):
        stats.percentile([1, 2], 101)
    with pytest.raises(ValueError):
        stats.percentile([1, 2], -1)


def test_p99_of_100_values_is_between_the_top_two():
    data = list(range(1, 101))
    assert stats.percentile(data, 99) == pytest.approx(99.01)


def test_summarize_fields():
    s = stats.summarize([1, 2, 3, 4])
    assert s["n"] == 4
    assert s["mean"] == 2.5
    assert s["min"] == 1 and s["max"] == 4
    assert s["p50"] == 2.5
    assert stats.summarize([])["n"] == 0
    assert stats.summarize([])["p50"] is None


def test_median_range_odd_even_and_none_values_dropped():
    r = stats.median_range([3, 1, 2])
    assert (r["median"], r["min"], r["max"], r["n"]) == (2, 1, 3, 3)
    r = stats.median_range([4, 1, 2, 3])
    assert r["median"] == 2.5
    r = stats.median_range([1, None, 3])
    assert r["n"] == 2 and r["median"] == 2
    assert stats.median_range([])["median"] is None


def test_fmt_range():
    assert stats.fmt_range(stats.median_range([1.0, 2.0, 3.0])) == "2.0 (1.0-3.0)"
    assert stats.fmt_range(stats.median_range([]), 1) == "-"
    assert stats.fmt_range(stats.median_range([0.123, 0.456, 0.2]), 3) == "0.200 (0.123-0.456)"


def test_wilson_interval_known_values():
    lo, hi = stats.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert stats.wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = stats.wilson_interval(0, 10)
    assert lo == 0.0 and hi < 0.35


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert math.isclose(stats.mcnemar_exact(3, 9), stats.mcnemar_exact(9, 3))


def test_pooled():
    assert stats.pooled([[1, 2], [3], []]) == [1, 2, 3]
