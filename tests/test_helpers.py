import datetime as dt

import numpy as np
import pandas as pd
import polars as pl
import pytest
import shapely.geometry as sg
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from gtfs_kit_polars import helpers as gkh

from .context import cairns, cairns_dates, cairns_trip_stats, gtfs_kit_polars


def test_timestr_to_seconds_0():
    timestr1 = "01:01:01"
    seconds1 = 3600 + 60 + 1
    timestr2 = "25:01:01"
    assert gkh.timestr_to_seconds_0(timestr1) == seconds1
    assert gkh.timestr_to_seconds_0(timestr2, mod24=True) == seconds1
    # Test error handling
    assert gkh.timestr_to_seconds_0(seconds1) is None


def test_seconds_to_timestr_0():
    timestr1 = "01:01:01"
    seconds1 = 3600 + 60 + 1
    timestr2 = "25:01:01"
    seconds2 = 25 * 3600 + 60 + 1
    assert gkh.seconds_to_timestr_0(seconds1) == timestr1
    assert gkh.seconds_to_timestr_0(seconds2) == timestr2
    assert gkh.seconds_to_timestr_0(seconds2, mod24=True) == timestr1
    assert gkh.seconds_to_timestr_0(timestr1) is None


def test_timestr_to_seconds():
    df = pl.DataFrame({
        "t": ["00:00:00", "01:02:03", "24:00:00", "27:15:30"]
    })
    # No mod24
    out = df.with_columns(sec=gkh.timestr_to_seconds("t")).to_dict(as_series=False)["sec"]
    assert out == [0, 3723, 86400, 98130]

    # With mod24 -> wrap at 24h
    out_mod = df.with_columns(sec=gkh.timestr_to_seconds("t", mod24=True)).to_dict(as_series=False)["sec"]
    assert out_mod == [0, 3723, 0, 11730]  # 98130 % 86400 = 11730


def test_seconds_to_timestr():
    df = pl.DataFrame({
        "s": [0, 3723, 86400, 98130]  # 27:15:30
    })
    # No mod24
    out = df.with_columns(t=gkh.seconds_to_timestr("s")).to_dict(as_series=False)["t"]
    assert out == ["00:00:00", "01:02:03", "24:00:00", "27:15:30"]

    # With mod24 -> wrap at 24h
    out_mod = df.with_columns(t=gkh.seconds_to_timestr("s", mod24=True)).to_dict(as_series=False)["t"]
    assert out_mod == ["00:00:00", "01:02:03", "00:00:00", "03:15:30"]


def test_datestr_to_date():
    datestr = "20140102"
    date = dt.date(2014, 1, 2)
    assert gkh.datestr_to_date(datestr) == date


def test_date_to_datestr():
    datestr = "20140102"
    date = dt.date(2014, 1, 2)
    assert gkh.date_to_datestr(date) == datestr


def test_replace_date():
    # original datetimes with different dates but distinct times
    dts = [
        dt.datetime(2024, 5, 1, 8, 30, 0),
        dt.datetime(2024, 5, 2, 9, 45, 15),
    ]
    df = pl.DataFrame({"datetime": dts})

    # target date (YYYYMMDD)
    target = "20250102"

    # Eager
    out_df = gkh.replace_date(df, target)
    assert out_df.dtypes == df.dtypes
    assert out_df.height == 2
    assert out_df["datetime"][0] == dt.datetime(2025, 1, 2, 8, 30, 0)
    assert out_df["datetime"][1] == dt.datetime(2025, 1, 2, 9, 45, 15)

    # Lazy
    out_lf = gkh.replace_date(df.lazy(), target)
    collected = out_lf.collect()
    assert collected["datetime"][0] == dt.datetime(2025, 1, 2, 8, 30, 0)
    assert collected["datetime"][1] == dt.datetime(2025, 1, 2, 9, 45, 15)

def test_is_metric():
    assert gkh.is_metric("m")
    assert gkh.is_metric("km")
    assert not gkh.is_metric("ft")
    assert not gkh.is_metric("mi")
    assert not gkh.is_metric("bingo")


def test_get_convert_dist():
    di = "mi"
    do = "km"
    fn = gkh.get_convert_dist(di, do)
    f = pl.DataFrame({"dist": [1]})
    assert f.select(dist=fn("dist"))["dist"].to_list() == [1.609_344]


def test_get_max_runs():
    x = [7, 1, 2, 7, 7, 1, 2]
    get = gkh.get_max_runs(x)
    expect = np.array([[0, 1], [3, 5]])
    assert_array_equal(get, expect)


def test_are_equal():
    f = pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert gkh.are_equal(f, f)

    g = pl.DataFrame({"b": [4, 2], "a": [3, 1]})
    assert gkh.are_equal(f, g)

    h = pl.DataFrame({"a": [1, 5], "b": [2, 4]})
    assert not gkh.are_equal(f, h)

    h = pl.DataFrame({})
    assert not gkh.are_equal(f, h)

def test_is_not_null():
    c = "foo"

    f = pl.DataFrame(schema={"bar": pl.Int64, c: pl.Int64})
    assert not gkh.is_not_null(f, c)

    f = pl.DataFrame({"bar": [1], c: [None]})
    assert not gkh.is_not_null(f, c)

    f = pl.DataFrame({"bar": [1, 2], c: [None, 2]})
    assert gkh.is_not_null(f, c)


def test_longest_subsequence():
    dates = [
        ("2015-02-03", "name1"),
        ("2015-02-04", "nameg"),
        ("2015-02-04", "name5"),
        ("2015-02-05", "nameh"),
        ("1929-03-12", "name4"),
        ("2023-07-01", "name7"),
        ("2015-02-07", "name0"),
        ("2015-02-08", "nameh"),
        ("2015-02-15", "namex"),
        ("2015-02-09", "namew"),
        ("1980-12-23", "name2"),
        ("2015-02-12", "namen"),
        ("2015-02-13", "named"),
    ]

    assert gkh.longest_subsequence(dates, "weak") == [
        ("2015-02-03", "name1"),
        ("2015-02-04", "name5"),
        ("2015-02-05", "nameh"),
        ("2015-02-07", "name0"),
        ("2015-02-08", "nameh"),
        ("2015-02-09", "namew"),
        ("2015-02-12", "namen"),
        ("2015-02-13", "named"),
    ]

    from operator import itemgetter

    assert gkh.longest_subsequence(dates, "weak", key=itemgetter(0)) == [
        ("2015-02-03", "name1"),
        ("2015-02-04", "nameg"),
        ("2015-02-04", "name5"),
        ("2015-02-05", "nameh"),
        ("2015-02-07", "name0"),
        ("2015-02-08", "nameh"),
        ("2015-02-09", "namew"),
        ("2015-02-12", "namen"),
        ("2015-02-13", "named"),
    ]

    indices = set(gkh.longest_subsequence(dates, key=itemgetter(0), index=True))
    assert [e for i, e in enumerate(dates) if i not in indices] == [
        ("2015-02-04", "nameg"),
        ("1929-03-12", "name4"),
        ("2023-07-01", "name7"),
        ("2015-02-15", "namex"),
        ("1980-12-23", "name2"),
    ]


def test_make_ids():
    assert gkh.make_ids(10, "s") == [
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
    ]
    assert gkh.make_ids(11, "s") == [
        "s00",
        "s01",
        "s02",
        "s03",
        "s04",
        "s05",
        "s06",
        "s07",
        "s08",
        "s09",
        "s10",
    ]


def test_combine_time_series():
    # Minute-level index (two bins)
    t0 = dt.datetime(2025, 1, 1, 0, 0, 0)
    t1 = t0 + dt.timedelta(minutes=1)
    idx = [t0, t1]


    # Indicators (wide frames: columns are entities + 'datetime')
    num_trips = pl.DataFrame({
        "datetime": idx,
        "R1-0": [1, 2],
        "R1-1": [0, 1],
    })
    num_trip_starts = pl.DataFrame({
        "datetime": idx,
        "R1-0": [1, 0],
        "R1-1": [None, 1],  # will coerce None->0
    })
    num_trip_ends = pl.DataFrame({
        "datetime": idx,
        "R1-0": [0, 1],
        "R1-1": [0, 0],
    })
    # Distance/duration chosen so speed = 2.0 where duration>0
    service_distance = pl.DataFrame({
        "datetime": idx,
        "R1-0": [0.5, 0.5],
        "R1-1": [0.0, 1.0],
    })
    service_duration = pl.DataFrame({
        "datetime": idx,
        "R1-0": [0.25, 0.25],
        "R1-1": [0.0, 0.5],
    })

    series_by_indicator = {
        "num_trips": num_trips,
        "num_trip_starts": num_trip_starts,
        "num_trip_ends": num_trip_ends,
        "service_distance": service_distance,
        "service_duration": service_duration,
    }

    out_lf = gkh.combine_time_series(
        series_by_indicator,
        kind="route",
        split_directions=True,
    )
    out = out_lf.collect()

    # Expected rows: 2 timestamps * 2 directions = 4
    assert out.height == 4

    # Required columns & ordering
    expected_front = ["datetime", "route_id", "direction_id"]
    expected_inds = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
        "service_speed",
    ]
    for c in expected_front + expected_inds:
        assert c in out.columns

    # Sorted by datetime then route_id then direction_id
    out_sorted = out.sort(["datetime", "route_id", "direction_id"])
    gkh.are_equal(out, out_sorted)

    # Direction split & ID decoding
    assert set(out.select("route_id").unique().to_series().to_list()) == {"R1"}
    assert set(out.select("direction_id").unique().to_series().to_list()) == {0, 1}

    # Nones coerced to zero in numeric indicators
    assert out.select(pl.col("num_trip_starts").min()).item() >= 0

    # Specifically check the None location: at t0, direction 1 had None -> 0
    row_t0_dir1 = out.filter(
        (pl.col("datetime") == t0) & (pl.col("direction_id") == 1)
    ).row(0, named=True)
    assert row_t0_dir1["num_trip_starts"] == 0

    # service_speed = distance / duration; when duration==0 => 0.0
    # Check a bin with duration>0 (should be 2.0): t1, dir 0 -> 0.5 / 0.25 = 2.0
    row_t1_dir0 = out.filter(
        (pl.col("datetime") == t1) & (pl.col("direction_id") == 0)
    ).row(0, named=True)
    assert pytest.approx(row_t1_dir0["service_speed"]) == 2.0

    # Check a bin with duration==0 (should be 0.0): t0, dir 1 -> 0.0 / 0.0 -> 0.0 after fill
    assert row_t0_dir1["service_speed"] == 0.0

def test_downsample():
    ts = cairns.compute_network_time_series(
        cairns_dates, num_minutes=60
    ).collect()
    f = gkh.downsample(ts, num_minutes=60).collect()
    assert ts.equals(f)

    f = gkh.downsample(ts, num_minutes=3*60).collect()

    # Should have correct num rows
    assert f.height == ts.height / 3
    # Should have correct frequency
    assert gkh.get_bin_size(f) == 3*60
