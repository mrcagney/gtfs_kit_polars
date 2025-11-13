import numpy as np
import pandas as pd
import polars as pl
import pytest

from gtfs_kit_polars import stop_times as gks

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_trip_stats,
    gtfs_kit_polars,
    sample,
)


def test_get_stop_times():
    feed = cairns.copy()
    date = cairns_dates[0]
    f = gks.get_stop_times(feed, date).collect()
    # Should have a reasonable shape
    assert f.height <= feed.stop_times.collect().height
    # Should have correct columns
    assert set(f.columns) == set(feed.stop_times.collect_schema().names())


def test_get_start_and_end_times():
    feed = cairns.copy()
    date = cairns_dates[0]
    st = gks.get_stop_times(feed, date).collect()
    times = gks.get_start_and_end_times(feed, date)
    # Should be strings
    for t in times:
        assert isinstance(t, str)
        # Should lie in stop times
        assert (
            t
            in st.to_pandas()[["departure_time", "arrival_time"]]
            .dropna()
            .values.flatten()
        )

    # Should get null times in some cases
    times = gks.get_start_and_end_times(feed, "19690711")
    for t in times:
        assert t is None
    feed.stop_times = feed.stop_times.with_columns(departure_time=pl.lit(None))
    times = gks.get_start_and_end_times(feed)
    assert times[0] is None


def test_stop_times_to_geojson():
    feed = cairns.copy()
    trip_ids = feed.trips.head(2).collect()["trip_id"].to_list()
    gj = gks.stop_times_to_geojson(feed, trip_ids)
    assert isinstance(gj, dict)

    n = (
        feed.stop_times.filter(pl.col("trip_id").is_in(trip_ids))
        .unique(subset=["trip_id", "stop_id"])
        .collect()
        .height
    )
    assert len(gj["features"]) == n

    gj = gks.stop_times_to_geojson(feed, ["bingo"])
    assert len(gj["features"]) == 0


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_append_dist_to_stop_times():
    feed1 = cairns.copy()
    st1 = feed1.stop_times.collect()
    feed2 = gks.append_dist_to_stop_times(feed1)
    st2 = feed2.stop_times.collect()

    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = set(st1.columns) | {"shape_dist_traveled"}
    cols2 = set(st2.columns)
    assert cols1 == cols2

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for group in st2.partition_by("trip_id"):
        sdt = group.sort("stop_sequence")["shape_dist_traveled"].to_list()
        if any(sdt):
            assert sdt == sorted(sdt)

    # Trips with no shapes should have null distances
    shape_id = feed1.shapes.collect()["shape_id"][0]
    trip_id = feed1.trips.filter(pl.col("shape_id") == shape_id).collect()["trip_id"][0]
    feed1.trips = feed1.trips.with_columns(
        shape_id=pl.when(pl.col("shape_id") == shape_id)
        .then(None)
        .otherwise(pl.col("shape_id"))
    )
    feed2 = feed1.append_dist_to_stop_times()
    assert (
        feed2.stop_times.filter(pl.col("trip_id") == trip_id)
        .collect()["shape_dist_traveled"]
        .is_null()
        .all()
    )

    # Again, check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for group in feed2.stop_times.collect().partition_by("trip_id"):
        sdt = group.sort("stop_sequence")["shape_dist_traveled"].to_list()
        if any(sdt):
            assert sdt == sorted(sdt)
