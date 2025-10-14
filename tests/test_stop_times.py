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
    f = gks.get_stop_times(feed, date)
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have a reasonable shape
    assert f.shape[0] <= feed.stop_times.shape[0]
    # Should have correct columns
    assert set(f.columns) == set(feed.stop_times.columns)


def test_get_start_and_end_times():
    feed = cairns.copy()
    date = cairns_dates[0]
    st = gks.get_stop_times(feed, date).collect()
    times = gks.get_start_and_end_times(feed, date)
    # Should be strings
    for t in times:
        assert isinstance(t, str)
        # Should lie in stop times
        assert t in st.to_pandas()[["departure_time", "arrival_time"]].dropna().values.flatten()

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
    st1 = feed1.stop_times
    feed2 = gks.append_dist_to_stop_times(feed1)
    st2 = feed2.stop_times

    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = set(st1.columns) | {"shape_dist_traveled"}
    cols2 = set(st2.columns)
    assert cols1 == cols2

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for trip, group in st2.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        sdt = group.shape_dist_traveled.values.tolist()
        assert sdt == sorted(sdt)

    # Trips with no shapes should have NaN distances
    trip_id = feed1.stop_times["trip_id"].iat[0]
    feed1.trips.loc[lambda x: x["trip_id"] == trip_id, "shape_id"] = np.nan

    feed2 = feed1.append_dist_to_stop_times()
    assert (
        feed2.stop_times.loc[lambda x: x["trip_id"] == trip_id, "shape_dist_traveled"]
        .isna()
        .all()
    )

    # Again, check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for trip, group in feed2.stop_times.groupby("trip_id"):
        group = group.sort_values("stop_sequence")
        sdt = group.shape_dist_traveled.values.tolist()
        assert sdt == sorted(sdt)
