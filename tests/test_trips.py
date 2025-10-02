import pandas as pd
import numpy as np
import pytest
import polars as pl
import geopandas as gpd
import folium as fl

from .context import (
    gtfs_kit_polars,
    DATA_DIR,
    cairns,
    cairns_shapeless,
    cairns_dates,
    cairns_trip_stats,
)
from gtfs_kit_polars import constants as cs
from gtfs_kit_polars import trips as gkt
from gtfs_kit_polars import calendar as gkc
from gtfs_kit_polars import helpers as gkh
from gtfs_kit_polars import stop_times as gks


def test_get_active_services():
    feed = cairns
    week = feed.get_first_week()
    s = gkt.get_active_services(feed, week[0])
    assert set() < set(s) < set(feed.trips.collect()["service_id"])

    # Should work with only one of `feed.calendar` and `feed.calendar_dates`
    for table in ["calendar", "calendar_dates"]:
        feed = cairns.copy()
        setattr(feed, table, None)
        week = feed.get_first_week()
        s = gkt.get_active_services(feed, week[0])
        assert set() < set(s) < set(feed.trips.collect()["service_id"])


def test_get_trips():
    feed = cairns.copy()
    date = cairns_dates[0]
    trips1 = gkt.get_trips(feed, date).collect()
    # Should have correct num rows
    assert trips1.height <= feed.trips.collect().height
    # Should have correct columns
    assert set(trips1.columns) == set(feed.trips.collect_schema().names())

    trips2 = gkt.get_trips(feed, date, "07:30:00").collect()
    # Should have the correct num rows
    assert trips2.height <= trips2.height
    # Should have correct columns
    assert set(trips2.columns) == set(feed.trips.collect_schema().names())

    feed = cairns.copy()
    g = gkt.get_trips(feed, as_geo=True).collect()
    assert gkh.get_srid(g) == cs.WGS84
    assert g.height == feed.trips.collect().height
    assert set(g.columns) == set(feed.trips.collect_schema().names()) | {"geometry"}

    with pytest.raises(ValueError):
        gkt.get_trips(cairns_shapeless, as_geo=True)


def test_compute_trip_activity():
    feed = cairns.copy()
    dates = gkc.get_first_week(feed)
    ta = gkt.compute_trip_activity(feed, dates + ["19990101"]).collect()
    # Should have the correct num rows and correct columns()
    assert ta.height == feed.trips.collect().height
    assert set(ta.columns) == {"trip_id"} | set(dates)
    # Date columns should contain only zeros and ones
    assert (
        ta.unpivot(index=[], on=dates, value_name="v")
        .select(pl.col("v").is_in([0, 1]).all())
        .item()
    )


def test_compute_busiest_date():
    feed = cairns.copy()
    dates = gkc.get_first_week(feed)[:1]
    date = gkt.compute_busiest_date(feed, dates + ["999"])
    # Busiest day should lie in first week
    assert date in dates


def test_name_stop_patterns():
    feed = cairns.copy()
    t = gkt.name_stop_patterns(feed).collect()
    expect_cols = set(feed.trips.collect_schema().names()) | {"stop_pattern_name"}
    assert set(t.columns) == expect_cols

    # Should still work without direction ID
    feed.trips = feed.trips.drop("direction_id")
    t = gkt.name_stop_patterns(feed)
    assert set(t.columns) == expect_cols - {"direction_id"}


def test_compute_trip_stats():
    feed = cairns.copy()
    n = 3
    rids = feed.routes.collect()["route_id"].to_list()[:n]
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids).collect()

    # Should have correct number of rows
    trips = feed.trips.filter(pl.col("route_id").is_in(rids)).collect()
    assert trip_stats.height == trips.height

    # Should contain the correct columns
    expect_cols = {
        "trip_id",
        "direction_id",
        "route_id",
        "route_short_name",
        "route_type",
        "shape_id",
        "stop_pattern_name",
        "num_stops",
        "start_time",
        "end_time",
        "start_stop_id",
        "end_stop_id",
        "distance",
        "duration",
        "speed",
        "is_loop",
    }
    assert set(trip_stats.columns) == expect_cols

    # Shapeless feeds should have null entries for distance column
    feed = cairns_shapeless.copy()
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids).collect()
    assert trip_stats["distance"].n_unique() == 1
    assert trip_stats["distance"].unique().item(0) is None

    # Should contain the correct trips
    assert set(trip_stats["trip_id"].to_list()) == set(trips["trip_id"].to_list())

    # Missing the optional ``direction_id`` column in ``feed.trips``
    # should give ``direction_id`` column in stats with all NaNs
    feed = cairns.copy()
    feed.trips = feed.trips.drop("direction_id")
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids).collect()
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats["direction_id"].is_null().all()

    # Missing the optional ``shape_id`` column in ``feed.trips``
    # should give ``shape_id`` column in stats with all NaNs
    feed = cairns.copy()
    feed.trips = feed.trips.drop("shape_id")
    trip_stats = gkt.compute_trip_stats(feed, route_ids=rids).collect()
    assert set(trip_stats.columns) == expect_cols
    assert trip_stats["shape_id"].is_null().all()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_locate_trips():
    feed = cairns.copy()
    feed = gks.append_dist_to_stop_times(feed)
    date = cairns_dates[0]
    times = ["08:00:00"]
    f = gkt.locate_trips(feed, date, times).collect()

    g = gkt.get_trips(feed, date, times[0]).collect()
    # Should have the correct number of rows
    assert f.height == g.height
    # Should have the correct columns
    expect_cols = {
        "route_id",
        "trip_id",
        "direction_id",
        "shape_id",
        "time",
        "rel_dist",
        "lon",
        "lat",
    }
    assert set(f.columns) == expect_cols

    # Missing feed.trips.shape_id should raise an error
    feed = cairns_shapeless.copy()
    feed.trips = feed.trips.drop("shape_id")
    with pytest.raises(ValueError):
        gkt.locate_trips(feed, date, times)


def test_trips_to_geojson():
    feed = cairns.copy()
    trip_ids = feed.trips.trip_id.loc[:1]
    n = len(trip_ids)
    gj = gkt.trips_to_geojson(feed, trip_ids)
    assert len(gj["features"]) == n

    gj = gkt.trips_to_geojson(feed, trip_ids, include_stops=True)
    k = (
        feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)]
        .drop_duplicates(subset=["trip_id", "stop_id"])
        .shape[0]
    )
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        gkt.trips_to_geojson(cairns_shapeless)

    with pytest.raises(ValueError):
        gkt.trips_to_geojson(cairns, trip_ids=["bingo"])


def test_map_trips():
    feed = cairns.copy()
    tids = feed.trips["trip_id"].values[:2]
    m = gkt.map_trips(feed, tids, show_stops=True)
    assert isinstance(m, fl.Map)
