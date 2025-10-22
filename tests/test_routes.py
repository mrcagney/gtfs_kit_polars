import itertools as it

import folium as fl
import polars as pl
import polars_st as st
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from gtfs_kit_polars import constants as cs
from gtfs_kit_polars import routes as gkr
from gtfs_kit_polars import helpers as gkh

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_shapeless,
    cairns_trip_stats,
    gtfs_kit_polars,
)

sample = gtfs_kit_polars.read_feed(DATA_DIR / "sample_gtfs_2.zip", dist_units="km")


def test_get_routes():
    feed = cairns.copy()
    date = cairns_dates[0]
    f = gkr.get_routes(feed, date).collect()
    # Should have the correct height
    assert f.height <= feed.routes.collect().height
    # Should have correct columns
    assert set(f.columns) == set(feed.routes.collect().columns)

    g = gkr.get_routes(feed, date, "07:30:00").collect()
    # Should have the correct height
    assert g.height <= f.height
    # Should have correct columns
    assert set(g.columns) == set(f.columns)

    # Test geo options
    feed = cairns.copy()
    g = gkr.get_routes(feed, as_geo=True, use_utm=True).collect()
    assert gkh.get_srid(g) != cs.WGS84

    g = gkr.get_routes(feed, as_geo=True, split_directions=True).collect()
    assert gkh.get_srid(g) == cs.WGS84
    assert (
        g.height
        == feed.trips.select("route_id", "direction_id").unique().collect().height
    )

    with pytest.raises(ValueError):
        gkr.get_routes(cairns_shapeless, as_geo=True)

    # Test written by Gilles Cuyaubere
    feed = gtfs_kit_polars.Feed(dist_units="km")
    feed.agency = pl.LazyFrame(
        {"agency_id": ["agency_id_0"], "agency_name": ["agency_name_0"]}
    )
    feed.routes = pl.LazyFrame(
        {
            "route_id": ["route_id_0"],
            "agency_id": ["agency_id_0"],
            "route_short_name": [None],
            "route_long_name": ["route_long_name_0"],
            "route_desc": [None],
            "route_type": [1],
            "route_url": [None],
            "route_color": [None],
            "route_text_color": [None],
        }
    )
    feed.trips = pl.LazyFrame(
        {
            "route_id": ["route_id_0"],
            "service_id": ["service_id_0"],
            "trip_id": ["trip_id_0"],
            "trip_headsign": [None],
            "trip_short_name": [None],
            "direction_id": [None],
            "block_id": [None],
            "wheelchair_accessible": [None],
            "bikes_allowed": [None],
            "trip_desc": [None],
            "shape_id": ["shape_id_0"],
        }
    )
    feed.shapes = pl.LazyFrame(
        {
            "shape_id": ["shape_id_0", "shape_id_0"],
            "shape_pt_lon": [2.36, 2.37],
            "shape_pt_lat": [48.82, 48.82],
            "shape_pt_sequence": [0, 1],
        }
    )
    feed.stops = pl.LazyFrame(
        {
            "stop_id": ["stop_id_0", "stop_id_1"],
            "stop_name": ["stop_name_0", "stop_name_1"],
            "stop_desc": [None, None],
            "stop_lat": [48.82, 48.82],
            "stop_lon": [2.36, 2.37],
            "zone_id": [None, None],
            "stop_url": [None, None],
            "location_type": [0, 0],
            "parent_station": [None, None],
            "wheelchair_boarding": [None, None],
        }
    )
    feed.stop_times = pl.LazyFrame(
        {
            "trip_id": ["trip_id_0", "trip_id_0"],
            "arrival_time": ["11:40:00", "11:45:00"],
            "departure_time": ["11:40:00", "11:45:00"],
            "stop_id": ["stop_id_0", "stop_id_1"],
            "stop_sequence": [0, 1],
            "stop_time_desc": [None, None],
            "pickup_type": [None, None],
            "drop_off_type": [None, None],
        }
    )

    g = gkr.get_routes(feed, as_geo=True)
    assert gkh.get_srid(g) == cs.WGS84

    # Turning a route shapes into point or None geometries
    # should yield empty route geometries and not raise an error
    rid = feed.routes.collect()["route_id"].item(0)
    shids = feed.trips.filter(pl.col("route_id") == rid).collect()["shape_id"].to_list()
    feed.shapes = feed.shapes.filter(pl.col("shape_id").is_in(shids)).unique("shape_id")
    assert (
        feed.get_routes(as_geo=True)
        .filter(pl.col("route_id") == rid)
        .with_columns(is_empty=st.geom().st.is_empty())
        .collect()["is_empty"]
        .all()
    )


def test_build_route_timetable():
    feed = cairns.copy()
    route_id = feed.routes.head().collect()["route_id"].item(0)
    dates = cairns_dates
    f = gkr.build_route_timetable(feed, route_id, dates).collect()

    # Should have the correct columns
    expect_cols = (
        set(feed.trips.collect_schema().names())
        | set(feed.stop_times.collect_schema().names())
        | {"date"}
    )
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert set(f["date"].to_list()) == set(dates)

    # Empty check
    f = gkr.build_route_timetable(feed, route_id, ["19990101"]).collect()
    assert gkh.is_empty(f)


def test_routes_to_geojson():
    feed = cairns.copy()
    n = 3
    route_ids = feed.routes.head(n).collect()["route_id"].to_list()
    route_short_names = (
        feed.routes.filter(pl.col("route_id") == route_ids[-1])
        .collect()["route_short_name"]
        .to_list()
    )
    route_ids = route_ids[: n - 1]
    gj = gkr.routes_to_geojson(
        feed, route_ids=route_ids, route_short_names=route_short_names
    )
    assert len(gj["features"]) == n

    gj = gkr.routes_to_geojson(
        feed,
        route_ids=route_ids,
        route_short_names=route_short_names,
        include_stops=True,
    )
    k = (
        feed.stop_times.join(feed.trips, "trip_id")
        .filter(pl.col("route_id").is_in(route_ids))
        .select("stop_id")
        .unique()
        .collect()
        .height
    )
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        gkr.routes_to_geojson(cairns_shapeless)

    gj = gkr.routes_to_geojson(cairns, route_ids=["bingo"])
    assert len(gj["features"]) == 0


def test_map_routes():
    feed = cairns.copy()
    rids = feed.routes.head(1).collect()["route_id"].to_list()
    rsns = feed.routes.head(2).collect()["route_short_name"].to_list()[-2:]
    m = gkr.map_routes(feed, route_ids=rids, route_short_names=rsns, show_stops=True)
    assert isinstance(m, fl.Map)

    with pytest.raises(ValueError):
        gkr.map_routes(feed)


def test_compute_route_stats_0():
    ts1 = cairns_trip_stats
    ts2 = ts1.with_columns(direction_id=pl.lit(None, dtype=pl.Int8))
    for ts, split_directions in it.product([ts1, ts2], [True, False]):
        if (
            split_directions
            and ts.select(pl.col("direction_id").is_null().all()).item()
        ):
            # Should raise an error
            with pytest.raises(ValueError):
                gkr.compute_route_stats_0(
                    ts, split_directions=split_directions
                ).collect()
            continue

        rs = gkr.compute_route_stats_0(ts, split_directions=split_directions).collect()

        # Should be a table of the correct shape
        n = ts1["route_id"].n_unique()
        N = (2 * n) if split_directions else n
        assert rs.height <= N

        # Should contain the correct columns
        expect_cols = {
            "route_id",
            "route_short_name",
            "route_type",
            "num_trips",
            "num_trip_ends",
            "num_trip_starts",
            "num_stop_patterns",
            "is_loop",
            "start_time",
            "end_time",
            "max_headway",
            "min_headway",
            "mean_headway",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "mean_trip_distance",
            "mean_trip_duration",
        }
        if split_directions:
            expect_cols.add("direction_id")
        else:
            expect_cols.add("is_bidirectional")

        assert set(rs.columns) == expect_cols

    # Empty check (both split settings)
    for split_directions in (True, False):
        rs = gkr.compute_route_stats_0(
            pl.DataFrame(), split_directions=split_directions
        ).collect()
        assert rs.height == 0


def test_compute_route_stats():
    feed = cairns.copy()
    dates = cairns_dates + ["19990101"]
    n = 3
    rids = cairns_trip_stats["route_id"].unique().to_list()[:n]
    trip_stats = cairns_trip_stats.filter(pl.col("route_id").is_in(rids))

    for split_directions in [True, False]:
        rs = gkr.compute_route_stats(
            feed, dates, trip_stats, split_directions=split_directions
        ).collect()

        # Should have correct num rows
        N = 2 * n * len(dates) if split_directions else n * len(dates)
        assert rs.height <= N

        # Should contain the correct columns
        expect_cols = {
            "date",
            "route_id",
            "route_short_name",
            "route_type",
            "num_trips",
            "num_trip_ends",
            "num_trip_starts",
            "num_stop_patterns",
            "is_loop",
            "start_time",
            "end_time",
            "max_headway",
            "min_headway",
            "mean_headway",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "mean_trip_distance",
            "mean_trip_duration",
        }
        if split_directions:
            expect_cols.add("direction_id")
        else:
            expect_cols.add("is_bidirectional")

        assert set(rs.columns) == expect_cols

        # Should only contains valid dates
        set(rs["date"].to_list()) == set(cairns_dates)

        # Non-feed date should yield empty table
        rs = gkr.compute_route_stats(
            feed, ["19990101"], trip_stats, split_directions=split_directions
        ).collect()
        assert rs.is_empty()


def test_compute_route_time_series_0():
    ts1 = cairns_trip_stats
    ts2 = ts1.with_columns(direction_id=pl.lit(None, dtype=pl.Int8))
    nroutes = ts1["route_id"].n_unique()

    for ts, split_directions in it.product([ts1, ts2], [True, False]):
        if (
            split_directions
            and ts.select(pl.col("direction_id").is_null().all()).item()
        ):
            with pytest.raises(ValueError):
                gkr.compute_route_stats_0(
                    ts, split_directions=split_directions
                ).collect()
            continue

        rs = gkr.compute_route_stats_0(ts, split_directions=split_directions).collect()
        rts = gkr.compute_route_time_series_0(
            ts, split_directions=split_directions, num_minutes=60
        ).collect()

        # row count checks (24 hourly bins per route)
        if split_directions:
            assert rts.height >= nroutes * 24
        else:
            assert rts.height == nroutes * 24

        # column checks
        expect_cols = {
            "datetime",
            "route_id",
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_directions:
            expect_cols.add("direction_id")
        assert set(rts.columns) == expect_cols

        # per-route service distance consistency (only when not split)
        if not split_directions:
            routes = rs["route_id"].unique().to_list()
            for rid in routes:
                get = (
                    rts.filter(pl.col("route_id") == rid)
                    .select(pl.col("service_distance").sum())
                    .item()
                )
                expect = (
                    rs.filter(pl.col("route_id") == rid)
                    .select(pl.col("service_distance").sum())
                    .item()
                )
                assert get == pytest.approx(expect)

    # Empty check (both split settings)
    for split_directions in (True, False):
        rts = gkr.compute_route_time_series_0(
            pl.DataFrame(), split_directions=split_directions, num_minutes=60
        ).collect()
        assert rts.is_empty()


def test_compute_route_time_series():
    feed = cairns.copy()
    dates = cairns_dates
    rids = cairns_trip_stats["route_id"].unique().to_list()[:3]
    trip_stats = cairns_trip_stats.filter(pl.col("route_id").is_in(rids))

    for split_directions in [True, False]:
        rs = gkr.compute_route_stats(
            feed, dates, trip_stats, split_directions=split_directions
        ).collect()
        rts = gkr.compute_route_time_series(
            feed,
            dates + ["19990101"],  # Invalid last date
            trip_stats,
            split_directions=split_directions,
            num_minutes=12 * 60,
        ).collect()

        # Should have correct num rows
        n = rs.height
        if split_directions:
            assert rts.height >= n * 2
        else:
            # date span * num routes * num time chunks
            assert rts.height == n * 2

        # Should have correct columns
        expect_cols = {
            "datetime",
            "route_id",
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_directions:
            expect_cols |= {"direction_id"}
        assert set(rts.columns) == expect_cols

        # Each route have a correct num_trip_starts
        if not split_directions:
            routes = rs["route_id"].unique().to_list()
            for rid in routes:
                get = (
                    rts.filter(pl.col("route_id") == rid)
                    .select(pl.col("num_trip_starts").sum())
                    .item()
                )
                expect = (
                    rs.filter(pl.col("route_id") == rid)
                    .select(pl.col("num_trip_starts").sum())
                    .item()
                )
                assert get == pytest.approx(expect)

        # Non-feed dates should yield empty table
        rts = gkr.compute_route_time_series(
            feed, ["19990101"], trip_stats, split_directions=split_directions
        ).collect()
        assert rts.is_empty()
