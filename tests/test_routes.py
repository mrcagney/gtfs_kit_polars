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
    f = gkr.get_routes(feed, date)
    # Should have the correct shape
    assert f.shape[0] <= feed.routes.shape[0]
    assert f.shape[1] == feed.routes.shape[1]
    # Should have correct columns
    assert set(f.columns) == set(feed.routes.columns)

    g = gkr.get_routes(feed, date, "07:30:00")
    # Should have the correct shape
    assert g.shape[0] <= f.shape[0]
    assert g.shape[1] == f.shape[1]
    # Should have correct columns
    assert set(g.columns) == set(feed.routes.columns)

    # Test GDF options
    feed = cairns.copy()
    g = gkr.get_routes(feed, as_geo=True, use_utm=True)
    assert isinstance(g, gpd.GeoDataFrame)
    assert g.crs != cs.WGS84

    g = gkr.get_routes(feed, as_geo=True, split_directions=True)
    assert g.crs == cs.WGS84
    assert (
        g.shape[0]
        == feed.trips[["route_id", "direction_id"]].drop_duplicates().shape[0]
    )

    with pytest.raises(ValueError):
        gkr.get_routes(cairns_shapeless, as_geo=True)

    # Test written by Gilles Cuyaubere
    feed = gtfs_kit_polars.Feed(dist_units="km")
    feed.agency = pd.DataFrame(
        {"agency_id": ["agency_id_0"], "agency_name": ["agency_name_0"]}
    )
    feed.routes = pd.DataFrame(
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
    feed.trips = pd.DataFrame(
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
    feed.shapes = pd.DataFrame(
        {
            "shape_id": ["shape_id_0", "shape_id_0"],
            "shape_pt_lon": [2.36, 2.37],
            "shape_pt_lat": [48.82, 48.82],
            "shape_pt_sequence": [0, 1],
        }
    )
    feed.stops = pd.DataFrame(
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
    feed.stop_times = pd.DataFrame(
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
    assert g.crs == cs.WGS84

    # Turning a route's shapes into point geometries,
    # should yield an empty route geometry and should not throw an error
    rid = feed.routes["route_id"].iat[0]
    shids = feed.trips.loc[lambda x: x["route_id"] == rid, "shape_id"]
    f0 = feed.shapes.loc[lambda x: x["shape_id"].isin(shids)].drop_duplicates(
        "shape_id"
    )
    f1 = feed.shapes.loc[lambda x: ~x["shape_id"].isin(shids)]
    feed.shapes = pd.concat([f0, f1])
    assert (
        feed.get_routes(as_geo=True)
        .loc[lambda x: x["route_id"] == rid, "geometry"]
        .iat[0]
        is None
    )


def test_build_route_timetable():
    feed = sample.copy()
    route_id = feed.routes["route_id"].values[0]
    dates = feed.get_first_week()[:2]
    f = gkr.build_route_timetable(feed, route_id, dates)

    # Should have the correct columns
    expect_cols = set(feed.trips.columns) | set(feed.stop_times.columns) | set(["date"])
    assert set(f.columns) == expect_cols

    # Should only have feed dates
    assert f.date.unique().tolist() == dates

    # Empty check
    f = gkr.build_route_timetable(feed, route_id, dates[2:])
    assert f.is_empty()


def test_routes_to_geojson():
    feed = cairns.copy()
    route_ids = feed.routes.route_id.loc[:1]
    n = len(route_ids)

    gj = gkr.routes_to_geojson(feed, route_ids)
    assert len(gj["features"]) == n

    gj = gkr.routes_to_geojson(feed, route_ids, include_stops=True)
    k = (
        feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
        .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
        .nunique()
    )
    assert len(gj["features"]) == n + k

    with pytest.raises(ValueError):
        gkr.routes_to_geojson(cairns_shapeless)

    with pytest.raises(ValueError):
        gkr.routes_to_geojson(cairns, route_ids=["bingo"])


def test_map_routes():
    feed = cairns.copy()
    rids = feed.routes["route_id"].iloc[:1]
    rsns = feed.routes["route_short_name"].iloc[-2:]
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
    dates = cairns_dates  # Spans three consecutive dates
    n = 3
    rids = cairns_trip_stats["route_id"].unique().to_list()[:n]
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
        if split_directions:
            assert rts.shape[0] >= len(dates) * n * 2
        else:
            # date span * num routes * num time chunks
            assert rts.shape[0] == len(dates) * n * 2

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
