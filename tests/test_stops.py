import itertools

import polars as pl
import folium as fl
import polars_st as st
import geopandas as gp
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gtfs_kit_polars import calendar as gkc
from gtfs_kit_polars import stops as gks
from gtfs_kit_polars import helpers as gkh


from .context import DATA_DIR, cairns, cairns_dates, gtfs_kit_polars

sample = gtfs_kit_polars.read_feed(DATA_DIR / "sample_gtfs_2.zip", dist_units="km")


def test_geometrize_stops():
    stops = cairns.stops
    geo_stops = gks.geometrize_stops(stops, use_utm=True).collect()
    # Should have correct height
    assert geo_stops.height == stops.collect().height
    # Should have correct columns
    assert set(geo_stops.columns) == (
        set(stops.collect_schema().names()) | {"geometry"}
    ) - {"stop_lon", "stop_lat"}


def test_ungeometrize_stops():
    stops = cairns.stops
    geo_stops = gks.geometrize_stops(stops)
    stops2 = gks.ungeometrize_stops(geo_stops)
    # Test columns are correct
    assert set(stops2.collect_schema().names()) == set(stops.collect_schema().names())
    # Data frames should be equal after sorting columns
    gkh.are_equal(stops, stops2)


def test_get_stops():
    feed = cairns.copy()
    date = cairns_dates[0]
    trip_ids = feed.trips.head(1).collect()["trip_id"].to_list()
    route_ids = feed.routes.head(1).collect()["route_id"].to_list()
    frames = [
        gks.get_stops(feed).collect(),
        gks.get_stops(feed, date=date).collect(),
        gks.get_stops(feed, trip_ids=trip_ids).collect(),
        gks.get_stops(feed, route_ids=route_ids).collect(),
        gks.get_stops(feed, date=date, trip_ids=trip_ids).collect(),
        gks.get_stops(feed, date=date, route_ids=route_ids).collect(),
        gks.get_stops(
            feed, date=date, trip_ids=trip_ids, route_ids=route_ids
        ).collect(),
    ]
    for f in frames:
        # Should have correct num rows
        assert f.height <= feed.stops.collect().height
        # Should have correct columns
        set(f.columns) == set(feed.stops.collect_schema().names())

    # Number of rows should be reasonable
    assert frames[0].height <= frames[1].height
    assert frames[2].height <= frames[4].height
    assert frames[4].height == frames[6].height

    g = gks.get_stops(feed, as_geo=True)
    assert gkh.get_srid(g) == 4326

    g = gks.get_stops(feed, as_geo=True, use_utm=True)
    assert gkh.get_srid(g) != 4326


def test_compute_stop_activity():
    feed = cairns.copy()
    dates = cairns_dates
    sa = gks.compute_stop_activity(feed, dates + ["19990101"]).collect()
    # Should have the correct height
    assert sa.height == feed.stops.collect().height
    # Should have correct columns
    assert set(sa.columns) == {"stop_id"} | set(dates)
    # Date columns should contain only zeros and ones
    assert (
        sa.unpivot(index=[], on=dates, value_name="v")
        .select(pl.col("v").is_in([0, 1]).all())
        .item()
    )


def test_build_stop_timetable():
    feed = cairns.copy()
    stop_id = feed.stops.head(1).collect()["stop_id"][0]
    dates = cairns_dates
    f = gks.build_stop_timetable(feed, stop_id, dates + ["19990101"]).collect()
    print(f)

    # Should have the correct columns
    assert set(f.columns) == set(feed.trips.collect().columns) | set(
        feed.stop_times.collect().columns
    ) | {"date"}

    # Should only have feed dates
    assert set(f["date"].unique()) == set(dates)

    # Empty check
    f = gks.build_stop_timetable(feed, stop_id, [])
    assert gkh.is_empty(f)


def test_build_geometry_by_stop():
    d = gks.build_geometry_by_stop(cairns)
    assert isinstance(d, dict)
    assert len(d) == cairns.stops.collect()["stop_id"].n_unique()


def test_stops_to_geojson():
    feed = cairns.copy()
    stop_ids = feed.stops.head(2).collect()["stop_id"].to_list()
    gj = gks.stops_to_geojson(feed, stop_ids)
    assert isinstance(gj, dict)
    assert len(gj["features"]) == len(stop_ids)

    gj = gks.stops_to_geojson(feed, ["bingo"])
    assert len(gj["features"]) == 0


def test_get_stops_in_area():
    feed = cairns.copy()
    area = st.read_file(DATA_DIR / "cairns_square_stop_750070.geojson")
    stops = gks.get_stops_in_area(feed, area).collect()
    assert stops["stop_id"].to_list() == ["750070"]


def test_map_stops():
    feed = cairns.copy()
    m = gks.map_stops(feed, feed.stops.head(5).collect()["stop_id"].to_list())
    assert isinstance(m, fl.Map)


def test_compute_stop_stats_0():
    feed = cairns.copy()
    stop_times = feed.stop_times.head(250)
    trips1 = feed.trips
    # Set some null directions
    trips2 = trips1.with_columns(pl.lit(None, dtype=pl.Int32).alias("direction_id"))

    for trips, split_directions in itertools.product(
        [trips1, trips2],
        [True, False],
    ):
        if (
            split_directions
            and trips.select(pl.col("direction_id").is_null().all()).collect().item()
        ):
            # Should raise an error when all direction_id are NULL and we split by direction
            with pytest.raises(ValueError):
                gks.compute_stop_stats_0(
                    stop_times, trips, split_directions=split_directions
                )
            continue

        stop_stats = gks.compute_stop_stats_0(
            stop_times, trips, split_directions=split_directions
        ).collect()

        # Should have the correct columns
        expect_cols = {
            "stop_id",
            "num_routes",
            "num_trips",
            "max_headway",
            "min_headway",
            "mean_headway",
            "start_time",
            "end_time",
        }
        if split_directions:
            expect_cols |= {"direction_id"}
        assert set(stop_stats.columns) == expect_cols

        # Should contain the correct stops
        expect_stops = set(stop_times.collect()["stop_id"].to_list())
        got_stops = set(stop_stats["stop_id"].to_list())
        assert got_stops == expect_stops

    # Empty check
    stats = gks.compute_stop_stats_0(stop_times, pl.DataFrame())
    assert gkh.is_empty(stats)


def test_compute_stop_stats():
    dates = cairns_dates
    feed = cairns.copy()
    n = 3
    sids = feed.stops.head(n).collect()["stop_id"].to_list()
    for split_directions in [True, False]:
        f = gks.compute_stop_stats(
            feed, dates + ["19990101"], stop_ids=sids, split_directions=split_directions
        ).collect()

        # Should contain the correct columns
        expect_cols = {
            "date",
            "stop_id",
            "num_routes",
            "num_trips",
            "max_headway",
            "min_headway",
            "mean_headway",
            "start_time",
            "end_time",
        }
        if split_directions:
            expect_cols.add("direction_id")

        assert set(f.columns) == expect_cols

        # Should contain the correct stops
        g = (
            gks.get_stops(feed, date=dates[0])
            .filter(pl.col("stop_id").is_in(sids))
            .collect()
        )
        assert set(f["stop_id"].to_list()) == set(g["stop_id"].to_list())

        # Should have correct dates
        set(f["date"].to_list()) == set(dates)

        # Non-feed dates should yield empty DataFrame
        f = gks.compute_stop_stats(
            feed, ["19990101"], split_directions=split_directions
        )
        assert gkh.is_empty(f)


def test_compute_stop_time_series_0():
    feed = cairns.copy()
    stop_times = feed.stop_times.head(250)
    trips1 = feed.trips
    # Set some null directions
    trips2 = trips1.with_columns(pl.lit(None, dtype=pl.Int32).alias("direction_id"))
    nstops = stop_times.collect()["stop_id"].n_unique()
    for trips, split_directions in itertools.product([trips1, trips2], [True, False]):
        if (
            split_directions
            and trips.select(pl.col("direction_id").is_null().all()).collect().item()
        ):
            # Should raise an error
            with pytest.raises(ValueError):
                gks.compute_stop_time_series_0(
                    stop_times, trips, split_directions=split_directions
                )
            continue

        ss = gks.compute_stop_stats_0(
            stop_times, trips, split_directions=split_directions
        ).collect()
        sts = gks.compute_stop_time_series_0(
            stop_times, trips, num_minutes=12 * 60, split_directions=split_directions
        ).collect()

        # Should have correct num rows and column names
        if split_directions:
            expect_cols = {"datetime", "stop_id", "direction_id", "num_trips"}
            assert sts.height <= nstops * 2
        else:
            expect_cols = {"datetime", "stop_id", "num_trips"}
            assert sts.height == nstops * 2
        assert set(sts.columns) == expect_cols

        # Each stop should have a correct total trip count
        if not split_directions:
            for stop_id, group in ss.partition_by("stop_id", as_dict=True).items():
                stop_id = stop_id[0]
                assert (
                    sts.filter(pl.col("stop_id") == stop_id)["num_trips"].sum()
                    == group["num_trips"].sum()
                )

    # Empty check
    sts = gks.compute_stop_time_series_0(
        feed.stop_times, pl.DataFrame(), split_directions=split_directions
    )
    assert gkh.is_empty(sts)


def test_compute_stop_time_series():
    feed = cairns.copy()
    dates = cairns_dates
    n = 3
    stop_ids = feed.stops.head(n).collect()["stop_id"].to_list()

    for split_directions in [True, False]:
        ss = gks.compute_stop_stats(
            feed, dates, stop_ids=stop_ids, split_directions=split_directions
        ).collect()
        sts = gks.compute_stop_time_series(
            feed,
            dates + ["19990101"],
            stop_ids=stop_ids,
            num_minutes=12 * 60,
            split_directions=split_directions,
        ).collect()

        # Should have correct num rows and column names
        k = len(stop_ids) * len(dates) * 2
        if split_directions:
            expect_cols = {"datetime", "stop_id", "direction_id", "num_trips"}
            assert sts.height <= k
        else:
            expect_cols = {"datetime", "stop_id", "num_trips"}
            assert sts.height == k
        assert set(sts.columns) == expect_cols

        # Each stop should have a correct total trip count
        if not split_directions:
            for stop_id, group in ss.partition_by("stop_id", as_dict=True).items():
                stop_id = stop_id[0]
                assert (
                    sts.filter(pl.col("stop_id") == stop_id)["num_trips"].sum()
                    == group["num_trips"].sum()
                )

        # Dates should be correct
        set(sts["datetime"].dt.strftime("%Y%m%d").to_list()) == set(dates)

    # Empty check
    sts = gks.compute_stop_time_series(
        feed, dates=["19990101"], split_directions=split_directions
    )
    assert gkh.is_empty(sts)
