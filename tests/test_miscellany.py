import numpy as np
import pandas as pd
import polars as pl
import polars_st as st
import pytest
import shapely.geometry as sg
from pandas.testing import assert_series_equal

from gtfs_kit_polars import constants as gkc
from gtfs_kit_polars import helpers as gkh
from gtfs_kit_polars import miscellany as gkm
from gtfs_kit_polars import shapes as gks

from .context import (
    DATA_DIR,
    cairns,
    cairns_dates,
    cairns_trip_stats,
    gtfs_kit_polars,
    nyc_subway,
    sample,
)


def test_list_fields():
    feed = sample.copy()

    with pytest.raises(ValueError):
        gkm.list_fields(feed, "bad_table")

    for table in [None, "stops"]:
        f = gkm.list_fields(feed, table).collect()
        expect_cols = {
            "table",
            "column",
            "num_values",
            "num_nonnull_values",
            "num_unique_values",
            "min_value",
            "max_value",
        }
        assert set(f.columns) == expect_cols
        assert f.height


def test_describe():
    f = gkm.describe(sample)
    assert set(f.collect_schema().names()) == {"indicator", "value"}


def test_assess_quality():
    f = gkm.assess_quality(sample)
    assert set(f.collect_schema().names()) == {"indicator", "value"}


def test_convert_dist():
    # Test with no distances
    feed1 = cairns.copy()  # No distances here
    feed2 = gkm.convert_dist(feed1, "mi")
    assert feed2.dist_units == "mi"

    # Test with distances and identity conversion
    feed1 = gks.append_dist_to_shapes(feed1)
    feed2 = gkm.convert_dist(feed1, feed1.dist_units)
    assert feed1 == feed2

    # Test with proper conversion
    feed2 = gkm.convert_dist(feed1, "m")
    (feed2.shapes.collect()["shape_dist_traveled"] / 1000).to_list() == pytest.approx(list(feed1.shapes.collect()["shape_dist_traveled"]))



def test_compute_network_stats_0():
    feed = cairns.copy()
    k = cairns_trip_stats.height
    trip_stats = (
        # Add another route type besides 3
        cairns_trip_stats
        .with_row_index("i")
        .with_columns(
            route_type = (
                pl.when(pl.col("i") < k//2)
                .then(pl.lit(2, dtype=pl.Int64))
                .otherwise(pl.col("route_type"))
            )
        )
        .drop("i")
    )
    for split_route_types in [True, False]:
        f = gkm.compute_network_stats_0(
            feed.stop_times, trip_stats, split_route_types=split_route_types
        ).collect()

        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert f.height == n
        # Should contain the correct columns
        expect_cols = {
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "num_routes",
            "num_stops",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols


def test_compute_network_stats():
    feed = cairns.copy()
    dates = cairns_dates
    k = cairns_trip_stats.height
    trip_stats = (
        # Add another route type besides 3
        cairns_trip_stats
        .with_row_index("i")
        .with_columns(
            route_type = (
                pl.when(pl.col("i") < k//2)
                .then(pl.lit(2, dtype=pl.Int64))
                .otherwise(pl.col("route_type"))
            )
        )
        .drop("i")
    )
    for split_route_types in [True, False]:
        f = gkm.compute_network_stats(
            feed, dates + ["19990101"], trip_stats, split_route_types=split_route_types
        ).collect()
        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert f.height == n * len(dates)

        # Should have the correct dates
        assert set(f["date"].to_list()) == set(cairns_dates)

        # Should have correct columns
        expect_cols = {
            "num_trips",
            "num_trip_starts",
            "num_trip_ends",
            "num_routes",
            "num_stops",
            "peak_num_trips",
            "peak_start_time",
            "peak_end_time",
            "service_duration",
            "service_distance",
            "service_speed",
            "date",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols

        # Non-feed dates should yield empty DataFrame
        f = gkm.compute_network_stats(
            feed, ["19990101"], cairns_trip_stats, split_route_types=split_route_types
        )
        assert gkh.is_empty(f)


def test_compute_network_time_series():
    feed = cairns.copy()
    dates = cairns_dates
    k = feed.routes.collect().height
    feed.routes = (
        # Add another route type besides 3
        feed.routes
        .with_row_index("i")
        .with_columns(
            route_type = (
                pl.when(pl.col("i") < k//2)
                .then(pl.lit(2, dtype=pl.Int64))
                .otherwise(pl.col("route_type"))
            )
        )
        .drop("i")
    )
    for split_route_types in [True, False]:
        f = gkm.compute_network_time_series(
            feed,
            dates + ["19990101"],
            num_minutes=12*60,
            split_route_types=split_route_types,
        ).collect()

        # Should have correct num rows
        n = 2 if split_route_types else 1
        assert f.height == n * len(dates) * 2

        # Should have correct columns
        expect_cols = {
            "datetime",
            "num_trip_starts",
            "num_trip_ends",
            "num_trips",
            "service_distance",
            "service_duration",
            "service_speed",
        }
        if split_route_types:
            expect_cols.add("route_type")

        assert set(f.columns) == expect_cols

        # Should have correct dates
        assert set(f["datetime"].dt.strftime("%Y%m%d").to_list()) == set(dates)

        # Empty check
        f = gkm.compute_network_time_series(
            feed, ["19990101"], split_route_types=split_route_types
        )
        assert gkh.is_empty(f)


def test_create_shapes():
    feed1 = cairns.copy()

    # Remove a trip shape: set shape_id to null for the chosen trip
    trip_id = feed1.trips.head(1).collect()["trip_id"].item(0)
    feed1.trips = (
        feed1.trips.with_columns(
            shape_id = pl.when(pl.col("trip_id") == trip_id)
                         .then(pl.lit(None, pl.Utf8))
                         .otherwise(pl.col("shape_id"))
        )
    )

    # Run create_shapes on only-missing
    feed2 = gkm.create_shapes(feed1)

    # Should create only 1 new shape
    shapes2 = (
        feed2.shapes.select("shape_id").unique().collect().get_column("shape_id")
    )
    shapes1 = (
        feed1.shapes.select("shape_id").unique().collect().get_column("shape_id")
    )
    assert len(set(shapes2.to_list()) - set(shapes1.to_list())) == 1

    # Run create_shapes on all trips
    feed2 = gkm.create_shapes(feed1, all_trips=True)

    # Number of shapes should equal number of unique stop sequences
    st_grouped = (
        feed1.stop_times
        .sort("trip_id", "stop_sequence")
        .group_by("trip_id")
        .agg(stop_seq = pl.col("stop_id").implode())
        .select("stop_seq")
        .collect()
    )
    stop_seqs = {tuple(seq) for seq in st_grouped.get_column("stop_seq").to_list()}

    n_shapes = (
        feed2.shapes.select(pl.col("shape_id").n_unique().alias("n"))
        .collect()
        .item()
    )
    assert n_shapes == len(stop_seqs)


def test_compute_bounds():
    feed = cairns.copy()
    minlon, minlat, maxlon, maxlat = gkm.compute_bounds(feed)
    # Bounds should be in the ball park
    assert 145 < minlon < 146
    assert 145 < maxlon < 146
    assert -18 < minlat < -15
    assert -18 < maxlat < -15

    # A one-stop bounds should be the stop
    stop_id = feed.stops.head(1).collect()["stop_id"].item()
    minlon, minlat, maxlon, maxlat = gkm.compute_bounds(feed, [stop_id])
    expect_lon, expect_lat = feed.stops.filter(pl.col("stop_id") == stop_id).select("stop_lon", "stop_lat").collect().row(0)
    assert minlon == expect_lon
    assert minlat == expect_lat
    assert maxlon == expect_lon
    assert maxlat == expect_lat


def test_compute_convex_hull():
    feed = cairns.copy()
    hull = gkm.compute_convex_hull(feed)
    assert isinstance(hull, sg.Polygon)
    # Hull should encompass all stops
    m = sg.MultiPoint(feed.stops.select("stop_lon", "stop_lat").collect().to_pandas().values)
    assert hull.contains(m)

    # A one-stop hull should be the stop
    stop_id = feed.stops.head(1).collect()["stop_id"].item(0)
    hull = gkm.compute_convex_hull(feed, [stop_id])
    lon, lat = list(hull.coords)[0]
    expect_lon, expect_lat = feed.stops.filter(pl.col("stop_id") == stop_id).select("stop_lon", "stop_lat").collect().row(0)
    assert lon == expect_lon
    assert lat == expect_lat


def test_compute_centroid():
    feed = cairns.copy()
    centroid = gkm.compute_centroid(feed)
    assert isinstance(centroid, sg.Point)
    # Centroid should lie within bounds
    lon, lat = centroid.coords[0]
    bounds = gkm.compute_bounds(feed)
    assert bounds[0] < lon < bounds[2]
    assert bounds[1] < lat < bounds[3]

    # A one-stop centroid should be the stop
    stop_id = feed.stops.head(1).collect()["stop_id"].item(0)
    centroid = gkm.compute_centroid(feed, [stop_id])
    lon, lat = centroid.coords[0]
    expect_lon, expect_lat = feed.stops.filter(pl.col("stop_id") == stop_id).select("stop_lon", "stop_lat").collect().row(0)
    assert lon == expect_lon
    assert lat == expect_lat

def test_restrict_to_trips():
    feed1 = cairns

    # take first two trip_ids
    trip_ids = (
        feed1.trips.select("trip_id").limit(2).collect().get_column("trip_id").to_list()
    )

    feed2 = gkm.restrict_to_trips(feed1, trip_ids)

    # Should have correct trips
    trips2 = feed2.trips.select("trip_id").collect().get_column("trip_id").to_list()
    assert set(trips2) == set(trip_ids)

    # Should have correct shapes
    shape_ids = (
        feed1.trips
        .join(pl.LazyFrame({"trip_id": trip_ids}), on="trip_id", how="semi")
        .select("shape_id")
        .drop_nulls()
        .collect()
        .get_column("shape_id")
        .to_list()
    )
    shapes2 = (
        feed2.shapes.select("shape_id").collect().get_column("shape_id").to_list()
        if feed2.shapes is not None else []
    )
    assert set(shapes2) == set(shape_ids)

    # Should have correct stops in stop_times
    stop_ids = (
        feed1.stop_times
        .join(pl.LazyFrame({"trip_id": trip_ids}), on="trip_id", how="semi")
        .select("stop_id")
        .collect()
        .get_column("stop_id")
        .to_list()
    )
    stop_ids2 = (
        feed2.stop_times.select("stop_id").collect().get_column("stop_id").to_list()
    )
    assert set(stop_ids2) == set(stop_ids)

    # Fake trip id → all non-agency tables empty
    feed2 = gkm.restrict_to_trips(feed1, ["fake"])
    for table in gkc.DTYPES:
        if table != "agency" and getattr(feed1, table) is not None:
            tbl = getattr(feed2, table)
            assert tbl.collect().height == 0

    # NYC subway: ensure parent station inclusion
    feed = nyc_subway

    # pick a stop with a parent_station
    stop_id, parent_id = (
        feed.stops
        .filter(pl.col("parent_station").is_not_null())
        .select("stop_id", "parent_station")
        .collect()
        .row(0)
    )

    # get one trip that visits that stop
    trip_id = (
        feed.stop_times
        .filter(pl.col("stop_id") == stop_id)
        .select("trip_id")
        .limit(1)
        .collect()
        .item()
    )

    # restrict feed and verify parent station retained
    feed2 = gkm.restrict_to_trips(feed, [trip_id])

    stops2 = feed2.stops.select("stop_id").collect().get_column("stop_id").to_list()
    assert parent_id in stops2

    if feed2.transfers is not None:
        transfers_from = (
            feed2.transfers.select("from_stop_id")
            .collect()
            .get_column("from_stop_id")
            .to_list()
        )
        assert parent_id in transfers_from


def test_restrict_to_routes():
    feed1 = cairns.copy()
    route_ids = feed1.routes.head(2).collect()["route_id"].to_list()
    feed2 = gkm.restrict_to_routes(feed1, route_ids)
    # Should have correct routes
    assert set(feed2.routes.collect()["route_id"]) == set(route_ids)
    # Should have correct trips
    trip_ids = feed1.trips.filter(pl.col("route_id").is_in(route_ids)).collect()["trip_id"].to_list()
    assert set(feed2.trips.collect()["trip_id"]) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips.filter(pl.col("trip_id").is_in(trip_ids)).collect()["shape_id"].to_list()
    assert set(feed2.shapes.collect()["shape_id"]) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times.filter(pl.col("trip_id").is_in(trip_ids)).collect()["stop_id"].to_list()
    assert set(feed2.stop_times.collect()["stop_id"]) == set(stop_ids)

def test_restrict_to_agencies():
    feed1 = cairns.copy()

    # --- Build routes with an agency_id column; first row belongs to OP2, others OP1
    new_routes = (
        feed1.routes
        .with_columns(agency_id = pl.lit("OP1"))
        .with_row_index("i")
        .with_columns(
            agency_id = pl.when(pl.col("i") == 0)
                          .then(pl.lit("OP2"))
                          .otherwise(pl.col("agency_id"))
        )
        .drop("i")
    )

    # --- Build agency with OP1 for existing rows, then add a single OP2 row
    base_agency = feed1.agency.with_columns(agency_id = pl.lit("OP1"))
    op2_row = (
        base_agency
        .select(pl.all().first())          # take first row's values to match schema
        .with_columns(agency_id = pl.lit("OP2"))
    )
    new_agency = pl.concat([base_agency, op2_row], how="vertical_relaxed")

    # Reassign on feed copy
    feed1.routes = new_routes
    feed1.agency = new_agency

    # --- Exercise
    feed2 = gkm.restrict_to_agencies(feed1, ["OP2"])

    # Should only contain OP2 in the agency table
    agencies2 = (
        feed2.agency.select("agency_id").unique()
        .collect().get_column("agency_id").to_list()
    )
    assert set(agencies2) == {"OP2"}

    # Routes should match the original first row (except for the inserted agency_id col)
    cols_wo_agency = [c for c in feed1.routes.collect_schema().names() if c != "agency_id"]

    row_feed2 = (
        feed2.routes.select(cols_wo_agency).limit(1)
        .collect().row(0)
    )
    row_feed1 = (
        feed1.routes.select(cols_wo_agency).limit(1)
        .collect().row(0)
    )
    assert set(row_feed2) == set(row_feed1)

def test_restrict_to_dates():
    feed1 = cairns.copy()
    dates = feed1.get_first_week()[6:]
    feed2 = gkm.restrict_to_dates(feed1, dates)
    # Should have correct agency
    assert gkh.are_equal(feed2.agency, feed1.agency)
    # Should have correct dates
    assert set(feed2.get_dates()) < set(feed1.get_dates())
    # Should have correct trips
    assert set(feed2.trips.collect()["trip_id"].to_list()) < set(feed1.trips.collect()["trip_id"].to_list())
    # Should have correct routes
    assert set(feed2.routes.collect()["route_id"].to_list()) < set(feed1.routes.collect()["route_id"].to_list())
    # Should have correct shapes
    assert set(feed2.shapes.collect()["shape_id"].to_list()) < set(feed1.shapes.collect()["shape_id"].to_list())
    # Should have correct stops
    assert set(feed2.stops.collect()["stop_id"].to_list()) < set(feed1.stops.collect()["stop_id"].to_list())

    # Try again with date out of range
    dates = ["19990101"]
    feed2 = gkm.restrict_to_dates(feed1, dates)
    assert gkh.are_equal(feed2.agency, feed1.agency)
    assert gkh.is_empty(feed2.trips)
    assert gkh.is_empty(feed2.routes)
    assert gkh.is_empty(feed2.shapes)
    assert gkh.is_empty(feed2.stops)
    assert gkh.is_empty(feed2.stop_times)

def test_restrict_to_area():
    feed1 = cairns.copy()
    area = st.read_file(DATA_DIR / "cairns_square_stop_750070.geojson")
    feed2 = gkm.restrict_to_area(feed1, area)
    # Should have correct routes
    rsns = ["120", "120N"]
    assert set(feed2.routes.collect()["route_short_name"]) == set(rsns)
    # Should have correct trips
    route_ids = feed1.routes.filter(pl.col("route_short_name").is_in(rsns)).collect()["route_id"].to_list()
    trip_ids = feed1.trips.filter(pl.col("route_id").is_in(route_ids)).collect()["trip_id"].to_list()
    assert set(feed2.trips.collect()["trip_id"]) == set(trip_ids)
    # Should have correct shapes
    shape_ids = feed1.trips.filter(pl.col("trip_id").is_in(trip_ids)).collect()["shape_id"]
    assert set(feed2.shapes.collect()["shape_id"]) == set(shape_ids)
    # Should have correct stops
    stop_ids = feed1.stop_times.filter(pl.col("trip_id").is_in(trip_ids)).collect()["stop_id"]
    assert set(feed2.stop_times.collect()["stop_id"]) == set(stop_ids)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compute_screen_line_counts():
    feed = cairns.append_dist_to_stop_times()
    dates = cairns_dates[:2]

    # Load screen line
    path = DATA_DIR / "cairns_screen_lines.geojson"
    screen_lines = st.read_file(path)
    f = gkm.compute_screen_line_counts(feed, screen_lines, dates)

    # Should have correct columns
    expect_cols = {
        "date",
        "screen_line_id",
        "trip_id",
        "direction_id",
        "route_id",
        "route_short_name",
        "route_type",
        "shape_id",
        "crossing_direction",
        "crossing_dist_m",
        "crossing_time",
    }
    assert set(f.columns) == expect_cols

    # Should have both directions
    assert set(f.crossing_direction.unique()) == {-1, 1}

    # Should only have feed dates
    assert set(f["date"].values) == set(dates)

    # Empty check
    f = gkm.compute_screen_line_counts(feed, screen_lines, ["20010101"])
    assert f.is_empty()
