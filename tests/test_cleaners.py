import numpy as np
import polars as pl
import pytest

from gtfs_kit_polars import cleaners as gkc
from gtfs_kit_polars import helpers as gkh

from .context import gtfs_kit_polars, sample


def set_val(f, col, val, rows=[1]):
    return (
        gkh.make_lazy(f)
        .with_row_index("_i", offset=1)
        .with_columns(
            (
                pl.when(pl.col("_i").is_in(rows))
                .then(pl.lit(val))
                .otherwise(pl.col(col))
                .alias(col)
            )
        )
        .drop("_i")
    )

def first_val(f, col):
    return gkh.make_lazy(f).select(pl.col(col)).head(1).collect().item()

def series_all_true(f, expr):
    return gkh.make_lazy(f).select(expr.all()).collect().item()

# --------------------------------------------------------------------------

def test_clean_column_names():
    f = sample.routes
    g = gkc.clean_column_names(f)
    gkh.are_equal(f, g)

    f = sample.routes.with_columns(**{" route_id  ": pl.col("route_id")}).drop("route_id")
    g = gkc.clean_column_names(f).collect()
    assert "route_id" in g.columns
    assert " route_id  " not in g.columns

def test_clean_ids():
    f1 = sample.copy()

    # set routes.route_id[0] = "  ho   ho ho "
    f1.routes = set_val(f1.routes, "route_id", "  ho   ho ho ")

    f2 = gkc.clean_ids(f1)
    expect_rid = "ho_ho_ho"
    assert first_val(f2.routes, "route_id") == expect_rid

    # idempotent: running again doesn't change anything
    f3 = gkc.clean_ids(f2)
    gkh.are_equal(f3.routes, f2.routes)
    gkh.are_equal(f3.trips, f2.trips)


def test_extend_id():
    f1 = sample.copy()

    # Original: how many trips have route_id == "AAMV" ?
    n_orig = gkh.height(f1.trips.filter(pl.col("route_id") == "AAMV"))
    assert n_orig == 4

    # Prefix all route_id
    f2 = gkc.extend_id(f1, "route_id", "prefix_")
    assert series_all_true(f2.routes, pl.col("route_id").str.starts_with("prefix_"))

    n_pref = gkh.height(f2.trips.filter(pl.col("route_id") == "prefix_AAMV"))
    assert n_pref == 4

    # Suffix now
    f3 = gkc.extend_id(f2, "route_id", "_suffix", prefix=False)
    assert series_all_true(f3.routes, pl.col("route_id").str.ends_with("_suffix"))

    n_both = gkh.height(
        f3.trips.filter(pl.col("route_id") == "prefix_AAMV_suffix")
    )
    assert n_both == 4

    # Bad column name should raise
    with pytest.raises(ValueError):
        gkc.extend_id(f2, "direction_id", "_suffix", prefix=False)


def test_clean_times():
    f1 = sample.copy()

    # stop_times.departure_time[0] = "7:00:00"
    f1.stop_times = set_val(f1.stop_times, "departure_time", "7:00:00")

    # frequencies.start_time[0] = "7:00:00"
    f1.frequencies = set_val(f1.frequencies, "start_time", "7:00:00")

    f2 = gkc.clean_times(f1)

    assert first_val(f2.stop_times, "departure_time") == "07:00:00"
    assert first_val(f2.frequencies, "start_time") == "07:00:00"



def test_clean_route_short_names():
    f1 = sample.copy()

    # Should have no effect on a fine feed
    f2 = gkc.clean_route_short_names(f1)
    gkh.are_equal(f2.routes.select("route_short_name"), f1.routes.select("route_short_name"))

    # Make route short name duplicates
    f1.routes = set_val(f1.routes, "route_short_name", None, [1, 2])
    f1.routes = set_val(f1.routes, "route_short_name", "  he llo  ", [3])
    f2 = gkc.clean_route_short_names(f1)

    # Should have unique route short names
    assert f2.routes.collect()["route_short_name"].n_unique() == f2.routes.collect().height

    # None should be replaced by n/a and route IDs
    expect_rsns = [f"n/a-{r}" for r in sample.routes.head(2).collect()["route_id"].to_list()]
    assert f2.routes.head(2).collect()["route_short_name"].to_list() == expect_rsns

    # Should have names without leading or trailing whitespace
    assert not f2.routes.collect()["route_short_name"].str.starts_with(" ").any()
    assert not f2.routes.collect()["route_short_name"].str.ends_with(" ").any()

def test_drop_zombies():
    # 1) No-op on a healthy feed
    f1 = sample.copy()
    f2 = gkc.drop_zombies(f1)
    assert gkh.are_equal(f2.routes, f1.routes)

    # 2) Drops stops (loc_type 0/NaN) with no stop_times
    f1 = sample.copy()
    # make all location_type nulls, then remove one stop's stop_times
    f1.stops = f1.stops.with_columns(location_type=pl.lit(None, dtype=pl.Float64))
    stop_id = f1.stops.select("stop_id").limit(1).collect()["stop_id"][0]
    f1.stop_times = f1.stop_times.filter(pl.col("stop_id") != stop_id)
    f2 = gkc.drop_zombies(f1)
    stops_after = set(f2.stops.select("stop_id").collect()["stop_id"])
    assert stop_id not in stops_after

    # 3) Undefined parent stations → null
    f1 = sample.copy()
    f1.stops = f1.stops.with_columns(parent_station=pl.lit("bingo"))
    f2 = gkc.drop_zombies(f1)
    is_all_null = (
        f2.stops.select(pl.col("parent_station").is_null().all())
        .collect()
        .item()
    )
    assert is_all_null

    # 4) Make a route's trips “zombie” (no stop_times), ensure they and the route drop
    f1 = sample.copy()
    rid = f1.routes.select("route_id").limit(1).collect()["route_id"][0]
    # set those trips' IDs to a value not present in stop_times
    f1.trips = f1.trips.with_columns(
        trip_id=pl.when(pl.col("route_id") == rid)
        .then(pl.lit("hoopla"))
        .otherwise(pl.col("trip_id"))
    )
    f2 = gkc.drop_zombies(f1)
    trips_after = set(f2.trips.select("trip_id").collect()["trip_id"].to_list())
    routes_after = set(f2.routes.select("route_id").collect()["route_id"].to_list())
    assert "hoopla" not in trips_after
    assert rid not in routes_after




def test_build_aggregate_routes_dict():
    # Equalize all route short names
    routes = sample.routes.with_columns(route_short_name=pl.lit("bingo"))
    nid_by_oid = gkc.build_aggregate_routes_dict(routes, route_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_0"}


def test_aggregate_routes():
    feed1 = sample.copy()
    # Equalize all route short names
    feed1.routes["route_short_name"] = "bingo"
    feed2 = gkc.aggregate_routes(feed1)

    # feed2 should have only one route ID
    assert feed2.routes.shape[0] == 1

    # Feeds should have same trip DataFrames excluding route IDs
    feed1.trips["route_id"] = feed2.trips["route_id"]
    assert gkh.are_equal(feed1.trips, feed2.trips)

    # Feeds should have same fare rules DataFrames excluding route IDs
    feed1.fare_rules["route_id"] = feed2.fare_rules["route_id"]
    assert gkh.are_equal(feed1.fare_rules, feed2.fare_rules)

    # Feeds should have equal attributes excluding routes, trips, and fare rules
    # DataFrames
    feed2.routes = feed1.routes
    feed2.trips = feed1.trips
    feed2.fare_rules = feed1.fare_rules
    assert feed1 == feed2


def test_build_aggregate_stops_dict():
    stops = sample.stops.copy()
    # Equalize all stop codes
    stops["stop_code"] = "bingo"
    nid_by_oid = gkc.build_aggregate_stops_dict(stops, stop_id_prefix="bongo_")
    assert set(nid_by_oid.values()) == {"bongo_0"}


def test_aggregate_stops():
    feed1 = sample.copy()
    # Equalize all stop codes
    feed1.stops["stop_code"] = "bingo"
    feed2 = gkc.aggregate_stops(feed1)

    # feed2 should have only one stop ID
    assert feed2.stops.shape[0] == 1

    # Feeds should have same stop times, excluding stop IDs
    feed1.stop_times["stop_id"] = feed2.stop_times.stop_id
    assert gkh.are_equal(feed1.stop_times, feed2.stop_times)

    # Feeds should have equal attributes excluding
    # stops stop times DataFrames
    feed2.stops = feed1.stops
    feed2.stop_times = feed1.stop_times
    assert feed1 == feed2


def test_clean():
    f1 = sample.copy()
    rid = f1.routes["route_id"].iat[0]
    f1.routes.loc[0, "route_id"] = " " + rid + "   "
    f2 = gkc.clean(f1)
    assert f2.routes.loc[0, "route_id"] == rid
    gkh.are_equal(f2.trips, sample.trips)


def test_drop_invalid_columns():
    f1 = sample.copy()
    f1.routes["bingo"] = "bongo"
    f1.trips["wingo"] = "wongo"
    f2 = gkc.drop_invalid_columns(f1)
    print(f2.routes)
    assert f2 == sample
