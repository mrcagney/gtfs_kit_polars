"""
Functions about cleaning feeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def clean_column_names(f: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Strip the whitespace from all column names in the given table
    and return the result.
    """
    return f.rename({c: c.strip() for c in f.collect_schema().names()})


def clean_ids(feed: "Feed") -> "Feed":
    """
    In the given Feed, strip whitespace from all string IDs and
    then replace every remaining whitespace chunk with an underscore.
    Return the resulting Feed.
    """
    # Alter feed inputs only, and build a new feed from them.
    # The derived feed attributes, such as feed.trips_i,
    # will be automatically handled when creating the new feed.
    feed = feed.copy()

    for table, d in cs.DTYPES.items():
        f = getattr(feed, table)
        if f is None:
            continue
        names = f.collect_schema().names()
        for col in d:
            if col in names and d[col] == pl.Utf8 and col.endswith("_id"):
                f = f.with_columns(pl.col(col).str.strip_chars().str.replace_all(r"\s+", "_").alias(col))
                setattr(feed, table, f)

    return feed


def extend_id(feed: "Feed", id_col: str, extension: str, *, prefix=True) -> "Feed":
    """
    Add a prefix (if ``prefix``) or a suffix (otherwise) to all values of column
    ``id_col`` across all tables of this Feed.
    This can be helpful when preparing to merge multiple GTFS feeds with colliding
    route IDs, say.

    Raises a ValueError if ``id_col`` values are strings,
    e.g. if ``id_col`` is 'direction_id'.
    """
    feed = feed.copy()

    for table, d in cs.DTYPES.items():
        t = getattr(feed, table)
        if t is not None and id_col in d:
            if d[id_col] != pl.Utf8:
                raise ValueError(f"{id_col} must be a string column")
            elif prefix:
                t = t.with_columns((pl.lit(extension) + pl.col(id_col)).alias(id_col))
                setattr(feed, table, t)
            else:
                t = t.with_columns((pl.col(id_col) + pl.lit(extension)).alias(id_col))
                setattr(feed, table, t)

    return feed


def clean_times(feed: "Feed") -> "Feed":
    """
    In the given Feed, convert H:MM:SS time strings to HH:MM:SS time
    strings to make sorting by time work as expected.
    Return the resulting Feed.
    """
    feed = feed.copy()

    tables_and_columns = [
        ("stop_times", ["arrival_time", "departure_time"]),
        ("frequencies", ["start_time", "end_time"]),
    ]

    for table, columns in tables_and_columns:
        f = getattr(feed, table)
        if f is None:
            continue

        updates = []
        for col in columns:
            # strip, then if length == 7 (e.g. "7:00:00") pad to "07:00:00"
            s = pl.col(col)
            stripped = s.str.strip_chars()
            updates.append(
                pl.when(s.is_null())
                .then(None)
                .otherwise(
                    pl.when(s.str.len_chars() == 7)
                    .then(pl.lit("0") + stripped)
                    .otherwise(stripped)
                )
                .alias(col)
            )
        if updates:
            f = f.with_columns(updates)

        setattr(feed, table, f)

    return feed

def clean_route_short_names(feed: "Feed") -> "Feed":
    """
    In ``feed.routes``, assign 'n/a' to missing route short names and
    strip whitespace from route short names.
    Then disambiguate each route short name that is duplicated by
    appending '-' and its route ID.
    Return the resulting Feed.
    """
    feed = feed.copy()
    r = feed.routes
    if r is None:
        return feed

    # Normalize short names: fill nulls with "n/a" and strip whitespace
    r = r.with_columns(
        route_short_name=(
            pl.when(pl.col("route_short_name").is_null())
            .then(pl.lit("n/a"))
            .otherwise(pl.col("route_short_name"))
            .str.strip_chars()
        )
    )

    # Mark duplicates and disambiguate by appending "-<route_id>"
    r = (
        r
        .with_columns(
            dup=(pl.len().over("route_short_name") > 1)
        )
        .with_columns(
            route_short_name=(
                pl.when(pl.col("dup"))
                .then(pl.col("route_short_name") + pl.lit("-") + pl.col("route_id"))
                .otherwise(pl.col("route_short_name"))
            )
        )
        .drop("dup")
    )
    feed.routes = r
    return feed

def drop_zombies(feed: "Feed") -> "Feed":
    """
    In the given Feed, do the following in order and return the resulting Feed.

    1. Drop agencies with no routes.
    2. Drop stops of location type 0 or None with no stop times.
    3. Remove undefined parent stations from the ``parent_station`` column.
    4. Drop trips with no stop times.
    5. Drop shapes with no trips.
    6. Drop routes with no trips.
    7. Drop services with no trips.

    """
    feed = feed.copy()

    # 1) Agencies with routes only
    if feed.agency is not None:
        r_ag = feed.routes.select("agency_id").unique()
        feed.agency = feed.agency.join(r_ag, on="agency_id", how="inner")

    # 2) Drop stops of location_type 0/None that have no stop_times
    st_stop_ids = feed.stop_times.select("stop_id").unique().collect()["stop_id"].to_list()
    base_keep = pl.col("stop_id").is_in(st_stop_ids)
    if "location_type" in feed.stops.collect_schema().names():
        keep = base_keep | ~pl.col("location_type").is_in([0, None])
    else:
        keep = base_keep
    feed.stops = feed.stops.filter(keep)

    # 3) Clean undefined parent_station → null
    if "parent_station" in feed.stops.collect_schema().names():
        stop_ids_for_parent = feed.stops.select("stop_id").unique().collect()["stop_id"].to_list()
        feed.stops = feed.stops.with_columns(
            parent_station=pl.when(
                pl.col("parent_station").is_in(stop_ids_for_parent)
            )
            .then(pl.col("parent_station"))
            .otherwise(None)
        )

    # 4) Keep only trips that appear in stop_times
    st_trip_ids = feed.stop_times.select("trip_id").unique()
    feed.trips = feed.trips.join(st_trip_ids, on="trip_id", how="inner")

    # 5) Keep only shapes that appear in trips
    if feed.shapes is not None:
        trip_shape_ids = feed.trips.select("shape_id").unique()
        feed.shapes = feed.shapes.join(trip_shape_ids, on="shape_id", how="inner")

    # 6) Keep only routes that appear in trips
    trip_route_ids = feed.trips.select("route_id").unique()
    feed.routes = feed.routes.join(trip_route_ids, on="route_id", how="inner")

    # 7) Keep only services that appear in trips
    service_ids = feed.trips.select("service_id").unique()
    if feed.calendar is not None:
        feed.calendar = feed.calendar.join(service_ids, on="service_id", how="inner")
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.join(
            service_ids, on="service_id", how="inner"
        )

    return feed



def build_aggregate_routes_dict(
    routes: pl.DataFrame | pl.LazyFrame, by: str = "route_short_name", route_id_prefix: str = "route_"
) -> dict[str, str]:
    """
    Given a DataFrame of routes, group the routes by route short name, say,
    and assign new route IDs using the given prefix.
    Return a dictionary of the form <old route ID> -> <new route ID>.
    Helper function for :func:`aggregate_routes`.

    More specifically, group ``routes`` by the ``by`` column, and for each group make
    one new route ID for all the old route IDs in that group based on the given
    ``route_id_prefix`` string and a running count, e.g. ``'route_013'``.
    """
    routes = hp.make_lazy(routes).collect()
    if by not in routes.columns:
        raise ValueError(f"Column {by} not in routes.")

    # Create new route IDs
    n = routes.select(by).n_unique()
    nids = hp.make_ids(n, route_id_prefix)
    nid_by_oid = dict()
    i = 0
    for group in routes.partition_by(by):
        d = {oid: nids[i] for oid in group["route_id"].to_list()}
        nid_by_oid.update(d)
        i += 1

    return nid_by_oid

def build_aggregate_routes_table(
    routes: pl.DataFrame | pl.LazyFrame,
    by: str = "route_short_name",
    route_id_prefix: str = "route_",
) -> pl.LazyFrame:
    """
    Group routes by the ``by`` column and assign one new route ID per group
    using the given prefix. Return a table with columns

    - ``route_id``
    - ``new_route_id``

    """
    routes = hp.make_lazy(routes)
    schema = routes.collect_schema().names()
    if by not in schema:
        raise ValueError(f"Column {by} not in routes.")
    if "route_id" not in schema:
        raise ValueError("Column 'route_id' not in routes.")

    # Distinct groups (small), deterministically ordered
    groups = (
        routes
        .select(pl.col(by))
        .unique()
        .collect()
    )

    # One new id per group (uses your helper for padding/format)
    group_map = pl.LazyFrame({by: groups[by], "new_route_id": hp.make_ids(groups.height, route_id_prefix)})

    # Join back to get old->new mapping, then drop dups
    return (
        routes
        .join(group_map, on=by, how="left")
        .select(
            "route_id",
            "new_route_id",
        )
        .unique()
    )

def aggregate_routes(
    feed: "Feed", by: str = "route_short_name", route_id_prefix: str = "route_"
) -> "Feed":
    """
    Aggregate routes by route short name, say, and assign new route IDs using the
    given prefix.

    More specifically, create new route IDs with the function
    :func:`build_aggregate_routes_dict` and the parameters ``by`` and
    ``route_id_prefix`` and update the old route IDs to the new ones in all the relevant
    Feed tables.
    Return the resulting Feed.
    """
    feed = feed.copy()

    # Make new route IDs
    routes = feed.routes
    nid_by_oid = build_aggregate_routes_dict(routes, by, route_id_prefix)

    # Update route IDs in routes
    routes["route_id"] = routes.route_id.map(lambda x: nid_by_oid[x])
    routes = routes.groupby(by).first().reset_index()
    feed.routes = routes

    # Update route IDs in trips
    trips = feed.trips
    trips["route_id"] = trips.route_id.map(lambda x: nid_by_oid[x])
    feed.trips = trips

    # Update route IDs of fare rules
    if feed.fare_rules is not None and "route_id" in feed.fare_rules.columns:
        fr = feed.fare_rules
        fr["route_id"] = fr.route_id.map(lambda x: nid_by_oid[x])
        feed.fare_rules = fr

    return feed


def build_aggregate_stops_dict(
    stops: pd.DataFrame, by: str = "stop_code", stop_id_prefix: str = "stop_"
) -> dict[str, str]:
    """
    Given a DataFrame of stops, group the stops by stop code, say,
    and assign new stop IDs using the given prefix.
    Return a dictionary of the form <old stop ID> -> <new stop ID>.
    Helper function for :func:`aggregate_stops`.

    More specifically, group ``stops`` by the ``by`` column, and for each group make
    one new stop ID for all the old stops IDs in that group based on the given
    ``stop_id_prefix`` string and a running count, e.g. ``'stop_013'``.
    """
    if by not in stops.columns:
        raise ValueError(f"Column {by} not in stops.")

    # Create new stop IDs
    n = stops.groupby(by).ngroups
    nids = hp.make_ids(n, stop_id_prefix)
    nid_by_oid = dict()
    i = 0
    for col, group in stops.groupby(by):
        d = {oid: nids[i] for oid in group.stop_id.values}
        nid_by_oid.update(d)
        i += 1

    return nid_by_oid


def aggregate_stops(
    feed: "Feed", by: str = "stop_code", stop_id_prefix: str = "stop_"
) -> "Feed":
    """
    Aggregate stops by stop code, say, and assign new stop IDs using the
    given prefix.

    More specifically, create new stop IDs with the function
    :func:`build_aggregate_stops_dict` and the parameters ``by`` and
    ``stop_id_prefix`` and update the old stop IDs to the new ones in all the relevant
    Feed tables.
    Return the resulting Feed.
    """
    feed = feed.copy()

    # Make new stop ID by old stop ID dict
    stops = feed.stops
    nid_by_oid = build_aggregate_stops_dict(stops, by, stop_id_prefix)

    # Apply dict
    stops["stop_id"] = stops.stop_id.map(nid_by_oid)
    if "parent_station" in stops:
        stops["parent_station"] = stops.parent_station.map(nid_by_oid)

    stops = stops.groupby(by).first().reset_index()
    feed.stops = stops

    # Update stop IDs of stop times
    stop_times = feed.stop_times
    stop_times["stop_id"] = stop_times.stop_id.map(lambda x: nid_by_oid[x])
    feed.stop_times = stop_times

    # Update route IDs of transfers
    if feed.transfers is not None:
        transfers = feed.transfers
        transfers["to_stop_id"] = transfers.to_stop_id.map(lambda x: nid_by_oid[x])
        transfers["from_stop_id"] = transfers.from_stop_id.map(lambda x: nid_by_oid[x])
        feed.transfers = transfers

    return feed


def clean(feed: "Feed") -> "Feed":
    """
    Apply the following functions to the given Feed in order and return the resulting
    Feed.

    #. :func:`clean_ids`
    #. :func:`clean_times`
    #. :func:`clean_route_short_names`
    #. :func:`drop_zombies`

    """
    feed = feed.copy()
    ops = ["clean_ids", "clean_times", "clean_route_short_names", "drop_zombies"]
    for op in ops:
        feed = globals()[op](feed)

    return feed


def drop_invalid_columns(feed: "Feed") -> "Feed":
    """
    Drop all DataFrame columns of the given Feed that are not
    listed in the GTFS.
    Return the resulting Feed.
    """
    feed = feed.copy()
    for table, d in cs.DTYPES.items():
        f = getattr(feed, table)
        if f is None:
            continue
        valid_columns = set(d.keys())
        for col in f.columns:
            if col not in valid_columns:
                print(f"{table}: dropping invalid column {col}")
                del f[col]
        setattr(feed, table, f)

    return feed
