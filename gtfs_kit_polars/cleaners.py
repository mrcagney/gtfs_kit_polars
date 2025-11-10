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
                f = f.with_columns(
                    pl.col(col)
                    .str.strip_chars()
                    .str.replace_all(r"\s+", "_")
                    .alias(col)
                )
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
        r.with_columns(dup=(pl.len().over("route_short_name") > 1))
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
    st_stop_ids = (
        feed.stop_times.select("stop_id").unique().collect()["stop_id"].to_list()
    )
    base_keep = pl.col("stop_id").is_in(st_stop_ids)
    if "location_type" in feed.stops.collect_schema().names():
        keep = base_keep | ~pl.col("location_type").is_in([0, None])
    else:
        keep = base_keep
    feed.stops = feed.stops.filter(keep)

    # 3) Clean undefined parent_station → null
    if "parent_station" in feed.stops.collect_schema().names():
        stop_ids_for_parent = (
            feed.stops.select("stop_id").unique().collect()["stop_id"].to_list()
        )
        feed.stops = feed.stops.with_columns(
            parent_station=pl.when(pl.col("parent_station").is_in(stop_ids_for_parent))
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

    # Distinct groups (small)
    groups = routes.select(pl.col(by)).unique().collect()

    # One new id per group (uses your helper for padding/format)
    group_map = pl.LazyFrame(
        {by: groups[by], "new_route_id": hp.make_ids(groups.height, route_id_prefix)}
    )

    # Join back to get old->new mapping, then drop dups
    return (
        routes.join(group_map, on=by, how="left")
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
    :func:`build_aggregate_routes_table` and the parameters ``by`` and
    ``route_id_prefix`` and update the old route IDs to the new ones in all the relevant
    Feed tables.
    Return the resulting Feed.
    """
    feed = feed.copy()

    # Build mapping: old route_id -> new_route_id  (LazyFrame with cols ["route_id","new_route_id"])
    route_map = build_aggregate_routes_table(
        feed.routes, by=by, route_id_prefix=route_id_prefix
    )

    # Update routes
    feed.routes = (
        feed.routes.join(route_map, on="route_id", how="left")
        .with_columns(pl.col("new_route_id").alias("route_id"))
        .drop("new_route_id")
        # Collapse to one row per group
        .sort([by, "route_id"])
        .unique(subset=[by], keep="first")
    )

    # Update trips
    feed.trips = (
        feed.trips.join(route_map, on="route_id", how="left")
        .with_columns(pl.col("new_route_id").alias("route_id"))
        .drop("new_route_id")
    )

    # Update fare_rules
    if (
        feed.fare_rules is not None
        and "route_id" in feed.fare_rules.collect_schema().names()
    ):
        feed.fare_rules = (
            feed.fare_rules.join(route_map, on="route_id", how="left")
            .with_columns(
                # If some fare rules had no route bound, keep original (left join)
                pl.when(pl.col("new_route_id").is_not_null())
                .then(pl.col("new_route_id"))
                .otherwise(pl.col("route_id"))
                .alias("route_id")
            )
            .drop("new_route_id")
        )
    return feed


def build_aggregate_stops_table(
    stops: pl.DataFrame | pl.LazyFrame,
    by: str = "stop_code",
    stop_id_prefix: str = "stop_",
) -> pl.LazyFrame:
    """
    Group stops by the ``by`` column and assign one new stop ID per group
    using the given prefix. Return a table with columns

    - ``stop_id``
    - ``new_stop_id``

    """
    stops = hp.make_lazy(stops)
    schema = stops.collect_schema().names()
    if by not in schema:
        raise ValueError(f"Column {by} not in stops.")
    if "stop_id" not in schema:
        raise ValueError("Column 'stop_id' not in stops.")

    # Distinct groups (deterministic order)
    groups = stops.select(pl.col(by)).unique().sort(by).collect()

    # One new id per group
    group_map = pl.LazyFrame(
        {
            by: groups[by],
            "new_stop_id": hp.make_ids(groups.height, stop_id_prefix),
        }
    )

    # Map old -> new, then dedupe
    return (
        stops.join(group_map, on=by, how="left")
        .select("stop_id", "new_stop_id")
        .unique()
    )


def aggregate_stops(
    feed: "Feed",
    by: str = "stop_code",
    stop_id_prefix: str = "stop_",
) -> "Feed":
    """
    Aggregate stops by the column `by` and assign new stop IDs using the given prefix.
    Update IDs in stops, stop_times, and transfers. Return the resulting Feed.
    """
    feed = feed.copy()

    # Build 'stop_id' -> 'new_stop_id' mapping (table)
    mapping = build_aggregate_stops_table(
        feed.stops, by=by, stop_id_prefix=stop_id_prefix
    )

    # Update stops
    stops = (
        feed.stops.join(mapping, on="stop_id", how="left")
        .with_columns(stop_id=pl.coalesce([pl.col("new_stop_id"), pl.col("stop_id")]))
        .drop("new_stop_id")
    )

    if "parent_station" in stops.collect_schema().names():
        parent_map = mapping.rename(
            {"stop_id": "parent_station", "new_stop_id": "new_parent_station"}
        )
        stops = (
            stops.join(parent_map, on="parent_station", how="left")
            .with_columns(
                parent_station=pl.coalesce(
                    [pl.col("new_parent_station"), pl.col("parent_station")]
                )
            )
            .drop("new_parent_station")
        )
    feed.stops = (
        stops
        # Collapse to one row per group
        .sort([by, "stop_id"]).unique(subset=[by], keep="first")
    )

    # Update stop times
    feed.stop_times = (
        feed.stop_times.join(mapping, on="stop_id", how="left")
        .with_columns(stop_id=pl.coalesce([pl.col("new_stop_id"), pl.col("stop_id")]))
        .drop("new_stop_id")
    )

    # Update transfers
    if feed.transfers is not None:
        map_to = mapping.rename(
            {"stop_id": "to_stop_id", "new_stop_id": "new_to_stop_id"}
        )
        map_from = mapping.rename(
            {"stop_id": "from_stop_id", "new_stop_id": "new_from_stop_id"}
        )
        feed.transfers = (
            feed.transfers.join(map_to, on="to_stop_id", how="left")
            .join(map_from, on="from_stop_id", how="left")
            .with_columns(
                to_stop_id=pl.coalesce(
                    [pl.col("new_to_stop_id"), pl.col("to_stop_id")]
                ),
                from_stop_id=pl.coalesce(
                    [pl.col("new_from_stop_id"), pl.col("from_stop_id")]
                ),
            )
            .drop("new_to_stop_id", "new_from_stop_id")
        )

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
        drop_cols = set(f.collect_schema().names()) - valid_columns
        setattr(feed, table, f.drop(drop_cols))

    return feed
