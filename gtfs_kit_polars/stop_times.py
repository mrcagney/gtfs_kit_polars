"""
Functions about stop times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import polars as pl

from . import helpers as hp

# Help type checkers but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_stop_times(feed: "Feed", date: str | None = None) -> pl.LazyFrame:
    """
    Return ``feed.stop_times``.
    If a date (YYYYMMDD date string) is given, then subset the result to only those
    stop times with trips active on the date.
    """
    if date is None:
        st = feed.stop_times
    else:
        st = feed.stop_times.join(feed.get_trips(date), on="trip_id", how="semi")

    return st


def get_start_and_end_times(feed: "Feed", date: str | None = None) -> tuple[str]:
    """
    Return the first departure time and last arrival time
    (HH:MM:SS time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date (YYYYMMDD string) if specified.
    """
    st = feed.get_stop_times(date)
    return (
        st.select("departure_time").drop_nulls().min().collect().row(0)[0],
        st.select("arrival_time").drop_nulls().max().collect().row(0)[0],
    )


def stop_times_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str | None] = None,
) -> dict:
    """
    Return a GeoJSON FeatureCollection of Point features
    representing all the trip-stop pairs in ``feed.stop_times``.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    For every trip, drop duplicate stop IDs within that trip.
    In particular, a looping trip will lack its final stop.

    If an iterable of trip IDs is given, then subset to those trips, silently dropping
    invalid trip IDs.
    """
    from .stops import get_stops

    if trip_ids is None or not list(trip_ids):
        trip_ids = feed.trips.trip_id

    st = feed.stop_times.filter(pl.col("trip_id").is_in(trip_ids))
    g = (
        get_stops(feed, as_geo=True)
        .join(st, "stop_id")
        .unique(subset=["trip_id", "stop_id"])
        .sort("trip_id", "stop_sequence")
    )
    if g is None or hp.is_empty(g):
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = g.collect().st.__geo_interface__

    return result


def append_dist_to_stop_times(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` column in
    ``feed.stop_times`` in terms of the distance units ``feed.dist_units``.
    Trips without shapes will have NaN distances.
    Return the resulting Feed.
    Uses ``feed.shapes``, so if that is missing, then return the original feed.

    This does not always give accurate results.
    The algorithm works as follows.
    Compute the ``shape_dist_traveled`` field by using Shapely to
    measure the distance of a stop along its trip LineString.
    If for a given trip this process produces a non-monotonically
    increasing, hence incorrect, list of (cumulative) distances, then
    fall back to estimating the distances as follows.

    Set the first distance to 0, the last to the length of the trip shape,
    and leave the remaining ones computed above.
    Choose the longest increasing subsequence of that new set of
    distances and use them and their corresponding departure times to linearly
    interpolate the rest of the distances.
    """
    if getattr(feed, "shapes", None) is None or hp.is_empty(feed.shapes):
        return feed

    # Prepare data, building geometry tables in UTM (meters)
    shapes_geo = feed.get_shapes(as_geo=True, use_utm=True).select(
        "shape_id", "geometry"
    )
    stops_geo = feed.get_stops(as_geo=True, use_utm=True).select("stop_id", "geometry")
    convert = hp.get_convert_dist("m", feed.dist_units)  # returns a Polars expr fn
    final_cols = [
        c
        for c in feed.stop_times.collect_schema().names()
        if c != "shape_dist_traveled"
    ] + ["shape_dist_traveled"]

    stop_times = (
        feed.stop_times.join(
            feed.trips.select("trip_id", "shape_id"), on="trip_id", how="left"
        )
        .join(shapes_geo.rename({"geometry": "shape_geom"}), on="shape_id", how="left")
        .join(stops_geo.rename({"geometry": "stop_geom"}), on="stop_id", how="left")
        .sort("trip_id", "stop_sequence")
        .with_columns(
            dtime=hp.timestr_to_seconds("departure_time"),
            # distances along the linestring (in meters), and line length
            dist=pl.when(
                pl.col("shape_geom").is_not_null() & pl.col("stop_geom").is_not_null()
            )
            .then(pl.col("shape_geom").st.project(pl.col("stop_geom")))
            .otherwise(None)
            .cast(pl.Float64),
            shape_length=pl.col("shape_geom").st.length().cast(pl.Float64),
        )
        # Fix reversed direction trips
        .with_columns(
            first_dist=pl.col("dist").first().over("trip_id"),
            last_dist=pl.col("dist").last().over("trip_id"),
        )
        .with_columns(
            need_flip=(pl.col("last_dist") < pl.col("first_dist"))
            & pl.col("shape_length").is_not_null(),
        )
        .with_columns(
            dist=pl.when(pl.col("need_flip"))
            .then(pl.col("shape_length") - pl.col("dist"))
            .otherwise(pl.col("dist"))
        )
        .drop("shape_geom", "stop_geom", "first_dist", "last_dist", "need_flip")
    )

    # Separate trips that have bad distances and fix them
    bad_trips = (
        stop_times.with_columns(
            step=(pl.col("dist") - pl.col("dist").shift(1)).over("trip_id"),
            overrun=(pl.col("dist") > (pl.col("shape_length") + pl.lit(100.0))),
        )
        .with_columns(
            has_bad_step=(pl.col("step") < 0) & pl.col("dist").is_not_null(),
            has_overrun=pl.col("overrun").any().over("trip_id"),
        )
        .filter(pl.col("has_bad_step") | pl.col("has_overrun"))
        .select("trip_id")
        .unique()
    )

    def compute_dist(g: pl.DataFrame) -> pl.DataFrame:
        D = g["shape_length"][0]
        dists0 = g["dist"].to_list()

        # Do nothings for defunct cases
        if np.isnan(D) or len(dists0) <= 1:
            return g

        times = g["dtime"].to_numpy()
        dists = np.array([0.0] + dists0[1:-1] + [float(D)], dtype=float)
        ix = hp.longest_subsequence(dists, index=True)
        good_dists = np.take(dists, ix)
        good_times = np.take(times, ix)
        new_dists = np.interp(times, good_times, good_dists).astype(float)

        return g.with_columns(dist=pl.Series(new_dists))

    fixed = (
        stop_times.join(bad_trips, on="trip_id", how="inner")
        .sort("trip_id", "stop_sequence")
        .group_by("trip_id")
        .map_groups(compute_dist, schema=stop_times.collect_schema())
    )

    good = stop_times.join(bad_trips, on="trip_id", how="anti")

    # Assemble stop times
    stop_times = (
        pl.concat([good, fixed])
        .with_columns(
            shape_dist_traveled=pl.when(pl.col("dist").is_not_null())
            .then(convert(pl.col("dist")))
            .otherwise(None)
        )
        .sort("trip_id", "stop_sequence")
        .select(final_cols)
    )

    new_feed = feed.copy()
    new_feed.stop_times = stop_times
    return new_feed
