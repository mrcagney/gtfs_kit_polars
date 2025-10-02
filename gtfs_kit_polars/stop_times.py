"""
Functions about stop times.
"""

from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import json

import polars as pl
import pandas as pd
import numpy as np

from . import helpers as hp

# Help mypy but avoid circular imports
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


# TODO: Rewrite because way too slow
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
    if feed.shapes is None or hp.is_empty(feed.shapes):
        return feed

    # Geometry dicts in meters
    geom_by_stop = feed.build_geometry_by_stop(use_utm=True)
    geom_by_shape = feed.build_geometry_by_shape(use_utm=True)

    # Memoize per (shape_id, stop_id)
    dist_by_stop_by_shape = {k: {} for k in geom_by_shape}

    # def compute_dist(group):
    #     g = group.copy()

    #     # Compute the distances of the stops along this trip and memoize.
    #     shape = g.shape_id.iat[0]
    #     linestring = geom_by_shape[shape]
    #     dists = []
    #     for stop in g.stop_id.values:
    #         if stop in dist_by_stop_by_shape[shape]:
    #             d = dist_by_stop_by_shape[shape][stop]
    #         else:
    #             d = linestring.project(geom_by_stop[stop])
    #             dist_by_stop_by_shape[shape][stop] = d
    #         dists.append(d)

    #     s = sorted(dists)
    #     D = linestring.length
    #     dists_are_reasonable = all([d < D + 100 for d in dists])

    #     if dists_are_reasonable and s == dists:
    #         # Good
    #         g["shape_dist_traveled"] = dists
    #     elif dists_are_reasonable and s == dists[::-1]:
    #         # Good after reversal.
    #         # This happens when the direction of the linestring
    #         # opposes the direction of the vehicle trip.
    #         dists = dists[::-1]
    #         g["shape_dist_traveled"] = dists
    #     else:
    #         # Bad. Redo using interpolation on a good subset of dists.
    #         dists = np.array([0] + dists[1:-1] + [D])
    #         ix = hp.longest_subsequence(dists, index=True)
    #         good_dists = np.take(dists, ix)
    #         g["shape_dist_traveled"] = np.interp(
    #             g["dtime"], g.iloc[ix]["dtime"], good_dists
    #         )

    #         # Update dist dictionary with new and improved dists
    #         for row in g[["stop_id", "shape_dist_traveled"]].itertuples(index=False):
    #             dist_by_stop_by_shape[shape][row.stop_id] = row.shape_dist_traveled

    #     return g

    def compute_dist(group: pl.DataFrame) -> pl.DataFrame:
        g = group.clone()
        shape = g["shape_id"][0]
        if shape is None or (isinstance(shape, float) and np.isnan(shape)):
            # No shape -> all NaN
            g = g.with_columns(pl.lit(None, pl.Float64).alias("shape_dist_traveled"))
        else:
            linestring = geom_by_shape[shape]
            dists = []
            cache = dist_by_stop_by_shape[shape]
            for stop in g["stop_id"]:
                if stop in cache:
                    d = cache[stop]
                else:
                    d = linestring.project(geom_by_stop[stop])
                    cache[stop] = d
                dists.append(d)

            s = sorted(dists)
            D = linestring.length
            dists_are_reasonable = all(d < D + 100 for d in dists)

            if dists_are_reasonable and s == dists:
                pass  # already good
            elif dists_are_reasonable and s == dists[::-1]:
                dists = dists[::-1]  # reversed trip direction vs. shape direction
            else:
                # Bad. Redo using interpolation on a good subset of dists.
                arr = np.array([0] + dists[1:-1] + [D], dtype=float)
                ix = hp.longest_subsequence(arr, index=True)
                good_dists = np.take(arr, ix)
                times = g["dtime"].to_numpy()
                good_times = times[ix]
                dists = np.interp(times, good_times, good_dists).tolist()

                # Update cache with improved values
                for stop, d in zip(g["stop_id"], dists):
                    cache[stop] = d

            g = g.with_columns(pl.Series("shape_dist_traveled", dists))

        return g

    # Apply per-trip computation in eager, then convert units and return to lazy
    # Columns from original stop_times to keep and final set with the new distance column
    cols = [
        c
        for c in feed.stop_times.collect_schema().names()
        if c != "shape_dist_traveled"
    ]
    final_cols = cols + ["shape_dist_traveled"]
    convert_dist = hp.get_convert_dist("m", feed.dist_units)
    st = (
        feed.stop_times.select(cols)
        .join(feed.trips.select("trip_id", "shape_id"), on="trip_id", how="left")
        .with_columns(dtime=hp.timestr_to_seconds("departure_time"))
        .sort("trip_id", "stop_sequence")
        .collect()
        # Cheating here
        # .to_pandas()
        # .groupby("trip_id")
        # .apply(compute_dist, include_groups=False)
        # .reset_index()
        # .pipe(pl.from_pandas)
        .group_by("trip_id")
        .map_groups(compute_dist)
        # Convert from meters to feed units
        .with_columns(
            pl.when(pl.col("shape_dist_traveled").is_not_null())
            .then(convert_dist(pl.col("shape_dist_traveled")))
            .otherwise(None)
            .alias("shape_dist_traveled")
        )
        .select(final_cols)
        .lazy()
    )
    # Create new feed
    new_feed = feed.copy()
    new_feed.stop_times = st

    return new_feed


def get_start_and_end_times(feed: "Feed", date: str | None = None) -> list[str]:
    """
    Return the first departure time and last arrival time
    (HH:MM:SS time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date (YYYYMMDD string) if specified.
    """
    st = feed.get_stop_times(date)
    return (st["departure_time"].dropna().min(), st["arrival_time"].dropna().max())


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

    If an iterable of trip IDs is given, then subset to those trips.
    If some of the given trip IDs are not found in the feed, then raise a ValueError.
    """
    from .stops import get_stops

    if trip_ids is None or not list(trip_ids):
        trip_ids = feed.trips.trip_id

    D = set(trip_ids) - set(feed.trips.trip_id)
    if D:
        raise ValueError(f"Trip IDs {D} not found in feed.")

    st = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)]

    g = (
        get_stops(feed, as_geo=True)
        .loc[lambda x: x["stop_id"].isin(st["stop_id"].unique())]
        .merge(st)
        .sort_values(["trip_id", "stop_sequence"])
        .drop_duplicates(subset=["trip_id", "stop_id"])
    )

    return hp.drop_feature_ids(json.loads(g.to_json()))
