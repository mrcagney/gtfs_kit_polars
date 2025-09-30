"""
Functions about trips.
"""

from __future__ import annotations

import datetime as dt
import functools as ft
import json
from typing import TYPE_CHECKING, Iterable

import folium as fl
import folium.plugins as fp
import numpy as np
import pandas as pd
import polars as pl
import shapely.geometry as sg
import shapely.ops as so

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def get_active_services(feed: "Feed", date: str) -> list[str]:
    """
    Given a Feed and a date string in YYYYMMDD format,
    return the service IDs that are active on the date.
    """

    # helper: empty one-col LazyFrame
    def _empty():
        return pl.LazyFrame(schema={"service_id": pl.Utf8})

    # Weekday column name (e.g., "monday")
    weekday_str = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()

    # calendar: services scheduled on this date (start<=date<=end and weekday==1)
    active_1 = _empty()
    if feed.calendar is not None:
        active_1 = feed.calendar.filter(
            (pl.col("start_date") <= date)
            & (pl.col("end_date") >= date)
            & (pl.col(weekday_str) == 1)
        ).select("service_id")

    # calendar_dates: services explicitly added on date (exception_type==1)
    active_2 = _empty()
    # calendar_dates: services explicitly removed on date (exception_type==2)
    removed = _empty()
    if feed.calendar_dates is not None:
        active_2 = feed.calendar_dates.filter(
            (pl.col("date") == date) & (pl.col("exception_type") == 1)
        ).select("service_id")
        removed = feed.calendar_dates.filter(
            (pl.col("date") == date) & (pl.col("exception_type") == 2)
        ).select("service_id")

    # Union active parts, then set-difference removed
    return (
        pl.concat([active_1, active_2])
        .unique()
        .join(removed, on="service_id", how="anti")
        .collect()["service_id"]
        .to_list()
    )


def get_trips(
    feed: "Feed",
    date: str | None = None,
    time: str | None = None,
    *,
    as_geo: bool = False,
    use_utm: bool = False,
) -> pl.LazyFrame:
    """
    Return ``feed.trips``.
    If date (YYYYMMDD date string) is given then subset the result to trips
    that start on that date.
    If a time (HH:MM:SS string, possibly with HH > 23) is given in addition to a date,
    then further subset the result to trips in service at that time.

    If ``as_geo`` and ``feed.shapes`` is not None, then return the trips as a
    GeoDataFrame of LineStrings representating trip shapes.
    Use local UTM CRS if ``use_utm``; otherwise it the WGS84 CRS.
    If ``as_geo`` and ``feed.shapes`` is ``None``, then raise a ValueError.
    """
    if feed.trips is None:
        return None

    f = feed.trips
    if date is not None:
        f = f.filter(pl.col("service_id").is_in(get_active_services(feed, date)))

        if time is not None:
            # Get trips active during given time
            f = (
                f.join_where(
                    feed.stop_times["trip_id", "departure_time"],
                    pl.col("trip_id") == feed.stop_times["trip_id"],
                )
                .group_by("trip_id")
                .with_columns(
                    is_active=(
                        (pl.col("departure_time").min() <= time)
                        & (pl.col("departure_time").max() >= time)
                    )
                )
                .filter(pl.col("is_active"))
                .drop("departure_time", "is_active")
                .distinct(on="trip_id")
            )

    if as_geo:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")
        else:
            from .shapes import get_shapes

            f = (
                get_shapes(feed, as_geo=True, use_utm=use_utm)
                .select(["shape_id", "geometry", "crs"])
                .join(f, on=f["shape_id"], how="right")
                .select(list(f.collect_schema().names()) + ["geometry", "crs"])
            )

    return f


def compute_trip_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark trips as active or inactive on the given dates (YYYYMMDD date strings).
    Return a table with the columns

    - ``'trip_id'``
    - ``dates[0]``: 1 if the trip is active on ``dates[0]``;
      0 otherwise
    - ``dates[1]``: 1 if the trip is active on ``dates[1]``;
      0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``;
      0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an
    empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pd.DataFrame()

    # Get trip activity table for each day
    frames = [feed.trips[["trip_id"]]]
    for date in dates:
        frames.append(get_trips(feed, date)[["trip_id"]].assign(**{date: 1}))

    # Merge daily trip activity tables into a single table
    f = ft.reduce(lambda left, right: left.merge(right, how="outer"), frames).fillna(
        {date: 0 for date in dates}
    )
    f[dates] = f[dates].astype(int)
    return f


def compute_busiest_date(feed: "Feed", dates: list[str]) -> str:
    """
    Given a list of dates (YYYYMMDD date strings), return the first date that has the
    maximum number of active trips.
    """
    f = feed.compute_trip_activity(dates)
    s = [(f[c].sum(), c) for c in f.columns if c != "trip_id"]
    return max(s)[1]


def name_stop_patterns(feed: "Feed") -> pl.LazyFrame:
    """
    For each (route ID, direction ID) pair, find the distinct stop patterns of its
    trips, and assign them each an integer *pattern rank* based on the stop pattern's
    frequency rank, where 1 is the most frequent stop pattern, 2 is the second most
    frequent, etc.
    Return the table ``feed.trips`` with the additional column
    ``stop_pattern_name``, which equals the trip's 'direction_id' concatenated with a
    dash and its stop pattern rank.

    If ``feed.trips`` has no 'direction_id' column, then temporarily create one equal
    to all zeros, proceed as above, then delete the column.
    """
    t = feed.trips
    has_dir = "direction_id" in t.collect_schema().names()
    tt = t if has_dir else t.with_columns(pl.lit(0).alias("direction_id"))

    # Per-trip stop pattern (ordered stop_ids joined with "-")
    s = (
        feed.stop_times.sort(["trip_id", "stop_sequence"])
        .group_by("trip_id", maintain_order=True)
        .agg(stop_ids=pl.col("stop_id").implode())
        .with_columns(stop_pattern=pl.col("stop_ids").list.join("-"))
        .select(["trip_id", "stop_pattern"])
    )

    # Attach trip metadata
    f = s.join(
        tt.select(["route_id", "trip_id", "direction_id"]), on="trip_id", how="inner"
    )

    # Count frequency of each stop_pattern within (route_id, direction_id)
    c = (
        f.group_by(["route_id", "direction_id", "stop_pattern"])
        .agg(n=pl.len())
        .with_columns(
            # rank 1 = most frequent within (route_id, direction_id)
            rank=pl.col("n")
            .rank(method="dense", descending=True)
            .over(["route_id", "direction_id"])
        )
        .select(["route_id", "direction_id", "stop_pattern", "rank"])
    )

    # Join ranks back and build "direction-rank" names
    g = (
        f.join(c, on=["route_id", "direction_id", "stop_pattern"], how="left")
        .with_columns(
            stop_pattern_name=pl.concat_str(
                [pl.col("direction_id").cast(pl.Utf8), pl.col("rank").cast(pl.Utf8)],
                separator="-",
            )
        )
        .select(["trip_id", "stop_pattern_name"])
    )

    out = tt.join(g, on="trip_id", how="left")
    if not has_dir:
        out = out.drop("direction_id")
    return out


def compute_trip_stats(
    feed: "Feed",
    route_ids: list[str | None] = None,
    *,
    compute_dist_from_shapes: bool = False,
):
    """
    Return a DataFrame with the following columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: null if missing from feed
    - ``'shape_id'``: null if missing from feed
    - ``'stop_pattern_name'``: output from :func:`name_stop_patterns`
    - ``'num_stops'``: number of stops on trip
    - ``'start_time'``: first departure time of the trip
    - ``'end_time'``: last departure time of the trip
    - ``'start_stop_id'``: stop ID of the first stop of the trip
    - ``'end_stop_id'``: stop ID of the last stop of the trip
    - ``'is_loop'``: 1 if the start and end stop are less than 400m apart and
      0 otherwise
    - ``'distance'``: distance of the trip;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all null entries if ``feed.shapes is None``
    - ``'duration'``: duration of the trip in hours
    - ``'speed'``: distance/duration

    If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at
    least one non-null value and ``compute_dist_from_shapes == False``,
    then use that column to compute the distance column.
    Else if ``feed.shapes is not None``, then compute the distance
    column using the shapes and Shapely.
    Otherwise, set the distances to null.

    If route IDs are given, then restrict to trips on those routes.

    Notes
    -----
    - Assume the following feed attributes are not ``None``:

        * ``feed.trips``
        * ``feed.routes``
        * ``feed.stop_times``
        * ``feed.shapes`` (optionally)

    - Calculating trip distances with ``compute_dist_from_shapes=True``
      seems pretty accurate.  For example, calculating trip distances on
      `this Portland feed
      <https://transitfeeds.com/p/trimet/43/1400947517>`_
      using ``compute_dist_from_shapes=False`` and
      ``compute_dist_from_shapes=True``,
      yields a difference of at most 0.83km from the original values.

    """
    # Trips with stop pattern names
    t = name_stop_patterns(feed)
    if route_ids is not None:
        t = t.filter(pl.col("route_id").is_in(route_ids))

    # Ensure columns exist (nulls if missing)
    if "direction_id" not in t.collect_schema().names():
        t = t.with_columns(pl.lit(None).alias("direction_id"))
    if "shape_id" not in t.collect_schema().names():
        t = t.with_columns(pl.lit(None).alias("shape_id"))

    # Join with stop_times and convert departure times to seconds
    t = (
        t.select("route_id", "trip_id", "direction_id", "shape_id", "stop_pattern_name")
        .join(
            feed.routes.select("route_id", "route_short_name", "route_type"),
            on="route_id",
        )
        .join(feed.stop_times, on="trip_id")
        .sort("trip_id", "stop_sequence")
        .with_columns(dtime=hp.timestr_to_seconds_pl("departure_time"))
    )
    # Compute most trip stats
    stops_g = feed.get_stops(as_geo=True, use_utm=True)
    trip_stats = (
        t.group_by("trip_id")
        .agg(
            route_id=pl.col("route_id").first(),
            route_short_name=pl.col("route_short_name").first(),
            route_type=pl.col("route_type").first(),
            direction_id=pl.col("direction_id").first(),
            shape_id=pl.col("shape_id").first(),
            stop_pattern_name=pl.col("stop_pattern_name").first(),
            num_stops=pl.col("stop_id").count(),
            start_time=pl.col("dtime").min(),
            end_time=pl.col("dtime").max(),
            start_stop_id=pl.col("stop_id").first(),
            end_stop_id=pl.col("stop_id").last(),
            duration_s=pl.col("dtime").max() - pl.col("dtime").min(),
        )
        .join(
            stops_g.select(start_stop_id="stop_id", start_geom="geometry"),
            on="start_stop_id",
            how="left",
        )
        .join(
            stops_g.select(end_stop_id="stop_id", end_geom="geometry"),
            on="end_stop_id",
            how="left",
        )
        .with_columns(
            duration=pl.col("duration_s") / 3600.0,
            is_loop=(
                pl.col("start_geom").st.distance(pl.col("end_geom")) < 400
            ).fill_null(False),
        )
        .drop("duration_s")
    )
    # Compute distance
    if (
        hp.is_not_null(feed.stop_times, "shape_dist_traveled")
        and not compute_dist_from_shapes
    ):
        conv = (
            hp.get_convert_dist(feed.dist_units, "km")
            if hp.is_metric(feed.dist_units)
            else hp.get_convert_dist(feed.dist_units, "mi")
        )
        d = (
            t.group_by("trip_id")
            .agg(distance=pl.col("shape_dist_traveled").max())
            .with_columns(
                # Could speed this up with native Polars conv function
                distance=conv(pl.col("distance"))
            )
        )
        trip_stats = trip_stats.join(d, "trip_id", how="left")

    elif feed.shapes is not None:
        conv = hp.get_convert_dist("m", feed.dist_units)
        d = (
            feed.get_shapes(as_geo=True, use_utm=True)
            .with_columns(
                is_simple=pl.col("geometry").st.is_simple(),
                D=pl.col("geometry").st.length(),
            )
            .join(
                trip_stats.select("trip_id", "shape_id", "start_geom", "end_geom"),
                "shape_id",
            )
            .with_columns(
                d=(
                    # If simple linestring, then compute its length
                    pl.when("is_simple")
                    .then("D")
                    # Otherwise, compute distance from first stop to last stop along linestring
                    .otherwise(
                        pl.col("geometry").st.project("end_geom")
                        - pl.col("geometry").st.project("start_geom")
                    )
                )
            )
            # Assign distance based on ``d`` and ``D``
            .with_columns(
                distance=(
                    pl.when((0 < pl.col("d")) & (pl.col("d") < pl.col("D") + 100))
                    .then("d")
                    .otherwise("D")
                )
            )
            # Convert to feed dist units
            .select(
                "trip_id",
                # Could speed this up with native Polars conv function
                distance=conv(pl.col("distance")),
            )
        )
        trip_stats = trip_stats.join(d, "trip_id", how="left")
    else:
        trip_stats = trip_stats.with_columns(distance=pl.lit(None, dtype=pl.Float64))

    # Compute speed and finalize
    return (
        trip_stats.drop("start_geom", "end_geom")
        .with_columns(
            speed=pl.col("distance") / pl.col("duration"),
            start_time=hp.seconds_to_timestr_pl("start_time"),
            end_time=hp.seconds_to_timestr_pl("end_time"),
        )
        .sort("route_id", "direction_id", "start_time")
    )


def locate_trips(feed: "Feed", date: str, times: list[str]) -> pd.DataFrame:
    """
    Return the positions of all trips active on the
    given date (YYYYMMDD date string) and times (HH:MM:SS time strings,
    possibly with HH > 23).

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``'route_id'``
    - ``'direction_id'``: all NaNs if ``feed.trips.direction_id`` is
      missing
    - ``'time'``
    - ``'rel_dist'``: number between 0 (start) and 1 (end)
      indicating the relative distance of the trip along its path
    - ``'lon'``: longitude of trip at given time
    - ``'lat'``: latitude of trip at given time

    Assume ``feed.stop_times`` has an accurate
    ``shape_dist_traveled`` column.
    """
    if not hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        raise ValueError(
            "feed.stop_times needs to have a non-null shape_dist_traveled "
            "column. You can create it, possibly with some inaccuracies, "
            "via feed2 = feed.append_dist_to_stop_times()."
        )

    if "shape_id" not in feed.trips.columns:
        raise ValueError("feed.trips.shape_id must exist.")

    # Start with stop times active on date
    f = feed.get_stop_times(date)
    f["departure_time"] = f["departure_time"].map(hp.timestr_to_seconds)

    # Compute relative distance of each trip along its path
    # at the given time times.
    # Use linear interpolation based on stop departure times and
    # shape distance traveled.
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    sample_times = np.array([hp.timestr_to_seconds(s) for s in times])

    def compute_rel_dist(group):
        dists = sorted(group["shape_dist_traveled"].values)
        times = sorted(group["departure_time"].values)
        ts = sample_times[(sample_times >= times[0]) & (sample_times <= times[-1])]
        ds = np.interp(ts, times, dists)
        return pd.DataFrame({"time": ts, "rel_dist": ds / dists[-1]})

    # return f.groupby('trip_id', group_keys=False).\
    #   apply(compute_rel_dist).reset_index()
    g = f.groupby("trip_id").apply(compute_rel_dist, include_groups=False).reset_index()

    # Delete extraneous multi-index column
    del g["level_1"]

    # Convert times back to time strings
    g["time"] = g["time"].map(lambda x: hp.seconds_to_timestr(x))

    # Merge in more trip info and
    # compute longitude and latitude of trip from relative distance
    t = feed.trips.copy()
    if "direction_id" not in t.columns:
        t["direction_id"] = np.nan

    h = pd.merge(g, t[["trip_id", "route_id", "direction_id", "shape_id"]])
    if not h.shape[0]:
        # Return a DataFrame with the promised headers but no data.
        # Without this check, result below could be an empty DataFrame.
        h["lon"] = pd.Series()
        h["lat"] = pd.Series()
        return h

    def get_lonlat(group):
        shape = group.name
        linestring = geometry_by_shape[shape]
        lonlats = [
            linestring.interpolate(d, normalized=True).coords[0]
            for d in group["rel_dist"].values
        ]
        group["lon"], group["lat"] = zip(*lonlats)
        return group

    return (
        h.groupby("shape_id")
        .apply(get_lonlat, include_groups=False)
        .reset_index()
        .drop("level_1", axis=1)  # Where did this column come from?
    )


def trips_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
    *,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection of LineString features representing
    all the Feed's trips.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If ``include_stops``, then include the trip stops as Point features.
    If an iterable of trip IDs is given, then subset to those trips.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If the Feed has no shapes, then raise a ValueError.
    """
    if trip_ids is None or not list(trip_ids):
        trip_ids = feed.trips.trip_id

    D = set(trip_ids) - set(feed.trips.trip_id)
    if D:
        raise ValueError(f"Trip IDs {D} not found in feed.")

    # Get trips
    g = get_trips(feed, as_geo=True).loc[lambda x: x["trip_id"].isin(trip_ids)]
    trips_gj = json.loads(g.to_json())

    # Get stops if desired
    if include_stops:
        st_gj = feed.stop_times_to_geojson(trip_ids)
        trips_gj["features"].extend(st_gj["features"])

    return hp.drop_feature_ids(trips_gj)


def map_trips(
    feed: "Feed",
    trip_ids: Iterable[str],
    color_palette: list[str] = cs.COLORS_SET2,
    *,
    show_stops: bool = False,
    show_direction: bool = False,
):
    """
    Return a Folium map showing the given trips and (optionally)
    their stops.
    If any of the given trip IDs are not found in the feed, then raise a ValueError.
    If ``include_direction``, then use the Folium plugin PolyLineTextPath to draw arrows
    on each trip polyline indicating its direction of travel; this fails to work in some
    browsers, such as Brave 0.68.132.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Create colors
    n = len(trip_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, trip_id in enumerate(trip_ids):
        collection = trips_to_geojson(feed, [trip_id], include_stops=show_stops)

        group = fl.FeatureGroup(name=f"Trip {trip_id}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop if present
            if f["geometry"]["type"] == "Point":
                lon, lat = f["geometry"]["coordinates"]
                fl.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    fill=True,
                    color=color,
                    weight=1,
                    popup=fl.Popup(hp.make_html(prop)),
                ).add_to(group)

            # Add trip
            else:
                path = fl.PolyLine(
                    [[x[1], x[0]] for x in f["geometry"]["coordinates"]],
                    color=color,
                    popup=hp.make_html(prop),
                )

                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

                if show_direction:
                    # Direction arrows, assuming, as GTFS does, that
                    # trip direction equals LineString direction
                    fp.PolyLineTextPath(
                        path,
                        "        \u27a4        ",
                        repeat=True,
                        offset=5.5,
                        attributes={"fill": color, "font-size": "18"},
                    ).add_to(group)

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    # Folium wants a different ordering
    bounds = [(bounds[1], bounds[0]), (bounds[3], bounds[2])]
    my_map.fit_bounds(bounds)

    return my_map
