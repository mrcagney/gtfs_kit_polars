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

    def empty():
        return pl.LazyFrame(schema={"service_id": pl.Utf8})

    # Weekday column name (e.g., "monday")
    weekday_str = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()

    active_1 = empty()
    if feed.calendar is not None:
        # Filter calendar to services active on date
        active_1 = feed.calendar.filter(
            (pl.col("start_date") <= date)
            & (pl.col("end_date") >= date)
            & (pl.col(weekday_str) == 1)
        ).select("service_id")

    active_2 = empty()
    removed = empty()
    if feed.calendar_dates is not None:
        # Filter calendar_dates to services active on date
        active_2 = feed.calendar_dates.filter(
            (pl.col("date") == date) & (pl.col("exception_type") == 1)
        ).select("service_id")
        # Filter calendar_dates to services removed on date
        removed = feed.calendar_dates.filter(
            (pl.col("date") == date) & (pl.col("exception_type") == 2)
        ).select("service_id")

    # Union active services, then set-difference removed services
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
                f.join(
                    feed.stop_times.select("trip_id", "departure_time"),
                    on="trip_id",
                )
                .with_columns(
                    is_active=(
                        (pl.col("departure_time").min() <= time)
                        & (pl.col("departure_time").max() >= time)
                    ).over("trip_id")
                )
                .filter(pl.col("is_active"))
                .drop("departure_time", "is_active")
                .unique("trip_id")
            )

    if as_geo:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")
        else:
            from .shapes import get_shapes

            f = (
                get_shapes(feed, as_geo=True, use_utm=use_utm)
                .select("shape_id", "geometry")
                .join(f, "shape_id", how="right")
                .select(list(f.collect_schema().names()) + ["geometry"])
            )

    return f


def compute_trip_activity(feed: "Feed", dates: list[str]) -> pd.LazyFrame:
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
    empty table.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pl.LazyFrame(schema={"trip_id": pl.Utf8})

    # Get trip activity table for each day
    frames = [feed.trips.select("trip_id")] + [
        get_trips(feed, date).select("trip_id").with_columns(**{date: pl.lit(1)})
        for date in dates
    ]

    # Join daily trip activity tables into a single table
    return ft.reduce(
        lambda left, right: left.join(right, "trip_id", how="full", coalesce=True),
        frames,
    ).with_columns(pl.col(dates).fill_null(0).cast(pl.Int8))


def compute_busiest_date(feed: "Feed", dates: list[str]) -> str:
    """
    Given a list of dates (YYYYMMDD date strings), return the first date that has the
    maximum number of active trips.
    """
    f = feed.compute_trip_activity(dates)
    s = [(f.select(c).sum(), c) for c in f.collect_schema().names() if c != "trip_id"]
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
    has_dir = "direction_id" in feed.trips.collect_schema().names()
    # Add direction ID placeholder when needed to ease calcs
    trips = (
        feed.trips
        if has_dir
        else feed.trips.with_columns(pl.lit(0).alias("direction_id"))
    )

    # Per-trip stop pattern (ordered stop_ids joined with "-")
    f = (
        # Get stop patterns
        feed.stop_times.sort(["trip_id", "stop_sequence"])
        .group_by("trip_id", maintain_order=True)
        .agg(stop_ids=pl.col("stop_id").implode())
        .with_columns(stop_pattern=pl.col("stop_ids").list.join("-"))
        .select(["trip_id", "stop_pattern"])
        # Attached trip metadat
        .join(
            trips.select("route_id", "trip_id", "direction_id"),
            on="trip_id",
            how="inner",
        )
    )
    # Rank each stop_pattern frequency within (route_id, direction_id)
    ranks = (
        f.group_by("route_id", "direction_id", "stop_pattern")
        .agg(n=pl.len())
        .with_columns(
            # rank 1 = most frequent within (route_id, direction_id)
            rank=pl.col("n")
            .rank(method="dense", descending=True)
            .over(["route_id", "direction_id"])
        )
        .select("route_id", "direction_id", "stop_pattern", "rank")
    )
    # Assign pattern names based on rank
    f = (
        f.join(ranks, on=["route_id", "direction_id", "stop_pattern"], how="left")
        .with_columns(
            stop_pattern_name=pl.concat_str(
                [pl.col("direction_id").cast(pl.Utf8), pl.col("rank").cast(pl.Utf8)],
                separator="-",
            )
        )
        .select("trip_id", "stop_pattern_name")
    )

    result = trips.join(f, on="trip_id", how="left")
    if not has_dir:
        result = result.drop("direction_id")
    return result


def compute_trip_stats(
    feed: "Feed",
    route_ids: list[str | None] = None,
    *,
    compute_dist_from_shapes: bool = False,
) -> pl.LazyFrame:
    """
    Return a table with the following columns:

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
    - ``'is_loop'``: True if the start and end stop are less than 400m apart and
      False otherwise
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
        t = t.with_columns(pl.lit(None, pl.Utf8).alias("shape_id"))

    # Join with stop_times and convert departure times to seconds
    t = (
        t.select("route_id", "trip_id", "direction_id", "shape_id", "stop_pattern_name")
        .join(
            feed.routes.select("route_id", "route_short_name", "route_type"),
            on="route_id",
        )
        .join(feed.stop_times, on="trip_id")
        .sort("trip_id", "stop_sequence")
        .with_columns(dtime=hp.timestr_to_seconds("departure_time"))
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
            start_time=hp.seconds_to_timestr("start_time"),
            end_time=hp.seconds_to_timestr("end_time"),
        )
        .sort("route_id", "direction_id", "start_time")
    )


def locate_trips(feed, date: str, times: list[str]) -> pl.LazyFrame:
    """
    Return the positions of all trips active on the
    given date (YYYYMMDD date string) and times (HH:MM:SS time strings,
    possibly with HH > 23).

    Return a DataFrame with the columns

    - ``'trip_id'``
    - ``'shape_id'``
    - ``'route_id'``
    - ``'direction_id'``: null if ``feed.trips.direction_id`` is
      missing
    - ``'time'``
    - ``'rel_dist'``: number between 0 (start) and 1 (end)
      indicating the relative distance of the trip along its path
    - ``'lon'``: longitude of trip at given time
    - ``'lat'``: latitude of trip at given time

    Assume ``feed.stop_times`` has an accurate ``shape_dist_traveled`` column.
    """
    # Validate required columns
    if "shape_dist_traveled" not in feed.stop_times.collect_schema().names():
        raise ValueError(
            "feed.stop_times needs non-null shape_dist_traveled; "
            "create it with feed.append_dist_to_stop_times()"
        )
    if "shape_id" not in feed.trips.collect_schema().names():
        raise ValueError("feed.trips.shape_id must exist.")

    # Trips active on date (ensure direction_id exists)
    trips = feed.get_trips(date)
    if "direction_id" not in trips.collect_schema().names():
        trips = trips.with_columns(pl.lit(None, dtype=pl.Int32).alias("direction_id"))

    # Trip lengths and target times
    trip_lengths = feed.stop_times.group_by("trip_id").agg(
        trip_length=pl.col("shape_dist_traveled").max()
    )
    times = pl.DataFrame(
        [{"time": t, "time_s": hp.timestr_to_seconds_0(t)} for t in times],
        schema={"time": pl.Utf8, "time_s": pl.Int32},
    ).lazy()

    st = (
        # Reshape stop times into segments with time bounds [t0, t1] and
        # distance bounds [d0, d1]
        feed.stop_times.with_columns(
            dtime=hp.timestr_to_seconds("departure_time"),
            shape_dist_traveled=pl.col("shape_dist_traveled").cast(pl.Float64),
        )
        .sort("dtime", "stop_sequence")
        .with_columns(
            t0=pl.col("dtime"),
            t1=pl.col("dtime").shift(-1).over("trip_id"),
            d0=pl.col("shape_dist_traveled"),
            d1=pl.col("shape_dist_traveled").shift(-1).over("trip_id"),
        )
        .filter(pl.col("t1").is_not_null())
        # explode-friendly shape
        .group_by("trip_id")
        .agg(
            t0=pl.col("t0"),
            t1=pl.col("t1"),
            d0=pl.col("d0"),
            d1=pl.col("d1"),
        )
        .explode(["t0", "t1", "d0", "d1"])
        # Interpolate relative distances at requested times per trip (in stop-time domain)
        .join(trip_lengths, on="trip_id")
        .join(feed.trips.select("trip_id", "shape_id"), on="trip_id")
        .join(times, how="cross")
        .filter((pl.col("time_s") >= pl.col("t0")) & (pl.col("time_s") <= pl.col("t1")))
        # Could have introduced duplicates on boundary overlap above so dedup later.
        # Linearly interpolate distances of sample times.
        .with_columns(
            denom=(pl.col("t1") - pl.col("t0")).cast(pl.Float64),
        )
        .with_columns(
            r=pl.when(pl.col("denom") == 0.0)
            .then(0.0)
            .otherwise(
                (pl.col("time_s") - pl.col("t0")).cast(pl.Float64) / pl.col("denom")
            ),
        )
        .with_columns(
            rel_dist=(pl.col("d0") + pl.col("r") * (pl.col("d1") - pl.col("d0")))
            / pl.col("trip_length"),
        )
        .select("trip_id", "shape_id", "time", "rel_dist")
        # Add in trip info and restrict to trips on given date
        .join(
            trips.select("trip_id", "route_id", "direction_id"),
            on="trip_id",
        )
    )
    # Ensure shapes have cumulative distances; compute if missing
    if "shape_dist_traveled" not in feed.shapes.collect_schema().names():
        from . import shapes as shapes_mod

        feed = shapes_mod.append_dist_to_shapes(feed)

    # Get shape segemnts with distance bounds [d0, d1] and lon-lat bounds [lon0, lon1], [lat0, lat1]
    shapes = (
        feed.shapes.sort(["shape_id", "shape_pt_sequence"])
        .with_columns(
            d0=pl.col("shape_dist_traveled"),
            d1=pl.col("shape_dist_traveled").shift(-1).over(["shape_id"]),
            lon0=pl.col("shape_pt_lon"),
            lon1=pl.col("shape_pt_lon").shift(-1).over(["shape_id"]),
            lat0=pl.col("shape_pt_lat"),
            lat1=pl.col("shape_pt_lat").shift(-1).over(["shape_id"]),
        )
        .filter(pl.col("d1").is_not_null())
        .select("shape_id", "d0", "d1", "lon0", "lon1", "lat0", "lat1")
    )

    # Shape lengths (to scale rel_dist -> absolute along-shape dist)
    shape_lengths = feed.shapes.group_by("shape_id").agg(
        shape_length=pl.col("shape_dist_traveled").max()
    )

    # Interpolate lon/lat for each (trip_id, time) via matching shape segment
    return (
        st.join(shape_lengths, on="shape_id")
        .with_columns(shape_rel_dist=pl.col("rel_dist") * pl.col("shape_length"))
        .join(shapes, on="shape_id")
        .filter(
            (pl.col("shape_rel_dist") >= pl.col("d0"))
            & (pl.col("shape_rel_dist") <= pl.col("d1"))
        )
        # Could have introduced duplicates on boundary overlap above so dedup later,
        # Linearly interpolate lon-lats from relative shape relative dists.
        .with_columns(
            denom=(pl.col("d1") - pl.col("d0")).cast(pl.Float64),
        )
        .with_columns(
            r=pl.when(pl.col("denom") == 0.0)
            .then(0.0)
            .otherwise(
                (pl.col("shape_rel_dist") - pl.col("d0")).cast(pl.Float64)
                / pl.col("denom")
            ),
        )
        .with_columns(
            lon=pl.col("lon0") + pl.col("r") * (pl.col("lon1") - pl.col("lon0")),
            lat=pl.col("lat0") + pl.col("r") * (pl.col("lat1") - pl.col("lat0")),
        )
        .select(
            "trip_id",
            "shape_id",
            "route_id",
            "direction_id",
            "time",
            "rel_dist",
            "lon",
            "lat",
        )
        .unique(subset=["trip_id", "time"], keep="first")
    )


def trips_to_geojson(
    feed: "Feed",
    trip_ids: Iterable[str] | None = None,
    *,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection (in WGS84 coordinates) of LineString features
    representing all the Feed's trips.

    If ``include_stops``, then include the trip stops as Point features.
    If an iterable of trip IDs is given, then subset to those trips, which could yield
    an empty FeatureCollection in case of invalid trip IDs.
    """
    g = get_trips(feed, as_geo=True)
    if trip_ids:
        g = g.filter(pl.col("trip_id").is_in(trip_ids))

    if g is None or hp.is_empty(g):
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = g.collect().st.__geo_interface__
        if include_stops:
            st_gj = feed.stop_times_to_geojson(trip_ids)
            result["features"].extend(st_gj["features"])

    return result


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
