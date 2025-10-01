"""
Functions about routes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

import folium as fl
import polars as pl
import polars_st as st
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg
import shapely.ops as so

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def build_route_timetable(
    feed: "Feed", route_id: str, dates: list[str]
) -> pd.DataFrame:
    """
    Return a timetable for the given route and dates (YYYYMMDD date strings).

    Return a DataFrame with whose columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times`` plus ``'date'``.
    The trip IDs are restricted to the given route ID.
    The result is sorted first by date and then by grouping by
    trip ID and sorting the groups by their first departure time.

    Skip dates outside of the Feed's dates.

    If there is no route activity on the given dates, then return
    an empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    final_cols = (
        ["date"] + feed.trips.columns.tolist() + feed.stop_times.columns.tolist()
    )
    if not dates:
        return pd.DataFrame(columns=final_cols)

    t = pd.merge(feed.trips, feed.stop_times)
    t = t[t["route_id"] == route_id].copy()
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to trips active on date
        ids = a.loc[a[date] == 1, "trip_id"]
        f = t[t["trip_id"].isin(ids)].copy()
        f["date"] = date
        # Groupby trip ID and sort groups by their minimum departure time.
        # For some reason NaN departure times mess up the transform below.
        # So temporarily fill NaN departure times as a workaround.
        f["dt"] = f["departure_time"].ffill().map(hp.timestr_to_seconds)
        f["min_dt"] = f.groupby("trip_id")["dt"].transform("min")
        frames.append(f)

    return (
        pd.concat(frames)
        .sort_values(["date", "min_dt", "stop_sequence"], ignore_index=True)
        .filter(final_cols)
    )


def get_routes(
    feed: "Feed",
    date: str | None = None,
    time: str | None = None,
    *,
    as_geo: bool = False,
    use_utm: bool = False,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Return ``feed.routes`` or a subset thereof.
    If a YYYYMMDD date string is given, then restrict routes to only those active on
    the date.
    If a HH:MM:SS time string is given, possibly with HH > 23, then restrict routes
    to only those active during the time.

    Given a Feed, return a GeoDataFrame with all the columns of ``feed.routes``
    plus a geometry column of (Multi)LineStrings, each of which represents the
    corresponding routes's shape.

    If ``as_geo`` and ``feed.shapes`` is not ``None``,
    then return a GeoDataFrame with all the columns of ``feed.routes``
    plus a geometry column of (Multi)LineStrings, each of which represents the
    corresponding routes's union of trip shapes.
    The GeoDataFrame will have a local UTM CRS if ``use_utm``; otherwise it will have
    CRS WGS84.
    If ``split_directions`` and ``as_geo``, then add the column ``direction_id`` and
    split each route into the union of its direction 0 shapes
    and the union of its direction 1 shapes.
    If ``as_geo`` and ``feed.shapes`` is ``None``, then raise a ValueError.
    """
    from .trips import get_trips

    trips = get_trips(feed, date=date, time=time, as_geo=as_geo, use_utm=use_utm)
    f = feed.routes[lambda x: x["route_id"].isin(trips["route_id"])]

    if as_geo:
        if feed.shapes is None:
            raise ValueError("This Feed has no shapes.")

        if split_directions:
            groupby_cols = ["route_id", "direction_id"]
            final_cols = f.columns.tolist() + ["direction_id", "geometry"]
        else:
            groupby_cols = ["route_id"]
            final_cols = f.columns.tolist() + ["geometry"]

        # def merge_lines(group):
        #     d = {}
        #     d["geometry"] = so.linemerge(group["geometry"].tolist())
        #     return pd.Series(d)

        def merge_lines(group):
            lines = [
                g
                for g in group["geometry"]
                if g.geom_type in ["LineString", "MultiLineString"]
            ]
            if not lines:
                return pd.Series({"geometry": None})
            return pd.Series({"geometry": so.linemerge(lines)})

        f = (
            trips
            # Drop unnecessary duplicate shapes
            .drop_duplicates(subset=["shape_id", "route_id"])
            .filter(groupby_cols + ["geometry"])
            .groupby(groupby_cols)
            .apply(merge_lines, include_groups=False)
            .reset_index()
            .merge(f, how="right")
            .pipe(gpd.GeoDataFrame)
            .set_crs(trips.crs)
            .filter(final_cols)
        )

    return f


def routes_to_geojson(
    feed: "Feed",
    route_ids: Iterable[str | None] = None,
    *,
    split_directions: bool = False,
    include_stops: bool = False,
) -> dict:
    """
    Return a GeoJSON FeatureCollection of MultiLineString features representing this Feed's routes.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If ``include_stops``, then include the route stops as Point features .
    If an iterable of route IDs is given, then subset to those routes.
    If the subset is empty, then return a FeatureCollection with an empty list of
    features.
    If the Feed has no shapes, then raise a ValueError.
    If any of the given route IDs are not found in the feed, then raise a ValueError.
    """
    if route_ids is None or not list(route_ids):
        route_ids = feed.routes.route_id

    D = set(route_ids) - set(feed.routes.route_id)
    if D:
        raise ValueError(f"Route IDs {D} not found in feed.")

    # Get routes
    g = get_routes(feed, as_geo=True, split_directions=split_directions).loc[
        lambda x: x["route_id"].isin(route_ids)
    ]
    collection = json.loads(g.to_json())

    # Get stops if desired
    if len(route_ids) and include_stops:
        stop_ids = (
            feed.stop_times.merge(feed.trips.filter(["trip_id", "route_id"]))
            .loc[lambda x: x.route_id.isin(route_ids), "stop_id"]
            .unique()
        )
        stops_gj = feed.stops_to_geojson(stop_ids=stop_ids)
        collection["features"].extend(stops_gj["features"])

    return hp.drop_feature_ids(collection)


def map_routes(
    feed: "Feed",
    route_ids: Iterable[str] | None = None,
    route_short_names: Iterable[str] | None = None,
    color_palette: Iterable[str] = cs.COLORS_SET2,
    *,
    show_stops: bool = False,
):
    """
    Return a Folium map showing the given routes and (optionally) their stops.
    At least one of ``route_ids`` and ``route_short_names`` must be given.
    If both are given, then combine the two into a single set of routes.
    If any of the given route IDs are not found in the feed, then raise a ValueError.
    """
    # Compile route IDs
    R = set()
    if route_short_names is not None:
        R |= set(
            feed.routes.loc[
                lambda x: x["route_short_name"].isin(route_short_names), "route_id"
            ]
        )
    if route_ids is not None:
        R |= set(route_ids)
    route_ids = sorted(R)
    if not R:
        raise ValueError("Route IDs or route short names must be given")

    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron", prefer_canvas=True)

    # Create route colors
    n = len(route_ids)
    colors = [color_palette[i % len(color_palette)] for i in range(n)]

    # Collect route bounding boxes to set map zoom later
    bboxes = []

    # Create a feature group for each route and add it to the map
    for i, route_id in enumerate(route_ids):
        collection = feed.routes_to_geojson(
            route_ids=[route_id], include_stops=show_stops
        )

        # Use route short name for group name if possible; otherwise use route ID
        route_name = route_id
        for f in collection["features"]:
            if "route_short_name" in f["properties"]:
                route_name = f["properties"]["route_short_name"]
                break

        group = fl.FeatureGroup(name=f"Route {route_name}")
        color = colors[i]

        for f in collection["features"]:
            prop = f["properties"]

            # Add stop
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

            # Add path
            else:
                prop["color"] = color
                path = fl.GeoJson(
                    f,
                    name=prop["route_short_name"],
                    style_function=lambda x: {"color": x["properties"]["color"]},
                )
                path.add_child(fl.Popup(hp.make_html(prop)))
                path.add_to(group)
                bboxes.append(sg.box(*sg.shape(f["geometry"]).bounds))

        group.add_to(my_map)

    fl.LayerControl().add_to(my_map)

    # Fit map to bounds
    bounds = so.unary_union(bboxes).bounds
    bounds2 = [bounds[1::-1], bounds[3:1:-1]]  # Folium expects this ordering
    my_map.fit_bounds(bounds2)

    return my_map


def compute_route_stats_0(
    trip_stats: pl.DataFrame | pl.LazyFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pl.LazyFrame:
    """
    Compute stats for the given subset of trip stats (of the form output by the
    function :func:`.trips.compute_trip_stats`).

    Ignore trips with zero duration, because they are defunct.

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a table with the columns

    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: present if only if ``split_directions``
    - ``'num_trips'``: number of trips on the route in the subset
    - ``'num_trip_starts'``: number of trips on the route with
      nonnull start times
    - ``'num_trip_ends'``: number of trips on the route with nonnull
      end times that end before 23:59:59
    - ``'num_stop_patterns'``: number of stop pattern across trips
    - ``'is_loop'``: 1 if at least one of the trips on the route has
      its ``is_loop`` field equal to 1; 0 otherwise
    - ``'is_bidirectional'``: 1 if the route has trips in both
      directions; 0 otherwise; present if only if not ``split_directions``
    - ``'start_time'``: start time of the earliest trip on the route
    - ``'end_time'``: end time of latest trip on the route
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip starts on the route between
      ``headway_start_time`` and ``headway_end_time`` on the given
      dates
    - ``'min_headway'``: minimum of the durations (in minutes)
      mentioned above
    - ``'mean_headway'``: mean of the durations (in minutes)
      mentioned above
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service (for the given direction, or for both directions when
      ``split_directions==False``)
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs
    - ``'service_duration'``: total of the duration of each trip on
      the route in the given subset of trips; measured in hours
    - ``'service_distance'``: total of the distance traveled by each
      trip on the route in the given subset of trips;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_speed'``: service_distance/service_duration
    - ``'mean_trip_distance'``: service_distance/num_trips
    - ``'mean_trip_duration'``: service_duration/num_trips

    If ``trip_stats`` is empty, return an empty table.

    Raise a ValueError if ``split_directions`` and no non-NaN
    direction ID values present
    """
    # Coerce trip stats to LazyFrame
    f = trip_stats if isinstance(trip_stats, pl.LazyFrame) else trip_stats.lazy()

    # Handle defunct case
    schema = {
        "route_id": pl.Utf8,
        "route_short_name": pl.Utf8,
        "route_type": pl.Int32,
        "num_trips": pl.Int32,
        "num_trip_starts": pl.Int32,
        "num_trip_ends": pl.Int32,
        "num_stop_patterns": pl.Int32,
        "is_loop": pl.Boolean,
        "start_time": pl.Utf8,
        "end_time": pl.Utf8,
        "max_headway": pl.Float64,
        "min_headway": pl.Float64,
        "mean_headway": pl.Float64,
        "peak_num_trips": pl.Int32,
        "peak_start_time": pl.Utf8,
        "peak_end_time": pl.Utf8,
        "service_distance": pl.Float64,
        "service_duration": pl.Float64,
        "service_speed": pl.Float64,
        "mean_trip_distance": pl.Float64,
        "mean_trip_duration": pl.Float64,
    }
    final_cols = list(schema.keys())
    if split_directions:
        schema |= {"direction_id": pl.Int32}
        final_cols.insert(1, "direction_id")
    else:
        schema |= {"is_bidirectional": pl.Boolean}
        final_cols.insert(1, "is_bidirectional")

    if hp.is_empty(f):
        return pl.LazyFrame(schema=schema)

    # Handle generic case
    if split_directions:
        if "direction_id" not in f.collect_schema().names():
            f = f.with_columns(pl.lit(None, dtype=pl.Int32).alias("direction_id"))
        has_dir = f.select(pl.col("direction_id").is_not_null().any()).collect().item()
        if not has_dir:
            raise ValueError(
                "At least one trip stats direction ID value must be non-NULL."
            )
        group_cols = ["route_id", "direction_id"]
    else:
        group_cols = ["route_id"]

    # Prepare trips subset
    needed_cols = group_cols + [
        "trip_id",
        "route_short_name",
        "route_type",
        "start_time",
        "end_time",
        "duration",
        "distance",
        "stop_pattern_name",
        "is_loop",
    ]
    if (
        "direction_id" not in group_cols
        and "direction_id" in f.collect_schema().names()
    ):
        needed_cols.append("direction_id")

    f = (
        f.filter(pl.col("duration") > 0)
        .select(needed_cols)
        .with_columns(
            start_s=hp.timestr_to_seconds("start_time"),
            end_s=hp.timestr_to_seconds("end_time"),
        )
    )
    if split_directions:
        f = f.filter(pl.col("direction_id").is_not_null()).with_columns(
            pl.col("direction_id").cast(pl.Int32)
        )

    # Compute basic stats
    basic_stats = f.group_by(group_cols).agg(
        route_short_name=pl.first("route_short_name"),
        route_type=pl.first("route_type"),
        num_trips=pl.len(),
        num_trip_starts=pl.col("start_s").is_not_null().sum(),
        num_trip_ends=pl.col("end_s").is_not_null().sum(),
        num_stop_patterns=pl.col("stop_pattern_name").n_unique(),
        is_loop=pl.col("is_loop").any(),
        start_s_min=pl.col("start_s").min(),
        end_s_max=pl.col("end_s").max(),
        service_distance=pl.col("distance").sum(),
        service_duration=pl.col("duration").sum(),
        is_bidirectional=None
        if split_directions
        else (pl.col("direction_id").n_unique() > 1),
    )
    # Compute headway stats
    h_start = hp.timestr_to_seconds_0(headway_start_time)
    h_end = hp.timestr_to_seconds_0(headway_end_time)
    headway_stats = (
        f.filter(pl.col("start_s").is_between(h_start, h_end, closed="both"))
        .select(group_cols + ["start_s"])
        .sort(group_cols + ["start_s"])
        .with_columns(
            prev=pl.col("start_s").shift(1).over(group_cols),
        )
        .with_columns(
            headway_m=(pl.col("start_s") - pl.col("prev")) / 60.0,
        )
        .filter(pl.col("headway_m").is_not_null())
        .group_by(group_cols)
        .agg(
            max_headway=pl.col("headway_m").max(),
            min_headway=pl.col("headway_m").min(),
            mean_headway=pl.col("headway_m").mean(),
        )
    )
    # Compute peak stats, the tricky part.
    # Create events table with +1 for trip starts, -1 for trip ends
    events = (
        f.select(group_cols + ["start_s", "end_s"])
        .unpivot(
            index=group_cols,
            on=["start_s", "end_s"],
            variable_name="event_type",
            value_name="t",
        )
        .filter(pl.col("t").is_not_null())
        .with_columns(
            delta=pl.when(pl.col("event_type") == "start_s").then(1).otherwise(-1)
        )
        .sort(
            group_cols + ["t", "delta"],
            descending=False,
        )
        # Get cumulative sum of all trips in service per time slot
        .group_by(group_cols + ["t"], maintain_order=True)
        .agg(
            num_trip_starts=pl.col("delta").sum(),
        )
        .with_columns(num_trips=pl.col("num_trip_starts").cum_sum().over(group_cols))
        .drop("num_trip_starts")
    )
    peak_vals = events.group_by(group_cols).agg(
        peak_num_trips=pl.col("num_trips").max()
    )
    peak_periods = (
        events.join(peak_vals, on=group_cols)
        .with_columns(t_next=pl.col("t").shift(-1).over(group_cols))
        .with_columns(duration=pl.col("t_next") - pl.col("t"))
        .filter(pl.col("num_trips") == pl.col("peak_num_trips"))
        .filter(pl.col("duration") == pl.max("duration").over(group_cols))
        .sort(group_cols + ["t"])
        .group_by(group_cols)
        .agg(
            peak_start_s=pl.first("t"),
            peak_end_s=pl.first("t_next"),
        )
    )
    peak_stats = peak_vals.join(peak_periods, group_cols, how="left")

    # Collate stats
    return (
        basic_stats.join(headway_stats, on=group_cols, how="left")
        .join(peak_stats, on=group_cols, how="left")
        .with_columns(
            start_time=hp.seconds_to_timestr("start_s_min"),
            end_time=hp.seconds_to_timestr("end_s_max"),
            peak_start_time=hp.seconds_to_timestr("peak_start_s"),
            peak_end_time=hp.seconds_to_timestr("peak_end_s"),
            service_speed=pl.col("service_distance")
            / pl.col("service_duration").replace(0, None),
            mean_trip_distance=pl.col("service_distance") / pl.col("num_trips"),
            mean_trip_duration=pl.col("service_duration") / pl.col("num_trips"),
        )
        .select(final_cols)
    )


def compute_route_stats(
    feed: "Feed",
    dates: list[str],
    trip_stats: pl.DataFrame | pl.LazyFrame | None = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pl.LazyFrame:
    """
    Compute route stats for all the trips that lie in the given subset
    of trip stats, which defaults to ``feed.compute_trip_stats()``,
    and that start on the given dates (YYYYMMDD date strings).

    If ``split_directions``, then separate the stats by trip direction (0 or 1).
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'date'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'direction_id'``: present if only if ``split_directions``
    - ``'num_trips'``: number of trips on the route in the subset
    - ``'num_trip_starts'``: number of trips on the route with
      nonnull start times
    - ``'num_trip_ends'``: number of trips on the route with nonnull
      end times that end before 23:59:59
    - ``'num_stop_patterns'``: number of stop pattern across trips
    - ``'is_loop'``: 1 if at least one of the trips on the route has
      its ``is_loop`` field equal to 1; 0 otherwise
    - ``'is_bidirectional'``: 1 if the route has trips in both
      directions; 0 otherwise; present if only if not ``split_directions``
    - ``'start_time'``: start time of the earliest trip on the route
    - ``'end_time'``: end time of latest trip on the route
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip starts on the route between
      ``headway_start_time`` and ``headway_end_time`` on the given
      dates
    - ``'min_headway'``: minimum of the durations (in minutes)
      mentioned above
    - ``'mean_headway'``: mean of the durations (in minutes)
      mentioned above
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service (for the given direction, or for both directions when
      ``split_directions==False``)
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs
    - ``'service_duration'``: total of the duration of each trip on
      the route in the given subset of trips; measured in hours
    - ``'service_distance'``: total of the distance traveled by each
      trip on the route in the given subset of trips;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_speed'``: service_distance/service_duration when defined; 0 otherwise
    - ``'mean_trip_distance'``: service_distance/num_trips
    - ``'mean_trip_duration'``: service_duration/num_trips


    Exclude dates with no active trips, which could yield the empty DataFrame.

    If not ``split_directions``, then compute each route's stats,
    except for headways, using its trips running in both directions.
    For headways, (1) compute max headway by taking the max of the
    max headways in both directions; (2) compute mean headway by
    taking the weighted mean of the mean headways in both
    directions.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.
    - The route stats for date d contain stats for trips that start on
      date d only and ignore trips that start on date d-1 and end on
      date d.
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present.

    """
    null_stats = compute_route_stats_0(
        feed.trips.head(0), split_directions=split_directions
    )
    final_cols = ["date"] + list(null_stats.collect_schema().names())
    null_stats = null_stats.with_columns(date=None).select(final_cols)
    dates = feed.subset_dates(dates)

    # Handle defunct case
    if not dates:
        return null_stats

    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()
    elif isinstance(trip_stats, pl.DataFrame):
        trip_stats = trip_stats.lazy()

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFrame.
    stats_by_ids = {}
    activity = feed.compute_trip_activity(dates)
    frames = []
    for date in dates:
        ids = tuple(
            sorted(
                activity.filter(pl.col(date) > 0)
                .select("trip_id")
                .collect()["trip_id"]
                .to_list()
            )
        )
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].with_columns(date=pl.lit(date))
        elif ids:
            # Compute stats afresh
            t = trip_stats.filter(pl.col("trip_id").is_in(ids))
            stats = compute_route_stats_0(
                t,
                split_directions=split_directions,
                headway_start_time=headway_start_time,
                headway_end_time=headway_end_time,
            ).with_columns(date=pl.lit(date))
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    sort_by = (
        ["date", "route_id", "direction_id"]
        if split_directions
        else ["date", "route_id"]
    )
    return pl.concat(frames).select(final_cols).sort(sort_by)


def compute_route_time_series_0(
    trip_stats: pd.DataFrame,
    date_label: str = "20010101",
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats in a 24-hour time series form at the given Pandas frequency
    for the given subset of trip stats of the
    form output by the function :func:`.trips.compute_trip_stats`.

    If ``split_directions``, then separate each routes's stats by trip direction.
    Use the given YYYYMMDD date label as the date in the time series index.

    Return a long-format DataFrame with the columns

    - ``datetime``: datetime object
    - ``route_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: number of trips in service on the route
      at any time within the time bin
    - ``num_trip_starts``: number of trips that start within
      the time bin
    - ``num_trip_ends``: number of trips that end within the
      time bin, ignoring trips that end past midnight
    - ``service_distance``: sum of the service distance accrued
      during the time bin across all trips on the route;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
    - ``service_duration``: sum of the service duration accrued
      during the time bin across all trips on the route;
      measured in hours
    - ``service_speed``: ``service_distance/service_duration``
      for the route


    Notes
    -----
    - Trips that lack start or end times are ignored, so the the
      aggregate ``num_trips`` across the day could be less than the
      ``num_trips`` column of :func:`compute_route_stats_0`
    - All trip departure times are taken modulo 24 hours.
      So routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series,
      except for their ``num_trip_ends`` indicator.
      Trip endings past 23:59:59 are not binned so that resampling the
      ``num_trips`` indicator works efficiently.
    - Note that the total number of trips for two consecutive time bins
      t1 < t2 is the sum of the number of trips in bin t2 plus the
      number of trip endings in bin t1.
      Thus we can downsample the ``num_trips`` indicator by keeping
      track of only one extra count, ``num_trip_ends``, and can avoid
      recording individual trip IDs.
    - All other indicators are downsampled by summing.
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    final_cols = [
        "datetime",
        "route_id",
        "num_trips",
        "num_trip_starts",
        "num_trip_ends",
        "service_distance",
        "service_duration",
        "service_speed",
    ]
    if split_directions:
        final_cols.insert(2, "direction_id")

    null_stats = pd.DataFrame([], columns=final_cols)

    # Handle defunct case
    if trip_stats.is_empty():
        return null_stats

    tss = trip_stats.copy()
    if split_directions:
        tss = tss.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if tss.is_empty():
            raise ValueError(
                "At least one trip stats direction ID value must be non-NaN."
            )

        # Alter route IDs to encode direction:
        # <route ID>-0 and <route ID>-1 or <route ID>-NA
        tss["route_id"] = (
            tss["route_id"] + "-" + tss["direction_id"].map(lambda x: str(int(x)))
        )

    routes = tss["route_id"].unique()

    # Build a dictionary of time series and then merge them all
    # at the end.
    # Assign a uniform generic date for the index
    indicators = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_duration",
        "service_distance",
    ]

    # Bin start and end times
    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    def timestr_to_min(x):
        return hp.timestr_to_seconds(x, mod24=True) // 60

    tss["start_index"] = tss["start_time"].map(timestr_to_min)
    tss["end_index"] = tss["end_time"].map(timestr_to_min)

    # Bin each trip according to its start and end time and weight
    routes = sorted(tss["route_id"].dropna().unique().tolist())
    series_by_route_by_indicator = {
        indicator: {route: [0 for i in range(num_bins)] for route in routes}
        for indicator in indicators
    }
    for row in tss.itertuples(index=False):
        route = row.route_id
        start = row.start_index
        end = row.end_index
        distance = row.distance

        # Ignore defunct trips
        if pd.isna(start) or pd.isna(end) or start == end:
            continue

        # Get bins to fill
        if start < end:
            bins_to_fill = bins[start:end]
        else:
            bins_to_fill = bins[start:] + bins[:end]

        # Bin trip and calculate indicators.
        # Num trip starts.
        series_by_route_by_indicator["num_trip_starts"][route][start] += 1

        # Num trip ends.
        # Don't mark trip ends for trips that run past midnight;
        # allows for easy resampling of num_trips later.
        if start <= end:
            series_by_route_by_indicator["num_trip_ends"][route][end] += 1

        # Do rest of indicators
        for indicator in indicators[2:]:
            if indicator == "num_trips":
                weight = 1
            elif indicator == "service_duration":
                weight = 1 / 60
            else:
                weight = distance / len(bins_to_fill)
            for b in bins_to_fill:
                series_by_route_by_indicator[indicator][route][b] += weight

    # Build per-indicator DataFrames indexed by minute across the provided date
    rng = pd.date_range(
        pd.to_datetime(f"{date_label} 00:00:00"), periods=24 * 60, freq="Min"
    )
    series_by_indicator = {
        indicator: pd.DataFrame(
            series_by_route_by_indicator[indicator], index=rng
        ).fillna(0)
        for indicator in indicators
    }

    # Combine into a single long-form time series per route (and direction if requested);
    # hp.combine_time_series is expected to compute derived fields like service_speed
    g = hp.combine_time_series(
        series_by_indicator, kind="route", split_directions=split_directions
    )
    # Downsample to requested frequency (sum for counts/durations/distances; speed handled by helper)
    return hp.downsample(g, freq=freq)


def compute_route_time_series(
    feed: "Feed",
    dates: list[str],
    trip_stats: pd.DataFrame | None = None,
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute route stats in time series form for the trips that lie in
    the trip stats subset, which defaults to the output of
    :func:`.trips.compute_trip_stats`, and that start on the given dates
    (YYYYMMDD date strings).

    If ``split_directions``, then separate each routes's stats by trip direction.
    Specify the time series frequency with a Pandas frequency string, e.g. ``'5Min'``.

    Return a time series DataFrame with the following columns.

    - ``datetime``: datetime object
    - ``route_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: number of trips in service on the route
      at any time within the time bin
    - ``num_trip_starts``: number of trips that start within
      the time bin
    - ``num_trip_ends``: number of trips that end within the
      time bin, ignoring trips that end past midnight
    - ``service_distance``: sum of the service distance accrued
      during the time bin across all trips on the route;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
    - ``service_duration``: sum of the service duration accrued
      during the time bin across all trips on the route;
      measured in hours
    - ``service_speed``: ``service_distance/service_duration``
      for the route

    Exclude dates that lie outside of the Feed's date range.
    If all dates lie outside the Feed's date range, then return an
    empty DataFrame.

    Notes
    -----
    - If you've already computed trip stats in your workflow, then you should pass
      that table into this function to speed things up significantly.
    - See the notes for :func:`compute_route_time_series_0`
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    null_stats = compute_route_time_series_0(
        pd.DataFrame(), split_directions=split_directions
    )

    # Handle defunct case
    if not dates:
        return null_stats

    activity = feed.compute_trip_activity(dates)
    if trip_stats is None:
        trip_stats = feed.compute_trip_stats()
    else:
        trip_stats = trip_stats.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary re-computations.
    # Store in dictionary of the form
    # trip ID sequence -> stats table
    null_stats = pd.DataFrame()
    stats_by_ids = {}
    activity = feed.compute_trip_activity(dates)
    frames = []
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].pipe(hp.replace_date, date=date)
        elif ids:
            # Compute stats afresh
            t = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
            stats = compute_route_time_series_0(
                t, split_directions=split_directions, freq=freq, date_label=date
            ).pipe(hp.replace_date, date=date)
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    return pd.concat(frames, ignore_index=True)
