"""
Functions about stops.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING, Iterable

import folium as fl
import folium.plugins as fp
import polars as pl
import polars_st as st
import geopandas as gpd
import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


#: Leaflet circleMarker parameters for mapping stops
STOP_STYLE = {
    "radius": 8,
    "fill": "true",
    "color": cs.COLORS_SET2[1],
    "weight": 1,
    "fillOpacity": 0.75,
}


def geometrize_stops(
    stops: pl.DataFrame | pl.LazyFrame, *, use_utm: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    """
    Given a GTFS stops Table, convert it to a geotable with a "geometry"
    column of LineStrings and a "srid" column with the (constant) srid of the geographic
    projection, e.g. 'EPSG:4326' for the WGS84 srid.
    Return the resulting geotable, which will no longer have
    the columns ``'stop_lon'`` and ``'stop_lat'``.

    If ``use_utm``, then use local UTM coordinates for the geometries.
    """
    g = stops.with_columns(
        geometry=st.point(pl.concat_arr("stop_lon", "stop_lat")).st.set_srid(cs.WGS84)
    ).drop(["stop_lon", "stop_lat"])
    if use_utm:
        g = hp.to_srid(g, hp.get_utm_srid(g))

    return g


def ungeometrize_stops(
    stops_g: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    The inverse of :func:`geometrize_stops`.

    If ``stops_g`` is in UTM coordinates (has a UTM srid property),
    then convert those UTM coordinates back to WGS84 coordinates,
    which is the standard for a GTFS shapes table.
    """
    return (
        hp.to_srid(stops_g, cs.WGS84)
        .with_columns(
            stop_lon=pl.col("geometry").st.x(),
            stop_lat=pl.col("geometry").st.y(),
        )
        .drop("geometry")
    )


def get_stops(
    feed: "Feed",
    date: str | None = None,
    trip_ids: Iterable[str] | None = None,
    route_ids: Iterable[str] | None = None,
    *,
    in_stations: bool = False,
    as_geo: bool = False,
    use_utm: bool = False,
) -> pl.LazyFrame:
    """
    Return ``feed.stops``.
    If a YYYYMMDD date string is given, then subset to stops
    active (visited by trips) on that date.
    If trip IDs are given, then subset further to stops visited by those
    trips.
    If route IDs are given, then ignore the trip IDs and subset further
    to stops visited by those routes.
    If ``in_stations``, then subset further stops in stations if station data
    is available.
    If ``as_gdf``, then return the result as a GeoDataFrame with a 'geometry'
    column of points instead of 'stop_lat' and 'stop_lon' columns.
    The GeoDataFrame will have a UTM CRS if ``use_utm`` and a WGS84 CRS otherwise.
    """
    s = feed.stops

    if date is not None:
        # Get active stop_ids for the given date
        stop_ids = feed.get_stop_times(date).select("stop_id").unique()
        s = s.join(stop_ids, on="stop_id", how="semi")

    if trip_ids is not None:
        # Filter by trip_ids
        stop_ids = (
            feed.stop_times.filter(pl.col("trip_id").is_in(trip_ids))
            .select("stop_id")
            .unique()
        )
        s = s.join(stop_ids, on="stop_id", how="semi")

    elif route_ids is not None:
        # Filter by route_ids (overrides trip_ids)
        trip_ids = (
            feed.trips.filter(pl.col("route_id").is_in(route_ids))
            .select("trip_id")
            .unique()
        )
        stop_ids = (
            feed.stop_times.join(trip_ids, on="trip_id", how="semi")
            .select("stop_id")
            .unique()
        )
        s = s.join(stop_ids, on="stop_id", how="semi")

    if in_stations:
        # Check if required columns exist
        if {"location_type", "parent_station"} <= set(s.collect_schema().names()):
            s = s.filter(
                (pl.col("location_type") != 1) & ~pl.col("parent_station").is_null()
            )

    if as_geo:
        s = geometrize_stops(s, use_utm=use_utm)

    return s


def compute_stop_activity(feed: "Feed", dates: list[str]) -> pd.DataFrame:
    """
    Mark stops as active or inactive on the given dates (YYYYMMDD date strings).
    A stop is *active* on a given date if some trips that starts on the
    date visits the stop (possibly after midnight).

    Return a table with the columns

    - stop_id
    - ``dates[0]``: 1 if the stop has at least one trip visiting it
      on ``dates[0]``; 0 otherwise
    - ``dates[1]``: 1 if the stop has at least one trip visiting it
      on ``dates[1]``; 0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the stop has at least one trip visiting it
      on ``dates[-1]``; 0 otherwise

    If all dates lie outside the Feed period, then return an empty table.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pl.LazyFrame(schema={"stop_id": pl.Utf8})

    return (
        feed.compute_trip_activity(dates)
        .join(feed.stop_times.select("trip_id", "stop_id"), "trip_id")
        .group_by("stop_id")
        .agg([pl.col(d).max().alias(d) for d in dates])
        # ensure 0/1 ints as in the pandas version
        # .with_columns([pl.col(d).cast(pl.UInt8) for d in dates])
        .select(["stop_id", *dates])
    )


def build_stop_timetable(feed: "Feed", stop_id: str, dates: list[str]) -> pl.LazyFrame:
    """
    Return a timetable for the given stop ID and dates (YYYYMMDD date strings)

    Return a DataFrame whose columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times`` plus ``'date'``, and the stop IDs are restricted to the given
    stop ID.
    The result is sorted by date then departure time.
    """
    dates = feed.subset_dates(dates)
    if not dates:
        return pl.LazyFrame(schema={"stop_id": pl.Utf8})

    t = feed.trips.join(feed.stop_times, "trip_id").filter(pl.col("stop_id") == stop_id)
    a = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        # Slice to stops active on date
        b = a.filter(pl.col(date) == 1)
        f = t.join(b, "trip_id", how="semi").with_columns(date=pl.lit(date))
        frames.append(f)

    return (
        pl.concat(frames)
        .with_columns(dtime=hp.timestr_to_seconds("departure_time"))
        .sort("date", "dtime")
        .drop("dtime")
    )


def build_geometry_by_stop(
    feed: "Feed", stop_ids: Iterable[str] | None = None, *, use_utm: bool = False
) -> dict:
    """
    Return a dictionary of the form <stop ID> -> <Shapely Point representing stop>.
    """
    g = get_stops(feed, as_geo=True, use_utm=use_utm).with_columns(
        geometry=pl.col("geometry").st.to_shapely()
    )
    if stop_ids is not None:
        g = g.filter(pl.col("stop_id").is_in(stop_ids))
    return dict(g.select("stop_id", "geometry").collect().rows())


def stops_to_geojson(feed: "Feed", stop_ids: Iterable[str | None] = None) -> dict:
    """
    Return a GeoJSON FeatureCollection of Point features
    representing all the stops in ``feed.stops``.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of stop IDs is given, then subset to those stops.
    """
    g = get_stops(feed, as_geo=True)
    if stop_ids is not None:
        g = g.filter(pl.col("stop_id").is_in(stop_ids))
    if g is None or hp.is_empty(g):
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = g.collect().st.__geo_interface__

    return result


def get_stops_in_area(
    feed: "Feed",
    area: st.GeoLazyFrame | st.GeoDataFrame,
) -> st.GeoLazyFrame:
    """
    Return the subset of ``feed.stops`` that contains all stops that intersect the
    given geotable of polygons.
    """
    area = hp.make_lazy(area)
    return feed.stops.join(
        get_stops(feed, as_geo=True).st.sjoin(hp.to_srid(area, cs.WGS84)),
        "stop_id",
        how="semi",
    )


def map_stops(feed: "Feed", stop_ids: Iterable[str], stop_style: dict = STOP_STYLE):
    """
    Return a Folium map showing the given stops of this Feed.
    If some of the given stop IDs are not found in the feed, then raise a ValueError.
    """
    # Initialize map
    my_map = fl.Map(tiles="cartodbpositron")

    # Add stops to feature group
    stops = feed.stops.filter(pl.col("stop_id").is_in(stop_ids)).fill_null("n/a")

    # Add stops with clustering
    callback = f"""\
    function (row) {{
        var imarker;
        marker = L.circleMarker(new L.LatLng(row[0], row[1]),
            {stop_style}
        );
        marker.bindPopup(
            '<b>Stop name</b>: ' + row[2] + '<br>' +
            '<b>Stop code</b>: ' + row[3] + '<br>' +
            '<b>Stop ID</b>: ' + row[4]
        );
        return marker;
    }};
    """
    fp.FastMarkerCluster(
        data=stops.select(["stop_lat", "stop_lon", "stop_name", "stop_code", "stop_id"])
        .collect()
        .to_numpy()
        .tolist(),
        callback=callback,
        disableClusteringAtZoom=14,
    ).add_to(my_map)

    # Fit map to stop bounds
    bounds = [
        (stops.select("stop_lat").min(), stops.select("stop_lon").min()),
        (stops.select("stop_lat").max(), stops.select("stop_lon").max()),
    ]
    my_map.fit_bounds(bounds, padding=[1, 1])

    return my_map


def compute_stop_stats_0(
    stop_times_subset: pl.DataFrame | pl.LazyFrame,
    trip_subset: pl.DataFrame | pl.LazyFrame,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pl.LazyFrame:
    """
    Given a subset of a stop times Table and a subset of a trips
    Table, return a Table that provides summary stats about the
    stops in the inner join of the two Tables.

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a Table with the columns

    - stop_id
    - direction_id: present if and only if ``split_directions``
    - num_routes: number of routes visiting stop
      (in the given direction)
    - num_trips: number of trips visiting stop
      (in the givin direction)
    - max_headway: maximum of the durations (in minutes)
      between trip departures at the stop between
      ``headway_start_time`` and ``headway_end_time``
    - min_headway: minimum of the durations (in minutes) mentioned
      above
    - mean_headway: mean of the durations (in minutes) mentioned
      above
    - start_time: earliest departure time of a trip from this stop
    - end_time: latest departure time of a trip from this stop

    Notes
    -----
    - If ``trip_subset`` is empty, then return an empty Table.
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present.
    """
    # Handle defunct case
    final_cols = [
        "stop_id",
        "num_routes",
        "num_trips",
        "max_headway",
        "min_headway",
        "mean_headway",
        "start_time",
        "end_time",
    ]
    if split_directions:
        final_cols.insert(1, "direction_id")

    if hp.is_empty(trip_subset):
        return pl.LazyFrame({c: [] for c in final_cols})

    # Handle generic case
    headway_start = hp.timestr_to_seconds_0(headway_start_time)
    headway_end = hp.timestr_to_seconds_0(headway_end_time)

    # Join stop times and trips
    f = (
        hp.make_lazy(stop_times_subset)
        .join(hp.make_lazy(trip_subset), "trip_id")
        .with_columns(dtime=hp.timestr_to_seconds("departure_time"))
    )

    # Handle direction split
    if split_directions:
        # Ensure direction_id present (nullable int32)
        if "direction_id" not in (f.collect_schema().names()):
            f = f.with_columns(direction_id=pl.lit(None, dtype=pl.Int32))
        f_nonnull = f.filter(pl.col("direction_id").is_not_null())
        if f_nonnull.select(pl.len()).collect().item() == 0:
            raise ValueError("At least one trip direction ID value must be non-NULL.")
        group_cols = ["stop_id", "direction_id"]
        f = f_nonnull
    else:
        group_cols = ["stop_id"]

    # Basic stats per stop(/direction)
    basic_stats = f.group_by(group_cols).agg(
        num_routes=pl.col("route_id").n_unique().cast(pl.UInt32),
        num_trips=pl.len().cast(pl.UInt32),
        start_time_s=pl.col("dtime").min(),
        end_time_s=pl.col("dtime").max(),
    )
    # Headway stats within [start, end]
    headway_stats = (
        f.filter((pl.col("dtime") >= headway_start) & (pl.col("dtime") <= headway_end))
        .select(group_cols + ["dtime"])
        .sort(by=group_cols + ["dtime"])
        .with_columns(prev_dtime=pl.col("dtime").shift(1).over(group_cols))
        .with_columns(headway_s=(pl.col("dtime") - pl.col("prev_dtime")))
        .filter(pl.col("headway_s").is_not_null())
        .group_by(group_cols)
        .agg(
            max_headway=(pl.col("headway_s").max() / 60.0).cast(pl.Float64),
            min_headway=(pl.col("headway_s").min() / 60.0).cast(pl.Float64),
            mean_headway=(pl.col("headway_s").mean() / 60.0).cast(pl.Float64),
        )
    )
    # Join & finalize times as strings
    return (
        basic_stats.join(headway_stats, group_cols, how="left")
        .with_columns(
            start_time=hp.seconds_to_timestr("start_time_s"),
            end_time=hp.seconds_to_timestr("end_time_s"),
        )
        .drop("start_time_s", "end_time_s")
        .select(final_cols)
    )


def compute_stop_stats(
    feed: "Feed",
    dates: list[str],
    stop_ids: list[str | None] = None,
    headway_start_time: str = "07:00:00",
    headway_end_time: str = "19:00:00",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stats for all stops for the given dates (YYYYMMDD date strings).
    Optionally, restrict to the stop IDs given.

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.
    Use the headway start and end times to specify the time period for computing
    headway stats.

    Return a DataFrame with the columns

    - ``'date'``
    - ``'stop_id'``
    - ``'direction_id'``: present if and only if ``split_directions``
    - ``'num_routes'``: number of routes visiting the stop
      (in the given direction) on the date
    - ``'num_trips'``: number of trips visiting stop
      (in the givin direction) on the date
    - ``'max_headway'``: maximum of the durations (in minutes)
      between trip departures at the stop between
      ``headway_start_time`` and ``headway_end_time`` on the date
    - ``'min_headway'``: minimum of the durations (in minutes) mentioned
      above
    - ``'mean_headway'``: mean of the durations (in minutes) mentioned
      above
    - ``'start_time'``: earliest departure time of a trip from this stop
      on the date
    - ``'end_time'``: latest departure time of a trip from this stop on
      the date

    Exclude dates with no active stops, which could yield the empty DataFrame.
    """
    dates = feed.subset_dates(dates)
    null_stats = compute_stop_stats_0(
        pd.DataFrame(), pd.DataFrame(), split_directions=split_directions
    )
    final_cols = ["date"] + null_stats.columns.tolist()

    # Handle defunct case
    if not dates:
        return null_stats

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFarme.
    stats_by_ids = {}

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

    activity = feed.compute_trip_activity(dates)

    frames = []
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            stats = (
                stats_by_ids[ids]
                # Assign date
                .assign(date=date)
            )
        elif ids:
            # Compute stats
            t = feed.trips
            trips = t[t["trip_id"].isin(ids)].copy()
            stats = (
                compute_stop_stats_0(
                    stop_times_subset,
                    trips,
                    split_directions=split_directions,
                    headway_start_time=headway_start_time,
                    headway_end_time=headway_end_time,
                )
                # Assign date
                .assign(date=date)
            )
            # Memoize stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Assemble stats into a single DataFrame
    return pd.concat(frames).filter(final_cols)


def compute_stop_time_series_0(
    stop_times_subset: pd.DataFrame,
    trips_subset: pd.DataFrame,
    freq: str = "h",
    date_label: str = "20010101",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute stop stats in a 24-hour time series form at the given Pandas frequency
    for stops in the inner join of the given subset of stop times and trips.

    If ``split_directions``, then separate each stop's stats by trip direction.
    Use the given YYYYMMDD date label as the date in the time series index.

    Return a long-format DataFrame with columns

    - ``datetime``: datetime object for the given date and frequency chunks
    - ``stop_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: the number of trips that visit the stop in the time bin and
      have a nonnull departure time from the stop

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_0`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trips_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    final_cols = [
        "datetime",
        "stop_id",
        "num_trips",
    ]
    if split_directions:
        final_cols.insert(2, "direction_id")
    null_stats = pd.DataFrame([], columns=final_cols)

    # Handle defunct case
    if trips_subset.is_empty():
        return null_stats

    f = pd.merge(stop_times_subset, trips_subset)

    if split_directions:
        if "direction_id" not in f.columns:
            f["direction_id"] = pd.NA
        f = f.loc[lambda x: x.direction_id.notnull()].assign(
            direction_id=lambda x: x.direction_id.astype(int)
        )
        if f.is_empty():
            raise ValueError("At least one trip direction ID value must be non-NaN.")

        # Alter stop IDs to encode trip direction:
        # <stop ID>-0 and <stop ID>-1
        f["stop_id"] = f["stop_id"] + "-" + f["direction_id"].map(str)

    # Bin each stop departure time
    bins = [i for i in range(24 * 60)]  # One bin for each minute
    num_bins = len(bins)

    def timestr_to_min(x):
        return hp.timestr_to_seconds(x, mod24=True) // 60

    f["departure_index"] = f["departure_time"].map(timestr_to_min)

    # Create one time series for each stop
    series_by_stop = {
        stop: [0 for i in range(num_bins)] for stop in f["stop_id"].unique()
    }

    for stop, group in f.groupby("stop_id"):
        counts = Counter((bin, 0) for bin in bins) + Counter(
            group["departure_index"].values
        )
        series_by_stop[stop] = [counts[bin] for bin in bins]

    # Build per-indicator DataFrames indexed by minute across the provided date
    rng = pd.date_range(
        pd.to_datetime(f"{date_label} 00:00:00"), periods=24 * 60, freq="Min"
    )
    series_by_indicator = {
        "num_trips": pd.DataFrame(series_by_stop, index=rng).fillna(0)
    }

    # Combine into a single long-form time series per route (and direction if requested);
    # hp.combine_time_series is expected to compute derived fields like service_speed
    g = hp.combine_time_series(
        series_by_indicator, kind="stop", split_directions=split_directions
    )
    # Downsample to requested frequency (sum for counts/durations/distances; speed handled by helper)
    return hp.downsample(g, freq=freq)


def compute_stop_time_series(
    feed: "Feed",
    dates: list[str],
    stop_ids: list[str | None] = None,
    freq: str = "h",
    *,
    split_directions: bool = False,
) -> pd.DataFrame:
    """
    Compute time series for the given stops (defaults to all stops in Feed)
    on the given dates (YYYYMMDD date strings) at the
    given frequency (Pandas frequency string, e.g. ``'5Min'``).
    Return a long-format DataFrame with columns

    - ``datetime``: datetime object for the given date and frequency chunks
    - ``stop_id``
    - ``direction_id``: direction of route; presest if and only if ``split_directions``
    - ``num_trips``: the number of trips that visit the stop in the time bin and
      have a nonnull departure time from the stop

    Exclude dates that lie outside of the Feed's date range.
    If all dates lie outside the Feed's date range, then return an
    empty DataFrame

    If ``split_directions``, then separate the stop stats by direction (0 or 1)
    of the trips visiting the stops.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - Stop times with null departure times are ignored, so the aggregate
      of ``num_trips`` across the day could be less than the
      ``num_trips`` column in :func:`compute_stop_stats_0`
    - All trip departure times are taken modulo 24 hours,
      so routes with trips that end past 23:59:59 will have all
      their stats wrap around to the early morning of the time series.
    - 'num_trips' should be resampled with ``how=np.sum``
    - If ``trips_subset`` is empty, then return an empty DataFrame
    - Raise a ValueError if ``split_directions`` and no non-null
      direction ID values present

    """
    dates = feed.subset_dates(dates)
    null_stats = compute_stop_time_series_0(
        pd.DataFrame(), pd.DataFrame(), split_directions=split_directions
    )
    # Handle defunct case
    if not dates:
        return null_stats

    activity = feed.compute_trip_activity(dates)

    # Restrict stop times to stop IDs if specified
    if stop_ids is not None:
        stop_times_subset = feed.stop_times.loc[
            lambda x: x["stop_id"].isin(stop_ids)
        ].copy()
    else:
        stop_times_subset = feed.stop_times.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence -> stats table
    frames = []
    stats_by_ids = {}
    for date in dates:
        ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
        if ids in stats_by_ids:
            # Reuse stats with updated date
            stats = stats_by_ids[ids].pipe(hp.replace_date, date=date)
        elif ids:
            # Compute stats afresh
            trips_subset = feed.trips.loc[lambda x: x["trip_id"].isin(ids)].copy()
            stats = compute_stop_time_series_0(
                stop_times_subset,
                trips_subset,
                split_directions=split_directions,
                freq=freq,
                date_label=date,
            ).pipe(hp.replace_date, date=date)
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    return pd.concat(frames, ignore_index=True)
