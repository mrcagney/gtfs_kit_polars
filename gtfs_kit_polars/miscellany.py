"""
Functions about miscellany.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def list_fields(feed: "Feed", table_name: str | None = None) -> pl.DataFrame:
    """
    Return a DataFrame summarizing all GTFS tables in the given feed
    or in the given table name if specified.

    The resulting DataFrame has the following columns.

    - ``'table'``: name of the GTFS table, e.g. ``'stops'``
    - ``'column'``: name of a column in the table,
      e.g. ``'stop_id'``
    - ``'num_values'``: number of values in the column
    - ``'num_nonnull_values'``: number of nonnull values in the
      column
    - ``'num_unique_values'``: number of unique values in the
      column, excluding null values
    - ``'min_value'``: minimum value in the column
    - ``'max_value'``: maximum value in the column

    If the table is not in the feed, then return an empty DataFrame
    If the table is not valid, raise a ValueError
    """
    gtfs_tables = list(cs.DTYPES.keys())
    if table_name is not None:
        if table_name not in gtfs_tables:
            raise ValueError(f"{table_name} is not a GTFS table")
        else:
            table_names = [table_name]
    else:
        table_names = gtfs_tables

    frames = []
    schema = {
        "table": pl.Utf8,
        "column": pl.Utf8,
        "num_values": pl.Int64,
        "num_nonnull_values": pl.Int64,
        "num_unique_values": pl.Int64,
        "min_value": pl.Utf8,
        "max_value": pl.Utf8,
    }
    final_cols = list(schema.keys())
    for table_name in table_names:
        t = getattr(feed, table_name)
        if t is None:
            continue

        frames_0 = []
        for col_name in t.collect_schema().names():
            col = pl.col(col_name)
            r = (
                t.select(
                    column=pl.lit(col_name),
                    num_nonnull_values=col.count(),
                    num_unique_values=col.drop_nulls().n_unique(),
                    min_value=col.min().cast(pl.Utf8),
                    max_value=col.max().cast(pl.Utf8),
                    num_nulls=col.is_null().sum(),
                )
                .with_columns(
                    num_values=pl.col("num_nonnull_values") + pl.col("num_nulls"),
                    table=pl.lit(table_name),
                )
                .select(final_cols)
            )
            frames_0.append(r)

        s = pl.concat(frames_0) if frames_0 else pl.LazyFrame(schema=schema)
        frames.append(s)

    return pl.concat(frames, how="vertical") if frames else pl.LazyFrame(schema=schema)


def describe(feed: "Feed", sample_date: str | None = None) -> pd.LazyFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of routes.
    Specialize some those indicators to the given YYYYMMDD sample date string,
    e.g. number of routes active on the date.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    """
    from . import calendar as cl

    d = dict()
    dates = cl.get_dates(feed)
    d["agencies"] = feed.agency.collect()["agency_name"].to_list()
    d["timezone"] = feed.agency.collect()["agency_timezone"].item(0)
    d["start_date"] = dates[0]
    d["end_date"] = dates[-1]
    d["num_routes"] = hp.height(feed.routes)
    d["num_trips"] = hp.height(feed.trips)
    d["num_stops"] = hp.height(feed.stops)
    if feed.shapes is not None:
        d["num_shapes"] = feed.shapes.collect()["shape_id"].n_unique()
    else:
        d["num_shapes"] = 0

    if sample_date is None or sample_date not in feed.get_dates():
        sample_date = cl.get_first_week(feed)[3]
    d["sample_date"] = sample_date
    d["num_routes_active_on_sample_date"] = hp.height(feed.get_routes(sample_date))
    trips = feed.get_trips(sample_date)
    d["num_trips_active_on_sample_date"] = hp.height(trips)
    d["num_stops_active_on_sample_date"] = hp.height(feed.get_stops(sample_date))

    return pl.LazyFrame(
        {
            "indicator": list(d.keys()),
            "value": [str(v) for v in d.values()],  # everything as Utf8
        }
    )


def assess_quality(feed: "Feed") -> pd.DataFrame:
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of trips missing shapes.

    The resulting DataFrame has the columns

    - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
    - ``'value'``: value of the indicator, e.g. 27

    This function is odd but useful for seeing roughly how broken a feed is
    This function is not a GTFS validator.
    """
    r = feed.routes
    st = feed.stop_times
    t = feed.trips

    # Safer column checks on LazyFrames
    trips_cols = set(t.collect_schema().names())
    st_cols = set(st.collect_schema().names())

    has_dir = "direction_id" in trips_cols
    has_shape = "shape_id" in trips_cols
    has_sdt = "shape_dist_traveled" in st_cols

    # route short-name duplicates
    routes_stats = r.select(
        total_rows=pl.len(),
        unique_short=(
            pl.when(pl.col("route_short_name").is_null())
            .then(pl.lit(None))
            .otherwise(pl.col("route_short_name"))
            .drop_nulls()
            .n_unique()
        ),
    ).select(
        num_route_short_names_duplicated=pl.col("total_rows") - pl.col("unique_short"),
        frac_route_short_names_duplicated=(
            pl.col("total_rows") - pl.col("unique_short")
        )
        / pl.col("total_rows"),
    )

    # stop_time distances missing
    st_dists = st.select(
        num_stop_time_dists_missing=(
            pl.col("shape_dist_traveled").is_null() if has_sdt else pl.lit(True)
        ).sum(),
        frac_stop_time_dists_missing=(
            pl.col("shape_dist_traveled").is_null() if has_sdt else pl.lit(True)
        ).mean(),
        st_total=pl.len(),
    )

    # trips direction_id missing
    trips_dir = t.select(
        num_direction_ids_missing=(
            pl.col("direction_id").is_null() if has_dir else pl.lit(True)
        ).sum(),
        frac_direction_ids_missing=(
            pl.col("direction_id").is_null() if has_dir else pl.lit(True)
        ).mean(),
        trips_total=pl.len(),
    )

    # trips missing shapes
    if getattr(feed, "shapes", None) is not None and has_shape:
        trips_shapes = t.select(
            num_trips_missing_shapes=pl.col("shape_id").is_null().sum(),
            frac_trips_missing_shapes=pl.col("shape_id").is_null().mean(),
        )
    else:
        trips_shapes = t.select(
            num_trips_missing_shapes=pl.len(),
            frac_trips_missing_shapes=pl.lit(1.0),
        )

    # any departure_time missing
    dep_any = st.select(
        num_departure_times_missing=pl.col("departure_time").is_null().sum(),
        frac_departure_times_missing=pl.col("departure_time").is_null().mean(),
    )

    # first departure per trip missing
    firsts = (
        st.sort(["trip_id", "stop_sequence"])
        .with_columns(rn=pl.row_index().over("trip_id"))
        .filter(pl.col("rn") == 0)
    )
    first_missing = firsts.select(
        num_first_departure_times_missing=pl.col("departure_time").is_null().sum(),
    )
    st_total_only = st.select(st_count=pl.len())
    first_missing = first_missing.join(st_total_only, how="cross").select(
        "num_first_departure_times_missing",
        frac_first_departure_times_missing=pl.col("num_first_departure_times_missing")
        / pl.col("st_count"),
    )

    # last departure per trip missing (note descending flag list)
    lasts = (
        st.sort(by=["trip_id", "stop_sequence"], descending=[False, True])
        .with_columns(rn=pl.row_index().over("trip_id"))
        .filter(pl.col("rn") == 0)
    )
    last_missing = lasts.select(
        num_last_departure_times_missing=pl.col("departure_time").is_null().sum(),
    )
    last_missing = last_missing.join(st_total_only, how="cross").select(
        "num_last_departure_times_missing",
        frac_last_departure_times_missing=pl.col("num_last_departure_times_missing")
        / pl.col("st_count"),
    )

    # stitch & assess
    scalars = (
        routes_stats.join(st_dists, how="cross")
        .join(trips_dir, how="cross")
        .join(trips_shapes, how="cross")
        .join(dep_any, how="cross")
        .join(first_missing, how="cross")
        .join(last_missing, how="cross")
        .with_columns(
            assessment=pl.when(
                (pl.col("frac_first_departure_times_missing") >= 0.1)
                | (pl.col("frac_last_departure_times_missing") >= 0.1)
                | (pl.col("frac_trips_missing_shapes") >= 0.8)
            )
            .then(pl.lit("bad feed"))
            .when(
                (pl.col("frac_direction_ids_missing") > 0)
                | (pl.col("frac_stop_time_dists_missing") > 0)
                | (pl.col("num_route_short_names_duplicated") > 0)
            )
            .then(pl.lit("probably a fixable feed"))
            .otherwise(pl.lit("good feed"))
        )
    )

    # long form (indicator, value) as Utf8
    return (
        scalars.select(
            [
                "num_route_short_names_duplicated",
                "frac_route_short_names_duplicated",
                "num_stop_time_dists_missing",
                "frac_stop_time_dists_missing",
                "num_direction_ids_missing",
                "frac_direction_ids_missing",
                "num_trips_missing_shapes",
                "frac_trips_missing_shapes",
                "num_departure_times_missing",
                "frac_departure_times_missing",
                "num_first_departure_times_missing",
                "frac_first_departure_times_missing",
                "num_last_departure_times_missing",
                "frac_last_departure_times_missing",
                "assessment",
            ]
        )
        .melt(variable_name="indicator", value_name="value")
        .with_columns(pl.col("value").cast(pl.Utf8))
    )


def convert_dist(feed: "Feed", new_dist_units: str) -> "Feed":
    """
    Convert the distances recorded in the ``shape_dist_traveled``
    columns of the given Feed to the given distance units.
    New distance units must lie in :const:`.constants.DIST_UNITS`.
    Return the resulting Feed.
    """
    feed = feed.copy()

    if feed.dist_units == new_dist_units:
        # Nothing to do
        return feed

    old_dist_units = feed.dist_units
    feed.dist_units = new_dist_units

    d = hp.get_convert_dist(old_dist_units, new_dist_units)

    if hp.is_not_null(feed.stop_times, "shape_dist_traveled"):
        feed.stop_times = feed.stop_times.with_columns(
            shape_dist_traveled=d(pl.col("shape_dist_traveled"))
        )

    if hp.is_not_null(feed.shapes, "shape_dist_traveled"):
        feed.shapes = feed.shapes.with_columns(
            shape_dist_traveled=d(pl.col("shape_dist_traveled"))
        )

    return feed


def compute_network_stats_0(
    stop_times: pd.DataFrame | pl.LazyFrame,
    trip_stats: pl.DataFrame | pl.LazyFrame,
    *,
    split_route_types=False,
) -> pl.LazyFrame:
    """
    Compute some network stats for the trips common to the
    given subset of stop times and given subset of trip stats
    of the form output by the function :func:`.trips.compute_trip_stats`

    Return a table with the columns

    - ``'route_type'`` (optional): presest if and only if ``split_route_types``
    - ``'num_stops'``: number of stops active on the date
    - ``'num_routes'``: number of routes active on the date
    - ``'num_trips'``: number of trips that start on the date
    - ``'num_trip_starts'``: number of trips with nonnull start
      times on the date
    - ``'num_trip_ends'``: number of trips with nonnull start times
      and nonnull end times on the date, ignoring trips that end
      after 23:59:59 on the date
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service on the date
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs on the date
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs on the date
    - ``'service_distance'``: sum of the service distances for the
      active routes on the date;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: sum of the service durations for the
      active routes on the date; measured in hours
    - ``'service_speed'``: service_distance/service_duration on the
      date

    Exclude dates with no active stops, which could yield the empty DataFrame.

    Helper function for :func:`compute_network_stats`.
    """
    # Handle defunct case
    schema = {
        "num_stops": pl.Int32,
        "num_routes": pl.Int32,
        "num_trips": pl.Int32,
        "num_trip_starts": pl.Int32,
        "num_trip_ends": pl.Int32,
        "service_distance": pl.Float64,
        "service_duration": pl.Float64,  # hours
        "service_speed": pl.Float64,
        "peak_num_trips": pl.Int32,
        "peak_start_time": pl.Utf8,  # HH:MM:SS
        "peak_end_time": pl.Utf8,  # HH:MM:SS
    }
    if split_route_types:
        schema = {"route_type": pl.Int32, **schema}

    null_stats = pl.LazyFrame(schema=schema)
    final_cols = list(schema.keys())

    if hp.is_empty(stop_times) or hp.is_empty(trip_stats):
        return null_stats

    # Handle generic case
    trip_stats = hp.make_lazy(trip_stats)
    stop_times = hp.make_lazy(stop_times)

    # Compute basic stats
    f = trip_stats.filter(pl.col("duration") > 0).with_columns(
        start_s=hp.timestr_to_seconds("start_time"),
        end_s=hp.timestr_to_seconds("end_time"),
    )

    if not split_route_types:
        # Unify calcs with dummy route_type
        f = f.with_columns(route_type=pl.lit(1))

    group_cols = ["route_type"]
    basic_stats = f.group_by(group_cols).agg(
        num_routes=pl.col("route_id").n_unique(),
        num_trips=pl.len(),
        num_trip_starts=pl.col("start_s").is_not_null().sum(),
        num_trip_ends=pl.col("end_s").is_not_null().sum(),
        service_distance=pl.col("distance").sum(),
        service_duration=pl.col("duration").sum(),
    )

    # Comput num stop stats
    num_stops_stats = (
        stop_times.select("trip_id", "stop_id")
        .join(f.select(group_cols + ["trip_id"]).unique(), "trip_id")
        .group_by(group_cols)
        .agg(num_stops=pl.col("stop_id").n_unique())
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
    result = (
        basic_stats.join(num_stops_stats, group_cols, how="left")
        .join(peak_stats, group_cols, how="left")
        .with_columns(
            service_speed=(
                pl.when(pl.col("service_duration") > 0)
                .then(pl.col("service_distance") / pl.col("service_duration"))
                .otherwise(pl.lit(0.0))
            ),
            peak_start_time=hp.seconds_to_timestr("peak_start_s"),
            peak_end_time=hp.seconds_to_timestr("peak_end_s"),
        )
        .select(final_cols)
    )
    return result.sort(group_cols) if split_route_types else result


def compute_network_stats(
    feed: "Feed",
    dates: list[str],
    trip_stats: pl.LazyFrame | pl.DataFrame | None = None,
    *,
    split_route_types=False,
) -> pl.LazyFrame:
    """
    Compute some network stats for the given subset of trip stats, which defaults to
    `feed.compute_trip_stats()`, and for the given dates (YYYYMMDD date stings).

    Return a table with the columns

    - ``'date'``
    - ``'route_type'`` (optional): presest if and only if ``split_route_types``
    - ``'num_stops'``: number of stops active on the date
    - ``'num_routes'``: number of routes active on the date
    - ``'num_trips'``: number of trips that start on the date
    - ``'num_trip_starts'``: number of trips with nonnull start
      times on the date
    - ``'num_trip_ends'``: number of trips with nonnull start times
      and nonnull end times on the date, ignoring trips that end
      after 23:59:59 on the date
    - ``'peak_num_trips'``: maximum number of simultaneous trips in
      service on the date
    - ``'peak_start_time'``: start time of first longest period
      during which the peak number of trips occurs on the date
    - ``'peak_end_time'``: end time of first longest period during
      which the peak number of trips occurs on the date
    - ``'service_distance'``: sum of the service distances for the
      active routes on the date;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: sum of the service durations for the
      active routes on the date; measured in hours
    - ``'service_speed'``: service_distance/service_duration on the
      date

    Exclude dates with no active stops, which could yield the empty DataFrame.

    The route and trip stats for date d contain stats for trips that
    start on date d only and ignore trips that start on date d-1 and
    end on date d.

    Notes
    -----
    - If you've already computed trip stats in your workflow,
      then passing it into this function will speed it up.

    """
    dates = feed.subset_dates(dates)

    # Handle defunct case
    null_stats = compute_network_stats_0(
        feed.stop_times.head(0), pl.LazyFrame(), split_route_types=split_route_types
    )
    if not dates:
        return null_stats

    final_cols = ["date"] + list(null_stats.collect_schema().names())

    # Collect stats for each date,
    # memoizing stats the sequence of trip IDs active on the date
    # to avoid unnecessary recomputations.
    # Store in a dictionary of the form
    # trip ID sequence -> stats DataFrame.
    trip_stats = (
        hp.make_lazy(trip_stats)
        if trip_stats is not None
        else feed.compute_trip_stats()
    )

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
            stats = compute_network_stats_0(
                feed.stop_times,
                trip_stats.filter(pl.col("trip_id").is_in(ids)),
                split_route_types=split_route_types,
            ).with_columns(date=pl.lit(date))
            # Remember stats
            stats_by_ids[ids] = stats
        else:
            stats = null_stats

        frames.append(stats)

    # Collate stats
    return pl.concat(frames).select(final_cols)


def compute_network_time_series(
    feed: "Feed",
    dates: list[str],
    trip_stats: pl.LazyFrame | pl.DataFrame | None = None,
    num_minutes: int = 60,
    *,
    split_route_types: bool = False,
) -> pd.DataFrame:
    """
    Compute some network stats in time series form for the given dates
    (YYYYMMDD date strings) and trip stats, which defaults to
    ``feed.compute_trip_stats()``.
    Use the given Pandas frequency string ``freq`` to specify the frequency of the
    resulting time series, e.g. '5Min'.
    If ``split_route_types``, then split stats by route type; otherwise don't.

    Return a long-form time series table with the columns

    - ``'datetime'``: datetime object
    - ``'route_type'``: integer; present if and only if ``split_route_types``
    - ``'num_trips'``: number of trips in service during during the
      time period
    - ``'num_trip_starts'``: number of trips with starting during the
      time period
    - ``'num_trip_ends'``: number of trips ending during the
      time period, ignoring the trips the end past midnight
    - ``'service_distance'``: distance traveled during the time
      period by all trips active during the time period;
      measured in kilometers if ``feed.dist_units`` is metric;
      otherwise measured in miles;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - ``'service_duration'``: duration traveled during the time
      period by all trips active during the time period;
      measured in hours
    - ``'service_speed'``: ``service_distance/service_duration`` when defined; 0
      otherwise

    Exclude dates that lie outside of the Feed's date range.
    If all the dates given lie outside of the Feed's date range,
    then return an empty DataFrame with the specified columns.

    Notes
    -----
    - If you've already computed trip stats in your workflow,
      then passing it into this function will speed it up.

    """
    rts = feed.compute_route_time_series(
        dates=dates, trip_stats=trip_stats, num_minutes=num_minutes
    )

    # Handle defunct case
    null_stats = rts.head(0).drop("route_id")
    if split_route_types:
        null_stats = null_stats.with_columns(route_type=None)

    if hp.is_empty(rts):
        return null_stats

    # Handle generic case
    metrics = [
        "num_trip_starts",
        "num_trip_ends",
        "num_trips",
        "service_distance",
        "service_duration",
    ]
    group_cols = ["datetime"]

    if split_route_types:
        rts = rts.join(
            feed.routes.select("route_id", "route_type"), "route_id", how="left"
        )
        group_cols.append("route_type")

    return (
        rts.group_by(group_cols)
        .agg(**{c: pl.col(c).sum() for c in metrics})
        .with_columns(
            service_speed=(
                pl.when(pl.col("service_duration") > 0)
                .then(pl.col("service_distance") / pl.col("service_duration"))
                .otherwise(pl.lit(0.0))
            ),
        )
        .sort(group_cols)
    )


def create_shapes(feed: "Feed", *, all_trips: bool = False) -> "Feed":
    """
    Given a feed, create a shape for every trip that is missing a
    shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips
    tables.

    If ``all_trips``, then create new shapes for all trips by
    connecting stops, and remove the old shapes.
    """
    if all_trips:
        trips_to_touch = feed.trips.select("trip_id")
    else:
        trips_to_touch = feed.trips.filter(pl.col("shape_id").is_null()).select(
            "trip_id"
        )

    # Get stop times for target trips
    f = (
        feed.stop_times.join(trips_to_touch, on="trip_id", how="semi")
        .select("trip_id", "stop_sequence", "stop_id")
        .sort("trip_id", "stop_sequence")
    )

    if hp.is_empty(f):
        # No stop times so nothing can be done
        return feed

    # For each trip, build its ordered stop sequence
    seqs = f.group_by("trip_id", maintain_order=True).agg(
        stop_seq=pl.col("stop_id").cast(pl.Utf8).implode()
    )

    # Create a canonical ID per unique stop sequence.
    # To make IDs deterministic, sort by a stable key derived from the sequence.
    uniq = (
        seqs.select("stop_seq")
        .unique()
        .with_columns(seq_key=pl.col("stop_seq").list.join("\x1f"))
        .sort("seq_key")
        .with_columns(
            shape_id_new=pl.concat_str([pl.lit("shape_"), pl.row_index().cast(pl.Utf8)])
        )
        .select("stop_seq", "shape_id_new")
    )

    # Map each trip to its shape_id_new via its stop sequence
    trip_to_shape = seqs.join(uniq, on="stop_seq", how="left").select(
        "trip_id", "shape_id_new", "stop_seq"
    )

    # Updated trips table
    if all_trips:
        trips = (
            feed.trips.join(
                trip_to_shape.select("trip_id", "shape_id_new"),
                on="trip_id",
                how="left",
            )
            .with_columns(shape_id=pl.col("shape_id_new"))
            .drop("shape_id_new")
        )
    else:
        trips = (
            feed.trips.join(
                trip_to_shape.select("trip_id", "shape_id_new"),
                on="trip_id",
                how="left",
            )
            .with_columns(
                shape_id=pl.coalesce([pl.col("shape_id"), pl.col("shape_id_new")])
            )
            .drop("shape_id_new")
        )

    # Build new shapes rows from the unique sequences:
    # one row per (shape_id_new, shape_pt_sequence, stop_id)
    shape_rows = (
        uniq.explode("stop_seq")
        .with_columns(
            shape_pt_sequence=pl.row_index().over("shape_id_new"),
            stop_id=pl.col("stop_seq"),
        )
        .select("shape_id_new", "shape_pt_sequence", "stop_id")
    )
    # Join to stops for lon/lat
    new_shapes = (
        shape_rows.join(
            feed.stops.select("stop_id", "stop_lon", "stop_lat"),
            on="stop_id",
            how="left",
        )
        .select(
            "shape_pt_sequence",
            shape_id=pl.col("shape_id_new"),
            shape_pt_lon=pl.col("stop_lon"),
            shape_pt_lat=pl.col("stop_lat"),
        )
        .sort(["shape_id", "shape_pt_sequence"])
    )

    # Final shapes table
    if getattr(feed, "shapes", None) is not None and not all_trips:
        shapes = pl.concat([feed.shapes, new_shapes], how="diagonal_relaxed")
    else:
        shapes = new_shapes

    # Return updated feed
    feed = feed.copy()
    feed.trips = trips
    feed.shapes = shapes
    return feed


def compute_bounds(feed: "Feed", stop_ids: list[str] | None = None) -> np.array:
    """
    Return the bounding box (Numpy array [min longitude, min latitude, max longitude,
    max latitude]) of the given Feed's stops or of the subset of stops
    specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_geo=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.total_bounds


def compute_convex_hull(feed: "Feed", stop_ids: list[str] | None = None) -> sg.Polygon:
    """
    Return a convex hull (Shapely Polygon) representing the convex hull of the given
    Feed's stops or of the subset of stops specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_geo=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.union_all().convex_hull


def compute_centroid(feed: "Feed", stop_ids: list[str] | None = None) -> sg.Point:
    """
    Return the centroid (Shapely Point) of the convex hull the given Feed's stops
    or of the subset of stops specified by the given stop IDs.
    """
    from .stops import get_stops

    g = get_stops(feed, as_geo=True)
    if stop_ids is not None:
        g = g.loc[lambda x: x["stop_id"].isin(stop_ids)]

    return g.union_all().convex_hull.centroid


def restrict_to_trips(feed: "Feed", trip_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. used by the trips of the given IDs.
    Return the resulting feed.

    If no valid trip IDs are given, which includes the case of the empty list,
    then the resulting feed will have all empty non-agency tables.

    This function is probably more useful internally than externally.
    """
    feed = feed.copy()
    has_agency_ids = "agency_id" in feed.routes.columns

    # Subset trips
    feed.trips = feed.trips.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Subset routes
    feed.routes = feed.routes.loc[lambda x: x.route_id.isin(feed.trips.route_id)].copy()

    # Subset stop times
    feed.stop_times = feed.stop_times.loc[lambda x: x.trip_id.isin(trip_ids)].copy()

    # Subset stops, collecting parent stations too
    stop_ids_0 = set(feed.stop_times["stop_id"])
    stop_ids_1 = set(
        feed.stops.loc[
            lambda x: x["stop_id"].isin(stop_ids_0), "parent_station"
        ].dropna()
    )
    stop_ids = stop_ids_0 | stop_ids_1
    feed.stops = feed.stops.loc[lambda x: x["stop_id"].isin(stop_ids)].copy()

    # Subset calendar
    service_ids = feed.trips["service_id"].unique()
    if feed.calendar is not None:
        feed.calendar = feed.calendar.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Subset agency
    if has_agency_ids:
        agency_ids = feed.routes["agency_id"]
        feed.agency = feed.agency.loc[lambda x: x.agency_id.isin(agency_ids)].copy()

    # Now for the optional files.
    # Subset calendar dates.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates.loc[
            lambda x: x.service_id.isin(service_ids)
        ].copy()

    # Subset frequencies
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies.loc[
            lambda x: x.trip_id.isin(trip_ids)
        ].copy()

    # Subset shapes
    if feed.shapes is not None:
        shape_ids = feed.trips.shape_id
        feed.shapes = feed.shapes.loc[lambda x: x.shape_id.isin(shape_ids)].copy()

    # Subset transfers
    if feed.transfers is not None:
        feed.transfers = feed.transfers.loc[
            lambda x: x.from_stop_id.isin(stop_ids) & x.to_stop_id.isin(stop_ids)
        ].copy()

    return feed


def restrict_to_routes(feed: "Feed", route_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips` and
    the trips with the given route IDs.
    Return the resulting feed.
    """
    trip_ids = feed.trips.loc[lambda x: x.route_id.isin(route_ids), "trip_id"].tolist()
    return restrict_to_trips(feed, trip_ids)


def restrict_to_agencies(feed: "Feed", agency_ids: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_routes` and
    the routes with the given agency IDs.
    Return the resulting feed.
    """
    # Build feed via `restrict_to_routes`
    feed = feed.copy()
    route_ids = feed.routes.loc[
        lambda x: x.agency_id.isin(agency_ids), "route_id"
    ].tolist()

    return feed.restrict_to_routes(route_ids)


def restrict_to_dates(feed: "Feed", dates: list[str]) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips` and
    the trips active on at least one of the given dates (YYYYMMDD strings).
    Return the resulting feed.
    """
    # Get every trip that is active on at least one of the dates
    trip_activity = feed.compute_trip_activity(dates)
    if trip_activity.is_empty():
        trip_ids = []
    else:
        trip_ids = trip_activity.loc[
            lambda x: x.filter(dates).sum(axis=1) > 0,
            "trip_id",
        ]

    return restrict_to_trips(feed, trip_ids)


def restrict_to_area(feed: "Feed", area: gpd.GeoDataFrame) -> "Feed":
    """
    Build a new feed by restricting this one via :func:`restrict_to_trips`
    and the trips that have at least one stop intersecting the given GeoDataFrame of
    polygons.
    Return the resulting feed.
    """
    from .stops import get_stops_in_area

    # Get IDs of stops within the polygon
    stop_ids = get_stops_in_area(feed, area).stop_id

    # Get all trips with at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st.loc[lambda x: x.stop_id.isin(stop_ids), "trip_id"]

    return restrict_to_trips(feed, trip_ids)


def _reshape_stop_times(stop_times: pd.DataFrame) -> pd.DataFrame:
    """
    Given a GTFS stop times DataFrame, reshape it to have only the following columns.

    - trip_id
    - stop_sequence
    - from_departure_time
    - to_departure_time
    - from_stop_id
    - to_stop_id
    - from_shape_dist_traveled (optional): present if and only if
      'shape_dist_traveled' column present in given stop times
    - to_shape_dist_traveled (optional): present if and only if 'shape_dist_traveled'
      column present in given stop times

    This is a helper function for :func:`compute_screen_line_counts`.
    """
    f = stop_times.sort_values(["trip_id", "stop_sequence"], ignore_index=True)
    g = f.groupby("trip_id")

    has_dist = "shape_dist_traveled" in f.columns

    # For each trip, create shifted columns for the "to" stop and its associated time and distance fields
    f["to_stop_id"] = g["stop_id"].shift(-1)
    f["to_departure_time"] = g["departure_time"].shift(-1)
    if has_dist:
        f["to_shape_dist_traveled"] = g["shape_dist_traveled"].shift(-1)

    # Drop rows where there is no "to" stop (i.e. the last stop in each trip)
    f = f.dropna(subset=["to_stop_id"])

    # Rename the original columns to reflect they represent the "from" stop in the segment
    f = f.rename(
        columns={
            "stop_id": "from_stop_id",
            "departure_time": "from_departure_time",
            "shape_dist_traveled": "from_shape_dist_traveled",
        }
    )

    return f.filter(
        [
            "trip_id",
            "stop_sequence",
            "from_departure_time",
            "to_departure_time",
            "from_stop_id",
            "to_stop_id",
            "from_shape_dist_traveled",
            "to_shape_dist_traveled",
        ]
    )


def compute_screen_line_counts(
    feed: "Feed",
    screen_lines: gpd.GeoDataFrame,
    dates: list[str],
    segmentize_m: float = 5,
    *,
    include_testing_cols: bool = False,
) -> pd.DataFrame:
    """
    Find all the Feed trips active on the given YYYYMMDD dates that intersect
    the given segment-associated screen lines of the form output by
    :func:`build_screen_lines`.
    Behind the scenes, use simple sub-LineStrings of the feed
    (with points separated by at ``segmentize_m`` meters)
    to compute screen line intersections.
    Using them instead of the Feed shapes avoids miscounting intersections in the
    case of non-simple (self-intersecting) shapes.

    For each trip crossing a screen line,
    compute the crossing time, crossing direction, etc. and return a DataFrame
    of results with the columns

    - ``'date'``: the YYYYMMDD date string given
    - ``'screen_line_id'``: ID of a screen line
    - ``'trip_id'``: ID of a trip that crosses the screen line
    - ``'shape_id'``: ID of the trip's shape
    - ``'direction_id'``: GTFS direction of trip
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'route_type'``
    - ``'shape_id'``
    - ``'crossing_direction'``: 1 or -1; 1 indicates trip travel from the
      left side to the right side of the screen line;
      -1 indicates trip travel in the  opposite direction
    - ``'crossing_time'``: time, according to the GTFS schedule, that the trip
      crosses the screen line
    - ``'crossing_dist_m'``: distance along the trip shape (not subshape) of the
      crossing; in meters

    If ``include_testing_columns``, then include the following extra columns for testing
    purposes.

    - ``'subshape_id'``: ID of the simple sub-LineString S of the trip's shape that
      crosses the screen line
    - ``'subshape_length_m'``: length of S in meters
    - ``'from_departure_time'``: departure time of the trip from the last stop before
      the screen line
    - ``'to_departure_time'``: departure time of the trip at from the first stop after
      the screen line
    - ``'subshape_dist_frac'``: proportion of S's length at which the screen line
      intersects S

    Notes:

    - Assume the Feed's stop times DataFrame has an accurate ``shape_dist_traveled``
      column.
    - Assume that trips travel in the same direction as their shapes, an assumption
      that is part of the GTFS.
    - Assume that the screen line is straight and simple.
    - The algorithm works as follows

        1. Find the Feed's simple subshapes (computed via :func:`shapes.split_simple`)
           that intersect the screen lines.
        2. For each such subshape and screen line, compute the intersection points,
           the distance of each point along the subshape, aka the *crossing distance*,
           and the orientation of the screen line relative to the subshape.
        3. Restrict to trips active on the given dates and for each trip associated to
           an intersecting subshape above, interpolate a trip stop time
           for the intersection point using the crossing distance, subshape length,
           cumulative subshape length, and trip stop times.

    """
    from .shapes import split_simple

    # Convert geoms to UTM
    crs = screen_lines.estimate_utm_crs()
    screen_lines = screen_lines.to_crs(crs)

    # Create screen line IDs if necessary
    n = screen_lines.shape[0]
    if "screen_line_id" not in screen_lines.columns:
        screen_lines["screen_line_id"] = hp.make_ids(n, "sl")

    # Make a vector in the direction of each screen line to calculate crossing
    # orientation. Does not work in case of a bent screen line.
    p1 = screen_lines["geometry"].map(lambda x: np.array(x.coords[0]))
    p2 = screen_lines["geometry"].map(lambda x: np.array(x.coords[-1]))
    screen_lines["screen_line_vector"] = p2 - p1

    # Get the simple subshapes that intersect the screen lines.
    # Need subshapes to have only small gaps between them,
    # so `segmentize_m` needs to be small.
    subshapes = (
        feed.get_shapes(as_geo=True, use_utm=True)
        .sjoin(screen_lines)
        .drop_duplicates("shape_id")
        .pipe(split_simple, segmentize_m=segmentize_m)
    )

    # Get intersection points of subshapes and screen lines
    g0 = (
        subshapes.sjoin(screen_lines.filter(["screen_line_id", "geometry"]))
        .merge(screen_lines, on="screen_line_id")
        .assign(
            int_point=lambda x: gpd.GeoSeries(x["geometry_x"], crs=crs).intersection(
                gpd.GeoSeries(x["geometry_y"], crs=crs)
            )
        )
    )

    # Unpack multipoint intersections to yield a new GeoDataFrame.
    # Should be very few multipoints.
    records = []
    for row in g0.itertuples(index=False):
        if isinstance(row.int_point, sg.Point):
            intersections = [row.int_point]
        else:
            intersections = row.int_point.geoms
        for int_point in intersections:
            record = {
                "subshape_id": row.subshape_id,
                "shape_id": row.shape_id,
                "subshape_length_m": row.subshape_length_m,
                "cum_length_m": row.cum_length_m,
                "screen_line_id": row.screen_line_id,
                "geometry": row.geometry_x,
                "int_point": int_point,
                "screen_line_vector": row.screen_line_vector,
            }
            records.append(record)

    g = gpd.GeoDataFrame.from_records(records).set_geometry("geometry").set_crs(crs)

    # Get distance (in meters) of each intersection point along subshape
    g["subshape_dist_frac"] = g.apply(
        lambda x: x["geometry"].project(x.int_point, normalized=True), axis=1
    )
    g["subshape_dist_m"] = g["subshape_dist_frac"] * g["subshape_length_m"]
    g["crossing_dist_m"] = (
        g["subshape_dist_m"] + g["cum_length_m"] - g["subshape_length_m"]
    )

    # Build a tiny vector along each subshape from the intersection point
    p2 = g.apply(
        lambda x: x["geometry"].interpolate(x["subshape_dist_m"] + 1), axis=1
    ).map(lambda x: np.array(x.coords[0]))
    p1 = g.int_point.map(lambda x: np.array(x.coords[0]))
    g["subshape_vector"] = p2 - p1

    # Compute crossing direction by taking the vector cross product of
    # the link vector and the screen line vector
    det = g.apply(
        lambda x: np.linalg.det(
            np.array([x["subshape_vector"], x["screen_line_vector"]])
        ),
        axis=1,
    )
    g["crossing_direction"] = det.map(lambda x: 1 if x >= 0 else -1)

    # Summarize work so far
    g = g[
        [
            "subshape_id",
            "shape_id",
            "screen_line_id",
            "subshape_dist_frac",
            "subshape_dist_m",
            "subshape_length_m",
            "crossing_direction",
            "crossing_dist_m",
        ]
    ]

    # Get stop times to compute crossing times
    feed = feed.convert_dist("m")
    frames = []
    for date in dates:
        st = (
            feed.get_stop_times(date)
            .pipe(_reshape_stop_times)
            .merge(feed.trips[["trip_id", "shape_id"]])
            # Keep only non-NaN departure times
            .loc[lambda x: x["from_departure_time"].notna()]
            .loc[lambda x: x["to_departure_time"].notna()]
            # Convert to seconds past midnight for upcoming crossing time calculation
            .assign(
                t1=lambda x: x["from_departure_time"].map(hp.timestr_to_seconds),
                t2=lambda x: x["to_departure_time"].map(hp.timestr_to_seconds),
            )
        )

        # Compute crossing times
        subframes = []
        for shape_id, group in g.groupby("shape_id"):
            f = (
                st.merge(group)
                # Only keep the times of the pair of stops on either side of each screen line,
                # whose distance along a trip shape is marked by column 'crossing_dist_m'
                .loc[lambda x: x["from_shape_dist_traveled"] <= x["crossing_dist_m"]]
                .loc[lambda x: x["crossing_dist_m"] <= x["to_shape_dist_traveled"]]
            )
            f["crossing_time"] = (
                f["t1"] + f["subshape_dist_frac"] * (f["t2"] - f["t1"])
            ).map(lambda x: hp.seconds_to_timestr(x))
            # Get distance along trip shape of crossing point
            subframes.append(f)

        if subframes:
            f = pd.concat(subframes).assign(date=date)
        else:
            f = pd.DataFrame()
        frames.append(f)

    f = pd.concat(frames)

    # Clean up
    final_cols = [
        "date",
        "segment_id",
        "segment_length",
        "screen_line_id",
        "shape_id",
        "trip_id",
        "direction_id",
        "route_id",
        "route_short_name",
        "route_type",
        "crossing_direction",
        "crossing_time",
        "crossing_dist_m",
    ]
    if include_testing_cols:
        final_cols += [
            "subshape_id",
            "subshape_length_m",
            "from_departure_time",
            "to_departure_time",
            "subshape_dist_frac",
            "subshape_dist_m",
        ]

    return (
        f
        # Append screen line info
        .merge(screen_lines.drop("geometry", axis=1))
        # Append extra trip info
        .merge(feed.trips[["trip_id", "direction_id", "route_id"]])
        .merge(feed.routes[["route_id", "route_short_name", "route_type"]])
        .filter(final_cols)
        .sort_values(
            ["screen_line_id", "trip_id", "crossing_dist_m"],
            ignore_index=True,
        )
    )
