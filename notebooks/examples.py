import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib as pb
    import json

    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import geopandas as gp
    import matplotlib
    import folium as fl

    import gtfs_kit_polars as gk

    DATA = pb.Path("data")
    DESK = pb.Path.home() / "Desktop"
    return DATA, DESK, gk, pl


@app.cell
def _(DATA, gk):
    # List feed

    gk.list_feed(DATA / "cairns_gtfs.zip")
    return


@app.cell
def _(DESK, gk):
    # Read feed and describe

    feed = gk.read_feed(DESK / "auckland_gtfs_20250918.zip", dist_units="km")

    # feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")
    # feed.describe()
    return (feed,)


@app.cell
def _(feed, gk):
    feed.get_shapes(as_geo=True).pipe(gk.to_wkt).collect()
    return


@app.cell
def _(feed):
    week = feed.get_first_week()
    dates = [week[0], week[6]]
    dates
    return


@app.cell
def _(feed):
    # Trip stats; reuse these for later speed ups

    trip_stats = feed.compute_trip_stats(compute_dist_from_shapes=False)
    trip_stats.collect()
    return (trip_stats,)


@app.cell
def _(gk, pl):
    # Fix this
    def compute_route_stats_0(
        trip_stats,
        headway_start_time: str = "07:00:00",
        headway_end_time: str = "19:00:00",
        *,
        split_directions: bool = False,
    ):
        """
        Compute stats for the given subset of trips stats (of the form output by the
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
        hp = gk
        # accept DataFrame or LazyFrame
        f = trip_stats.lazy() if isinstance(trip_stats, pl.DataFrame) else trip_stats

        # empty -> typed empty frame
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

        # group columns
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

        # prepare subset & seconds
        need = group_cols + [
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
        if "direction_id" not in group_cols and "direction_id" in f.collect_schema().names():
            need.append("direction_id")

        f = (
            f.filter(pl.col("duration") > 0)
            .select(need)
            .with_columns(
                start_s=hp.timestr_to_seconds("start_time"),
                end_s=hp.timestr_to_seconds("end_time"),
            )
        )
        if split_directions:
            f = f.filter(pl.col("direction_id").is_not_null()).with_columns(
                pl.col("direction_id").cast(pl.Int32)
            )

        # basic stats
        basic = f.group_by(group_cols, maintain_order=True).agg(
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

        # headway stats
        h_start = hp.timestr_to_seconds(headway_start_time)
        h_end = hp.timestr_to_seconds(headway_end_time)

        headways = (
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

        # peak stats via event sweep# Build events (+1 for start, -1 for end), sort properly within groups
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
                descending=[False] * len(group_cols) + [False, True],
            )
            .with_columns(
                running=pl.col("delta").cum_sum().over(group_cols),
                t_next=pl.col("t").shift(-1).over(group_cols),
            )
            .with_columns(
                duration=pl.col("t_next") - pl.col("t"),
            )
        )
        peak_vals = events.group_by(group_cols).agg(peak_num_trips=pl.col("running").max())
        peak_best = (
            events.join(peak_vals, on=group_cols, how="inner")
            .filter(pl.col("running") == pl.col("peak_num_trips"))
            .filter(pl.col("duration") > 0)
            .with_columns(max_dur=pl.max("duration").over(group_cols))
            .filter(pl.col("duration") == pl.col("max_dur"))
            .sort(group_cols + ["t"])
            .group_by(group_cols)
            .agg(
                peak_start_s=pl.first("t"),
                peak_end_s=pl.first("t_next"),
            )
        )

        # assemble
        return (
            basic.join(headways, on=group_cols, how="left")
            .join(peak_vals, on=group_cols, how="left")
            .join(peak_best, on=group_cols, how="left")
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
    return (compute_route_stats_0,)


@app.cell
def _(compute_route_stats_0, trip_stats):
    compute_route_stats_0(trip_stats).collect()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
