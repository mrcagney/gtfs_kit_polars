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
def _(gk, pl, stop_id):
    import datetime as dt
    import re
    import functools as ft
    from typing import Literal


    def combine_time_series(
        series_by_indicator,
        *,
        kind: Literal["route", "stop"],
        split_directions: bool = False,
    ):
        """
        Combine a dict of wide time series (one DataFrame per indicator, columns are entities)
        into a single long-form time series with columns

        - ``'datetime'``
        - ``'route_id'`` or ``'stop_id'``: depending on ``kind``
        - ``'direction_id'``: present if and only if ``split_directions``
        - one column per indicator provided in `series_by_indicator`
        - ``'service_speed'``: if both ``service_distance`` and ``service_duration`` present

        If ``split_directions``, then assume the original time series contains data
        separated by trip direction; otherwise, assume not.
        The separation is indicated by a suffix ``'-0'`` (direction 0) or ``'-1'``
        (direction 1) in the route ID or stop ID column values.
        """
        if not series_by_indicator:
            return pl.LazyFrame(schema={"datetime": pl.Datetime})

        indicators = list(series_by_indicator.keys())
        entity_col = "route_id" if kind == "route" else "stop_id"

        long_frames = []
        for ind, wf in series_by_indicator.items():
            if isinstance(wf, pl.LazyFrame):
                lz = wf
            else:
                lz = wf.lazy()
            value_cols = [c for c in lz.columns if c != "datetime"]
            if not value_cols:
                continue
            lf = lz.unpivot(
                index=["datetime"],
                on=value_cols,
                variable_name=entity_col,
                value_name=ind,
            )
            long_frames.append(lf)

        if not long_frames:
            return pl.LazyFrame(schema={"datetime": pl.Datetime, entity_col: pl.Utf8})

        f = ft.reduce(
            lambda left, right: left.join(
                right, on=["datetime", entity_col], how="full", coalesce=True
            ),
            long_frames,
        )

        if split_directions:
            f = (
                f.with_columns(
                    _base=pl.col(entity_col).str.extract(r"^(.*)-(0|1)$", group_index=1),
                    _dir=pl.col(entity_col)
                    .str.extract(r"^(.*)-(0|1)$", group_index=2)
                    .cast(pl.Int32),
                )
                .with_columns(
                    pl.when(pl.col("_base").is_not_null())
                    .then(pl.col("_base"))
                    .otherwise(pl.col(entity_col))
                    .alias(entity_col),
                    pl.col("_dir").alias("direction_id"),
                )
                .drop("_base", "_dir")
            )

        numeric_cols = [ind for ind in indicators if ind in f.columns]
        if numeric_cols:
            f = f.with_columns(
                [pl.col(c).cast(pl.Float64).fill_null(0.0).alias(c) for c in numeric_cols]
            )

        if "service_distance" in f.columns and "service_duration" in f.columns:
            f = f.with_columns(
                service_speed=(
                    pl.when(pl.col("service_duration") > 0)
                    .then(pl.col("service_distance") / pl.col("service_duration"))
                    .otherwise(0.0)
                ).cast(pl.Float64)
            )

        cols0 = ["datetime", entity_col]
        if split_directions:
            cols0.append("direction_id")

        cols = cols0 + [
            c
            for c in [
                "num_trip_starts",
                "num_trip_ends",
                "num_trips",
                "service_duration",
                "service_distance",
                "service_speed",
            ]
            if c in f.columns
        ]

        return f.select(cols).sort(cols0)


    def get_bin_size(time_series: pl.LazyFrame) -> float:
        """
        Return the number of minutes per bin of the given time series with datetime column
        'datetime'.
        Assume the time series is regularly sampled and therefore has a single bin size.
        Return None if there's only one unique datetime present.
        """
        times = (
            time_series.select("datetime")
            .sort("datetime")
            .unique()
            .collect()["datetime"]
            .to_list()[:2]
        )
        if len(times) >= 2:
            return (times[1] - times[0]).seconds / 60


    def downsample(time_series: pl.LazyFrame, num_minutes: int) -> pl.LazyFrame:
        """
        Downsample the given route, stop, or feed time series,
        (outputs of :func:`.routes.compute_route_time_series`,
        :func:`.stops.compute_stop_time_series`, or
        :func:`.miscellany.compute_feed_time_series`,
        respectively) to time bins of size ``num_minutes`` minutes.

        Return the given time series unchanged if it's empty or
        has only one time bin per date.
        Raise a value error if ``num_minutes`` does not evenly divide 1440
        (the number of minutes in a day) or if its not a multiple of the
        bin size of the given time series.
        """
        hp = gk
        # Coerce to LazyFrame
        time_series = (
            time_series.lazy() if isinstance(time_series, pl.DataFrame) else time_series
        )

        # Handle defunct cases
        if hp.is_empty(time_series):
            return time_series

        orig_num_minutes = get_bin_size(time_series)
        if orig_num_minutes is None:
            return time_series

        num_minutes = int(num_minutes)
        if num_minutes == orig_num_minutes:
            return time_series

        if 1440 % num_minutes != 0:
            raise ValueError("num_minutes must evenly divide 24*60")

        if num_minutes % orig_num_minutes != 0:
            raise ValueError(
                f"num_minutes must be a multiple of the original time series bin size "
                f"({orig_num_minutes} minutes)"
            )

        # Handle generic case
        cols = time_series.collect_schema().names()

        if stop_id in cols:
            # It's a stops time series
            metrics = ["num_trips"]
            dims = [c for c in cols if c not in (["datetime"] + metrics)]
            result = (
                time_series.with_columns(
                    datetime=pl.col("datetime")
                    .cast(pl.Datetime)
                    .dt.truncate(f"{num_minutes}m")
                )
                .group_by(dims + ["datetime"])
                .agg(num_trips=pl.col("num_trips").sum())
            )
        else:
            # It's a route or feed time series
            metrics = [
                "num_trips",
                "num_trip_starts",
                "num_trip_ends",
                "service_distance",
                "service_duration",
                "service_speed",
            ]
            dims = [c for c in cols if c not in (["datetime"] + metrics)]

            # Bin timestamps but also keep fine timestamps for last-value pick below
            ts = time_series.with_columns(
                big=pl.col("datetime").cast(pl.Datetime).dt.truncate(f"{num_minutes}m")
            )

            # Sums per coarse timestamp and last fine timestamp
            sums = ts.group_by(dims + ["big"]).agg(
                num_trip_starts=pl.col("num_trip_starts").sum(),
                num_trip_ends=pl.col("num_trip_ends").sum(),
                service_distance=pl.col("service_distance").sum(),
                service_duration=pl.col("service_duration").sum(),
                last_dt=pl.col("datetime").max(),
            )

            # Get last fine timestamp values per coarse timestamp values of
            # num_trips and num_trip_ends to use in aggregation:
            # num_trips = num_trips_last + sum(num_trip_ends in all but last fine timestamp)
            #           = num_trips_last + (num_trip_ends_sum - num_trip_ends_last)
            last_vals = (
                ts.select(dims + ["big", "datetime", "num_trips", "num_trip_ends"])
                .join(sums.select(dims + ["big", "last_dt"]), on=dims + ["big"])
                .filter(pl.col("datetime") == pl.col("last_dt"))
                .group_by(dims + ["big"])
                .agg(
                    num_trips_last=pl.col("num_trips").max(),
                    num_trip_ends_last=pl.col("num_trip_ends").max(),
                )
            )

            # Merge and compute final metrics
            result = sums.join(last_vals, on=dims + ["big"], how="left").with_columns(
                num_trips=pl.col("num_trips_last")
                + pl.col("num_trip_ends")
                - pl.col("num_trip_ends_last"),
                service_speed=(
                    pl.col("service_distance") / pl.col("service_duration")
                ).fill_null(0),
                datetime=pl.col("big"),
            )

        return result.select(["datetime"] + dims + metrics).sort(["datetime"] + dims)


    def compute_route_time_series_0(
        trip_stats,
        date_label: str = "20010101",
        num_minutes: int = 60,
        *,
        split_directions: bool = False,
    ):
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
        hp = gk

        tss = trip_stats.collect() if isinstance(trip_stats, pl.LazyFrame) else trip_stats

        # Handle defunct case
        schema = {
            "datetime": pl.Datetime,
            "route_id": pl.Utf8,
            "num_trips": pl.Float64,
            "num_trip_starts": pl.Float64,
            "num_trip_ends": pl.Float64,
            "service_distance": pl.Float64,
            "service_duration": pl.Float64,
            "service_speed": pl.Float64,
        }
        if split_directions:
            schema["direction_id"] = (pl.Int8,)

        null_stats = pl.LazyFrame(schema=schema)
        if hp.is_empty(tss):
            return null_stats

        # Split directions (encode into route_id) if requested
        if split_directions:
            tss = tss.filter(pl.col("direction_id").is_not_null()).with_columns(
                pl.col("direction_id").cast(pl.Int8),
                route_id=(
                    pl.col("route_id") + pl.lit("-") + pl.col("direction_id").cast(pl.Utf8)
                ),
            )
            if hp.is_empty(tss):
                raise ValueError(
                    "At least one trip stats direction ID value must be non-null."
                )

        # Precompute per-minute bins
        bins = list(range(24 * 60))
        num_bins = len(bins)

        # Routes present
        routes = sorted(tss["route_id"].unique().to_list())

        # Prepare indicator containers
        indicators = [
            "num_trip_starts",
            "num_trip_ends",
            "num_trips",
            "service_duration",
            "service_distance",
        ]
        series_by_route_by_indicator = {
            ind: {route: [0] * num_bins for route in routes} for ind in indicators
        }

        # ---- helper: HH:MM:SS -> minute-of-day (0..1439), null on bad input
        def timestr_to_min(col: str) -> pl.Expr:
            return hp.timestr_to_seconds(col, mod24=True) // 60
            # parts = pl.col(col).str.split_exact(":", 3)
            # h = parts.struct.field("field_0").cast(pl.Int64)
            # m = parts.struct.field("field_1").cast(pl.Int64)
            # s = parts.struct.field("field_2").cast(pl.Int64)
            # total = (h * 3600 + m * 60 + s)
            # # modulo 24h, then floor-div by 60 to get minute index
            # return ((total % (24 * 3600)) // 60).cast(pl.Int32)

        # Iterate rows (same semantics as pandas version)
        # Compute minute indices
        tss = tss.with_columns(
            start_index=timestr_to_min("start_time"),
            end_index=timestr_to_min("end_time"),
        )
        for row in tss.iter_rows(named=True):
            route = row["route_id"]
            start = row["start_index"]
            end = row["end_index"]
            distance = row.get("distance")

            if start is None or end is None or start == end:
                continue

            if start < end:
                bins_to_fill = bins[start:end]
            else:
                bins_to_fill = bins[start:] + bins[:end]

            # starts
            series_by_route_by_indicator["num_trip_starts"][route][start] += 1
            # ends (only when not wrapping past midnight)
            if start < end:
                series_by_route_by_indicator["num_trip_ends"][route][end] += 1

            # per-minute accruals
            L = len(bins_to_fill)
            for b in bins_to_fill:
                series_by_route_by_indicator["num_trips"][route][b] += 1
                series_by_route_by_indicator["service_duration"][route][b] += 1 / 60.0
                if distance is not None and L > 0:
                    series_by_route_by_indicator["service_distance"][route][b] += distance / L

        # Build per-indicator DataFrames indexed by minute over provided date_label
        base_dt = dt.datetime.strptime(date_label + " 00:00:00", "%Y%m%d %H:%M:%S")
        rng = [base_dt + dt.timedelta(minutes=i) for i in range(24 * 60)]

        series_by_indicator = {}
        for ind in indicators:
            cols = {"datetime": rng}
            for route in routes:
                cols[route] = series_by_route_by_indicator[ind][route]
            series_by_indicator[ind] = pl.LazyFrame(cols, strict=False)

        # Combine & downsample via your helpers; return a LazyFrame
        return combine_time_series(
            series_by_indicator, kind="route", split_directions=split_directions
        ).pipe(downsample, num_minutes=num_minutes)
    return (compute_route_time_series_0,)


@app.cell
def _(compute_route_time_series_0, trip_stats):
    compute_route_time_series_0(trip_stats).collect()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
