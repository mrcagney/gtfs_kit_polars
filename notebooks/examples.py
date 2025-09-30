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
    return DATA, DESK, gk, pd, pl


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
def _(compute_route_stats_0, pd, pl):
    def compute_route_stats(
        feed: "Feed",
        dates: list[str],
        trip_stats: pl.DataFrame|pl.LazyFrame| None = None,
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
        null_stats = null_stats.assign(date=None).filter(final_cols)
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
            ids = tuple(sorted(activity.loc[activity[date] > 0, "trip_id"].values))
            if ids in stats_by_ids:
                # Reuse stats with updated date
                stats = stats_by_ids[ids].assign(date=date)
            elif ids:
                # Compute stats afresh
                t = trip_stats.loc[lambda x: x.trip_id.isin(ids)].copy()
                stats = compute_route_stats_0(
                    t,
                    split_directions=split_directions,
                    headway_start_time=headway_start_time,
                    headway_end_time=headway_end_time,
                ).assign(date=date)
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
        return pd.concat(frames).filter(final_cols).sort_values(sort_by)

    return


@app.cell
def _(pl, trip_stats):
    trip_stats.filter(pl.col("route_id") == "106-202").collect()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
