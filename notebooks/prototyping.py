import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import datetime as dt
    import sys
    import os
    import dateutil.relativedelta as rd
    import json
    import pathlib as pb
    from typing import List
    import warnings

    import marimo as mo
    import polars as pl
    import polars_st as st
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import shapely as sl
    import shapely.geometry as sg
    import shapely.ops as so
    import folium as fl
    import plotly.express as px

    import gtfs_kit_polars as gk

    warnings.filterwarnings("ignore")

    DATA = pb.Path("data")
    return DATA, gk, pl, st


@app.cell
def _(DATA, gk):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    # feed = gk.read_feed(
    #     pb.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km"
    # )
    feed = gk.read_feed(
        DATA / "cairns_gtfs.zip", dist_units="km"
    ).append_dist_to_stop_times()
    feed.stop_times.head().collect()
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _(gk, pl, st):
    def compute_screen_line_counts(
        feed: "Feed",
        screen_lines: st.GeoLazyFrame | st.GeoDataFrame,
        dates: list[str],
        *,
        include_testing_cols: bool = False,
    ) -> pl.LazyFrame:
        """
        Find all the Feed trips active on the given YYYYMMDD dates that intersect
        the given screen lines (LineStrings) with optional ID column ``screen_line_id``.
        Behind the scenes, use simple sub-LineStrings of the feed
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
        hp = gk

        # Convert screen lines to UTM
        utm = hp.get_utm_srid(screen_lines)
        sl = hp.make_lazy(screen_lines).pipe(hp.to_srid, srid=utm)

        if "screen_line_id" not in sl.collect_schema().names():
            # Create screen line ID with index
            sl = sl.with_columns(
                screen_line_id=(
                    pl.lit("sl") + pl.int_range(0, pl.len()).cast(pl.Utf8).str.zfill(3)
                )
            )
        # Make a vector in the direction of each screen line to later calculate crossing
        # direction. Does not work in case of a bent screen line.
        sl = (
            sl.with_columns(n=st.geom().st.count_points())
            .with_columns(
                p1=st.geom().st.get_point(0),
                p2=st.geom().st.get_point(pl.col("n") - 1),
            )
            .with_columns(
                sl_dx=pl.col("p2").st.x() - pl.col("p1").st.x(),
                sl_dy=pl.col("p2").st.y() - pl.col("p1").st.y(),
            )
            .drop("p1", "p2")
        )
        intersections = (
            # Get simple subshapes
            feed.get_shapes(as_geo=True, use_utm=True)
            .select("shape_id", "geometry")
            .pipe(gk.split_simple)
            .join(sl, how="cross")
            # Intersect them with screen lines
            .filter(st.geom().st.intersects(pl.col("geometry_right")))
            .rename({"geometry_right": "screen_geom"})
            # Unpack all intersection points
            .with_columns(int_geom=st.geom().st.intersection(pl.col("screen_geom")))
            .with_columns(int_parts=pl.col("int_geom").st.parts())
            .explode("int_parts")
            .drop("int_geom")
            .rename({"int_parts": "int_point"})
            # Compute crossing distance (in meters) along each subshape then along each shape
            .with_columns(subshape_dist_m=st.geom().st.project(pl.col("int_point")))
            .with_columns(
                subshape_dist_frac=pl.col("subshape_dist_m") / pl.col("subshape_length_m"),
                crossing_dist_m=pl.col("subshape_dist_m")
                + pl.col("cum_length_m")
                - pl.col("subshape_length_m"),
            )
            # Make a 1-meter-long vector from intersection point p1
            # to a point p2 on subshape
            .with_columns(p1=pl.col("int_point"))
            # target distance along subshape, clamped to [0, subshape_length_m]
            .with_columns(
                targ=pl.min_horizontal(
                    pl.col("subshape_dist_m") + pl.lit(1), pl.col("subshape_length_m")
                )
            )
            .with_columns(p2=st.geom().st.interpolate(pl.col("targ")))
            .with_columns(
                seg_dx=pl.col("p2").st.x() - pl.col("p1").st.x(),
                seg_dy=pl.col("p2").st.y() - pl.col("p1").st.y(),
            )
            # The sign of the 2D cross product of the subshape vector with the screenline vector
            # determines the direction of crossing across the screen line
            .with_columns(
                det=pl.col("seg_dx") * pl.col("sl_dy") - pl.col("seg_dy") * pl.col("sl_dx"),
            )
            .with_columns(
                crossing_direction=pl.when(pl.col("det") >= 0)
                .then(pl.lit(1))
                .otherwise(pl.lit(-1)),
            )
            .select(
                "subshape_id",
                "shape_id",
                "screen_line_id",
                "subshape_dist_frac",
                "subshape_dist_m",
                "subshape_length_m",
                "crossing_direction",
                "crossing_dist_m",
            )
        )

        # Estimate crossing time from stop times table
        feed_m = feed.convert_dist("m")
        frames = []
        for date in dates:
            # Reshape stop times for easy time-distance interpolation
            stop_times_r = (
                feed_m.get_stop_times(date)
                .join(feed.trips.select("trip_id", "shape_id"), "trip_id")
                .select(
                    "trip_id",
                    "shape_id",
                    "stop_sequence",
                    "shape_dist_traveled",
                    "departure_time",
                )
                .sort(["trip_id", "stop_sequence"])
                .with_columns(
                    from_shape_dist_traveled=pl.col("shape_dist_traveled"),
                    to_shape_dist_traveled=pl.col("shape_dist_traveled")
                    .shift(-1)
                    .over("trip_id"),
                    from_departure_time=pl.col("departure_time"),
                    to_departure_time=pl.col("departure_time").shift(-1).over("trip_id"),
                )
                .drop("shape_dist_traveled", "departure_time")
                .filter(pl.col("to_shape_dist_traveled").is_not_null())
            )

            f = (
                intersections.join(stop_times_r, on="shape_id", how="inner")
                # Only keep the times of the pair of stops on either side of each screen line,
                # whose distance along a trip shape is marked by column 'crossing_dist_m'
                .filter(
                    (pl.col("from_departure_time").is_not_null())
                    & (pl.col("to_departure_time").is_not_null())
                    & (pl.col("from_shape_dist_traveled") <= pl.col("crossing_dist_m"))
                    & (pl.col("crossing_dist_m") <= pl.col("to_shape_dist_traveled"))
                )
                .with_columns(
                    t1=hp.timestr_to_seconds("from_departure_time"),
                    t2=hp.timestr_to_seconds("to_departure_time"),
                )
                .with_columns(
                    crossing_time=(
                        pl.col("t1")
                        + pl.col("subshape_dist_frac") * (pl.col("t2") - pl.col("t1"))
                    ),
                    date=pl.lit(date),
                )
            )
            frames.append(f)

        f = pl.concat(frames) if frames else pl.LazyFrame({"date": pl.Series([], pl.Utf8)})

        # Append trip and route info and clean up
        final_cols = [
            "date",
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
            f.join(
                feed.trips.select("trip_id", "direction_id", "route_id"),
                on="trip_id",
                how="left",
            )
            .join(
                feed.routes.select("route_id", "route_short_name", "route_type"),
                on="route_id",
                how="left",
            )
            .with_columns(crossing_time=hp.seconds_to_timestr("crossing_time"))
            .select(final_cols)
            .unique()
            .sort("screen_line_id", "trip_id", "crossing_dist_m")
        )
    return (compute_screen_line_counts,)


@app.cell
def _(DATA, compute_screen_line_counts, dates, feed, st):
    path = DATA / "cairns_screen_lines.geojson"
    screen_lines = st.read_file(path)  # .pipe(gk.make_lazy)
    # screen_lines.with_columns(
    #     p1=st.geom().st.get_point(0),
    #     p2=st.geom().st.get_point(1),
    # )

    compute_screen_line_counts(feed, screen_lines, dates).collect()
    return


@app.cell
def _(DATA, compute_screen_line_counts, feed, gk, st):
    import pytest


    def test_compute_screen_line_counts(cairns, dates):
        # Load screen line
        path = DATA / "cairns_screen_lines.geojson"
        screen_lines = st.read_file(path)
        f = compute_screen_line_counts(feed, screen_lines, dates)

        # Should have correct columns
        expect_cols = {
            "date",
            "screen_line_id",
            "trip_id",
            "direction_id",
            "route_id",
            "route_short_name",
            "route_type",
            "shape_id",
            "crossing_direction",
            "crossing_dist_m",
            "crossing_time",
        }
        assert set(f.columns) == expect_cols

        # Should have both directions
        assert set(f.crossing_direction.unique()) == {-1, 1}

        # Should only have feed dates
        assert set(f["date"].values) == set(dates)

        # Empty check
        f = compute_screen_line_counts(feed, screen_lines, ["20010101"])
        assert gk.is_empty(f)


    test_compute_screen_line_counts()
    return


@app.cell
def _(feed, gk):
    gk.split_simple(feed.get_shapes(as_geo=True, use_utm=False).head(5)).assign(
        is_simple=lambda x: x.is_simple
    )  # .collect().st.to_wkt()
    return


app._unparsable_cell(
    r"""
     s = gk.compute_route_time_series(feed, dates, trip_stats=trip_stats, freq=\"h\")
    ts
    """,
    name="_"
)


@app.cell
def _(gk, ts):
    gk.downsample(ts, freq="3h")
    return


@app.cell
def _(dates, feed, gk, trip_stats):
    gk.compute_network_stats(feed, dates, trip_stats=trip_stats)
    return


@app.cell
def _(dates, feed, trip_stats):
    feed.compute_route_stats(dates, trip_stats=trip_stats)
    return


@app.cell
def _(dates, feed, trip_stats):
    rts = feed.compute_route_time_series(dates, trip_stats=trip_stats, freq="6h")
    rts
    return


@app.cell
def _():
    # feed = gk.read_feed(DOWN / "gtfs_brevibus.zip", dist_units="km")
    # routes = feed.get_routes(as_geo=True)
    # print(routes)
    # feed = feed.aggregate_routes()
    # feed.map_routes(feed.routes["route_id"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
