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
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km").append_dist_to_stop_times()
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
        hp = gk

        # screen_lines: ensure lazy + UTM
        utm = hp.get_utm_srid(screen_lines)
        sl = hp.make_lazy(screen_lines).pipe(gk.to_srid, srid=utm)

        # Ensure screen_line_id using pure Polars (no Python UDF)
        names = sl.collect_schema().names()
        if "screen_line_id" not in names:
            sl = sl.with_columns(
                screen_line_id=(
                    pl.lit("sl") + pl.int_range(0, pl.len()).cast(pl.Utf8).str.zfill(3)
                )
            )

        # Precompute screen-line direction vector
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
        sl_meta_cols = [c for c in sl.collect_schema().names() if c != "geometry"]
        sl_meta = sl.select(sl_meta_cols)

        # Shapes (already UTM via use_utm=True)
        shapes = feed.get_shapes(as_geo=True, use_utm=True).select("shape_id", "geometry")

        # Simple subshapes (your ported function)
        sub = gk.split_simple(shapes).select(
            "shape_id", "subshape_id", "subshape_length_m", "cum_length_m", "geometry"
        )

        # Intersections between subshapes and screen lines (no UDF)
        cand = (
            sub.join(sl, how="cross")
            .filter(st.geom().st.intersects(pl.col("geometry_right")))
            .rename({"geometry_right": "screen_geom"})
        )

        ints = (
            cand.with_columns(int_geom=st.geom().st.intersection(pl.col("screen_geom")))
            .with_columns(int_parts=pl.col("int_geom").st.parts())
            .explode("int_parts")
            .drop("int_geom")
            .rename({"int_parts": "int_point"})
            .with_columns(subshape_dist_m=st.geom().st.project(pl.col("int_point")))
            .with_columns(
                subshape_dist_frac=pl.col("subshape_dist_m") / pl.col("subshape_length_m"),
                crossing_dist_m=pl.col("subshape_dist_m")
                + pl.col("cum_length_m")
                - pl.col("subshape_length_m"),
            )
            .with_columns(n=st.geom().st.count_points())
            .with_columns(
                seg_p1=st.geom().st.get_point(0),
                seg_p2=st.geom().st.get_point(pl.col("n") - 1),
            )
            .with_columns(
                seg_dx=pl.col("seg_p2").st.x() - pl.col("seg_p1").st.x(),
                seg_dy=pl.col("seg_p2").st.y() - pl.col("seg_p1").st.y(),
            )
            .with_columns(
                det=pl.col("seg_dx") * pl.col("sl_dy") - pl.col("seg_dy") * pl.col("sl_dx")
            )
            .with_columns(
                crossing_direction=pl.when(pl.col("det") >= 0)
                .then(pl.lit(1))
                .otherwise(pl.lit(-1))
            )
            .drop("seg_p1", "seg_p2", "seg_dx", "seg_dy", "det", "screen_geom")
        )
        g = ints.select(
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
        )

        # Interpolate crossing_time from stop_times per date
        feed_m = feed.convert_dist("m")
        frames = []
        for date in dates:
            edges = (
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

            j = (
                g.join(edges, on="shape_id", how="inner")
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
            frames.append(j)

        f = pl.concat(frames) if frames else pl.LazyFrame({"date": pl.Series([], pl.Utf8)})

        # Append screen line meta + trip/route info
        f = (
            f.join(sl_meta, on="screen_line_id", how="left")
            .join(
                feed.trips.select("trip_id", "direction_id", "route_id"),
                on="trip_id",
                how="left",
            )
            .join(
                feed.routes.select("route_id", "route_short_name", "route_type"),
                on="route_id",
                how="left",
            )
        )

        # Final column order (include any screen line meta except geometry)
        sl_meta_cols_no_geom = [c for c in sl_meta_cols if c != "geometry"]
        final_cols = [
            "date",
            *sl_meta_cols_no_geom,
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

        # Keep it lazy; convert seconds → timestr in the plan
        return (
            f.select([c for c in final_cols if c in f.collect_schema().names()])
            .with_columns(crossing_time=hp.seconds_to_timestr("crossing_time"))
            .unique()
            .sort(["screen_line_id", "trip_id", "crossing_dist_m"])
        )
    return (compute_screen_line_counts,)


@app.cell
def _(DATA, compute_screen_line_counts, dates, feed, st):
    path = DATA / "cairns_screen_lines.geojson"
    screen_lines = st.read_file(path)  # .pipe(gk.make_lazy)
    screen_lines.with_columns(
        p1=st.geom().st.get_point(0),
        p2=st.geom().st.get_point(1),
    )

    compute_screen_line_counts(feed, screen_lines, dates).collect()
    return


@app.cell
def _(feed, gk, pl, st):
    import pytest

    shapes_g = (
        gk.get_shapes(feed, as_geo=True, use_utm=True)
        .with_columns(
            length=st.geom().st.length(),
            is_simple=st.geom().st.is_simple(),
        )
        .collect()
    )
    # We should have some non-simple shapes to start with
    assert not shapes_g["is_simple"].all()

    s = gk.split_simple(shapes_g).collect()
    assert set(s.columns) == {
        "shape_id",
        "subshape_id",
        "subshape_sequence",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    }

    # All sublinestrings of result should be simple
    assert s.with_columns(is_simple=st.geom().st.is_simple())["is_simple"].all()

    # Check each shape group
    for shape_id, group in s.partition_by("shape_id", as_dict=True).items():
        shape_id = shape_id[0]
        ss = shapes_g.filter(pl.col("shape_id") == shape_id)
        # Each subshape should be shorter than shape
        assert (group["subshape_length_m"] <= ss["length"].sum()).all()
        # Cumulative length should equal shape length within 1%
        L = ss["length"][0]
        assert group["cum_length_m"].max() == pytest.approx(L, rel=0.001)
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
