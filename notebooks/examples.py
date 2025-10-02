import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import pathlib as pb
    import json

    import marimo as mo
    import polars as pl
    import polars_st as st
    import pandas as pd
    import numpy as np
    import geopandas as gp
    import matplotlib
    import folium as fl

    import gtfs_kit_polars as gk

    DATA = pb.Path("data")
    DESK = pb.Path.home() / "Desktop"
    return DATA, gk, pl, st


@app.cell
def _(DATA, gk):
    # List feed

    gk.list_feed(DATA / "cairns_gtfs.zip")
    return


@app.cell
def _(DATA, gk):
    # Read feed and describe

    # feed = gk.read_feed(DESK / "auckland_gtfs_20250918.zip", dist_units="km")
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m").append_dist_to_shapes()
    feed.shapes.collect()
    return (feed,)


@app.cell
def _(st):
    st.point([[20, 40]])
    return


@app.cell
def _(feed, pl, st):
    agg = (
        feed.shapes.head(1)
        .sort("shape_id", "shape_pt_sequence")
        .group_by("shape_id", maintain_order=True)
        .agg(
            coords=pl.concat_list("shape_pt_lon", "shape_pt_lat"),
            n=pl.len(),
        )
    )
    lines = (
        agg
        .filter(pl.col("n") >= 2)
        .with_columns(geometry = st.linestring("coords").st.set_srid(4326))
        .select("shape_id", "geometry")
    )
    points = (
        agg
        .filter(pl.col("n") == 1)
        .with_columns(
            geometry = st.point(pl.col("coords").list.get(0)).st.set_srid(4326)
        )
        .select("shape_id", "geometry")
    )
    shapes = pl.concat([lines, points])
    shapes.collect()
    return


@app.cell
def _(feed):
    week = feed.get_first_week()
    dates = [week[0], week[6]]
    dates
    return (dates,)


@app.cell
def _(dates, feed, pl):
    time = "23:00:00"
    t = (
        feed.get_trips(dates[0])
        .join(
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
        .drop("departure_time")
        .unique("trip_id")
    )
    t.collect()
    return


@app.cell
def _(feed):
    # Trip stats; reuse these for later speed ups

    trip_stats = feed.compute_trip_stats(compute_dist_from_shapes=False)
    trip_stats.collect()
    return


@app.cell
def _(dates, feed):
    feed.compute_route_time_series(
        dates, num_minutes=60 * 4, split_directions=True
    ).collect()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
