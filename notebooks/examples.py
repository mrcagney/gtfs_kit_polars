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
    return (dates,)


@app.cell
def _(feed):
    # Trip stats; reuse these for later speed ups

    trip_stats = feed.compute_trip_stats(compute_dist_from_shapes=False)
    trip_stats.collect()
    return


@app.cell
def _(DATA, gk, pl):
    ts = pl.read_csv(
        DATA / "cairns_trip_stats_2.csv",
        schema_overrides=(gk.DTYPES["trips"] | gk.DTYPES["routes"]),
    )
    #ts.write_csv(DATA / "cairns_trip_stats_2.csv")
    gk.compute_route_stats_0(ts).collect()
    return


@app.cell
def _(dates, feed):
    feed.compute_route_time_series(dates, num_minutes=60*4, split_directions=True).collect()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
