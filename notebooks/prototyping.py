import marimo

__generated_with = "0.16.5"
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
    return gk, pb, pl, st


@app.cell
def _(gk, pb):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    feed = gk.read_feed(
        pb.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km"
    )
    #feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km")
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _():
    # ts = feed.compute_route_time_series(dates, num_minutes=60).collect()
    return


@app.cell
def _(feed, gk, pl, st):
    g = (
        gk.split_simple_alt(feed.get_shapes(as_geo=True))
        .collect()
        .with_columns(is_simple=st.geom().st.is_simple())
    )
    g.filter(~pl.col("is_simple"))
    return


@app.cell
def _(feed, gk, pl, st):
    import pytest

    shapes_g = gk.get_shapes(feed, as_geo=True, use_utm=True).with_columns(
            length=st.geom().st.length(),
            is_simple=st.geom().st.is_simple(),
        ).collect()
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
