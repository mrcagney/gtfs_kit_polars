import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _(Path):
    import pathlib as pb
    import json

    import polars as pl 
    import polars_st as st
    import marimo as mo
    import matplotlib
    import folium as fl

    import gtfs_kit_polars as gk


    HERE = pb.Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    PROJECT_ROOT = HERE.parent  # notebooks/ -> project/
    DATA = (PROJECT_ROOT / "data").resolve()
    return DATA, fl, gk, json, mo, pl, st


@app.cell
def _(mo):
    mo.md(r"""
    # Notes

    - All outputs are LazyFrames, so we need to collect them to display them here
    """)
    return


@app.cell
def _(DATA, gk):
    # List feed

    gk.list_feed(DATA / "cairns_gtfs.zip").collect()
    return


@app.cell
def _(DATA, gk):
    # Read feed and describe

    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")
    feed.describe().collect()
    return (feed,)


@app.cell
def _(feed, mo):
    mo.output.append(feed.stop_times.head().collect())
    feed_1 = feed.append_dist_to_stop_times()
    mo.output.append(feed_1.stop_times.head().collect())
    return (feed_1,)


@app.cell
def _(feed_1):
    week = feed_1.get_first_week()
    dates = [week[0], week[6]]
    dates
    return (dates,)


@app.cell
def _(feed_1):
    # Trip stats; reuse these for later speed ups

    trip_stats = feed_1.compute_trip_stats().collect()
    trip_stats
    return


@app.cell
def _(dates, feed_1):
    # Pass in trip stats to avoid recomputing them

    network_stats = feed_1.compute_network_stats(dates).collect()
    network_stats
    return


@app.cell
def _(dates, feed_1):
    nts = feed_1.compute_network_time_series(dates, num_minutes=6*60).collect()
    nts
    return (nts,)


@app.cell
def _(gk, nts):
    gk.downsample(nts, num_minutes=12*60).collect()
    return


@app.cell
def _(dates, feed, feed_1):
    # Stop time series
    stop_ids = feed.stops.head(1).collect()["stop_id"].to_list()
    sts = feed_1.compute_stop_time_series(dates, stop_ids=stop_ids, num_minutes=12*60).collect()
    sts
    return (sts,)


@app.cell
def _(gk, sts):
    gk.downsample(sts, num_minutes=24*60).collect()
    return


@app.cell
def _(dates, feed_1):
    # Route time series

    rts = feed_1.compute_route_time_series(dates, num_minutes=12*60).collect()
    rts
    return


@app.cell
def _(dates, feed_1):
    # Route timetable

    route_id = feed_1.routes.head(1).collect()["route_id"].item(0)
    feed_1.build_route_timetable(route_id, dates).collect()
    return


@app.cell
def _(dates, feed_1):
    # Locate trips

    loc = feed_1.locate_trips(dates[0], times=["08:00:00", "09:00:00"]).collect()
    loc
    return


@app.cell
def _(feed_1):
    # Map routes

    rsns = feed_1.routes.head().collect()["route_short_name"].to_list()[2:4]
    feed_1.map_routes(route_short_names=rsns, show_stops=True)
    return


@app.cell
def _(feed):
    # Alternatively map routes without stops using GeoPandas's explore

    (
        feed.get_routes(as_geo=True).collect().st.to_geopandas().explore(
            column="route_short_name",
            style_kwds=dict(weight=3),
            highlight_kwds=dict(weight=8),
            tiles="CartoDB positron",
        )
    )
    return


@app.cell
def _(DATA, feed_1, fl, json, st):
    # Show screen line

    trip_id = "CNS2014-CNS_MUL-Weekday-00-4166247"
    m = feed_1.map_trips([trip_id], show_stops=True, show_direction=True)
    screen_line = st.read_file(DATA / "cairns_screen_line.geojson")
    screen_line_gj = json.loads(screen_line.st.to_geojson().row(0)[0])
    keys_to_remove = [
        key
        for key in m._children.keys()
        if key.startswith("layer_control_") or key.startswith("fit_bounds_")
    ]
    for key in keys_to_remove:
        m._children.pop(key)
    fg = fl.FeatureGroup(name="Screen lines")
    fl.GeoJson(
        screen_line_gj, style_function=lambda feature: {"color": "red", "weight": 2}
    ).add_to(fg)
    fg.add_to(m)
    fl.LayerControl().add_to(m)
    m.fit_bounds(fg.get_bounds())
    m
    return screen_line, trip_id


@app.cell
def _(dates, feed_1, pl, screen_line, trip_id):
    # Screen line counts

    slc = feed_1.compute_screen_line_counts(screen_line, dates=dates).collect()
    slc.filter(pl.col("trip_id") == trip_id)
    return


if __name__ == "__main__":
    app.run()
