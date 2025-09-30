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
    return DATA, fl, gk, gp, mo, np, pd, pl


@app.cell
def _(DATA, gk):
    # List feed

    gk.list_feed(DATA / "cairns_gtfs.zip")
    return


@app.cell
def _(DATA, gk):
    # Read feed and describe

    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="m")
    #feed.describe()
    return (feed,)


@app.cell
def _(feed, mo):
    mo.output.append(feed.stop_times)
    feed_1 = feed.append_dist_to_stop_times()
    mo.output.append(feed_1.stop_times)
    return (feed_1,)


@app.cell
def _(feed_1):
    week = feed_1.get_first_week()
    dates = [week[0], week[6]]
    dates
    return (dates,)


@app.cell
def _(gk, np, pl):
    def compute_trip_stats(
        feed: "Feed",
        route_ids: list[str | None] = None,
        *,
        compute_dist_from_shapes: bool = False,
    ):
        """
        Return a DataFrame with the following columns:

        - ``'trip_id'``
        - ``'route_id'``
        - ``'route_short_name'``
        - ``'route_type'``
        - ``'direction_id'``: null if missing from feed
        - ``'shape_id'``: null if missing from feed
        - ``'stop_pattern_name'``: output from :func:`name_stop_patterns`
        - ``'num_stops'``: number of stops on trip
        - ``'start_time'``: first departure time of the trip
        - ``'end_time'``: last departure time of the trip
        - ``'start_stop_id'``: stop ID of the first stop of the trip
        - ``'end_stop_id'``: stop ID of the last stop of the trip
        - ``'is_loop'``: 1 if the start and end stop are less than 400m apart and
          0 otherwise
        - ``'distance'``: distance of the trip;
          measured in kilometers if ``feed.dist_units`` is metric;
          otherwise measured in miles;
          contains all null entries if ``feed.shapes is None``
        - ``'duration'``: duration of the trip in hours
        - ``'speed'``: distance/duration

        If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at
        least one non-null value and ``compute_dist_from_shapes == False``,
        then use that column to compute the distance column.
        Else if ``feed.shapes is not None``, then compute the distance
        column using the shapes and Shapely.
        Otherwise, set the distances to null.

        If route IDs are given, then restrict to trips on those routes.

        Notes
        -----
        - Assume the following feed attributes are not ``None``:

            * ``feed.trips``
            * ``feed.routes``
            * ``feed.stop_times``
            * ``feed.shapes`` (optionally)

        - Calculating trip distances with ``compute_dist_from_shapes=True``
          seems pretty accurate.  For example, calculating trip distances on
          `this Portland feed
          <https://transitfeeds.com/p/trimet/43/1400947517>`_
          using ``compute_dist_from_shapes=False`` and
          ``compute_dist_from_shapes=True``,
          yields a difference of at most 0.83km from the original values.

        """
        hp = gk
    
        # Trips with stop pattern names
        t = gk.name_stop_patterns(feed)
        if route_ids is not None:
            t = t.filter(pl.col("route_id").is_in(route_ids))

        # Ensure columns exist (nulls if missing)
        if "direction_id" not in t.columns:
            t = t.with_columns(pl.lit(None).alias("direction_id"))
        if "shape_id" not in t.columns:
            t = t.with_columns(pl.lit(None).alias("shape_id"))

        # Attach route fields
        r = feed.routes.select("route_id", "route_short_name", "route_type")
        t = (
            t.select("route_id", "trip_id", "direction_id", "shape_id", "stop_pattern_name")
             .join(r, on="route_id", how="left")
        )

        # Stop_times with departure_time (seconds)
        cols_extra = ["shape_dist_traveled"] if "shape_dist_traveled" in feed.stop_times.columns else []
        u = (
            feed.stop_times
            .sort(["trip_id", "stop_sequence"])
            .with_columns(
                dtime=pl.col("departure_time").map_elements(hp.timestr_to_seconds, return_dtype=pl.Int64)
            )
            .select("trip_id", "stop_id", "stop_sequence", "dtime", *cols_extra)
        )

        f = t.join(u, on="trip_id", how="inner").sort(["trip_id", "stop_sequence"])

        # Per-trip stats (except distance)
        g = (
            f.group_by("trip_id", maintain_order=True)
             .agg(
                 route_id=pl.first("route_id"),
                 route_short_name=pl.first("route_short_name"),
                 route_type=pl.first("route_type"),
                 direction_id=pl.first("direction_id"),
                 shape_id=pl.first("shape_id"),
                 stop_pattern_name=pl.first("stop_pattern_name"),
                 num_stops=pl.len(),
                 start_time=pl.first("dtime"),
                 end_time=pl.last("dtime"),
                 start_stop_id=pl.first("stop_id"),
                 end_stop_id=pl.last("stop_id"),
             )
             .with_columns(duration=(pl.col("end_time") - pl.col("start_time")) / 3600.0)
        )

        # is_loop via UTM Shapely dicts (distance < 400 m)
        geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)

        def is_loop(a, b):
            try:
                return int(geometry_by_stop[a].distance(geometry_by_stop[b]) < 400.0)
            except Exception:
                return 0

        g = g.with_columns(
            is_loop=pl.struct(["start_stop_id", "end_stop_id"]).map_elements(
                lambda s: is_loop(s["start_stop_id"], s["end_stop_id"]),
                return_dtype=pl.Int8,
            )
        )

        # Distance
        use_sdt = (
            ("shape_dist_traveled" in f.columns)
            and (not compute_dist_from_shapes)
            and (f.select(pl.col("shape_dist_traveled").is_not_null().any()).collect().item())
        )

        if use_sdt:
            # Per original: use max(shape_dist_traveled) then convert to km/mi
            if hp.is_metric(feed.dist_units):
                conv = hp.get_convert_dist(feed.dist_units, "km")
            else:
                conv = hp.get_convert_dist(feed.dist_units, "mi")

            d = (
                f.group_by("trip_id")
                 .agg(mx=pl.col("shape_dist_traveled").max())
                 .with_columns(distance=pl.col("mx").map_elements(conv, return_dtype=pl.Float64))
                 .select("trip_id", "distance")
            )
            g = g.join(d, on="trip_id", how="left")

        elif feed.shapes is not None:
            # Use shapes via Shapely dict, in UTM, matching original logic
            geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)

            if hp.is_metric(feed.dist_units):
                to_units = hp.get_convert_dist("m", "km")
            else:
                to_units = hp.get_convert_dist("m", "mi")

            def _dist(shape_id, start_stop_id, end_stop_id):
                try:
                    line = geometry_by_shape[shape_id]
                except Exception:
                    return np.nan

                D = line.length
                if not getattr(line, "is_simple", True):
                    return to_units(D)

                try:
                    p1 = geometry_by_stop[start_stop_id]
                    p2 = geometry_by_stop[end_stop_id]
                except Exception:
                    return to_units(D)

                d1 = line.project(p1)
                d2 = line.project(p2)
                d = d2 - d1
                if 0 < d < D + 100:
                    return to_units(d)
                else:
                    return to_units(D)

            g = g.with_columns(
                distance=pl.struct(["shape_id", "start_stop_id", "end_stop_id"]).map_elements(
                    lambda s: _dist(s["shape_id"], s["start_stop_id"], s["end_stop_id"]),
                    return_dtype=pl.Float64,
                )
            )

        else:
            g = g.with_columns(distance=pl.lit(None, dtype=pl.Float64))

        # Final: speed and human-readable times
        g = g.with_columns(
            speed=(pl.col("distance") / pl.col("duration")),
            start_time=pl.col("start_time").map_elements(hp.seconds_to_timestr, return_dtype=pl.Utf8),
            end_time=pl.col("end_time").map_elements(hp.seconds_to_timestr, return_dtype=pl.Utf8),
        )

        return g.select(
            "trip_id",
            "route_id",
            "route_short_name",
            "route_type",
            "direction_id",
            "shape_id",
            "stop_pattern_name",
            "num_stops",
            "start_time",
            "end_time",
            "start_stop_id",
            "end_stop_id",
            "is_loop",
            "distance",
            "duration",
            "speed",
        ).sort(["route_id", "direction_id", "start_time"])
    return (compute_trip_stats,)


@app.cell
def _(compute_trip_stats, feed):
    # Trip stats; reuse these for later speed ups

    trip_stats = compute_trip_stats(feed)
    trip_stats
    return (trip_stats,)


@app.cell
def _(dates, feed_1, trip_stats):
    # Pass in trip stats to avoid recomputing them

    network_stats = feed_1.compute_network_stats(dates, trip_stats=trip_stats)
    network_stats
    return


@app.cell
def _(dates, feed_1, trip_stats):
    nts = feed_1.compute_network_time_series(dates, trip_stats=trip_stats, freq="6h")
    nts
    return (nts,)


@app.cell
def _(gk, nts):
    gk.downsample(nts, freq="12h")
    return


@app.cell
def _(dates, feed, feed_1):
    # Stop time series
    stop_ids = feed.stops.loc[:1, "stop_id"]
    sts = feed_1.compute_stop_time_series(dates, stop_ids=stop_ids, freq="12h")
    sts
    return (sts,)


@app.cell
def _(gk, sts):
    gk.downsample(sts, freq="d")
    return


@app.cell
def _(dates, feed_1, trip_stats):
    # Route time series

    rts = feed_1.compute_route_time_series(dates, trip_stats=trip_stats, freq="12h")
    rts
    return


@app.cell
def _(dates, feed_1):
    # Route timetable

    route_id = feed_1.routes["route_id"].iat[0]
    feed_1.build_route_timetable(route_id, dates)
    return


@app.cell
def _(dates, feed_1, pd):
    # Locate trips

    rng = pd.date_range("1/1/2000", periods=24, freq="h")
    times = [t.strftime("%H:%M:%S") for t in rng]
    loc = feed_1.locate_trips(dates[0], times)
    loc.head()
    return


@app.cell
def _(feed_1):
    # Map routes

    rsns = feed_1.routes["route_short_name"].iloc[2:4]
    feed_1.map_routes(route_short_names=rsns, show_stops=True)
    return


@app.cell
def _(feed):
    # Alternatively map routes without stops using GeoPandas's explore

    (
        feed.get_routes(as_geo=True).explore(
            column="route_short_name",
            style_kwds=dict(weight=3),
            highlight_kwds=dict(weight=8),
            tiles="CartoDB positron",
        )
    )
    return


@app.cell
def _(DATA, feed_1, fl, gp):
    # Show screen line

    trip_id = "CNS2014-CNS_MUL-Weekday-00-4166247"
    m = feed_1.map_trips([trip_id], show_stops=True, show_direction=True)
    screen_line = gp.read_file(DATA / "cairns_screen_line.geojson")
    keys_to_remove = [
        key
        for key in m._children.keys()
        if key.startswith("layer_control_") or key.startswith("fit_bounds_")
    ]
    for key in keys_to_remove:
        m._children.pop(key)
    fg = fl.FeatureGroup(name="Screen lines")
    fl.GeoJson(
        screen_line, style_function=lambda feature: {"color": "red", "weight": 2}
    ).add_to(fg)
    fg.add_to(m)
    fl.LayerControl().add_to(m)
    m.fit_bounds(fg.get_bounds())
    m
    return screen_line, trip_id


@app.cell
def _(dates, feed_1, screen_line, trip_id):
    # Screen line counts 

    slc = feed_1.compute_screen_line_counts(screen_line, dates=dates)
    slc.loc[lambda x: x["trip_id"] == trip_id]
    return


if __name__ == "__main__":
    app.run()
