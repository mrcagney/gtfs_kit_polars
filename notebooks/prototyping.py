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
    import shapely
    import shapely.geometry as sg
    import shapely.ops as so
    import folium as fl
    import plotly.express as px

    import gtfs_kit_polars as gk

    warnings.filterwarnings("ignore")

    DATA = pb.Path("data")
    return DATA, List, gk, np, pl, sg


@app.cell
def _(DATA, gk):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    # feed = gk.read_feed(pb.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km")
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km")
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _(gk, np, pl):
    def append_dist_to_stop_times(feed: "Feed") -> "Feed":
        """
        Calculate and append the optional ``shape_dist_traveled`` column in
        ``feed.stop_times`` in terms of the distance units ``feed.dist_units``.
        Trips without shapes will have NaN distances.
        Return the resulting Feed.
        Uses ``feed.shapes``, so if that is missing, then return the original feed.

        This does not always give accurate results.
        The algorithm works as follows.
        Compute the ``shape_dist_traveled`` field by using Shapely to
        measure the distance of a stop along its trip LineString.
        If for a given trip this process produces a non-monotonically
        increasing, hence incorrect, list of (cumulative) distances, then
        fall back to estimating the distances as follows.

        Set the first distance to 0, the last to the length of the trip shape,
        and leave the remaining ones computed above.
        Choose the longest increasing subsequence of that new set of
        distances and use them and their corresponding departure times to linearly
        interpolate the rest of the distances.
        """
        hp = gk
        if getattr(feed, "shapes", None) is None or hp.is_empty(feed.shapes):
            return feed

        # Prepare data, building geometry tables in UTM (meters)
        shapes_geo = feed.get_shapes(as_geo=True, use_utm=True).select(
            "shape_id", "geometry"
        )
        stops_geo = feed.get_stops(as_geo=True, use_utm=True).select("stop_id", "geometry")
        convert = hp.get_convert_dist("m", feed.dist_units)  # returns a Polars expr fn
        final_cols = [c for c in feed.stop_times.collect_schema().names() if c != "shape_dist_traveled"] + [
            "shape_dist_traveled"
        ]

        stop_times = (
            feed.stop_times.join(
                feed.trips.select("trip_id", "shape_id"), on="trip_id", how="left"
            )
            .join(shapes_geo.rename({"geometry": "shape_geom"}), on="shape_id", how="left")
            .join(stops_geo.rename({"geometry": "stop_geom"}), on="stop_id", how="left")
            .sort("trip_id", "stop_sequence")
            .with_columns(
                dtime=hp.timestr_to_seconds("departure_time"),
                # distances along the linestring (in meters), and line length
                dist=pl.when(
                    pl.col("shape_geom").is_not_null() & pl.col("stop_geom").is_not_null()
                )
                .then(pl.col("shape_geom").st.project(pl.col("stop_geom")))
                .otherwise(None)
                .cast(pl.Float64),
                shape_length=pl.col("shape_geom").st.length().cast(pl.Float64),
            )
            # Fix reversed direction trips
            .with_columns(
                first_dist=pl.col("dist").first().over("trip_id"),
                last_dist=pl.col("dist").last().over("trip_id"),
            )
            .with_columns(
                need_flip=(pl.col("last_dist") < pl.col("first_dist"))
                & pl.col("shape_length").is_not_null(),
            )
            .with_columns(
                dist=pl.when(pl.col("need_flip"))
                .then(pl.col("shape_length") - pl.col("dist"))
                .otherwise(pl.col("dist"))
            )
            .drop("shape_geom", "stop_geom", "first_dist", "last_dist", "need_flip")
        )

        # Separate trips that have bad distances and fix them
        bad_trips = (
            stop_times.with_columns(
                step=(pl.col("dist") - pl.col("dist").shift(1)).over("trip_id"),
                overrun=(pl.col("dist") > (pl.col("shape_length") + pl.lit(100.0))),
            )
            .with_columns(
                has_bad_step=(pl.col("step") < 0) & pl.col("dist").is_not_null(),
                has_overrun=pl.col("overrun").any().over("trip_id"),
            )
            .filter(pl.col("has_bad_step") | pl.col("has_overrun"))
            .select("trip_id")
            .unique()
        )
        trip_id = "CNS2014-CNS_MUL-Weekday-00-4180831"
        print(trip_id in bad_trips.collect()["trip_id"].to_list())

        def compute_dist(g: pl.DataFrame) -> pl.DataFrame:
            D = g["shape_length"][0]
            dists0 = g["dist"].to_list()

            # No geometry length or single stop → trivial
            if np.isnan(D) or len(dists0) <= 1:
                return g

            times = g["dtime"].to_numpy()
            dists = np.array([0.0] + dists0[1:-1] + [float(D)], dtype=float)
            ix = hp.longest_subsequence(dists, index=True)
            good_dists = np.take(dists, ix)
            good_times = np.take(times, ix)
            new_dists = np.interp(times, good_times, good_dists).astype(float)

            if g["trip_id"][0] == trip_id:
                print(times) 
                print(good_times) 
                print(good_dists)
            return g.with_columns(dist=pl.Series(new_dists))

        fixed = (
            stop_times.join(bad_trips, on="trip_id", how="inner")
            .sort("trip_id", "stop_sequence")
            .group_by("trip_id")
            .map_groups(compute_dist, schema=stop_times.collect_schema())
        )

        good = stop_times.join(bad_trips, on="trip_id", how="anti")

        # Assemble stop times
        stop_times = (
            pl.concat([good, fixed])
            .with_columns(
                shape_dist_traveled=pl.when(pl.col("dist").is_not_null())
                .then(convert(pl.col("dist")))
                .otherwise(None)
            )
            .sort("trip_id", "stop_sequence")
            .select(final_cols)
        )

        new_feed = feed.copy()
        new_feed.stop_times = stop_times
        return new_feed

    return (append_dist_to_stop_times,)


@app.cell
def _(append_dist_to_stop_times, feed, pl):
    s = append_dist_to_stop_times(feed).stop_times.collect()
    trip_id = "CNS2014-CNS_MUL-Weekday-00-4180831"
    s.filter(pl.col("trip_id") == trip_id)
    return


@app.cell
def _(List, Tuple, sg):
    def split_simple_0(line: sg.LineString) -> list[sg.LineString]:
        """
        Greedily build maximal simple LineString components.

        Strategy:
          - Try appending each next vertex.
          - If the tentative component stays .is_simple -> keep growing.
          - If it stops being simple:
              * If the new vertex is a previously visited vertex in the current component,
                emit the loop slice as its own component and restart from that vertex.
              * Otherwise, flush the whole current component and restart from the last edge
                (prev, next), so we don't drop the bridging segment.

        Returns maximal simple components (no self-intersections).
        """
        coords = list(map(tuple, getattr(line, "coords", [])))  # type: List[Tuple[float, float]]
        if len(coords) < 2:
            return []

        out: List[sg.LineString] = []

        current_pts: List[Tuple[float, float]] = [coords[0]]
        first_idx: dict[Tuple[float, float], int] = {coords[0]: 0}

        for i in range(1, len(coords)):
            p_prev = current_pts[-1]
            p_next = coords[i]

            # Skip consecutive duplicates
            if p_next == p_prev:
                continue

            # Try to extend greedily
            tentative = current_pts + [p_next]
            if sg.LineString(tentative).is_simple:
                current_pts.append(p_next)
                # record first visit to p_next if new
                if p_next not in first_idx:
                    first_idx[p_next] = len(current_pts) - 1
                continue

            # Not simple if we add p_next. Handle two cases:

            # 1) Loop closing at a previously seen vertex: emit the loop slice
            if p_next in first_idx:
                j = first_idx[p_next]  # first occurrence of p_next in current
                loop_pts = current_pts[j:] + [p_next]
                if len(loop_pts) >= 2:
                    out.append(sg.LineString(loop_pts))

                # Restart a fresh component from that vertex for the tail
                current_pts = [p_next]
                first_idx = {p_next: 0}
                continue

            # 2) General non-adjacent self-intersection:
            #    Flush current component, then start a new one from the bridging edge
            if len(current_pts) >= 2:
                out.append(sg.LineString(current_pts))
            current_pts = [p_prev, p_next]
            first_idx = {p_prev: 0, p_next: 1}

        # Flush final component
        if len(current_pts) >= 2:
            out.append(sg.LineString(current_pts))

        return out


    def test_split_simple_0():
        # ---- Test 1: straight line, no repeats -> single component
        line1 = sg.LineString([(0, 0), (1, 0), (2, 0)])
        expected1 = [
            [(0, 0), (1, 0), (2, 0)],
        ]

        # ---- Test 2: loop then tail (includes a consecutive duplicate that should be ignored)
        line2 = sg.LineString([(0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 1)])
        expected2 = [
            [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],  # closed loop
            [(0, 0), (0, 1)],  # tail
        ]

        # ---- Test 3: doubles back on an interior vertex
        line3 = sg.LineString([(0, 0), (2, 0), (2, 1), (1, 1), (1, 0), (2, 0), (3, 0)])
        expected3 = [
            [(0, 0), (2, 0), (2, 1), (1, 1), (1, 0), (2, 0)],
            [(2, 0), (3, 0)],
        ]

        for name, line, expected in [
            ("straight", line1, expected1),
            ("loop_then_tail", line2, expected2),
            ("double_back", line3, expected3),
        ]:
            parts = split_simple_0(line)

            assert isinstance(parts, list), f"{name}: must return list"
            assert len(parts) == len(expected), (
                f"{name}: expected {len(expected)} components, got {len(parts)}"
            )

            for i, (part, exp_coords) in enumerate(zip(parts, expected), start=1):
                assert isinstance(part, sg.LineString), f"{name} comp {i}: not a LineString"
                got = list(map(tuple, part.coords))
                assert got == exp_coords, (
                    f"{name} comp {i}: coords mismatch.\nGot: {got}\nExp: {exp_coords}"
                )
                assert part.is_simple, f"{name} comp {i}: returned LineString not simple"


    test_split_simple_0()
    return (split_simple_0,)


@app.cell
def _():
    return


@app.cell
def _(feed, split_simple_0):
    g = feed.get_shapes(as_geo=True, use_utm=True).head(2).collect().st.to_geopandas()
    ls = g["geometry"].to_list()[-1]
    if ls.is_simple:
        print("Nothing to do")
    else:
        print(list(ls.coords))
        segments = split_simple_0(ls)
        cum_length = 0
        for s in segments:
            d = s.length
            cum_length += d
            # print(s.is_simple, d)

    # print(ls.length, cum_length)
    # g.explore()
    # print(list(ls.coords))
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
    # routes = feed.get_routes(as_gdf=True)
    # print(routes)
    # feed = feed.aggregate_routes()
    # feed.map_routes(feed.routes["route_id"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
