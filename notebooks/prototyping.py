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
    return List, gk, pb, pl, sg


@app.cell
def _(gk, pb):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    feed = gk.read_feed(
        pb.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km"
    )
    # feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km")
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _(feed):
    '1659-69c424c4' in feed.stops.collect()["stop_id"]
    return


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
