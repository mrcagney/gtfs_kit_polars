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
    return DATA, gk, gpd, pd, pl, sg, shapely, so, st


@app.cell
def _(DATA, gk):
    # akl_url = "https://gtfs.at.govt.nz/gtfs.zip"
    # feed = gk.read_feed(akl_url, dist_units="km")
    #feed = gk.read_feed(pl.Path.home() / "Desktop" / "auckland_gtfs_20250918.zip", dist_units="km")
    feed = gk.read_feed(DATA / "cairns_gtfs.zip", dist_units="km")
    return (feed,)


@app.cell
def _(feed):
    dates = feed.get_first_week()
    dates = [dates[0], dates[6]]
    dates
    return (dates,)


@app.cell
def _(
    gpd,
    hp,
    merge_collinear_chain,
    pd,
    pl,
    sg,
    shapely,
    so,
    split_sequential,
    st,
):
    # Tweak these
    def get_self_intersections(ls: sg.LineString) -> sg.MultiPoint:
        from collections import Counter 

        # Split line at nodes
        mls = shapely.union_all(ls)
        segment_coords = []
        for segment in mls.geoms:
            segment_coords.extend(segment.coords)
        return sg.MultiPoint([sg.Point(x) for x, count in Counter(segment_coords).items() if count > 1])

    def get_self_intersections_2(ls: sg.LineString) -> sg.MultiPoint:
        from collections import Counter 

        # Split line at nodes
        mls = shapely.node(ls)
        segment_coords = []
        for segment in mls.geoms:
            segment_coords.extend(segment.coords)
        return sg.MultiPoint([sg.Point(x) for x, count in Counter(segment_coords).items() if count > 2])

    def split_simple_0(ls: sg.LineString) -> list[sg.LineString]:
        si = get_self_intersections_2(ls)

        # Split the (snapped) line at its self-intersection points
        ls_snapped = so.snap(ls, si, tolerance=1.0e-12) 
        parts = so.split(ls_snapped, si)

        # Order parts by their position along the original polyline
        def path_pos(g: sg.LineString) -> float:
            # Pick a point guaranteed on the part (midpoint along its length)
            mid_on_part = g.interpolate(0.5, normalized=True)
            return float(ls.project(mid_on_part))

        ordered = sorted((geom for geom in parts.geoms if geom.length > 0), key=path_pos)
        return ordered or [ls]
    
    def split_simple(shapes_g: st.GeoLazyFrame | st.GeoDataFrame) -> st.GeoLazyFrame:
        """
        Given GTFS shapes as a GeoDataFrame of the form output by :func:`geometrize_shapes`
        and possibly in a non-WGS84 CRS,
        split each non-simple LineString into maximal simple (non-self-intersecting)
        sub-LineStrings, and leave the simple LineStrings as is.

        Return a GeoDataFrame in the CRS of ``shapes_g`` with the columns

        - ``'shape_id'``: a unique identifier of the original LineString L
        - ``'subshape_id'``: a unique identifier of a simple sub-LineString S of L
        - ``'subshape_sequence'``: integer; indicates the order of S when joining up
          all simple sub-LineStrings to form L
        - ``'subshape_length_m'``: the length of S in meters
        - ``'cum_length_m'``: the length S plus the lengths of sub-LineStrings of L
          that come before S; in meters
        - ``'geometry'``: LineString geometry corresponding to S

        Within each 'shape_id' group, the subshapes will be sorted increasingly by
        'subshape_sequence'.
        """
        # Use GeoPandas for the time being
        srid = hp.get_srid(shapes_g)
        shapes_g = shapes_g.collect().st.to_geopandas() if isinstance(shapes_g, pl.LazyFrame) else shapes_g.st.to_geopandas()

        # Convert to UTM for meter calculations
        utm_crs = shapes_g.estimate_utm_crs()
        g = shapes_g.assign(is_simple=lambda x: x.is_simple).to_crs(utm_crs)

        final_cols = [
            "shape_id",
            "subshape_id",
            "subshape_sequence",
            "subshape_length_m",
            "cum_length_m",
            "geometry",
        ]

        # Simple shapes don't need splitting
        g0 = (
            g.loc[lambda x: x["is_simple"]]
            .assign(
                subshape_id=lambda x: x["shape_id"].astype(str) + "-0",
                subshape_sequence=0,
                subshape_length_m=lambda x: x.length,
                cum_length_m=lambda x: x["subshape_length_m"],
            )
            .filter(final_cols)
        )

        # Handle non-simple shapes by splitting at all internal nodes (sequential),
        # ordering, and maximally merging simple segments
        g1 = (
            g
            .loc[lambda df: ~df["is_simple"], ["shape_id", "geometry"]]
            .assign(
                geometry0=lambda df: df.geometry,
                parts=lambda df: df.geometry.apply(split_sequential),
            )
            .explode("parts", ignore_index=False)
            .assign(geometry=lambda x: x["parts"])
            .set_geometry("geometry")
            .assign(len_m=lambda df: df.geometry.length)
            .query("len_m > 0")
            .assign(
                start0=lambda df: df.geometry.apply(lambda ls: sg.Point(ls.coords[0])),
                end0=lambda df: df.geometry.apply(lambda ls: sg.Point(ls.coords[-1])),
            )
            .assign(
                 t0=lambda df: df.apply(lambda r: r["geometry0"].project(r["start0"]), axis=1),
                 t1=lambda df: df.apply(lambda r: r["geometry0"].project(r["end0"]), axis=1),
            )
            .sort_values(["shape_id", "t0", "t1", "len_m"])
            .drop(columns=["start0", "end0", "t0", "t1", "len_m"])
            .groupby("shape_id", group_keys=True, sort=False)
            .apply(lambda df: gpd.GeoDataFrame(
                {
                    "shape_id": df["shape_id"].iloc[0],
                    "geometry": merge_collinear_chain(list(df["geometry"])),
                },
                geometry="geometry",
                crs=utm_crs,
            ))
            .reset_index(drop=True)
            .assign(
                subshape_sequence=lambda df: df.groupby("shape_id").cumcount().astype("int32"),
                subshape_length_m=lambda df: df.geometry.length,
                subshape_id=lambda df: df["shape_id"].astype(str) + "-" + df["subshape_sequence"].astype(str),
                cum_length_m=lambda df: df.groupby("shape_id")["subshape_length_m"].cumsum(),
            )
            .reset_index(drop=True)
            .filter(final_cols)
        )
        result = (
            pd.concat([g0, g1])
            .sort_values(["shape_id", "subshape_sequence"], ignore_index=True)
            .to_crs(shapes_g.crs)
        )
        return result
        # Convert back to Polars ST
        return st.from_geopandas(result).with_columns(pl.col("geometry").st.set_srid(srid).alias("geometry")).lazy().pipe(hp.to_srid, srid)


    return (split_simple_0,)


@app.cell
def _(feed, split_simple_0):
    g = feed.get_shapes(as_geo=True, use_utm=True).head(3).collect().st.to_geopandas()
    ls = g["geometry"].to_list()[-1]
    if ls.is_simple:
        print("Nothing to do")
    else:
        segments = split_simple_0(ls)
        cum_length = 0
        for s in  segments:
            d = s.length
            cum_length += d 
            print(s.is_simple, d)
        
    print(ls.length, cum_length)
    g.explore()
    print(list(ls.coords))
    return


@app.cell
def _(feed, gk):
    gk.split_simple(feed.get_shapes(as_geo=True, use_utm=False).head(5)).assign(is_simple=lambda x: x.is_simple) #.collect().st.to_wkt()
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
