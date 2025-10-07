"""
Functions about shapes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterable

import geopandas as gpd
import pandas as pd
import polars as pl
import polars_st as st
import shapely
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help mypy but avoid circular imports
if TYPE_CHECKING:
    from .feed import Feed


def append_dist_to_shapes(feed: "Feed") -> "Feed":
    """
    Calculate and append the optional ``shape_dist_traveled`` field in
    ``feed.shapes`` in terms of the distance units ``feed.dist_units``.
    Return the resulting Feed.

    As a benchmark, using this function on `this Portland feed
    <https://transitfeeds.com/p/trimet/43/1400947517>`_
    produces a ``shape_dist_traveled`` column that differs by at most
    0.016 km in absolute value from of the original values.
    """
    if feed.shapes is None:
        raise ValueError("This function requires the feed to have a shapes.txt file")

    feed = feed.copy()

    lon, lat = feed.shapes.limit(1).select("shape_pt_lon", "shape_pt_lat").collect().row(0)
    utm_srid = hp.get_utm_srid_0(lon, lat)
    convert_dist = hp.get_convert_dist("m", feed.dist_units)
    feed.shapes = (
        # Build point geometries in WGS84 then convert to UTM
        feed.shapes
        .sort("shape_id", "shape_pt_sequence")
        .with_columns(
            geometry=st.point(pl.concat_arr("shape_pt_lon", "shape_pt_lat"))
            .st.set_srid(cs.WGS84)
            .st.to_srid(utm_srid)
        )
        # Get successive point distances in meters
        .with_columns(
            prev=pl.col("geometry").shift(1).over("shape_id"),
        )
        .with_columns(
            seg_m=(
                pl.when(pl.col("prev").is_null())
                .then(pl.lit(0.0))
                .otherwise(pl.col("geometry").st.distance(pl.col("prev")))
            )
        )
        .with_columns(cum_m=pl.col("seg_m").cum_sum().over("shape_id"))
        # Convert distances to feed units
        .with_columns(shape_dist_traveled=convert_dist(pl.col("cum_m")))
        # Clean up
        .drop("geometry", "prev", "seg_m", "cum_m")
    )
    return feed


def geometrize_shapes(
    shapes: pl.DataFrame | pl.LazyFrame, *, use_utm: bool = False
) -> st.GeoLazyFrame:
    """
    Given a GTFS shapes DataFrame, convert it to a GeoDataFrame of LineStrings
    and return the result, which will no longer have the columns
    ``'shape_pt_sequence'``, ``'shape_pt_lon'``,
    ``'shape_pt_lat'``, and ``'shape_dist_traveled'``.

    If ``use_utm``, then use local UTM coordinates for the geometries.
    """
    f = (
        hp.make_lazy(shapes)
        .sort("shape_id", "shape_pt_sequence")
        .group_by("shape_id", maintain_order=True)
        .agg(
            coords=pl.concat_list("shape_pt_lon", "shape_pt_lat"),
            n=pl.len(),
        )
    )
    lines = (
        f
        .filter(pl.col("n") >= 2)
        .with_columns(geometry=st.linestring("coords").st.set_srid(4326))
        .select("shape_id", "geometry")
    )
    defunct_lines = (
        f
        .filter(pl.col("n") == 1)
        .with_columns(
            geometry=st.linestring(pl.concat_list([pl.col("coords"), pl.col("coords")])).st.set_srid(4326)
        )
        .select("shape_id", "geometry")
    )
    g = pl.concat([defunct_lines, lines])

    if use_utm:
        g = hp.to_srid(g, hp.get_utm_srid(g))

    return g


def ungeometrize_shapes(shapes_g: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    The inverse of :func:`geometrize_shapes`.

    If ``shapes_g`` is in UTM coordinates (has a UTM SRID),
    convert those coordinates back to WGS84 (EPSG:4326), which is the
    standard for a GTFS shapes table.
    """
    return (
        hp.make_lazy(shapes_g)
        # Reproject to WGS84
        .with_columns(geometry=pl.col("geometry").st.to_srid(cs.WGS84))
        .select("shape_id", coords=pl.col("geometry").st.coordinates())
        .explode("coords")
        .with_columns(
            # coords is now a list [x, y] per row → extract scalars
            shape_pt_lon=pl.col("coords").list.get(0).cast(pl.Float64),
            shape_pt_lat=pl.col("coords").list.get(1).cast(pl.Float64),
        )
        .drop("coords")
        # Build 0-based sequence per shape_id in explode order
        .with_row_index("rc")
        .with_columns(
            shape_pt_sequence=(pl.col("rc") - pl.col("rc").min().over("shape_id"))
        )
        .drop("rc")
        .with_columns(pl.col("shape_id").cast(pl.Utf8))
        .select("shape_id", "shape_pt_sequence", "shape_pt_lon", "shape_pt_lat")
    )


def get_shapes(
    feed: "Feed", *, as_geo: bool = False, use_utm: bool = False
) -> pl.LazyFrame | None:
    """
    Get the shapes DataFrame for the given feed, which could be ``None``.
    If ``as_geo``, then return it as GeoDataFrame with a 'geometry' column
    of linestrings and no 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat',
    'shape_dist_traveled' columns.
    The GeoDataFrame will have a UTM CRS if ``use_utm``; otherwise it will have a
    WGS84 CRS.
    """
    f = feed.shapes
    if f is not None and as_geo:
        f = geometrize_shapes(f, use_utm=use_utm)
    return f


def build_geometry_by_shape(
    feed: "Feed", shape_ids: Iterable[str] | None = None, *, use_utm: bool = False
) -> dict:
    """
    Return a dictionary of the form
    <shape ID> -> <Shapely LineString representing shape>.
    If the Feed has no shapes, then return the empty dictionary.
    If ``use_utm``, then use local UTM coordinates; otherwise, use WGS84 coordinates.
    """
    if feed.shapes is None:
        return dict()

    g = get_shapes(feed, as_geo=True, use_utm=use_utm).with_columns(
        geometry=pl.col("geometry").st.to_shapely()
    )
    if shape_ids is not None:
        g = g.filter(pl.col("shape_id").is_in(shape_ids))
    return dict(g.select("shape_id", "geometry").collect().rows())


def shapes_to_geojson(feed: "Feed", shape_ids: Iterable[str] | None = None) -> dict:
    """
    Return a GeoJSON FeatureCollection of LineString features
    representing ``feed.shapes``.
    If the Feed has no shapes, then the features will be an empty list.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    If an iterable of shape IDs is given, then subset to those shapes.
    If the subset is empty, then return a FeatureCollection with an empty list of
    features.
    """
    g = get_shapes(feed, as_geo=True)
    if shape_ids is not None:
        g = g.filter(pl.col("shape_id").is_in(shape_ids))
    if g is None or hp.is_empty(g):
        result = {
            "type": "FeatureCollection",
            "features": [],
        }
    else:
        result = g.collect().st.__geo_interface__

    return result


def get_shapes_intersecting_geometry(
    feed: "Feed",
    geometry: sg.base.BaseGeometry,
    shapes_g: st.GeoDataFrame | st.GeoLazyFrame | None = None,
    *,
    as_geo: bool = False,
) -> pl.LazyFrame | None:
    """
    If the Feed has no shapes, then return None.
    Otherwise, return the subset of ``feed.shapes`` that contains all shapes that
    intersect the given Shapely WGS84 geometry, e.g. a Polygon or LineString.

    If ``as_geo``, then return the shapes as a GeoDataFrame.
    Specifying ``shapes_g`` will skip the first step of the
    algorithm, namely, geometrizing ``feed.shapes``.
    """
    if feed.shapes is None:
        return None

    if shapes_g is not None:
        g = hp.to_lazy(shapes_g)
    else:
        g = get_shapes(feed, as_geo=True)

    cols = g.collect_schema().names()

    # Convert geometry to WKB to please Polars ST
    wkb = shapely.wkb.dumps(geometry)
    g = (
        g
        .with_columns(
            hit = pl.col("geometry").st.intersects(
                st.from_wkb(pl.lit(wkb).cast(pl.Binary)).st.set_srid(4326)
            )
        )
        .filter(pl.col("hit"))
        .select(cols)
    )
    if as_geo:
        result = g
    else:
        result = ungeometrize_shapes(g)

    return result

# This will work once Polars ST or GEOS gets a ``split`` function
def split_simple_broken(
    shapes_g: st.GeoDataFrame | st.GeoLazyFrame, segmentize_m: float = 5.0
) -> st.GeoLazyFrame:
    """
    Given GTFS shapes as a GeoDataFrame of the form output by :func:`geometrize_shapes`
    and possibly in a non-WGS84 CRS,
    split each non-simple LineString into maximal simple (non-self-intersecting)
    sub-LineStrings, and leave the simple LineStrings as is.

    Return a GeoDataFrame in the CRS of ``shapes_g`` with the columns

    - ``'shape_id'``: GTFS ID of the original LineString L
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
    shapes_g = hp.make_lazy(shapes_g)
    orig_srid = hp.get_srid(shapes_g)
    utm_srid = hp.get_utm_srid(shapes_g)

    # Work in UTM (meters) and detect simple vs non-simple
    g = hp.to_srid(shapes_g, utm_srid).with_columns(is_simple=st.geom().st.is_simple())

    final_cols = [
        "subshape_id",
        "subshape_sequence",
        "shape_id",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    ]

    # Simple shapes don't need splitting
    g0 = (
        g.filter(pl.col("is_simple"))
        .with_columns(
            subshape_sequence=pl.lit(0, dtype=pl.Int32),
            subshape_length_m=st.geom().st.length(),
            cum_length_m=st.geom().st.length(),
            subshape_id=(pl.col("shape_id").cast(pl.Utf8) + pl.lit("-0")),
        )
        .select(final_cols)
    )

    # Split the non-simple shapes at self-intersections with GEOS's ``node`` command
    g1 = (
        g.filter(~pl.col("is_simple"))
        .with_columns(Ln = st.geom().st.node())  # fully noded MultiLineString
        # Split original linestring by the noded linework -> MultiLineString of path segments
        .with_columns(parts = st.split(pl.col("geometry"), pl.col("Ln")))
        .select("shape_id", "parts", geom_orig = pl.col("geometry"))
        .explode("parts")
        .rename({"parts": "geometry"})
        # Drop degenerate segments
        .with_columns(len_m = st.geom().st.length())
        .filter(pl.col("len_m") > 0)
        # Order by position along the original path
        .with_columns(
            start0 = pl.col("geometry").st.get_point(0),
            npts   = pl.col("geometry").st.count_points(),
            end0   = pl.col("geometry").st.get_point(pl.col("npts") - 1),
            t0     = pl.col("geom_orig").st.project(pl.col("start0")),
            t1     = pl.col("geom_orig").st.project(pl.col("end0")),
        )
        .sort(by=["shape_id", "t0", "t1", "len_m"])
        .with_columns(subshape_sequence=pl.row_index().over("shape_id").cast(pl.Int32))
        .drop("start0", "end0", "npts", "t0", "t1", "len_m", "geom_orig")
        .with_columns(
            subshape_length_m=st.geom().st.length(),
            subshape_id=(
                pl.col("shape_id").cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("subshape_sequence").cast(pl.Utf8)
            ),
        )
        .with_columns(
            cum_length_m=pl.col("subshape_length_m").cum_sum().over("shape_id")
        )
        .select(final_cols)
    )

    # Concat and project back to the original CRS
    return (
        pl.concat([g0, g1])
        .sort(["shape_id", "subshape_sequence"])
        .pipe(hp.to_srid, orig_srid)
    )

def self_intersection_points(ls: sg.LineString) -> sg.MultiPoint | None:
    """
    Return MultiPoint of self-intersection / self-touch vertices of a LineString.
    We node() the line, then collect vertices whose degree >= 3.
    """
    ln = shapely.node(ls)  # MultiLineString with vertices inserted at true intersections
    deg: dict[tuple[float, float], int] = {}
    if hasattr(ln, "geoms"):
        geoms = ln.geoms
    else:  # degenerate
        geoms = (ln,)
    for seg in geoms:
        c0 = seg.coords[0]
        c1 = seg.coords[-1]
        deg[c0] = deg.get(c0, 0) + 1
        deg[c1] = deg.get(c1, 0) + 1
    pts = [sg.Point(xy) for xy, d in deg.items() if d >= 3]
    return sg.MultiPoint(pts) if pts else None


# TODO: Fix to get working
def split_simple(shapes_g: st.GeoLazyFrame | st.GeoDataFrame) -> st.GeoLazyFrame:
    """
    Given GTFS shapes as a GeoDataFrame of the form output by :func:`geometrize_shapes`
    and possibly in a non-WGS84 CRS,
    split each non-simple LineString into maximal simple (non-self-intersecting)
    sub-LineStrings, and leave the simple LineStrings as is.
    Before splitting, segmentize (with Shapely's ``segmentize`` method)
    each non-simple LineString L by ``segmentize_m`` meters,
    which also sets the maximum gap size between L's simple sub-LineStrings.

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
    import shapely
    import shapely.ops as so

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

    def get_split_points(ls: sg.LineString) -> sg.MultiPoint | None:
        """
        Take all endpoints of the edges in node(ls), drop the original endpoints,
        snap/dedupe, return MultiPoint (or None). Splitting the original line on these
        points yields atomic simple edges along the original traversal.
        """
        if not isinstance(ls, sg.LineString) or len(ls.coords) < 3:
            return None

        # Node the line (adds vertices at all true intersections/touches)
        noded = shapely.node(ls)  # -> MultiLineString of atomic edges
        edges = getattr(noded, "geoms", (noded,))

        # Collect all edge endpoints
        pts = []
        for e in edges:
            pts.append(sg.Point(e.coords[0]))
            pts.append(sg.Point(e.coords[-1]))

        if not pts:
            return None

        # Exclude the original start/end to avoid zero-length splits
        start = sg.Point(ls.coords[0])
        end = sg.Point(ls.coords[-1])
        pts = [p for p in pts if not (p.equals(start) or p.equals(end))]

        if not pts:
            return None

        # Snap tolerance scaled to geometry length, with a small floor
        snap_tol = max(ls.length * 1e-9, 1e-12)
        snapped = so.snap(sg.MultiPoint(pts), ls, snap_tol)
        items = list(getattr(snapped, "geoms", [snapped]))

        # Deduplicate
        seen, uniq = set(), []
        for p in items:
            key = (round(p.x, 12), round(p.y, 12))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)

        return sg.MultiPoint(uniq) if uniq else None

    # Split the non-simple shapes at their self-intersection points
    g1 = (
        g.loc[lambda df: ~df["is_simple"], ["shape_id", "geometry"]]
        .assign(
            split_points=lambda df: df.geometry.apply(get_split_points),
            geometry0=lambda df: df.geometry,
        )
        .assign(
            parts=lambda df: df.apply(
                lambda r: (
                    list(so.split(r["geometry0"], r["split_points"]).geoms)
                    if r["split_points"] and not r["split_points"].is_empty
                    else [r["geometry0"]]
                ),
                axis=1,
            )
        )
        .explode("parts", ignore_index=False)
        .assign(geometry=lambda x: x["parts"])
        .set_geometry("geometry")
        # Drop degenerate/zero-length slivers
        .assign(len_m=lambda df: df.length)
        .query("len_m > 0")
        # Project endpoints along the original path to order segments
        .assign(
            start0=lambda df: df.geometry.apply(lambda ls: sg.Point(ls.coords[0])),
            end0=lambda df: df.geometry.apply(lambda ls: sg.Point(ls.coords[-1])),
        )
        .assign(
            t0=lambda df: df.apply(
                lambda r: r["geometry0"].project(r["start0"]), axis=1
            ),
            t1=lambda df: df.apply(lambda r: r["geometry0"].project(r["end0"]), axis=1),
        )
        .sort_values(["shape_id", "t0", "t1", "len_m"])
        .assign(subshape_sequence=lambda df: df.groupby("shape_id").cumcount())
        .assign(
            subshape_length_m=lambda df: df.length,
            subshape_id=lambda df: df["shape_id"].astype(str)
            + "-"
            + df["subshape_sequence"].astype(str),
            cum_length_m=lambda df: df.groupby("shape_id")["subshape_length_m"].cumsum(),
        )
        .reset_index(drop=True)
        .filter(final_cols)
    )
    result = (
        pd.concat([g0, g1])
        .sort_values(["shape_id", "subshape_sequence"], ignore_index=True)
    )
    # Convert to Polars ST
    return st.from_geopandas(result).with_columns(st.geom().st.set_srid(cs.WGS84).alias("geometry")).lazy().pipe(hp.to_srid, srid)

