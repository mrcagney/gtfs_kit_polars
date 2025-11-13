"""
Functions about shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import polars as pl
import polars_st as st
import shapely as sl
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp

# Help type checkers but avoid circular imports
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

    lon, lat = (
        feed.shapes.limit(1).select("shape_pt_lon", "shape_pt_lat").collect().row(0)
    )
    utm_srid = hp.get_utm_srid_0(lon, lat)
    convert_dist = hp.get_convert_dist("m", feed.dist_units)
    feed.shapes = (
        # Build point geometries in WGS84 then convert to UTM
        feed.shapes.sort("shape_id", "shape_pt_sequence")
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
    Given a GTFS shapes table, convert it to a geotable of LineStrings
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
        f.filter(pl.col("n") >= 2)
        .with_columns(geometry=st.linestring("coords").st.set_srid(4326))
        .select("shape_id", "geometry")
    )
    defunct_lines = (
        f.filter(pl.col("n") == 1)
        .with_columns(
            geometry=st.linestring(
                pl.concat_list([pl.col("coords"), pl.col("coords")])
            ).st.set_srid(4326)
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
    Get the shapes table for the given feed, which could be ``None``.
    If ``as_geo``, then return it as geotable with a 'geometry' column
    of LineStrings and no 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat',
    'shape_dist_traveled' columns.
    The geotable will have a UTM SRID if ``use_utm``; otherwise it will have a
    WGS84 SRID.
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
    shapes_g: st.GeoDataFrame | st.GeoLazyFrame = None,
    *,
    as_geo: bool = False,
) -> st.GeoLazyFrame | None:
    """
    If the Feed has no shapes, then return None.
    Otherwise, return the subset of ``feed.shapes`` that contains all shapes that
    intersect the given Shapely WGS84 geometry, e.g. a Polygon or LineString.

    If ``as_geo``, then return the shapes as a geotable.
    Specifying ``shapes_g`` will skip the first step of the
    algorithm, namely, geometrizing ``feed.shapes``.
    """
    if feed.shapes is None:
        return None

    if shapes_g is not None:
        g = shapes_g
    else:
        g = get_shapes(feed, as_geo=True)

    wkb = sl.wkb.dumps(geometry)
    g = g.filter(
        pl.col("geometry").st.intersects(
            st.from_wkb(pl.lit(wkb).cast(pl.Binary)).st.set_srid(cs.WGS84)
        )
    )
    if as_geo:
        result = g
    else:
        result = ungeometrize_shapes(g)

    return result


def split_simple_0(ls: sg.LineString) -> list[sg.LineString]:
    """
    Split the given LineString into simple sub-LineStrings by greedily building
    the segments from the curve points and binary search,
    checking for simplicity at every step.
    """
    coords = ls.coords
    segments = []
    n = len(coords)
    i = 0
    while i < n:
        # If only one coordinate remains, break  to
        # avoids making a degenerate LineString
        if i == n - 1:
            break
        # Start a binary search with at least two points
        lo, hi = i + 1, n - 1
        best = i + 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = sg.LineString(coords[i : mid + 1])
            if candidate.is_simple:
                best = mid  # candidate is simple; try extending further.
                lo = mid + 1
            else:
                hi = mid - 1
        segments.append(sg.LineString(coords[i : best + 1]))
        if best == n - 1:
            break
        # Start next segment from current best segment
        i = best
    return segments


def split_simple(shapes_g: st.GeoLazyFrame | st.GeoDataFrame) -> st.GeoLazyFrame:
    """
    Given a geotable of GTFS shapes of the form output by :func:`geometrize_shapes`
    with possibly non-WGS84 coordinates,
    split each non-simple LineString into large simple (non-self-intersecting)
    sub-LineStrings, and leave the simple LineStrings as is.

    Return a geotable in the coordinates of ``shapes_g`` with the columns

    - ``'shape_id'``: GTFS shape ID for a LineString L
    - ``'subshape_id'``: a unique identifier of a simple sub-LineString S of L
    - ``'subshape_sequence'``: integer; indicates the order of S when joining up
      all simple sub-LineStrings to form L
    - ``'subshape_length_m'``: the length of S in meters
    - ``'cum_length_m'``: the length S plus the lengths of sub-LineStrings of L
      that come before S; in meters
    - ``'geometry'``: LineString geometry corresponding to S

    Within each 'shape_id' group, the subshapes will be sorted increasingly by
    'subshape_sequence'.

    Notes
    -----
    - Simplicity checks and splitting are done in local UTM coordinates.
      Converting back to original coordinates can introduce rounding errors and
      non-simplicities.
      So test this function with a ``shapes_g`` in local UTM coordinates.
    - By construction, for each given LineString L with simple sub-LineStrings
      S_i, we have the inequality

        sum over i of length(S_i) <= length(L),

      where the lengths are expressed in meters.

    """
    # Log SRIDs
    orig_srid = hp.get_srid(shapes_g)
    utm_srid = hp.get_utm_srid(shapes_g)

    # Work in local UTM coordinates for meter-based operations and simplicity detection
    g = hp.to_srid(hp.make_lazy(shapes_g), utm_srid).with_columns(
        is_simple=st.geom().st.is_simple()
    )

    final_cols = [
        "shape_id",
        "subshape_id",
        "subshape_sequence",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    ]

    # Simple shapes remain unchanged
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

    # Non-simple shapes need splitting
    def split_group(df: pl.DataFrame) -> pl.DataFrame:
        shape_id = df["shape_id"][0]
        ls = sl.from_wkb(df["geometry"][0])  # Convert to Shapely
        parts = [sl.to_wkb(p) for p in split_simple_0(ls)]  # Convert back to WKB
        return pl.DataFrame(
            {
                "shape_id": [shape_id] * len(parts),
                "subshape_sequence": list(range(len(parts))),
                "geometry": parts,
            }
        )

    schema = g.select("shape_id", "geometry").collect_schema()
    g1 = (
        g.filter(~pl.col("is_simple"))
        .select("shape_id", "geometry")
        .group_by("shape_id")
        .map_groups(
            split_group,
            schema=pl.Schema(
                {
                    "shape_id": schema["shape_id"],
                    "subshape_sequence": pl.Int64,
                    "geometry": schema["geometry"],  # usually Binary/WKB
                }
            ),
        )
        .with_columns(geometry=st.from_wkb("geometry").st.set_srid(utm_srid))
        .with_columns(
            subshape_length_m=st.geom().st.length(),
            subshape_id=(
                pl.col("shape_id").cast(pl.Utf8)
                + pl.lit("-")
                + pl.col("subshape_sequence").cast(pl.Utf8)
            ),
        )
        .with_columns(
            cum_length_m=pl.col("subshape_length_m").cum_sum().over("shape_id"),
        )
        .select(final_cols)
    )

    # Collate and convert back to original coordinates
    return (
        pl.concat([g0, g1], how="vertical_relaxed")
        .sort(["shape_id", "subshape_sequence"])
        .pipe(hp.to_srid, orig_srid)
    )
