import json

import geopandas as gpd
import polars as pl
import polars_st as st
import pytest
import shapely.geometry as sg

from gtfs_kit_polars import constants as gkc
from gtfs_kit_polars import helpers as gkh
from gtfs_kit_polars import shapes as gks

from .context import DATA_DIR, cairns, cairns_shapeless, gtfs_kit_polars


def test_append_dist_to_shapes():
    feed1 = cairns.copy()
    s1 = feed1.shapes.collect()
    feed2 = gks.append_dist_to_shapes(feed1)
    s2 = feed2.shapes.collect()
    # Check columns are correct
    assert set(s2.columns) == set(s1.columns) | {"shape_dist_traveled"}

    # Check that within each trip the shape_dist_traveled column
    # is monotonically increasing
    for group in s2.partition_by("shape_id"):
        sdt = group["shape_dist_traveled"].to_list()
        assert sdt == sorted(sdt)


def test_geometrize_shapes():
    shapes = cairns.shapes.collect()
    geo_shapes = gks.geometrize_shapes(shapes).collect()
    # Should have the correct num rows
    assert geo_shapes.height == shapes["shape_id"].n_unique()
    # Should have the correct columns
    assert set(geo_shapes.columns) == (set(shapes.columns) | {"geometry"}) - {
        "shape_pt_lon",
        "shape_pt_lat",
        "shape_pt_sequence",
        "shape_dist_traveled",
    }
    # Should have correct SRID
    assert gkh.get_srid(geo_shapes) == gkc.WGS84

    # A shape with only one point should work
    shapes = cairns.shapes.collect().head(1)
    geo_shapes = gks.geometrize_shapes(shapes)
    print(geo_shapes.collect())
    # Should have correct SRID
    assert gkh.get_srid(geo_shapes) == gkc.WGS84


def test_ungeometrize_shapes():
    shapes = cairns.shapes.collect()
    geo_shapes = gks.geometrize_shapes(shapes).collect()
    shapes2 = gks.ungeometrize_shapes(geo_shapes).collect()

    # Test columns are correct
    assert set(shapes2.columns) == set(list(shapes.columns)) - set(["shape_dist_traveled"])

    # Data frames should agree on certain columns
    cols = ["shape_id", "shape_pt_lon", "shape_pt_lat"]
    assert shapes2.select(cols).equals(shapes.select(cols))


def test_get_shapes():
    g = gks.get_shapes(cairns, as_geo=True).collect()
    assert gkh.get_srid(g) == gkc.WGS84
    assert set(g.columns) == {"shape_id", "geometry"}
    assert gks.get_shapes(cairns_shapeless, as_geo=True) is None



def test_build_geometry_by_shape():
    d = gks.build_geometry_by_shape(cairns)
    assert isinstance(d, dict)
    assert len(d) == cairns.shapes.collect()["shape_id"].n_unique()
    assert gks.build_geometry_by_shape(cairns_shapeless) == {}


def test_shapes_to_geojson():
    feed = cairns
    shape_ids = feed.shapes.collect()["shape_id"].unique().to_list()[:2]
    collection = gks.shapes_to_geojson(feed, shape_ids)
    assert isinstance(collection, dict)
    assert len(collection["features"]) == len(shape_ids)

    assert gks.shapes_to_geojson(cairns_shapeless) == {
        "type": "FeatureCollection",
        "features": [],
    }


def test_get_shapes_intersecting_geometry():
    feed = cairns.copy()
    path = DATA_DIR / "cairns_square_stop_750070.geojson"
    polygon = sg.shape(json.load(path.open())["features"][0]["geometry"])
    pshapes = gks.get_shapes_intersecting_geometry(feed, polygon).collect()
    shape_ids = ["120N0005", "1200010", "1200001"]
    assert set(pshapes["shape_id"].unique()) == set(shape_ids)
    g = gks.get_shapes_intersecting_geometry(feed, polygon, as_geo=True).collect()
    assert gkh.get_srid(g) == gkc.WGS84
    assert set(g["shape_id"].unique()) == set(shape_ids)
    assert gks.get_shapes_intersecting_geometry(cairns_shapeless, polygon) is None

def test_split_simple():
    shapes_g = gks.get_shapes(cairns, as_geo=True, use_utm=True).head(20).collect()
    s = gks.split_simple(shapes_g).collect()
    assert set(s.columns) == {
        "shape_id",
        "subshape_id",
        "subshape_sequence",
        "subshape_length_m",
        "cum_length_m",
        "geometry",
    }
    # Should have some non-simple shapes to start with
    assert not shapes_g.with_columns(is_simple=st.geom().st.is_simple())["is_simple"].all()

    # All sublinestrings of result should be simple
    assert s.with_columns(is_simple=st.geom().st.is_simple())["is_simple"].all()

    # Check each shape group
    for shape_id, group in s.partition_by("shape_id", as_dict=True).items():
        shape_id = shape_id[0]
        ss = shapes_g.filter(pl.col("shape_id") == shape_id).with_columns(length=st.geom().st.length())
        # Each subshape should be shorter than shape
        assert (group["subshape_length_m"] <= ss["length"].sum()).all()
        # Cumulative length should equal shape length within 1%
        L = ss["length"][0]
        assert group["cum_length_m"].max() == pytest.approx(L, rel=0.01)

    # Create a (non-simple) bow-tie
    bowtie = st.GeoDataFrame(
        {
            "shape_id": ["test_shape"],
            "geometry": ["LINESTRING (0 0, 1 1, 0 1, 1 0)"],
        }
    ).with_columns(
        st.geom().st.set_srid(2193).alias("geometry")
    )
    print(bowtie)
    s = gks.split_simple(bowtie).collect()

    # No sub-linestring should have one coordinate
    for geom in s["geometry"].to_list():
        assert len(geom.coords) > 1, f"Found a degenerate one-point LineString: {geom}"

